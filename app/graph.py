"""LangGraph workflow: load -> extract -> rule_check -> erp_check -> route -> ...

::

    START -> load_document -> extract -> rule_check -> erp_check -> route
        no issues     -> save_order -> END          (status "saved")
        issues        -> human_review (interrupt)   (status "needs_review")
                           approve v<n> -> save_order          (status "saved")
                           reject  -> END                      (status "rejected")
                           refresh, expired, or ERP changed
                                   -> rule_check -> erp_check  (proposal v<n+1>)
                                   -> human_review              (status "needs_re_review")
        save_order finds the ERP changed inside the transaction
                      -> rule_check -> erp_check -> human_review (v<n+1>)
        extraction failed            -> END          (status "error")

The graph is compiled with ``MemorySaver`` so a run paused at ``human_review``
can be resumed later by ``thread_id`` with ``Command(resume={...})``.

Proposal versions: ``erp_check`` takes a snapshot (every value with source
and read time, a version number and a hash). ``human_review`` shows "at
proposal vs now". An approval must name the version and hash it is for;
``save_order`` re-reads price, customer and stock inside the save
transaction and, if anything differs, marks the version stale, re-runs the
checks and creates the next version for the approver. Only deterministic
code writes to the database.
"""

from __future__ import annotations

import json
import logging
import operator
import re
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from app import audit
from app.config import get_settings
from app.extract import extract_document, load_document
from app.llm import get_llm, model_name_for
from app.recheck import compare_with_current, take_snapshot
from app.schemas import (
    AgentVerdict,
    ErpSnapshot,
    Issue,
    ProcessResult,
    ProcessStatus,
    ProposalInfo,
    PurchaseDocument,
    RecheckResult,
)
from app.snapshot import expires_at, is_expired, values_as_dict
from app.tools import (
    AGENT_TOOLS,
    StaleProposalError,
    check_stock,
    create_order,
    find_customer,
    get_product_price,
)
from app.validate import run_all_checks

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 8
SKU_PATTERN = re.compile(r"\b[A-Z]{3}-\d{3}\b")

# Our Pydantic models travel through the checkpointer; register them explicitly
# so LangGraph's msgpack serializer accepts them without a warning.
CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[("app.schemas", "PurchaseDocument"), ("app.schemas", "Issue")]
)
PRICE_TOLERANCE_RATIO = 0.005  # 0.5 %: absorbs rounding of printed prices

ReviewAction = Literal["approve", "reject", "refresh"]


class GraphState(TypedDict, total=False):
    """State carried through the graph.

    ``log`` uses an additive reducer so every node appends a breadcrumb
    instead of overwriting the list.
    """

    file_path: str | None
    raw_text: str
    provider: str | None
    document: PurchaseDocument | None
    issues: list[Issue]
    customer_id: int | None
    needs_human: bool
    approved: bool | None
    order_id: int | None
    status: ProcessStatus | None
    erp_snapshot: dict[str, Any] | None  # ErpSnapshot of the current proposal (plain dict)
    proposal_version: int  # 0 before erp_check ran, then 1, 2, ...
    stale_reason: str | None  # why the previous version was marked stale
    refresh_pending: bool  # True when the next node run must produce a new proposal version
    recheck: dict[str, Any] | None  # last RecheckResult (plain dict)
    approver: str | None  # who approved the current version
    log: Annotated[list[str], operator.add]


# --------------------------------------------------------------------------
# erp_check helpers
# --------------------------------------------------------------------------


def erp_lookups(doc: PurchaseDocument) -> tuple[int | None, list[Issue]]:
    """Deterministic ERP lookups through the same tools the agent uses.

    This is the safety net under the LLM step: customer, SKU, stock and price
    are checked exactly, so a bad model answer cannot hide a real problem.
    Returns the matched customer id (or ``None``) and the issues found.
    """
    issues: list[Issue] = []
    customer = find_customer.invoke({"name": doc.customer_name})
    customer_id: int | None = customer["id"] if customer["found"] else None
    if customer_id is None:
        closest = ", ".join(customer.get("candidates", [])) or "none"
        issues.append(
            Issue(
                code="unknown_customer",
                message=f"Customer '{doc.customer_name}' not found in ERP (closest: {closest}).",
            )
        )
    else:
        status = str(customer.get("status") or "active").lower()
        if status != "active":
            issues.append(
                Issue(
                    code="customer_blocked",
                    message=f"Customer '{customer['name']}' has status '{status}' in the ERP.",
                )
            )
        limit = customer.get("credit_limit")
        if limit is not None and doc.computed_total > float(limit):
            issues.append(
                Issue(
                    code="credit_limit_exceeded",
                    message=(
                        f"Order total {doc.computed_total:.2f} exceeds the credit limit "
                        f"{float(limit):.2f} of '{customer['name']}'."
                    ),
                    severity="warning",
                )
            )

    for idx, line in enumerate(doc.lines, start=1):
        if line.sku is None:
            continue  # reported as missing_sku by rule_check
        stock = check_stock.invoke({"sku": line.sku, "quantity": line.quantity})
        if not stock["found"]:
            issues.append(
                Issue(
                    code="unknown_sku",
                    message=f"Line {idx}: SKU {line.sku} not found in catalogue.",
                )
            )
            continue
        if not stock["enough"]:
            issues.append(
                Issue(
                    code="insufficient_stock",
                    message=(
                        f"Line {idx}: only {stock['available']} units of {line.sku} in stock, "
                        f"{line.quantity} requested."
                    ),
                )
            )
        list_price = float(get_product_price.invoke({"sku": line.sku})["unit_price"])
        if abs(line.unit_price - list_price) > list_price * PRICE_TOLERANCE_RATIO:
            issues.append(
                Issue(
                    code="price_mismatch",
                    message=(
                        f"Line {idx}: unit price {line.unit_price:.2f} differs from list price "
                        f"{list_price:.2f} for {line.sku}."
                    ),
                    severity="warning",
                )
            )
    return customer_id, issues


AGENT_SYSTEM_PROMPT = """\
You are the validation agent of an ERP system. You receive a purchase order or
invoice extracted from a document, plus findings from automatic checks. Verify
the document against the ERP with the tools:

1. Call find_customer with the customer name.
2. For EVERY line call check_stock(sku, quantity) and get_product_price(sku).
3. Compare prices and descriptions with the catalogue; look for anything the
   automatic checks could have missed (a description that does not fit the
   SKU's product name, an implausible quantity).

Only report problems you actually verified with a tool. When you are done,
reply with a short plain-text summary.
"""

VERDICT_PROMPT = """\
Below is the extracted document, the automatic findings and the log of the ERP
tool calls you made with their results. Return your findings as structured
issues plus the matched customer id. Do not repeat the automatic findings.

## Extracted document
{document}

## Automatic findings
{findings}

## Tool call log
{tool_log}
"""


def run_erp_agent(
    llm: BaseChatModel, doc: PurchaseDocument, known_issues: list[Issue]
) -> AgentVerdict:
    """Tool-calling loop (max 8 iterations) followed by a structured verdict.

    The loop is written by hand so the control flow is explicit: the model is
    bound ONLY to the three read-only tools, each tool call is executed and
    answered with a ``ToolMessage``, and the loop ends when the model stops
    calling tools. A final structured-output call turns the conversation into
    an :class:`AgentVerdict`.
    """
    tools_by_name = {t.name: t for t in AGENT_TOOLS}
    model = llm.bind_tools(AGENT_TOOLS)
    document_json = doc.model_dump_json(indent=2)
    findings = "\n".join(f"- [{i.code}] {i.message}" for i in known_issues) or "- none"
    messages: list[Any] = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Document:\n{document_json}\n\nAutomatic findings:\n{findings}"),
    ]
    tool_log: list[str] = []

    for iteration in range(MAX_AGENT_ITERATIONS):
        response = model.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            tool = tools_by_name.get(call["name"])
            try:
                output: Any = (
                    tool.invoke(call["args"]) if tool else {"error": f"unknown tool {call['name']}"}
                )
            except Exception as exc:  # a failing tool must not kill the run
                output = {"error": str(exc)}
            tool_log.append(f"{call['name']}({json.dumps(call['args'])}) -> {json.dumps(output)}")
            messages.append(ToolMessage(content=json.dumps(output), tool_call_id=call["id"]))
        logger.debug("erp_check iteration %d: %d tool call(s)", iteration + 1, len(tool_calls))

    verdict_prompt = VERDICT_PROMPT.format(
        document=document_json, findings=findings, tool_log="\n".join(tool_log) or "(none)"
    )
    verdict = llm.with_structured_output(AgentVerdict, method="json_schema").invoke(
        [HumanMessage(content=verdict_prompt)]
    )
    return verdict if isinstance(verdict, AgentVerdict) else AgentVerdict.model_validate(verdict)


def _finding_key(issue: Issue) -> tuple[str, str]:
    """Identity of a finding: its code plus the SKU it talks about (if any).

    The agent often re-reports what the exact lookups already found, with
    different wording. Treating "same code, same SKU" as the same finding keeps
    the issue list readable for the reviewer.
    """
    skus = SKU_PATTERN.findall(issue.message.upper())
    return issue.code, skus[0] if skus else ""


def merge_issues(*groups: list[Issue]) -> list[Issue]:
    """Concatenate issue lists, keeping the first occurrence of each finding."""
    seen: set[tuple[str, str]] = set()
    merged: list[Issue] = []
    for group in groups:
        for issue in group:
            key = _finding_key(issue)
            if key not in seen:
                seen.add(key)
                merged.append(issue)
    return merged


def _thread_id(config: RunnableConfig | None) -> str:
    return str(((config or {}).get("configurable") or {}).get("thread_id", "?"))


def _decision(decision: Any) -> dict[str, Any]:
    """Normalise a resume value to ``{action, proposal_version, proposal_hash, actor}``.

    Accepts the structured form ``{"action": "approve", ...}``, the older
    ``{"approve": true}`` / ``True`` / ``"yes"`` forms (treated as approve
    without a version, which is refused), and ``{"action": "refresh"}``.
    """
    if isinstance(decision, dict):
        action = decision.get("action")
        if action not in ("approve", "reject", "refresh"):
            action = "approve" if decision.get("approve") else "reject"
        return {
            "action": action,
            "proposal_version": decision.get("proposal_version"),
            "proposal_hash": decision.get("proposal_hash"),
            "actor": decision.get("actor") or decision.get("approver"),
        }
    if isinstance(decision, str):
        approve = decision.strip().lower() in {"y", "yes", "approve", "true", "1"}
    else:
        approve = bool(decision)
    return {
        "action": "approve" if approve else "reject",
        "proposal_version": None,
        "proposal_hash": None,
        "actor": None,
    }


def _stale(
    state: GraphState,
    thread_id: str,
    snapshot: ErpSnapshot,
    reason: str,
    *,
    event: str = "stale",
    actor: str | None = None,
    current: dict[str, Any] | None = None,
    recheck: RecheckResult | None = None,
) -> dict[str, Any]:
    """Mark the current proposal version stale and send the run back to the checks."""
    audit.record(
        event,
        thread_id=thread_id,
        proposal_version=snapshot.version,
        proposal_hash=snapshot.snapshot_hash,
        actor=actor,
        reason=reason,
        at_proposal=values_as_dict(snapshot.values),
        current=current,
    )
    logger.info("review thread=%s v%s marked %s: %s", thread_id, snapshot.version, event, reason)
    return {
        "approved": False,
        "status": "needs_re_review",
        "stale_reason": reason,
        "refresh_pending": True,
        "recheck": recheck.model_dump() if recheck is not None else state.get("recheck"),
        "log": [f"human_review: v{snapshot.version} {event} ({reason})"],
    }


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


def build_graph(
    llm: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the workflow.

    Args:
        llm: Optional model override (tests inject a fake). When ``None`` the
            model for ``state["provider"]`` is created with :func:`get_llm`.
        checkpointer: Defaults to ``MemorySaver``; swap in a SQLite/Postgres
            saver when review threads must survive a restart.
    """

    def resolve_llm(state: GraphState) -> BaseChatModel:
        return llm if llm is not None else get_llm(state.get("provider"))

    def load_document_node(state: GraphState) -> dict[str, Any]:
        """Read ``file_path`` into ``raw_text`` unless text was supplied directly."""
        if state.get("raw_text"):
            return {"log": ["load_document: using provided text"]}
        path = state.get("file_path")
        if not path:
            return {
                "status": "error",
                "issues": [Issue(code="no_input", message="No document provided.")],
                "log": ["load_document: nothing to load"],
            }
        try:
            text = load_document(path)
        except (FileNotFoundError, ValueError) as exc:
            return {
                "status": "error",
                "issues": [Issue(code="load_failed", message=str(exc))],
                "log": [f"load_document: failed ({exc})"],
            }
        return {"raw_text": text, "log": [f"load_document: {path} ({len(text)} chars)"]}

    def extract_node(state: GraphState) -> dict[str, Any]:
        """LLM structured extraction (one retry inside ``extract_document``)."""
        if state.get("status") == "error":
            return {"log": ["extract: skipped"]}
        try:
            doc = extract_document(state["raw_text"], llm=resolve_llm(state))
        except Exception as exc:
            logger.exception("Extraction failed")
            return {
                "status": "error",
                "issues": [Issue(code="extraction_failed", message=str(exc))],
                "log": [f"extract: failed ({exc})"],
            }
        return {
            "document": doc,
            "log": [f"extract: {doc.doc_type} {doc.document_number}, {len(doc.lines)} lines"],
        }

    def rule_check_node(state: GraphState) -> dict[str, Any]:
        """Deterministic checks from ``app/validate.py`` (no LLM, no DB)."""
        doc = state.get("document")
        if doc is None:
            return {"log": ["rule_check: skipped"]}
        issues = run_all_checks(doc)
        return {"issues": issues, "log": [f"rule_check: {len(issues)} issue(s)"]}

    def erp_check_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        """ERP lookups + the LLM tool-calling agent; results are merged.

        Ends by taking the proposal snapshot: version ``n+1`` of this thread,
        every value with source and read time, and the proposal hash. A
        version above 1 (a refreshed proposal) always goes to a human.
        """
        doc = state.get("document")
        if doc is None:
            return {"log": ["erp_check: skipped"]}
        customer_id, lookup_issues = erp_lookups(doc)
        issues = merge_issues(state.get("issues", []), lookup_issues)
        try:
            verdict = run_erp_agent(resolve_llm(state), doc, issues)
            customer_id = customer_id if customer_id is not None else verdict.customer_id
            issues = merge_issues(issues, verdict.issues)
            note = f"erp_check: agent ran ({verdict.summary}); {len(issues)} issue(s) total"
        except Exception as exc:
            # The agent is an extra pair of eyes; if it fails we keep the exact
            # lookups and flag the run for a human rather than failing silently.
            logger.exception("erp_check agent failed")
            issues = merge_issues(
                issues,
                [
                    Issue(
                        code="agent_unavailable", message=f"Agent failed: {exc}", severity="warning"
                    )
                ],
            )
            note = "erp_check: agent failed, lookups only"
        version = int(state.get("proposal_version") or 0) + 1
        snapshot = take_snapshot(doc, version=version)
        needs_human = bool(issues) or version > 1
        audit.record(
            "proposed",
            thread_id=_thread_id(config),
            proposal_version=version,
            proposal_hash=snapshot.snapshot_hash,
            reason=state.get("stale_reason") if version > 1 else None,
            at_proposal=values_as_dict(snapshot.values),
        )
        return {
            "customer_id": customer_id,
            "issues": issues,
            "needs_human": needs_human,
            "approved": None,
            "approver": None,
            "refresh_pending": False,
            "status": (
                None if not needs_human else "needs_re_review" if version > 1 else "needs_review"
            ),
            "erp_snapshot": snapshot.model_dump(),
            "proposal_version": version,
            "log": [
                f"{note}; proposal v{version} {snapshot.snapshot_hash} at {snapshot.snapshot_at}"
            ],
        }

    def route(state: GraphState) -> Literal["save_order", "human_review", "end"]:
        """Conditional edge: clean documents are saved, the rest wait for a human."""
        if state.get("status") == "error":
            return "end"
        return "human_review" if state.get("needs_human") else "save_order"

    def human_review_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        """Pause until a human resumes the thread, then validate the decision.

        The interrupt payload carries the document, the issues, the proposal
        (version, hash, expiry) and a fresh "at proposal vs now" comparison.
        On resume:

        * ``reject``  -> status ``rejected``.
        * ``refresh`` -> the version is marked stale and the run goes back to
          ``rule_check`` to produce the next version.
        * ``approve`` -> must name the current version and hash; an expired
          or changed proposal is marked stale instead (next version); else
          the run proceeds to ``save_order``, which re-checks inside the
          transaction.
        """
        doc = state["document"]
        assert doc is not None
        snapshot = ErpSnapshot.model_validate(state["erp_snapshot"])
        thread_id = _thread_id(config)
        recheck = compare_with_current(doc, snapshot)
        decision = _decision(
            interrupt(
                {
                    "question": "Approve this order despite the issues?",
                    "document": doc.model_dump(),
                    "issues": [issue.model_dump() for issue in state.get("issues", [])],
                    "proposal": _proposal_info(state, snapshot).model_dump(),
                    **recheck.model_dump(),
                }
            )
        )
        actor = decision["actor"]

        if decision["action"] == "reject":
            audit.record(
                "rejected",
                thread_id=thread_id,
                proposal_version=snapshot.version,
                proposal_hash=snapshot.snapshot_hash,
                actor=actor,
            )
            logger.info("review thread=%s v%s decision=rejected", thread_id, snapshot.version)
            return {"approved": False, "status": "rejected", "log": ["human_review: rejected"]}

        recheck = compare_with_current(doc, snapshot)  # fresh: time passed during the pause
        current = values_as_dict(recheck.values, "now")
        if decision["action"] == "refresh":
            return _stale(
                state,
                thread_id,
                snapshot,
                "refresh requested by approver",
                event="refreshed",
                actor=actor,
                current=current,
                recheck=recheck,
            )

        # approve
        if (
            decision["proposal_version"] != snapshot.version
            or decision["proposal_hash"] != snapshot.snapshot_hash
        ):
            reason = (
                f"approval named v{decision['proposal_version']} {decision['proposal_hash']}, "
                f"current is v{snapshot.version} {snapshot.snapshot_hash}"
            )
            audit.record(
                "stale",
                thread_id=thread_id,
                proposal_version=snapshot.version,
                proposal_hash=snapshot.snapshot_hash,
                actor=actor,
                reason=reason,
                at_proposal=values_as_dict(snapshot.values),
                current=current,
            )
            logger.info("review thread=%s refused: %s", thread_id, reason)
            # Not a new version: the approver simply has to look at the current one.
            return {
                "approved": False,
                "status": "needs_re_review",
                "refresh_pending": False,
                "recheck": recheck.model_dump(),
                "log": [f"human_review: approval refused, {reason}"],
            }
        if recheck.expired:
            return _stale(
                state,
                thread_id,
                snapshot,
                f"proposal expired at {recheck.expires_at}",
                event="expired",
                actor=actor,
                current=current,
                recheck=recheck,
            )
        if recheck.stale or recheck.blocked:
            return _stale(
                state,
                thread_id,
                snapshot,
                f"ERP changed since proposal: {recheck.summary()}",
                actor=actor,
                current=current,
                recheck=recheck,
            )
        audit.record(
            "approved",
            thread_id=thread_id,
            proposal_version=snapshot.version,
            proposal_hash=snapshot.snapshot_hash,
            actor=actor,
            at_proposal=values_as_dict(snapshot.values),
            current=current,
        )
        logger.info(
            "review thread=%s v%s decision=approved by %s", thread_id, snapshot.version, actor
        )
        return {
            "approved": True,
            "approver": actor,
            "status": None,
            "recheck": recheck.model_dump(),
            "log": [f"human_review: v{snapshot.version} approved by {actor or 'anonymous'}"],
        }

    def after_review(
        state: GraphState,
    ) -> Literal["save_order", "human_review", "rule_check", "end"]:
        """Approved orders are saved; stale versions are re-checked; refusals wait."""
        if state.get("approved"):
            return "save_order"
        if state.get("status") != "needs_re_review":
            return "end"
        # A stale/expired/refreshed version gets a new one; a wrong version/hash
        # in the request just re-shows the current proposal.
        return "rule_check" if state.get("refresh_pending") else "human_review"

    def save_order_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        """Write through ``create_order``, which re-checks inside the transaction.

        ``create_order`` locks the customer and product rows, compares price
        and customer data with the proposal snapshot and checks stock. If
        anything differs it raises ``StaleProposalError`` and nothing is
        written: the version is marked stale and the run goes back to the
        checks for the next version. The save is idempotent per
        ``(thread_id, proposal_version)``.
        """
        doc = state["document"]
        customer_id = state.get("customer_id")
        thread_id = _thread_id(config)
        if doc is None or customer_id is None:
            return {
                "status": "error",
                "issues": merge_issues(
                    state.get("issues", []),
                    [
                        Issue(
                            code="save_failed",
                            message="Cannot save: customer is not matched in ERP.",
                        )
                    ],
                ),
                "log": ["save_order: skipped (no customer id)"],
            }
        snapshot = (
            ErpSnapshot.model_validate(state["erp_snapshot"]) if state.get("erp_snapshot") else None
        )
        lines = [
            {"sku": line.sku, "quantity": line.quantity, "unit_price": line.unit_price}
            for line in doc.lines
            if line.sku is not None
        ]
        try:
            order_id = create_order.invoke(
                {
                    "customer_id": customer_id,
                    "lines": lines,
                    "document_number": doc.document_number,
                    "snapshot": snapshot.model_dump() if snapshot else None,
                    "thread_id": thread_id,
                    "proposal_version": snapshot.version if snapshot else None,
                    "approved_by": state.get("approver"),
                }
            )
        except StaleProposalError as exc:
            assert snapshot is not None
            reason = f"ERP changed before save: {'; '.join(d.message for d in exc.drift)}"
            recheck = compare_with_current(doc, snapshot)
            result = _stale(
                state,
                thread_id,
                snapshot,
                reason,
                actor=state.get("approver"),
                current=values_as_dict(exc.current.values),
                recheck=recheck,
            )
            result["log"] = [f"save_order: refused, {reason}"]
            return result
        except Exception as exc:
            return {
                "status": "error",
                "issues": merge_issues(
                    state.get("issues", []),
                    [Issue(code="save_failed", message=f"Save failed: {exc}")],
                ),
                "log": [f"save_order: failed ({exc})"],
            }
        if snapshot is not None:
            audit.record(
                "saved",
                thread_id=thread_id,
                proposal_version=snapshot.version,
                proposal_hash=snapshot.snapshot_hash,
                actor=state.get("approver"),
                order_id=order_id,
                at_proposal=values_as_dict(snapshot.values),
            )
        return {
            "order_id": order_id,
            "status": "saved",
            "log": [f"save_order: created order {order_id}"],
        }

    def after_save(state: GraphState) -> Literal["rule_check", "end"]:
        """A stale save goes back to the checks for the next proposal version."""
        return "rule_check" if state.get("status") == "needs_re_review" else "end"

    workflow = StateGraph(GraphState)
    workflow.add_node("load_document", load_document_node)
    workflow.add_node("extract", extract_node)
    workflow.add_node("rule_check", rule_check_node)
    workflow.add_node("erp_check", erp_check_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("save_order", save_order_node)

    workflow.add_edge(START, "load_document")
    workflow.add_edge("load_document", "extract")
    workflow.add_edge("extract", "rule_check")
    workflow.add_edge("rule_check", "erp_check")
    workflow.add_conditional_edges(
        "erp_check", route, {"save_order": "save_order", "human_review": "human_review", "end": END}
    )
    workflow.add_conditional_edges(
        "human_review",
        after_review,
        {
            "save_order": "save_order",
            "human_review": "human_review",
            "rule_check": "rule_check",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "save_order", after_save, {"rule_check": "rule_check", "end": END}
    )
    return workflow.compile(checkpointer=checkpointer or MemorySaver(serde=CHECKPOINT_SERDE))


# --------------------------------------------------------------------------
# Public runners (CLI, API, evals)
# --------------------------------------------------------------------------


def _config(
    thread_id: str, provider: str | None = None, callbacks: list[Any] | None = None
) -> dict[str, Any]:
    """LangGraph run config: checkpoint thread + LangSmith tags/metadata."""
    provider = (provider or get_settings().llm_provider).lower()
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": callbacks or [],
        "tags": ["erp-agent", f"provider:{provider}"],
        "metadata": {"provider": provider, "model": model_name_for(provider)},
    }


def pending_interrupt(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the ``interrupt()`` payload if the run is paused, else ``None``."""
    interrupts = state.get("__interrupt__") or []
    return interrupts[0].value if interrupts else None


def recheck_from(mapping: dict[str, Any] | None) -> RecheckResult | None:
    """Rebuild a :class:`RecheckResult` from state or an interrupt payload."""
    if not mapping or "proposal_hash" not in mapping:
        return None
    return RecheckResult.model_validate(
        {key: mapping[key] for key in RecheckResult.model_fields if key in mapping}
    )


def _proposal_info(state: dict[str, Any], snapshot: ErpSnapshot) -> ProposalInfo:
    ttl = get_settings().proposal_ttl_hours
    return ProposalInfo(
        version=snapshot.version,
        hash=snapshot.snapshot_hash,
        created_at=snapshot.snapshot_at,
        expires_at=expires_at(snapshot.snapshot_at, ttl),
        expired=is_expired(snapshot.snapshot_at, ttl),
        stale_reason=state.get("stale_reason") if snapshot.version > 1 else None,
        approved_by=state.get("approver"),
    )


def to_process_result(state: dict[str, Any], thread_id: str) -> ProcessResult:
    """Translate raw graph state into the public :class:`ProcessResult`.

    While the run is paused, the re-check shown to the reviewer comes from the
    interrupt payload (it is fresher than anything stored in the state).
    """
    payload = pending_interrupt(state)
    status: ProcessStatus = state.get("status") or ("needs_review" if payload else "error")
    snapshot = (
        ErpSnapshot.model_validate(state["erp_snapshot"]) if state.get("erp_snapshot") else None
    )
    return ProcessResult(
        status=status,
        order_id=state.get("order_id"),
        thread_id=thread_id,
        document=state.get("document"),
        issues=list(state.get("issues") or []),
        log=list(state.get("log") or []),
        proposal=_proposal_info(state, snapshot) if snapshot else None,
        recheck=recheck_from(payload) or recheck_from(state.get("recheck")),
    )


def run_document(
    path: str | None = None,
    provider: str | None = None,
    thread_id: str | None = None,
    *,
    text: str | None = None,
    graph: CompiledStateGraph | None = None,
    callbacks: list[Any] | None = None,
) -> tuple[ProcessResult, dict[str, Any]]:
    """Run the workflow on a file (``path``) or raw ``text``.

    Returns the public :class:`ProcessResult` and the raw final state (which
    includes ``log`` and, when paused, ``__interrupt__``). A ``thread_id`` is
    generated when not given; it is needed to resume a paused run.
    """
    graph = graph or build_graph()
    thread_id = thread_id or str(uuid.uuid4())
    initial: GraphState = {
        "file_path": path,
        "raw_text": text or "",
        "provider": provider,
        "document": None,
        "issues": [],
        "customer_id": None,
        "needs_human": False,
        "approved": None,
        "order_id": None,
        "status": None,
        "erp_snapshot": None,
        "proposal_version": 0,
        "stale_reason": None,
        "refresh_pending": False,
        "recheck": None,
        "approver": None,
        "log": [],
    }
    state = graph.invoke(initial, config=_config(thread_id, provider, callbacks))
    return to_process_result(state, thread_id), state


def resume_document(
    graph: CompiledStateGraph,
    thread_id: str,
    approve: bool | None = None,
    *,
    action: ReviewAction | None = None,
    proposal_version: int | None = None,
    proposal_hash: str | None = None,
    actor: str | None = None,
) -> tuple[ProcessResult, dict[str, Any]]:
    """Resume a run paused at ``human_review`` with the human's decision.

    ``action`` is ``approve``, ``reject`` or ``refresh`` (``approve=True/False``
    is accepted as a shorthand). An approval must carry the ``proposal_version``
    and ``proposal_hash`` the approver looked at; otherwise, or when the ERP
    changed or the proposal expired, nothing is saved and the result has
    status ``needs_re_review`` with the next proposal version.
    """
    if action is None:
        action = "approve" if approve else "reject"
    checkpoint = graph.get_state({"configurable": {"thread_id": thread_id}})
    provider = checkpoint.values.get("provider") if checkpoint.values else None
    state = graph.invoke(
        Command(
            resume={
                "action": action,
                "proposal_version": proposal_version,
                "proposal_hash": proposal_hash,
                "actor": actor,
            }
        ),
        config=_config(thread_id, provider),
    )
    return to_process_result(state, thread_id), state


def review_view(graph: CompiledStateGraph, thread_id: str) -> ProcessResult | None:
    """Fresh "at proposal vs now" view of a paused run, or ``None`` for an unknown thread.

    Used by ``GET /documents/{thread_id}``: the ERP is re-read now, so an
    approver who opens the queue an hour later sees today's values.
    """
    checkpoint = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not checkpoint.values:
        return None
    state: dict[str, Any] = dict(checkpoint.values)
    doc, snapshot = state.get("document"), state.get("erp_snapshot")
    if checkpoint.next and doc is not None and snapshot:
        fresh = compare_with_current(doc, ErpSnapshot.model_validate(snapshot))
        state["recheck"] = fresh.model_dump()
        state["status"] = state.get("status") or "needs_review"
    return to_process_result(state, thread_id)


def is_waiting(graph: CompiledStateGraph, thread_id: str) -> bool | None:
    """``True`` if the thread is paused for review, ``False`` if finished, ``None`` if unknown."""
    checkpoint = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not checkpoint.values:
        return None
    return bool(checkpoint.next)
