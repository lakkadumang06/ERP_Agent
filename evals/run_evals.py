#!/usr/bin/env python
"""Compare LLM providers on the sample documents.

    python -m evals.run_evals --providers openai anthropic

For every requested provider whose API key is configured, the full graph
(extraction + rule checks + ERP agent) runs on each document in
``evals/dataset.json`` against a fresh seeded database. Runs that pause for
human review are rejected automatically: the eval measures, it never writes
questionable orders.

Metrics per provider
    field accuracy   header fields correct: customer_name, document_number, stated_total
    line accuracy    lines where sku, quantity and unit_price are all correct
    issue detection  error docs flagged with the expected issue code (out of 4)
    false alarms     clean docs that were sent to review (out of 6)
    avg latency      wall-clock seconds per document (first graph invoke)
    total cost       tokens from the models' usage metadata x PRICES (approximate)

Results are printed and written to ``evals/results.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: E402

from app.config import PROVIDERS, configure_logging, get_settings  # noqa: E402
from app.db import configure_engine  # noqa: E402
from app.graph import build_graph, pending_interrupt, resume_document, run_document  # noqa: E402
from app.llm import estimate_cost, model_name_for  # noqa: E402
from app.schemas import Issue, PurchaseDocument  # noqa: E402
from app.seed import seed  # noqa: E402
from app.tools import normalise_name  # noqa: E402

logger = logging.getLogger("evals")

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "dataset.json"
RESULTS_PATH = ROOT / "evals" / "results.md"
TOLERANCE = 0.011


@dataclass
class DocResult:
    """Scores for one document with one provider."""

    doc_id: str
    header_correct: int
    header_total: int
    lines_correct: int
    lines_total: int
    expected_codes: list[str]
    flagged: bool
    detected: bool  # error doc flagged with the right code
    false_alarm: bool  # clean doc sent to review
    issues: list[Issue]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    error: str | None = None


@dataclass
class ProviderResult:
    """Aggregated scores for one provider."""

    provider: str
    model: str
    docs: list[DocResult] = field(default_factory=list)

    def _ratio(self, num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def field_accuracy(self) -> float:
        return self._ratio(
            sum(d.header_correct for d in self.docs), sum(d.header_total for d in self.docs)
        )

    @property
    def line_accuracy(self) -> float:
        return self._ratio(
            sum(d.lines_correct for d in self.docs), sum(d.lines_total for d in self.docs)
        )

    @property
    def error_docs(self) -> int:
        return sum(1 for d in self.docs if d.expected_codes)

    @property
    def clean_docs(self) -> int:
        return sum(1 for d in self.docs if not d.expected_codes)

    @property
    def detected(self) -> int:
        return sum(1 for d in self.docs if d.detected)

    @property
    def false_alarms(self) -> int:
        return sum(1 for d in self.docs if d.false_alarm)

    @property
    def avg_latency(self) -> float:
        return self._ratio(sum(d.latency_s for d in self.docs), len(self.docs))

    @property
    def total_tokens(self) -> int:
        return sum(d.input_tokens + d.output_tokens for d in self.docs)

    @property
    def total_cost(self) -> float | None:
        if any(d.cost_usd is None for d in self.docs):
            return None
        return sum(d.cost_usd or 0.0 for d in self.docs)


def _close(a: float | None, b: float) -> bool:
    return a is not None and abs(a - b) <= TOLERANCE


def score_document(
    doc: PurchaseDocument | None, expected: dict[str, Any]
) -> tuple[int, int, int, int]:
    """Return (header_correct, header_total, lines_correct, lines_total)."""
    header_total, lines_total = 3, len(expected["lines"])
    if doc is None:
        return 0, header_total, 0, lines_total
    header_correct = sum(
        [
            normalise_name(doc.customer_name) == normalise_name(expected["customer_name"]),
            doc.document_number.strip().upper() == expected["document_number"].upper(),
            _close(doc.stated_total, expected["stated_total"]),
        ]
    )
    lines_correct = 0
    for idx, exp in enumerate(expected["lines"]):
        if idx >= len(doc.lines):
            break
        line = doc.lines[idx]
        if (
            (line.sku or "") == exp["sku"]
            and line.quantity == exp["quantity"]
            and _close(line.unit_price, exp["unit_price"])
        ):
            lines_correct += 1
    return header_correct, header_total, lines_correct, lines_total


def evaluate_document(graph: Any, provider: str, item: dict[str, Any]) -> DocResult:
    """Run one document through the graph and score it."""
    usage_cb = UsageMetadataCallbackHandler()
    thread_id = f"eval-{provider}-{item['id']}-{uuid.uuid4().hex[:6]}"
    error: str | None = None
    started = time.perf_counter()
    try:
        result, state = run_document(
            str(ROOT / item["file"]), provider, thread_id, graph=graph, callbacks=[usage_cb]
        )
    except Exception as exc:  # one failing document must not kill the whole eval
        logger.exception("Run failed for %s", item["id"])
        result, state, error = None, {}, str(exc)
    latency = time.perf_counter() - started

    if pending_interrupt(state) is not None:
        resume_document(graph, thread_id, action="reject")  # measure only, never save

    issues = list(result.issues) if result else []
    if result and result.status == "error" and not error:
        error = "; ".join(i.message for i in issues)
    flagged = bool(result and result.status in {"needs_review", "rejected"})
    found_codes = {i.code for i in issues}
    expected_codes = item["expected_issue_codes"]
    header_ok, header_n, lines_ok, lines_n = score_document(
        result.document if result else None, item["expected"]
    )

    usage = usage_cb.usage_metadata  # {model_name: {input_tokens, output_tokens, ...}}
    input_tokens = sum(u.get("input_tokens", 0) for u in usage.values())
    output_tokens = sum(u.get("output_tokens", 0) for u in usage.values())
    cost: float | None = 0.0
    for model, u in usage.items():
        per_model = estimate_cost(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
        cost = None if (per_model is None or cost is None) else cost + per_model

    return DocResult(
        doc_id=item["id"],
        header_correct=header_ok,
        header_total=header_n,
        lines_correct=lines_ok,
        lines_total=lines_n,
        expected_codes=expected_codes,
        flagged=flagged,
        detected=bool(expected_codes) and flagged and all(c in found_codes for c in expected_codes),
        false_alarm=not expected_codes and flagged,
        issues=issues,
        latency_s=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        error=error,
    )


def run_provider(provider: str, dataset: list[dict[str, Any]]) -> ProviderResult:
    """Evaluate every document with ``provider`` against a fresh database."""
    configure_engine(f"sqlite:///{Path(tempfile.mkdtemp()) / f'eval_{provider}.db'}")
    seed()
    graph = build_graph()
    result = ProviderResult(provider=provider, model=model_name_for(provider))
    for item in dataset:
        doc_result = evaluate_document(graph, provider, item)
        result.docs.append(doc_result)
        logger.info(
            "%s/%s: header %d/%d, lines %d/%d, flagged=%s, %.1fs, %d tokens",
            provider,
            item["id"],
            doc_result.header_correct,
            doc_result.header_total,
            doc_result.lines_correct,
            doc_result.lines_total,
            doc_result.flagged,
            doc_result.latency_s,
            doc_result.input_tokens + doc_result.output_tokens,
        )
    return result


def _cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def render_markdown(results: list[ProviderResult]) -> str:
    """Build the Markdown report (summary table + per-document tables)."""
    lines = [
        "# Evaluation results",
        "",
        f"Date: {date.today().isoformat()}. Models: "
        + ", ".join(f"{r.provider} = `{r.model}`" for r in results)
        + ".",
        "",
        "10 documents: 6 clean, 4 with one deliberate error each. Costs are **approximate**, "
        "computed from token usage and the public list prices in `app/llm.py` (`PRICES`).",
        "",
        "| Provider | Model | Field accuracy | Line accuracy | Issue detection | False alarms "
        "| Avg latency | Tokens | Total cost (approx.) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.provider} | `{r.model}` | {r.field_accuracy:.0%} | {r.line_accuracy:.0%} | "
            f"{r.detected}/{r.error_docs} | {r.false_alarms}/{r.clean_docs} | "
            f"{r.avg_latency:.1f}s | {r.total_tokens:,} | {_cost(r.total_cost)} |"
        )
    for r in results:
        lines += [
            "",
            f"## Per document: {r.provider} (`{r.model}`)",
            "",
            "| Document | Header fields | Lines | Expected issue | Flagged | Latency "
            "| Issues / error |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in r.docs:
            notes = d.error or "; ".join(f"{i.code}" for i in d.issues) or "-"
            lines.append(
                f"| {d.doc_id} | {d.header_correct}/{d.header_total} | "
                f"{d.lines_correct}/{d.lines_total} | "
                f"{', '.join(d.expected_codes) or '-'} | {'yes' if d.flagged else 'no'} | "
                f"{d.latency_s:.1f}s | {notes.replace('|', '/')} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ERP-Agent across LLM providers.")
    parser.add_argument(
        "--providers", nargs="+", choices=PROVIDERS, default=["openai", "anthropic"]
    )
    parser.add_argument("--output", default=str(RESULTS_PATH), help="Markdown output path")
    args = parser.parse_args(argv)

    configure_logging()
    settings = get_settings()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    results: list[ProviderResult] = []
    for provider in args.providers:
        if not settings.has_api_key(provider):
            logger.warning("Skipping %s: %s_API_KEY is not set", provider, provider.upper())
            continue
        logger.info(
            "=== Evaluating %s (%s) on %d documents ===",
            provider,
            model_name_for(provider),
            len(dataset),
        )
        results.append(run_provider(provider, dataset))

    if not results:
        logger.error("No provider evaluated. Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env.")
        return 1

    markdown = render_markdown(results)
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(markdown)  # noqa: T201
    print(f"Written to {args.output}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
