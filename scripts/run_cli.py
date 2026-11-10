#!/usr/bin/env python
"""Command-line runner.

    python scripts/run_cli.py data/sample_docs/err_over_stock.txt --provider anthropic

Runs the graph on one document and prints a readable summary. If the run
pauses for review it shows the proposal version and what changed in the ERP
since the proposal ("at proposal vs now"), asks "Approve? [y/N]" and resumes
the graph with that version and hash. If the ERP changed, or the proposal
expired, the graph produces the next version and the question is asked again.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

from app.config import PROVIDERS, configure_logging, get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.graph import build_graph, pending_interrupt, resume_document, run_document  # noqa: E402
from app.schemas import ProcessResult, RecheckResult  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process an invoice / purchase order with ERP-Agent."
    )
    parser.add_argument("document", help="Path to a .pdf or .txt document")
    parser.add_argument("--provider", choices=PROVIDERS, help="Override LLM_PROVIDER")
    parser.add_argument("--yes", action="store_true", help="Approve automatically (scripting)")
    parser.add_argument("--json", action="store_true", help="Also print the ProcessResult as JSON")
    return parser.parse_args(argv)


def print_summary(result: ProcessResult, log: list[str]) -> None:
    """Print the extracted document, the issues and the node log."""
    doc = result.document
    if doc is not None:
        title = doc.doc_type.replace("_", " ").title()
        print(f"\n{title} {doc.document_number} from {doc.customer_name}")  # noqa: T201
        for line in doc.lines:
            sku = line.sku or "???"
            amounts = f"{line.quantity:>5} x {line.unit_price:>8.2f} = {line.line_total:>9.2f}"
            print(f"  {sku:8} {amounts}  {line.description}")  # noqa: T201
        totals = f"stated {doc.stated_total:.2f} {doc.currency}, computed {doc.computed_total:.2f}"
        print(f"  total: {totals}")  # noqa: T201
    if result.issues:
        print("\nIssues:")  # noqa: T201
        for issue in result.issues:
            print(f"  - [{issue.severity}] {issue.code}: {issue.message}")  # noqa: T201
    print("\nLog:")  # noqa: T201
    for entry in log:
        print(f"  {entry}")  # noqa: T201


def print_recheck(recheck: RecheckResult | None) -> None:
    """Print the proposal version and every value at proposal time vs now."""
    if recheck is None:
        return
    head = f"Proposal v{recheck.proposal_version} {recheck.proposal_hash} ({recheck.snapshot_at})"
    print(f"\n{head}, expires {recheck.expires_at}")  # noqa: T201
    for row in recheck.values:
        label = f"{row.label} {row.sku}" if row.sku else row.label
        mark = "  CHANGED" if row.changed else ""
        print(  # noqa: T201
            f"  {label:28} {str(row.at_proposal):>12} -> {str(row.now):>12}  [{row.source}]{mark}"
        )
    if recheck.expired:
        print("  The proposal has expired and must be refreshed.")  # noqa: T201
    for item in recheck.drift:
        print(f"  - [{item.severity}] {item.message}")  # noqa: T201


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    init_db()
    settings = get_settings()
    provider = args.provider or settings.llm_provider
    print(f"Provider: {provider} ({settings.model_for(provider)})")  # noqa: T201

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    result, state = run_document(args.document, provider, thread_id, graph=graph)
    print_summary(result, state.get("log", []))

    while pending_interrupt(state) is not None:
        print_recheck(result.recheck)
        recheck, proposal = result.recheck, result.proposal
        if recheck is not None and recheck.needs_refresh:
            if args.yes or input("\nRefresh the proposal? [y/N] ").strip().lower() in {"y", "yes"}:
                action = "refresh"
            else:
                action = "reject"
        else:
            approve = args.yes or input("\nApprove? [y/N] ").strip().lower() in {"y", "yes"}
            action = "approve" if approve else "reject"
        result, state = resume_document(
            graph,
            thread_id,
            action=action,
            proposal_version=proposal.version if proposal else None,
            proposal_hash=proposal.hash if proposal else None,
            actor="cli",
        )
        print("\nAfter review:")  # noqa: T201
        for entry in state.get("log", [])[-3:]:
            print(f"  {entry}")  # noqa: T201

    print(
        f"\nStatus: {result.status}" + (f" (order id {result.order_id})" if result.order_id else "")
    )  # noqa: T201
    if args.json:
        print(result.model_dump_json(indent=2))  # noqa: T201
    return 0 if result.status in {"saved", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
