#!/usr/bin/env python
"""Command-line runner.

    python scripts/run_cli.py data/sample_docs/err_over_stock.txt --provider anthropic

Runs the graph on one document and prints a readable summary. If the run
pauses for review it asks "Approve? [y/N]" and resumes the graph.
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
from app.schemas import ProcessResult  # noqa: E402


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

    if pending_interrupt(state) is not None:
        approve = args.yes or input("\nApprove? [y/N] ").strip().lower() in {"y", "yes"}
        result, state = resume_document(graph, thread_id, approve)
        print("\nAfter review:")  # noqa: T201
        for entry in state.get("log", [])[-2:]:
            print(f"  {entry}")  # noqa: T201

    print(
        f"\nStatus: {result.status}" + (f" (order id {result.order_id})" if result.order_id else "")
    )  # noqa: T201
    if args.json:
        print(result.model_dump_json(indent=2))  # noqa: T201
    return 0 if result.status in {"saved", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
