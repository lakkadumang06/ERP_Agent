"""Smoke test for the eval runner with the fake model (no API keys)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import graph as graph_module
from app.config import Settings
from app.schemas import PurchaseDocument
from evals import run_evals
from tests.conftest import FakeERPModel


@pytest.fixture
def po_001() -> PurchaseDocument:
    data = json.loads(Path("evals/dataset.json").read_text())[0]["expected"]
    return PurchaseDocument(
        doc_type="purchase_order",
        document_number=data["document_number"],
        customer_name=data["customer_name"],
        lines=[
            {"description": "x", "line_total": line["quantity"] * line["unit_price"], **line}
            for line in data["lines"]
        ],
        stated_total=data["stated_total"],
    )


def test_dataset_has_ten_documents_with_expected_codes() -> None:
    data = json.loads(Path("evals/dataset.json").read_text())
    assert len(data) == 10
    assert sum(1 for d in data if d["expected_issue_codes"]) == 4
    assert all(Path(d["file"]).exists() for d in data)


def test_run_evals_end_to_end_with_fake_model(
    monkeypatch, tmp_path, po_001: PurchaseDocument
) -> None:
    fake = FakeERPModel(extracted=po_001)
    monkeypatch.setattr(graph_module, "get_llm", lambda provider=None: fake)
    monkeypatch.setattr(Settings, "has_api_key", lambda self, provider: True)

    output = tmp_path / "results.md"
    assert run_evals.main(["--providers", "anthropic", "--output", str(output)]) == 0
    markdown = output.read_text()
    assert "| anthropic |" in markdown and "Per document: anthropic" in markdown
    # The fake always returns po_001, so that document scores fully and is
    # saved; the summary row must exist with the 4 error / 6 clean denominators.
    assert "| po_001 | 3/3 | 3/3 | - | no |" in markdown
    assert "| anthropic | `" in markdown and "/4 |" in markdown and "/6 |" in markdown


def test_skips_provider_without_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(Settings, "has_api_key", lambda self, provider: False)
    assert run_evals.main(["--providers", "openai", "--output", str(tmp_path / "r.md")]) == 1
