"""End-to-end graph tests with a fake chat model (no network, no API keys)."""

from __future__ import annotations

import uuid

from app.graph import build_graph, erp_lookups, pending_interrupt, resume_document, run_document
from app.schemas import AgentVerdict, Issue, PurchaseDocument
from app.tools import check_stock
from tests.conftest import FakeERPModel


def _run(model: FakeERPModel):
    graph = build_graph(llm=model)
    thread_id = str(uuid.uuid4())
    result, state = run_document(None, None, thread_id, text="dummy document text", graph=graph)
    return result, state, graph, thread_id


def test_erp_lookups_flag_unknown_sku_and_customer(unknown_sku_doc: PurchaseDocument) -> None:
    customer_id, issues = erp_lookups(unknown_sku_doc)
    assert customer_id == 1
    assert [i.code for i in issues] == ["unknown_sku"]
    stranger = unknown_sku_doc.model_copy(update={"customer_name": "Golden Palm Trading LLC"})
    customer_id, issues = erp_lookups(stranger)
    assert customer_id is None
    assert "unknown_customer" in {i.code for i in issues}


def test_erp_lookups_flag_insufficient_stock_and_price(clean_doc: PurchaseDocument) -> None:
    doc = clean_doc.model_copy(deep=True)
    doc.lines[0].quantity = 999  # JWL-001 has 40 in stock
    doc.lines[1].unit_price = 99.0  # list price 12.50
    _, issues = erp_lookups(doc)
    assert {i.code for i in issues} == {"insufficient_stock", "price_mismatch"}


def test_clean_document_ends_saved(clean_doc: PurchaseDocument) -> None:
    result, state, _, _ = _run(FakeERPModel(extracted=clean_doc))
    assert pending_interrupt(state) is None
    assert result.status == "saved" and result.order_id is not None
    assert result.issues == []
    assert check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"] == 40 - 10
    assert [entry.split(":")[0] for entry in state["log"]] == [
        "load_document",
        "extract",
        "rule_check",
        "erp_check",
        "save_order",
    ]


def test_error_document_ends_in_interrupt(wrong_total_doc: PurchaseDocument) -> None:
    result, state, _, _ = _run(FakeERPModel(extracted=wrong_total_doc))
    payload = pending_interrupt(state)
    assert payload is not None
    assert result.status == "needs_review" and result.order_id is None
    assert payload["document"]["document_number"] == "PO-TEST-002"
    assert [i["code"] for i in payload["issues"]] == ["document_total_mismatch"]


def test_approve_resumes_and_saves(wrong_total_doc: PurchaseDocument) -> None:
    _, _, graph, thread_id = _run(FakeERPModel(extracted=wrong_total_doc))
    result, state = resume_document(graph, thread_id, approve=True)
    assert result.status == "saved" and result.order_id is not None
    assert state["approved"] is True


def test_reject_ends_rejected_without_saving(wrong_total_doc: PurchaseDocument) -> None:
    _, _, graph, thread_id = _run(FakeERPModel(extracted=wrong_total_doc))
    result, _ = resume_document(graph, thread_id, approve=False)
    assert result.status == "rejected" and result.order_id is None
    assert check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"] == 40


def test_unknown_sku_cannot_be_saved_even_if_approved(unknown_sku_doc: PurchaseDocument) -> None:
    _, _, graph, thread_id = _run(FakeERPModel(extracted=unknown_sku_doc))
    result, _ = resume_document(graph, thread_id, approve=True)
    # create_order is atomic: an unknown SKU raises, nothing is written.
    assert result.status == "error"
    assert result.issues[-1].code == "save_failed" and "JWL-999" in result.issues[-1].message
    assert check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"] == 40


def test_agent_issues_merged_and_tools_executed(clean_doc: PurchaseDocument) -> None:
    verdict = AgentVerdict(
        customer_id=1,
        issues=[
            Issue(
                code="description_mismatch", message="Line 1 description does not match catalogue."
            )
        ],
        summary="one semantic mismatch",
    )
    model = FakeERPModel(
        extracted=clean_doc,
        verdict=verdict,
        tool_calls_to_make=[
            {"name": "find_customer", "args": {"name": "Shree Ganesh Jewels"}},
            {"name": "check_stock", "args": {"sku": "JWL-001", "quantity": 10}},
            {"name": "not_a_tool", "args": {}},
        ],
    )
    result, state, _, _ = _run(model)
    assert result.status == "needs_review"
    assert [i.code for i in result.issues] == ["description_mismatch"]
    assert "one semantic mismatch" in state["log"][-1]


def test_extraction_retries_once_then_succeeds(clean_doc: PurchaseDocument) -> None:
    model = FakeERPModel(extracted=clean_doc, fail_extraction_times=1)
    result, _, _, _ = _run(model)
    assert model.extraction_attempts == 2 and result.status == "saved"


def test_extraction_failure_ends_in_error(clean_doc: PurchaseDocument) -> None:
    model = FakeERPModel(extracted=clean_doc, fail_extraction_times=5)
    result, _, _, _ = _run(model)
    assert model.extraction_attempts == 2
    assert result.status == "error" and result.issues[0].code == "extraction_failed"


def test_missing_file_ends_in_error(clean_doc: PurchaseDocument) -> None:
    result, _ = run_document(
        "data/does_not_exist.pdf", graph=build_graph(llm=FakeERPModel(extracted=clean_doc))
    )
    assert result.status == "error" and result.issues[0].code == "load_failed"


def test_real_sample_file_is_loaded(clean_doc: PurchaseDocument) -> None:
    _, state = run_document(
        "data/sample_docs/po_001.pdf", graph=build_graph(llm=FakeERPModel(extracted=clean_doc))
    )
    assert state["log"][0].startswith("load_document: data/sample_docs/po_001.pdf")
    assert "JWL-001" in state["raw_text"]


def test_merge_issues_collapses_same_finding_from_agent() -> None:
    from app.graph import merge_issues

    lookup = [
        Issue(
            code="insufficient_stock",
            message="Line 1: only 8 units of JWL-006 in stock, 25 requested.",
        )
    ]
    agent = [
        Issue(
            code="insufficient_stock", message="SKU JWL-006: only 8 units available, 25 requested."
        ),
        Issue(code="insufficient_stock", message="SKU JWL-005: only 1 unit available."),
        Issue(code="description_mismatch", message="JWL-006 description differs from catalogue."),
    ]
    merged = merge_issues(lookup, agent)
    assert [(i.code, SKU) for i in merged for SKU in [i.message.split("JWL-")[1][:3]]] == [
        ("insufficient_stock", "006"),
        ("insufficient_stock", "005"),
        ("description_mismatch", "006"),
    ]
