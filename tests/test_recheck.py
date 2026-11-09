"""Approval re-check: stale proposals are shown and refused (fake LLM, no API keys)."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.db import Customer, Product, get_session
from app.graph import build_graph, pending_interrupt, resume_document, review_view, run_document
from app.recheck import NO_DRIFT_HASH, compare_with_current, hash_drift, take_snapshot
from app.schemas import Drift, PurchaseDocument
from app.tools import check_stock
from tests.conftest import FakeERPModel


def set_price(sku: str, unit_price: float) -> None:
    """Change a catalogue price behind the agent's back (what a colleague would do)."""
    with get_session() as session:
        session.scalar(select(Product).where(Product.sku == sku)).unit_price = unit_price


def set_stock(sku: str, stock: int) -> None:
    with get_session() as session:
        session.scalar(select(Product).where(Product.sku == sku)).stock = stock


def stock_of(sku: str) -> int:
    return check_stock.invoke({"sku": sku, "quantity": 1})["available"]


def _paused(doc: PurchaseDocument):
    """Run ``doc`` through the graph until it waits for a human."""
    graph = build_graph(llm=FakeERPModel(extracted=doc))
    thread_id = str(uuid.uuid4())
    result, state = run_document(None, None, thread_id, text="dummy", graph=graph)
    assert result.status == "needs_review" and pending_interrupt(state) is not None
    return graph, thread_id, result


# --------------------------------------------------------------------------
# compare_with_current (pure logic over the read-only tools)
# --------------------------------------------------------------------------


def test_snapshot_records_price_stock_and_customer(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    assert snapshot.customer.found and snapshot.customer.id == 1
    assert [(s.sku, s.unit_price_catalogue, s.stock) for s in snapshot.lines] == [
        ("JWL-001", 24.5, 40),
        ("JWL-004", 12.5, 80),
    ]
    assert snapshot.snapshot_at.endswith("+00:00")


def test_no_change_means_no_drift(clean_doc: PurchaseDocument) -> None:
    result = compare_with_current(clean_doc, take_snapshot(clean_doc))
    assert result.drift == [] and not result.blocked
    assert result.drift_hash == NO_DRIFT_HASH and result.summary() == "no drift"


def test_price_change_above_tolerance_is_a_warning(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    set_price("JWL-001", 24.6)  # +0.4 %: inside the default 0.5 % tolerance
    assert compare_with_current(clean_doc, snapshot).drift == []
    set_price("JWL-001", 26.0)
    result = compare_with_current(clean_doc, snapshot)
    assert [(d.field, d.sku, d.severity) for d in result.drift] == [("price", "JWL-001", "warn")]
    assert result.drift[0].value_at_proposal == 24.5 and result.drift[0].value_now == 26.0
    assert not result.blocked
    assert compare_with_current(clean_doc, snapshot, tolerance=0.1).drift == []


def test_stock_drop_warns_or_blocks(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    set_stock("JWL-001", 15)  # still covers the 10 requested
    warn = compare_with_current(clean_doc, snapshot)
    assert [(d.field, d.severity, d.value_now) for d in warn.drift] == [("stock", "warn", 15)]
    set_stock("JWL-001", 3)
    block = compare_with_current(clean_doc, snapshot)
    assert block.blocked and block.drift[0].severity == "block"
    set_stock("JWL-001", 500)  # more stock is never drift
    assert compare_with_current(clean_doc, snapshot).drift == []


def test_customer_disappearing_blocks(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    with get_session() as session:
        session.delete(session.get(Customer, 1))
    result = compare_with_current(clean_doc, snapshot)
    assert result.blocked and result.drift[0].field == "customer"


def test_hash_is_stable_and_order_independent() -> None:
    a = Drift(field="price", sku="A", value_at_proposal=1, value_now=2, severity="warn", message="")
    b = Drift(field="stock", sku="B", value_at_proposal=9, value_now=8, severity="warn", message="")
    assert hash_drift([a, b]) == hash_drift([b, a]) != hash_drift([a]) != NO_DRIFT_HASH


# --------------------------------------------------------------------------
# Graph: approval against the version the human saw
# --------------------------------------------------------------------------


def test_no_drift_approval_saves_as_before(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    assert first.recheck is not None and first.recheck.drift == []
    result, state = resume_document(graph, thread_id, approve=True)
    assert result.status == "saved" and result.order_id is not None
    assert state["approved_drift_hash"] == NO_DRIFT_HASH
    assert stock_of("JWL-001") == 30


def test_price_change_before_approval_needs_re_review(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, _ = _paused(wrong_total_doc)
    set_price("JWL-001", 30.0)  # a colleague changes the price while the order waits
    result, state = resume_document(graph, thread_id, approve=True)  # approver saw no drift
    assert result.status == "needs_re_review" and result.order_id is None
    assert result.recheck is not None
    assert [(d.field, d.sku) for d in result.recheck.drift] == [("price", "JWL-001")]
    assert pending_interrupt(state) is not None  # waiting for the human again
    assert state["log"][-1].startswith("human_review: approval refused, drift not seen")
    assert stock_of("JWL-001") == 40  # nothing written


def test_approving_after_seeing_the_drift_saves(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, _ = _paused(wrong_total_doc)
    set_price("JWL-001", 30.0)
    view = review_view(graph, thread_id)  # what GET /documents/{thread_id} shows
    assert view is not None and view.recheck is not None and len(view.recheck.drift) == 1
    result, _ = resume_document(graph, thread_id, True, seen_drift_hash=view.recheck.drift_hash)
    assert result.status == "saved" and result.order_id is not None
    assert stock_of("JWL-001") == 30


def test_stale_hash_after_second_change_needs_re_review(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, _ = _paused(wrong_total_doc)
    set_price("JWL-001", 30.0)
    seen = review_view(graph, thread_id).recheck.drift_hash
    set_price("JWL-004", 20.0)  # a second change the approver has not seen
    result, _ = resume_document(graph, thread_id, True, seen_drift_hash=seen)
    assert result.status == "needs_re_review"
    assert {d.sku for d in result.recheck.drift} == {"JWL-001", "JWL-004"}
    # The loop continues: with the new hash the approval goes through.
    result, _ = resume_document(graph, thread_id, True, seen_drift_hash=result.recheck.drift_hash)
    assert result.status == "saved"


def test_stock_below_quantity_blocks_even_when_seen(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, _ = _paused(wrong_total_doc)
    set_stock("JWL-001", 5)  # 10 requested
    view = review_view(graph, thread_id)
    assert view.recheck.blocked
    result, _ = resume_document(graph, thread_id, True, seen_drift_hash=view.recheck.drift_hash)
    assert result.status == "needs_re_review" and result.recheck.blocked
    assert "blocking drift" in result.log[-1]
    assert stock_of("JWL-001") == 5
    # Reject still works and ends the thread.
    result, _ = resume_document(graph, thread_id, approve=False)
    assert result.status == "rejected"


def test_rejecting_needs_no_hash(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, _ = _paused(wrong_total_doc)
    set_price("JWL-001", 30.0)
    result, _ = resume_document(graph, thread_id, approve=False)
    assert result.status == "rejected" and result.order_id is None


def test_save_guard_refuses_drift_on_the_no_issue_path(
    monkeypatch, clean_doc: PurchaseDocument
) -> None:
    """Price changes between erp_check and save_order: the save guard catches it."""
    from app import graph as graph_module

    original = graph_module.take_snapshot

    def snapshot_then_change_price(doc: PurchaseDocument):
        snapshot = original(doc)
        set_price("JWL-001", 30.0)  # happens "while" the proposal is on its way to save_order
        return snapshot

    monkeypatch.setattr(graph_module, "take_snapshot", snapshot_then_change_price)
    graph = build_graph(llm=FakeERPModel(extracted=clean_doc))
    result, _ = run_document(None, text="dummy", graph=graph)
    assert result.status == "error" and result.issues[-1].code == "stale_proposal"
    assert result.recheck is not None and result.recheck.drift[0].field == "price"
    assert stock_of("JWL-001") == 40


def test_review_view_unknown_thread_is_none(clean_doc: PurchaseDocument) -> None:
    assert review_view(build_graph(llm=FakeERPModel(extracted=clean_doc)), "nope") is None
