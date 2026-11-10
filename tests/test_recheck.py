"""Stale-approval protection: snapshot, versions, re-check in the save transaction.

Fake LLM, no API keys. ``set_price`` / ``set_credit_limit`` / ``set_stock``
play the colleague who edits the ERP while a proposal waits for approval.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from app import snapshot as snapshot_module
from app.audit import entries_for
from app.db import Customer, Order, Product, get_session
from app.graph import build_graph, pending_interrupt, resume_document, review_view, run_document
from app.recheck import compare_with_current, take_snapshot
from app.schemas import PurchaseDocument
from app.tools import StaleProposalError, check_stock, create_order
from tests.conftest import FakeERPModel


def set_price(sku: str, unit_price: float) -> None:
    """Change a catalogue price behind the agent's back (what a colleague would do)."""
    with get_session() as session:
        session.scalar(select(Product).where(Product.sku == sku)).unit_price = unit_price


def set_stock(sku: str, stock: int) -> None:
    with get_session() as session:
        session.scalar(select(Product).where(Product.sku == sku)).stock = stock


def set_customer(customer_id: int, **fields) -> None:
    with get_session() as session:
        customer = session.get(Customer, customer_id)
        for key, value in fields.items():
            setattr(customer, key, value)


def stock_of(sku: str) -> int:
    return check_stock.invoke({"sku": sku, "quantity": 1})["available"]


def order_count() -> int:
    with get_session() as session:
        return len(list(session.scalars(select(Order))))


def _paused(doc: PurchaseDocument):
    """Run ``doc`` through the graph until it waits for a human."""
    graph = build_graph(llm=FakeERPModel(extracted=doc))
    thread_id = str(uuid.uuid4())
    result, state = run_document(None, None, thread_id, text="dummy", graph=graph)
    assert result.status == "needs_review" and pending_interrupt(state) is not None
    assert result.proposal is not None and result.proposal.version == 1
    return graph, thread_id, result


def approve(graph, thread_id, result, actor="alice"):
    """Approve exactly the proposal version the result shows."""
    return resume_document(
        graph,
        thread_id,
        action="approve",
        proposal_version=result.proposal.version,
        proposal_hash=result.proposal.hash,
        actor=actor,
    )


def events(thread_id: str) -> list[tuple[int, str]]:
    return [(e.proposal_version, e.event) for e in entries_for(thread_id)]


# --------------------------------------------------------------------------
# Step 1: snapshot with value, source, time; version and hash
# --------------------------------------------------------------------------


def test_snapshot_records_every_value_with_source_and_time(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    assert snapshot.version == 1 and len(snapshot.snapshot_hash) == 16
    assert snapshot.customer.found and snapshot.customer.id == 1
    assert snapshot.customer.credit_limit == 15000.0 and snapshot.customer.payment_terms == "net 30"
    assert snapshot.customer.status == "active" and "Surat" in snapshot.customer.address
    assert snapshot.customer.row_version == 1
    by_key = {v.key: v for v in snapshot.values}
    assert by_key["customer.credit_limit"].value == 15000.0
    assert by_key["customer.credit_limit"].source == "customers.credit_limit"
    assert by_key["price:JWL-001"].value == 24.5
    assert by_key["price:JWL-001"].source == "products.unit_price"
    assert by_key["stock:JWL-004"].value == 80 and not by_key["stock:JWL-004"].decision_relevant
    assert all(v.read_at == snapshot.snapshot_at for v in snapshot.values)
    assert snapshot.snapshot_at.endswith("+00:00")


def test_hash_depends_on_decision_values_only(clean_doc: PurchaseDocument) -> None:
    before = take_snapshot(clean_doc).snapshot_hash
    set_stock("JWL-001", 39)  # stock is informational: same hash
    assert take_snapshot(clean_doc).snapshot_hash == before
    set_price("JWL-001", 30.0)
    assert take_snapshot(clean_doc).snapshot_hash != before
    set_price("JWL-001", 24.5)
    set_customer(1, credit_limit=100.0)
    assert take_snapshot(clean_doc).snapshot_hash != before
    assert take_snapshot(clean_doc, version=2).version == 2


# --------------------------------------------------------------------------
# Step 2: at_proposal vs now, per value
# --------------------------------------------------------------------------


def test_no_change_means_nothing_changed(clean_doc: PurchaseDocument) -> None:
    result = compare_with_current(clean_doc, take_snapshot(clean_doc))
    assert not result.stale and not result.blocked and not result.expired
    assert result.drift == [] and result.summary() == "no drift"
    assert all(not v.changed for v in result.values)
    assert {v.key for v in result.values} >= {"customer.credit_limit", "price:JWL-001"}


def test_price_change_is_stale_with_at_proposal_and_now(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    set_price("JWL-001", 24.6)  # +0.4 %: inside the default 0.5 % tolerance
    assert not compare_with_current(clean_doc, snapshot).stale
    set_price("JWL-001", 26.0)
    result = compare_with_current(clean_doc, snapshot)
    row = next(v for v in result.values if v.key == "price:JWL-001")
    assert (row.at_proposal, row.now, row.source, row.changed) == (
        24.5,
        26.0,
        "products.unit_price",
        True,
    )
    assert result.stale and not result.blocked
    assert [(d.field, d.sku, d.severity) for d in result.drift] == [("price", "JWL-001", "warn")]
    assert not compare_with_current(clean_doc, snapshot, tolerance=0.1).stale


def test_customer_data_change_is_stale(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    set_customer(1, credit_limit=100.0, payment_terms="prepaid")
    result = compare_with_current(clean_doc, snapshot)
    changed = {v.key: (v.at_proposal, v.now) for v in result.values if v.changed}
    assert changed == {
        "customer.credit_limit": (15000.0, 100.0),
        "customer.payment_terms": ("net 30", "prepaid"),
    }
    assert result.stale and not result.blocked
    set_customer(1, status="blocked")
    assert compare_with_current(clean_doc, snapshot).blocked


def test_stock_drop_warns_or_blocks_but_is_not_stale(clean_doc: PurchaseDocument) -> None:
    snapshot = take_snapshot(clean_doc)
    set_stock("JWL-001", 15)  # still covers the 10 requested
    warn = compare_with_current(clean_doc, snapshot)
    assert [(d.field, d.severity, d.value_now) for d in warn.drift] == [("stock", "warn", 15)]
    assert next(v for v in warn.values if v.key == "stock:JWL-001").changed
    assert not warn.stale and not warn.blocked
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


# --------------------------------------------------------------------------
# Step 4: the save transaction re-reads price and customer data
# --------------------------------------------------------------------------


def _lines(doc: PurchaseDocument) -> list[dict]:
    return [
        {"sku": line.sku, "quantity": line.quantity, "unit_price": line.unit_price}
        for line in doc.lines
    ]


def test_create_order_refuses_when_price_changed_inside_transaction(clean_doc) -> None:
    snapshot = take_snapshot(clean_doc).model_dump()
    set_price("JWL-001", 30.0)
    with pytest.raises(StaleProposalError) as info:
        create_order.invoke({"customer_id": 1, "lines": _lines(clean_doc), "snapshot": snapshot})
    assert [d.field for d in info.value.drift] == ["price"]
    assert info.value.current.line("JWL-001").unit_price_catalogue == 30.0
    assert stock_of("JWL-001") == 40 and order_count() == 0


def test_create_order_refuses_when_customer_changed_inside_transaction(clean_doc) -> None:
    snapshot = take_snapshot(clean_doc).model_dump()
    set_customer(1, credit_limit=100.0)
    with pytest.raises(StaleProposalError):
        create_order.invoke({"customer_id": 1, "lines": _lines(clean_doc), "snapshot": snapshot})
    assert order_count() == 0


def test_create_order_is_idempotent_per_proposal_version(clean_doc) -> None:
    snapshot = take_snapshot(clean_doc).model_dump()
    args = {
        "customer_id": 1,
        "lines": _lines(clean_doc),
        "snapshot": snapshot,
        "thread_id": "t-1",
        "proposal_version": 1,
        "approved_by": "alice",
    }
    first = create_order.invoke(args)
    second = create_order.invoke(args)  # retry after a network blip
    assert first == second and order_count() == 1
    assert stock_of("JWL-001") == 30  # reduced once
    with get_session() as session:
        order = session.get(Order, first)
        assert (order.thread_id, order.proposal_version, order.approved_by) == ("t-1", 1, "alice")


# --------------------------------------------------------------------------
# Step 4 + 7: the graph, end to end
# --------------------------------------------------------------------------


def test_nothing_changes_saved_exactly_once(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    assert first.recheck is not None and not first.recheck.stale
    result, state = approve(graph, thread_id, first)
    assert result.status == "saved" and result.order_id is not None
    assert state["approved"] is True and state["approver"] == "alice"
    assert order_count() == 1 and stock_of("JWL-001") == 30
    assert events(thread_id) == [(1, "proposed"), (1, "approved"), (1, "saved")]
    with get_session() as session:
        order = session.get(Order, result.order_id)
        assert (order.thread_id, order.proposal_version, order.approved_by) == (
            thread_id,
            1,
            "alice",
        )


def test_price_change_while_waiting_marks_stale_and_creates_v2(wrong_total_doc) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    set_price("JWL-001", 30.0)  # a colleague changes the price while the order waits
    result, state = approve(graph, thread_id, first)  # approver approves v1
    assert result.status == "needs_re_review" and result.order_id is None
    assert pending_interrupt(state) is not None  # waiting for the human again
    assert result.proposal.version == 2 and result.proposal.hash != first.proposal.hash
    assert "ERP changed" in result.proposal.stale_reason
    assert result.recheck is not None and result.recheck.proposal_version == 2
    assert not result.recheck.stale  # v2 was built on today's values
    assert result.recheck.snapshot.line("JWL-001").unit_price_catalogue == 30.0
    assert order_count() == 0 and stock_of("JWL-001") == 40
    assert events(thread_id) == [(1, "proposed"), (1, "stale"), (2, "proposed")]
    stale = entries_for(thread_id)[1]
    assert stale.at_proposal["price:JWL-001"] == 24.5 and stale.current["price:JWL-001"] == 30.0
    # The approver reviews v2 and approves it: saved.
    result, _ = approve(graph, thread_id, result)
    assert result.status == "saved" and order_count() == 1
    # v2 re-ran the checks: the document price now differs from the catalogue.
    assert "price_mismatch" in {i.code for i in result.issues}


def test_credit_limit_change_while_waiting_marks_stale_and_creates_v2(wrong_total_doc) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    set_customer(1, credit_limit=100.0)
    result, _ = approve(graph, thread_id, first)
    assert result.status == "needs_re_review" and result.proposal.version == 2
    assert order_count() == 0
    assert "credit_limit_exceeded" in {i.code for i in result.issues}  # ERP check re-ran
    assert events(thread_id) == [(1, "proposed"), (1, "stale"), (2, "proposed")]


def test_change_between_precheck_and_transaction_is_caught_in_the_transaction(
    monkeypatch, wrong_total_doc: PurchaseDocument
) -> None:
    """The read-only pre-check passes; the locked re-read inside create_order does not."""
    from app import graph as graph_module

    real = graph_module.create_order

    class ChangePriceThenSave:
        def invoke(self, args):
            set_price("JWL-001", 30.0)  # happens after human_review's check, before the write
            return real.invoke(args)

    monkeypatch.setattr(graph_module, "create_order", ChangePriceThenSave())
    graph, thread_id, first = _paused(wrong_total_doc)
    result, _ = approve(graph, thread_id, first)
    assert result.status == "needs_re_review" and result.proposal.version == 2
    assert order_count() == 0 and stock_of("JWL-001") == 40
    assert result.log[-3].startswith("save_order: refused, ERP changed before save")
    assert events(thread_id) == [(1, "proposed"), (1, "approved"), (1, "stale"), (2, "proposed")]


def test_approving_an_old_version_is_refused_without_new_version(wrong_total_doc) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    result, _ = resume_document(
        graph, thread_id, action="approve", proposal_version=1, proposal_hash="0000000000000000"
    )
    assert result.status == "needs_re_review" and result.proposal.version == 1
    assert order_count() == 0
    assert result.log[-1].startswith("human_review: approval refused, approval named v1")
    # The legacy shorthand (no version) is refused the same way.
    result, _ = resume_document(graph, thread_id, approve=True)
    assert result.status == "needs_re_review" and order_count() == 0
    # Approving the version actually on screen works.
    result, _ = approve(graph, thread_id, first)
    assert result.status == "saved"


def test_stock_drop_while_waiting_still_blocks(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    set_stock("JWL-001", 5)  # 10 requested
    view = review_view(graph, thread_id)
    assert view.recheck.blocked and view.recheck.needs_refresh
    result, _ = approve(graph, thread_id, first)
    assert result.status == "needs_re_review" and result.proposal.version == 2
    assert "insufficient_stock" in {i.code for i in result.issues}
    assert order_count() == 0 and stock_of("JWL-001") == 5
    # Approving v2 cannot bypass stock either: create_order refuses in the transaction
    # (existing behaviour: save_failed, nothing written).
    result, _ = approve(graph, thread_id, result)
    assert result.status == "error" and result.issues[-1].code == "save_failed"
    assert "insufficient stock" in result.issues[-1].message
    assert order_count() == 0 and stock_of("JWL-001") == 5


def test_expired_proposal_must_be_refreshed(monkeypatch, wrong_total_doc) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    real_now = snapshot_module.utcnow()
    monkeypatch.setattr(snapshot_module, "utcnow", lambda: real_now + timedelta(hours=25))
    view = review_view(graph, thread_id)
    assert view.recheck.expired and view.proposal.expired and view.recheck.needs_refresh
    result, _ = approve(graph, thread_id, first)
    assert result.status == "needs_re_review" and result.proposal.version == 2
    assert "expired" in result.proposal.stale_reason and order_count() == 0
    assert events(thread_id) == [(1, "proposed"), (1, "expired"), (2, "proposed")]
    assert not result.recheck.expired  # v2 is fresh
    result, _ = approve(graph, thread_id, result)
    assert result.status == "saved" and order_count() == 1


def test_refresh_creates_next_version_on_request(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, first = _paused(wrong_total_doc)
    set_price("JWL-004", 20.0)
    view = review_view(graph, thread_id)
    assert view.recheck.stale
    result, _ = resume_document(graph, thread_id, action="refresh", actor="alice")
    assert result.status == "needs_re_review" and result.proposal.version == 2
    assert not result.recheck.stale and order_count() == 0
    assert events(thread_id) == [(1, "proposed"), (1, "refreshed"), (2, "proposed")]


def test_rejecting_needs_no_version(wrong_total_doc: PurchaseDocument) -> None:
    graph, thread_id, _ = _paused(wrong_total_doc)
    set_price("JWL-001", 30.0)
    result, _ = resume_document(graph, thread_id, approve=False)
    assert result.status == "rejected" and result.order_id is None
    assert events(thread_id) == [(1, "proposed"), (1, "rejected")]


def test_clean_document_is_saved_with_proposal_v1(clean_doc: PurchaseDocument) -> None:
    graph = build_graph(llm=FakeERPModel(extracted=clean_doc))
    result, _ = run_document(None, text="dummy", graph=graph)
    assert result.status == "saved" and result.proposal.version == 1
    assert events(result.thread_id) == [(1, "proposed"), (1, "saved")]


def test_review_view_unknown_thread_is_none(clean_doc: PurchaseDocument) -> None:
    assert review_view(build_graph(llm=FakeERPModel(extracted=clean_doc)), "nope") is None
