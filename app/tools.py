"""LangChain tools that give the agent access to the ERP database.

The docstrings matter: LangChain sends them to the LLM as the tool
descriptions, so each one states exactly what the tool does and returns.
Results are plain dicts so they serialise into tool messages, LangSmith
traces and MCP responses without conversion.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.db import Customer, Order, OrderLine, Product, get_session
from app.schemas import Drift, ErpSnapshot
from app.snapshot import (
    build_snapshot,
    compare_snapshots,
    customer_snapshot_from,
    line_snapshot_from,
)

logger = logging.getLogger(__name__)

# Minimum difflib similarity (0..1) for a customer-name match to count.
CUSTOMER_MATCH_CUTOFF = 0.75


def normalise_name(name: str) -> str:
    """Lower-case, drop punctuation and legal suffixes so names compare fairly."""
    cleaned = name.lower().replace(",", " ").replace(".", " ")
    for suffix in (" gmbh", " ag", " ltd", " llc", " pvt", " private limited", " limited"):
        cleaned = cleaned.replace(suffix, " ")
    return " ".join(cleaned.split())


def _get_product(session: Any, sku: str) -> Product | None:
    return session.scalar(select(Product).where(Product.sku == sku.strip().upper()))


@tool
def find_customer(name: str) -> dict[str, Any]:
    """Find a customer in the ERP by company name (fuzzy, cutoff 0.75).

    Small spelling differences and a missing legal suffix (GmbH, Ltd) still
    match. Returns ``{"found": true, "id", "name", "city", "country", "address",
    "status", "credit_limit", "payment_terms", "row_version"}`` for a match, or
    ``{"found": false, "candidates": [...]}`` with the three closest names.
    """
    query = normalise_name(name)
    with get_session() as session:
        customers = list(session.scalars(select(Customer)))
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, query, normalise_name(c.name)).ratio(), c)
            for c in customers
        ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if scored and scored[0][0] >= CUSTOMER_MATCH_CUTOFF:
        best = scored[0][1]
        return {
            "found": True,
            "id": best.id,
            "name": best.name,
            "city": best.city,
            "country": best.country,
            "address": best.address,
            "status": best.status,
            "credit_limit": best.credit_limit,
            "payment_terms": best.payment_terms,
            "row_version": best.row_version,
        }
    return {"found": False, "query": name, "candidates": [c.name for _, c in scored[:3]]}


@tool
def check_stock(sku: str, quantity: int) -> dict[str, Any]:
    """Check whether ``quantity`` units of product ``sku`` are available.

    Returns ``{"found": true, "sku", "available", "enough": bool}`` where
    ``enough`` means available >= quantity. Unknown SKUs return
    ``{"found": false, "sku"}``.
    """
    with get_session() as session:
        product = _get_product(session, sku)
        if product is None:
            return {"found": False, "sku": sku.strip().upper()}
        return {
            "found": True,
            "sku": product.sku,
            "available": product.stock,
            "enough": product.stock >= quantity,
        }


@tool
def get_product_price(sku: str) -> dict[str, Any]:
    """Return the current list price in EUR of product ``sku``.

    Returns ``{"found": true, "sku", "name", "unit_price", "row_version"}`` or
    ``{"found": false, "sku"}`` if the SKU is not in the catalogue.
    """
    with get_session() as session:
        product = _get_product(session, sku)
        if product is None:
            return {"found": False, "sku": sku.strip().upper()}
        return {
            "found": True,
            "sku": product.sku,
            "name": product.name,
            "unit_price": product.unit_price,
            "row_version": product.row_version,
        }


class OrderLineInput(BaseModel):
    """One line to write into an order."""

    sku: str = Field(description="Product SKU, e.g. 'JWL-001'.")
    quantity: int = Field(ge=1, description="Units ordered.")
    unit_price: float | None = Field(
        default=None, description="Agreed unit price; list price if omitted."
    )


class StaleProposalError(ValueError):
    """The ERP no longer matches the proposal snapshot; nothing was written.

    ``drift`` lists what differs and ``current`` is the snapshot read inside
    the transaction, so the caller can show the approver both views.
    """

    def __init__(self, drift: list[Drift], current: ErpSnapshot) -> None:
        self.drift = drift
        self.current = current
        super().__init__("; ".join(d.message for d in drift) or "proposal is stale")


def _existing_order_for(session: Any, thread_id: str | None, version: int | None) -> int | None:
    if not thread_id or version is None:
        return None
    return session.scalar(
        select(Order.id).where(Order.thread_id == thread_id, Order.proposal_version == version)
    )


@tool
def create_order(
    customer_id: int,
    lines: list[OrderLineInput],
    document_number: str | None = None,
    snapshot: dict[str, Any] | None = None,
    thread_id: str | None = None,
    proposal_version: int | None = None,
    approved_by: str | None = None,
) -> int:
    """Create a confirmed order and reduce stock in ONE transaction; return the order id.

    ``lines`` is a list of ``{sku, quantity, unit_price?}``. If the customer or
    any SKU is unknown, or stock is insufficient, a ``ValueError`` is raised
    and nothing is written. This tool is never given to the LLM; only the
    graph's save node calls it after validation and, if needed, approval.

    Approval safety, all inside the same transaction:

    * ``snapshot`` (an ``ErpSnapshot`` as dict): the customer and product rows
      are locked (``SELECT ... FOR UPDATE`` where the database supports it)
      and re-read; if price or customer data differ from the snapshot, or
      stock no longer covers a line, ``StaleProposalError`` is raised and
      nothing is written.
    * ``thread_id`` + ``proposal_version`` make the save idempotent: a second
      call for the same proposal version returns the existing order id, and
      the unique constraint on ``orders`` guarantees it even under a race.
    """
    if not lines:
        raise ValueError("an order needs at least one line")
    try:
        return _create_order_txn(
            customer_id, lines, document_number, snapshot, thread_id, proposal_version, approved_by
        )
    except IntegrityError:
        # Lost the race against a concurrent approve of the same proposal version.
        with get_session() as session:
            existing = _existing_order_for(session, thread_id, proposal_version)
        if existing is None:
            raise
        logger.info(
            "Order for %s v%s already exists (%s); replayed", thread_id, proposal_version, existing
        )
        return existing


def _create_order_txn(
    customer_id: int,
    lines: list[OrderLineInput],
    document_number: str | None,
    snapshot: dict[str, Any] | None,
    thread_id: str | None,
    proposal_version: int | None,
    approved_by: str | None,
) -> int:
    expected = ErpSnapshot.model_validate(snapshot) if snapshot else None
    with get_session() as session:
        existing = _existing_order_for(session, thread_id, proposal_version)
        if existing is not None:
            logger.info(
                "Order for %s v%s already exists (%s); replayed",
                thread_id,
                proposal_version,
                existing,
            )
            return existing

        # Lock and re-read the rows the decision depends on.
        customer = session.scalar(
            select(Customer).where(Customer.id == customer_id).with_for_update()
        )
        if customer is None:
            raise ValueError(f"customer id {customer_id} does not exist")
        parsed = [
            raw if isinstance(raw, OrderLineInput) else OrderLineInput(**raw) for raw in lines
        ]
        skus = sorted({line.sku.strip().upper() for line in parsed})
        products = {
            p.sku: p
            for p in session.scalars(select(Product).where(Product.sku.in_(skus)).with_for_update())
        }

        if expected is not None:
            current = build_snapshot(
                customer_snapshot_from(customer),
                [line_snapshot_from(sku, products.get(sku)) for sku in skus],
                version=expected.version,
            )
            quantities: dict[str, int] = {}
            for line in parsed:
                sku = line.sku.strip().upper()
                quantities[sku] = quantities.get(sku, 0) + line.quantity
            values, drift = compare_snapshots(
                expected, current, quantities, get_settings().recheck_price_tolerance
            )
            relevant = [d for d in drift if d.field != "stock" or d.severity == "block"]
            if any(v.changed and v.decision_relevant for v in values) or relevant:
                raise StaleProposalError(relevant or drift, current)

        order = Order(
            customer_id=customer_id,
            document_number=document_number,
            total=0.0,
            thread_id=thread_id,
            proposal_version=proposal_version,
            approved_by=approved_by,
        )
        total = 0.0
        for line in parsed:
            product = products.get(line.sku.strip().upper())
            if product is None:
                raise ValueError(f"SKU {line.sku.upper()} not found")
            if product.stock < line.quantity:
                raise ValueError(
                    f"insufficient stock for {product.sku}: {product.stock} available, "
                    f"{line.quantity} requested"
                )
            price = line.unit_price if line.unit_price is not None else product.unit_price
            product.stock -= line.quantity
            total += price * line.quantity
            order.lines.append(
                OrderLine(product_id=product.id, quantity=line.quantity, unit_price=price)
            )
        order.total = round(total, 2)
        session.add(order)
        session.flush()  # assigns order.id (or raises IntegrityError); the context manager commits
        order_id = order.id
    logger.info("Created order %s for customer %s (total %.2f)", order_id, customer_id, total)
    return order_id


# Tools the erp_check agent may call. create_order is intentionally excluded.
AGENT_TOOLS = [find_customer, check_stock, get_product_price]
