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

from app.db import Customer, Order, OrderLine, Product, get_session

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
    match. Returns ``{"found": true, "id", "name", "city"}`` for a match, or
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
        return {"found": True, "id": best.id, "name": best.name, "city": best.city}
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

    Returns ``{"found": true, "sku", "name", "unit_price"}`` or
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
        }


class OrderLineInput(BaseModel):
    """One line to write into an order."""

    sku: str = Field(description="Product SKU, e.g. 'JWL-001'.")
    quantity: int = Field(ge=1, description="Units ordered.")
    unit_price: float | None = Field(
        default=None, description="Agreed unit price; list price if omitted."
    )


@tool
def create_order(
    customer_id: int,
    lines: list[OrderLineInput],
    document_number: str | None = None,
) -> int:
    """Create a confirmed order and reduce stock in ONE transaction; return the order id.

    ``lines`` is a list of ``{sku, quantity, unit_price?}``. If the customer or
    any SKU is unknown, or stock is insufficient, a ``ValueError`` is raised
    and nothing is written. This tool is never given to the LLM; only the
    graph's save node calls it after validation and, if needed, approval.
    """
    if not lines:
        raise ValueError("an order needs at least one line")
    with get_session() as session:
        if session.get(Customer, customer_id) is None:
            raise ValueError(f"customer id {customer_id} does not exist")
        order = Order(customer_id=customer_id, document_number=document_number, total=0.0)
        total = 0.0
        for raw in lines:
            line = raw if isinstance(raw, OrderLineInput) else OrderLineInput(**raw)
            product = _get_product(session, line.sku)
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
        session.flush()  # assigns order.id; the context manager commits
        order_id = order.id
    logger.info("Created order %s for customer %s (total %.2f)", order_id, customer_id, total)
    return order_id


# Tools the erp_check agent may call. create_order is intentionally excluded.
AGENT_TOOLS = [find_customer, check_stock, get_product_price]
