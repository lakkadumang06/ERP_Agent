"""Approval re-check: protect the human reviewer from stale proposals.

A proposal can wait in the review queue for a long time. Meanwhile the
catalogue price, the stock level or the customer record can change, and the
approver would sign off on numbers that were only right when the proposal was
drafted. This module makes that gap visible and enforceable:

* :func:`take_snapshot` records the ERP values a proposal was built on.
* :func:`compare_with_current` re-reads the ERP through the same READ-ONLY
  tools and lists every difference as a :class:`Drift` item.
* :func:`hash_drift` gives a short fingerprint of a drift list, so the
  approver can prove which "then vs now" view they actually looked at.

Everything here is deterministic Python; no LLM is involved.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from app.config import get_settings
from app.schemas import (
    CustomerSnapshot,
    Drift,
    ErpSnapshot,
    LineSnapshot,
    PurchaseDocument,
    RecheckResult,
)
from app.tools import check_stock, find_customer, get_product_price


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def take_snapshot(doc: PurchaseDocument) -> ErpSnapshot:
    """Read the current ERP values for the document's customer and SKUs.

    Uses only the three read-only tools the agent is bound to, so the
    snapshot sees exactly what the agent saw.
    """
    customer = find_customer.invoke({"name": doc.customer_name})
    customer_snapshot = CustomerSnapshot(
        found=bool(customer["found"]),
        id=customer.get("id"),
        name=customer.get("name"),
    )
    lines: list[LineSnapshot] = []
    seen: set[str] = set()
    for line in doc.lines:
        if line.sku is None or line.sku in seen:
            continue
        seen.add(line.sku)
        stock = check_stock.invoke({"sku": line.sku, "quantity": line.quantity})
        if not stock["found"]:
            lines.append(LineSnapshot(sku=line.sku, found=False))
            continue
        price = get_product_price.invoke({"sku": line.sku})
        lines.append(
            LineSnapshot(
                sku=line.sku,
                found=True,
                unit_price_catalogue=float(price["unit_price"]),
                stock=int(stock["available"]),
            )
        )
    return ErpSnapshot(snapshot_at=_now_iso(), customer=customer_snapshot, lines=lines)


def hash_drift(drift: list[Drift]) -> str:
    """Stable 16-hex-digit fingerprint of a drift list (order-independent).

    The approver sends this value back with their decision. If the drift list
    at approval time hashes differently, they have not seen the latest changes.
    """
    canonical = sorted(json.dumps(d.model_dump(), sort_keys=True) for d in drift)
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:16]


NO_DRIFT_HASH = hash_drift([])


def _compare_customer(then: CustomerSnapshot, now: CustomerSnapshot) -> list[Drift]:
    """Customer missing or different now, when it was matched at proposal time."""
    if not then.found:
        return []  # already reported as unknown_customer; nothing new to show
    if not now.found:
        return [
            Drift(
                field="customer",
                value_at_proposal=then.name,
                value_now=None,
                severity="block",
                message=f"Customer '{then.name}' is no longer found in the ERP.",
            )
        ]
    if now.id != then.id:
        return [
            Drift(
                field="customer",
                value_at_proposal=then.name,
                value_now=now.name,
                severity="block",
                message=f"Customer now matches '{now.name}' instead of '{then.name}'.",
            )
        ]
    return []


def _compare_line(
    sku: str, quantity: int, then: LineSnapshot, now: LineSnapshot, tolerance: float
) -> list[Drift]:
    """Price and stock drift for one SKU."""
    if not then.found:
        return []  # already reported as unknown_sku; nothing new to show
    if not now.found:
        return [
            Drift(
                field="sku",
                sku=sku,
                value_at_proposal="in catalogue",
                value_now="missing",
                severity="block",
                message=f"{sku} has been removed from the catalogue.",
            )
        ]
    drift: list[Drift] = []
    then_price = float(then.unit_price_catalogue or 0.0)
    now_price = float(now.unit_price_catalogue or 0.0)
    if abs(now_price - then_price) > then_price * tolerance:
        drift.append(
            Drift(
                field="price",
                sku=sku,
                value_at_proposal=then_price,
                value_now=now_price,
                severity="warn",
                message=(
                    f"Catalogue price of {sku} changed from {then_price:.2f} to {now_price:.2f}."
                ),
            )
        )
    then_stock, now_stock = int(then.stock or 0), int(now.stock or 0)
    if now_stock < then_stock:
        enough = now_stock >= quantity
        drift.append(
            Drift(
                field="stock",
                sku=sku,
                value_at_proposal=then_stock,
                value_now=now_stock,
                severity="warn" if enough else "block",
                message=(
                    f"Stock of {sku} dropped from {then_stock} to {now_stock}"
                    + (
                        f"; {quantity} requested."
                        if enough
                        else f", below the {quantity} requested."
                    )
                ),
            )
        )
    return drift


def compare_with_current(
    doc: PurchaseDocument, snapshot: ErpSnapshot, tolerance: float | None = None
) -> RecheckResult:
    """Re-read the ERP and list what changed since ``snapshot``.

    Severity rules:

    * ``block``: stock is now below the requested quantity, the customer is
      missing or matches a different record, or a SKU vanished.
    * ``warn``: the catalogue price moved by more than ``tolerance`` (default
      ``RECHECK_PRICE_TOLERANCE``, 0.5 %), or stock dropped but still covers
      the quantity.

    Changes that cannot hurt the approver (stock went up, an unknown SKU is
    still unknown) are not drift.
    """
    tolerance = get_settings().recheck_price_tolerance if tolerance is None else tolerance
    current = take_snapshot(doc)
    drift = _compare_customer(snapshot.customer, current.customer)
    # Quantity per SKU: a SKU printed on two lines needs the sum to be in stock.
    quantities: dict[str, int] = {}
    for line in doc.lines:
        if line.sku is not None:
            quantities[line.sku] = quantities.get(line.sku, 0) + line.quantity
    for then in snapshot.lines:
        now = current.line(then.sku)
        if now is not None and then.sku in quantities:
            drift.extend(_compare_line(then.sku, quantities[then.sku], then, now, tolerance))
    return RecheckResult(
        snapshot_at=snapshot.snapshot_at,
        checked_at=current.snapshot_at,
        snapshot=snapshot,
        current=current,
        drift=drift,
        drift_hash=hash_drift(drift),
        blocked=any(d.severity == "block" for d in drift),
    )
