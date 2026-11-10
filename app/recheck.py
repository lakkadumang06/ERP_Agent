"""Approval re-check: protect the human reviewer from stale proposals.

A proposal can wait in the review queue for a long time. Meanwhile the
catalogue price, the stock level or the customer record can change, and the
approver would sign off on numbers that were only right when the proposal was
drafted. This module makes that gap visible:

* :func:`take_snapshot` records the ERP values a proposal was built on, each
  with its source and the time it was read, plus a version and a hash.
* :func:`compare_with_current` re-reads the ERP through the same READ-ONLY
  tools and returns every value "at proposal" vs "now", the drift items,
  and whether the proposal is stale, blocked or expired.

The enforcement happens elsewhere: ``create_order`` re-reads and compares
inside the save transaction (``app/tools.py``). Everything here is
deterministic Python; no LLM is involved.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import ErpSnapshot, LineSnapshot, PurchaseDocument, RecheckResult
from app.snapshot import (
    build_snapshot,
    compare_snapshots,
    customer_snapshot_from,
    expires_at,
    is_expired,
    now_iso,
    quantities_of,
)
from app.tools import check_stock, find_customer, get_product_price


def take_snapshot(doc: PurchaseDocument, version: int = 1) -> ErpSnapshot:
    """Read the current ERP values for the document's customer and SKUs.

    Uses only the three read-only tools the agent is bound to, so the
    snapshot sees exactly what the agent saw. Every value carries its
    source (``customers.credit_limit``, ``products.unit_price``, ...) and
    the UTC time it was read.
    """
    read_at = now_iso()
    customer = customer_snapshot_from(find_customer.invoke({"name": doc.customer_name}))
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
                row_version=price.get("row_version"),
            )
        )
    return build_snapshot(customer, lines, version=version, read_at=read_at)


def compare_with_current(
    doc: PurchaseDocument, snapshot: ErpSnapshot, tolerance: float | None = None
) -> RecheckResult:
    """Re-read the ERP and list what changed since ``snapshot``.

    * ``values``: one row per snapshot value with ``at_proposal``, ``now``,
      ``source`` and ``changed``.
    * ``drift``: human-readable items. ``block``: stock is now below the
      requested quantity, the customer is missing, blocked or a different
      record, or a SKU vanished. ``warn``: a decision-relevant value changed
      (catalogue price beyond ``RECHECK_PRICE_TOLERANCE``, credit limit,
      payment terms, address) or stock dropped but still covers the order.
    * ``stale``: a decision-relevant value changed, so the proposal hash no
      longer describes the ERP and the proposal must be refreshed.
    * ``expired``: older than ``PROPOSAL_TTL_HOURS``.
    """
    settings = get_settings()
    tolerance = settings.recheck_price_tolerance if tolerance is None else tolerance
    current = take_snapshot(doc, version=snapshot.version)
    values, drift = compare_snapshots(snapshot, current, quantities_of(doc), tolerance)
    return RecheckResult(
        snapshot_at=snapshot.snapshot_at,
        checked_at=current.snapshot_at,
        proposal_version=snapshot.version,
        proposal_hash=snapshot.snapshot_hash,
        expires_at=expires_at(snapshot.snapshot_at, settings.proposal_ttl_hours),
        expired=is_expired(snapshot.snapshot_at, settings.proposal_ttl_hours),
        snapshot=snapshot,
        current=current,
        values=values,
        drift=drift,
        stale=any(v.changed and v.decision_relevant for v in values),
        blocked=any(d.severity == "block" for d in drift),
    )
