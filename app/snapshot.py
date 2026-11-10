"""Proposal snapshots: the ERP values a decision depends on, with provenance.

Pure data logic, no database access and no LLM:

* :func:`customer_snapshot_from` / :func:`line_snapshot_from` turn ORM rows
  (or the dicts the read-only tools return) into snapshot records.
* :func:`build_snapshot` flattens them into :class:`SnapshotValue` items, each
  with its source (``table.field``) and the time it was read, and computes
  the proposal hash over the decision-relevant values.
* :func:`compare_snapshots` lists every value "at proposal" vs "now" and the
  drift between them. Used both by the review screen (read-only re-check)
  and by the save transaction (compare against freshly locked rows).
* :func:`is_expired` applies the proposal time-to-live.

Decision-relevant values (in the hash): catalogue price per SKU, customer
name/status/address/credit limit/payment terms. Stock is recorded and shown,
but it is not in the hash: the save transaction checks it against the
requested quantity itself, as it always did.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from app.schemas import (
    CustomerSnapshot,
    Drift,
    ErpSnapshot,
    LineSnapshot,
    SnapshotValue,
    ValueComparison,
)

CUSTOMER_FIELDS: tuple[tuple[str, str, str], ...] = (
    # (attribute on CustomerSnapshot, label, source)
    ("name", "Customer", "customers.name"),
    ("status", "Customer status", "customers.status"),
    ("credit_limit", "Credit limit", "customers.credit_limit"),
    ("payment_terms", "Payment terms", "customers.payment_terms"),
    ("address", "Address", "customers.address"),
)


def utcnow() -> datetime:
    """Current UTC time (a function so tests can monkeypatch it)."""
    return datetime.now(UTC)


def now_iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def _parse(ts: str) -> datetime:
    parsed = datetime.fromisoformat(ts)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# --------------------------------------------------------------------------
# building snapshots
# --------------------------------------------------------------------------


def customer_snapshot_from(row: Any) -> CustomerSnapshot:
    """Snapshot of a ``Customer`` ORM row, a ``find_customer`` result dict, or ``None``."""
    if row is None:
        return CustomerSnapshot(found=False)
    if isinstance(row, dict):
        if not row.get("found", True):
            return CustomerSnapshot(found=False)
        get = row.get
    else:
        get = lambda key, default=None: getattr(row, key, default)  # noqa: E731
    address = get("address") or ""
    city, country = get("city") or "", get("country") or ""
    place = country if city and city.lower() in address.lower() else f"{city} {country}".strip()
    full_address = ", ".join(part for part in (address, place) if part)
    return CustomerSnapshot(
        found=True,
        id=get("id"),
        name=get("name"),
        status=get("status"),
        address=full_address or None,
        credit_limit=None if get("credit_limit") is None else float(get("credit_limit")),
        payment_terms=get("payment_terms"),
        row_version=get("row_version"),
    )


def line_snapshot_from(sku: str, product: Any) -> LineSnapshot:
    """Snapshot of a ``Product`` ORM row (or ``None`` when the SKU is unknown)."""
    if product is None:
        return LineSnapshot(sku=sku, found=False)
    return LineSnapshot(
        sku=sku,
        found=True,
        unit_price_catalogue=float(product.unit_price),
        stock=int(product.stock),
        row_version=getattr(product, "row_version", None),
    )


def flatten_values(
    customer: CustomerSnapshot, lines: list[LineSnapshot], read_at: str
) -> list[SnapshotValue]:
    """One :class:`SnapshotValue` per decision input, with source and read time."""
    values: list[SnapshotValue] = []
    for attr, label, source in CUSTOMER_FIELDS:
        values.append(
            SnapshotValue(
                key=f"customer.{attr}",
                label=label,
                value=getattr(customer, attr) if customer.found else None,
                source=source,
                read_at=read_at,
            )
        )
    for line in lines:
        values.append(
            SnapshotValue(
                key=f"price:{line.sku}",
                label="Catalogue price",
                sku=line.sku,
                value=line.unit_price_catalogue if line.found else None,
                source="products.unit_price",
                read_at=read_at,
            )
        )
        values.append(
            SnapshotValue(
                key=f"stock:{line.sku}",
                label="Stock",
                sku=line.sku,
                value=line.stock if line.found else None,
                source="products.stock",
                read_at=read_at,
                decision_relevant=False,
            )
        )
    return values


def hash_values(values: list[SnapshotValue]) -> str:
    """16-hex-digit fingerprint of the decision-relevant values (order-independent)."""
    canonical = sorted(
        json.dumps({"key": v.key, "value": v.value}, sort_keys=True)
        for v in values
        if v.decision_relevant
    )
    return hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:16]


def build_snapshot(
    customer: CustomerSnapshot,
    lines: list[LineSnapshot],
    version: int = 1,
    read_at: str | None = None,
) -> ErpSnapshot:
    """Assemble an :class:`ErpSnapshot` with flattened values and hash."""
    read_at = read_at or now_iso()
    values = flatten_values(customer, lines, read_at)
    return ErpSnapshot(
        snapshot_at=read_at,
        version=version,
        snapshot_hash=hash_values(values),
        customer=customer,
        lines=lines,
        values=values,
    )


# --------------------------------------------------------------------------
# comparing snapshots
# --------------------------------------------------------------------------


def _customer_drift(then: CustomerSnapshot, now: CustomerSnapshot) -> list[Drift]:
    if not then.found:
        if now.found:
            return [
                Drift(
                    field="customer",
                    value_at_proposal=None,
                    value_now=now.name,
                    severity="warn",
                    message=f"Customer '{now.name}' is now found in the ERP (it was unknown).",
                )
            ]
        return []
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
    drift: list[Drift] = []
    for attr, label, _ in CUSTOMER_FIELDS:
        before, after = getattr(then, attr), getattr(now, attr)
        if before == after:
            continue
        blocked = attr == "status" and str(after).lower() != "active"
        drift.append(
            Drift(
                field="customer",
                value_at_proposal=before,
                value_now=after,
                severity="block" if blocked else "warn",
                message=f"{label} of '{then.name}' changed from {before!r} to {after!r}.",
            )
        )
    return drift


def _line_drift(
    sku: str, quantity: int, then: LineSnapshot, now: LineSnapshot, tolerance: float
) -> list[Drift]:
    if not then.found:
        if now.found:
            return [
                Drift(
                    field="sku",
                    sku=sku,
                    value_at_proposal="missing",
                    value_now="in catalogue",
                    severity="warn",
                    message=f"{sku} is now in the catalogue (it was unknown at proposal time).",
                )
            ]
        return []
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


def _price_changed(then: SnapshotValue, now_value: Any, tolerance: float) -> bool:
    if then.value is None or now_value is None:
        return then.value != now_value
    before = float(then.value)
    return abs(float(now_value) - before) > before * tolerance


def compare_snapshots(
    then: ErpSnapshot,
    now: ErpSnapshot,
    quantities: dict[str, int],
    tolerance: float,
) -> tuple[list[ValueComparison], list[Drift]]:
    """Per-value comparison plus the drift items for the reviewer.

    ``quantities`` maps SKU to the total quantity requested (a SKU printed on
    two lines needs the sum). A price change inside ``tolerance`` is not a
    change; any other difference in a decision-relevant value is.
    """
    now_values = {v.key: v for v in now.values}
    rows: list[ValueComparison] = []
    for then_value in then.values:
        now_value = now_values.get(then_value.key)
        current = now_value.value if now_value is not None else None
        if then_value.key.startswith("price:"):
            changed = _price_changed(then_value, current, tolerance)
        else:
            changed = then_value.value != current
        rows.append(
            ValueComparison(
                key=then_value.key,
                label=then_value.label,
                sku=then_value.sku,
                at_proposal=then_value.value,
                now=current,
                source=then_value.source,
                read_at=then_value.read_at,
                changed=changed,
                decision_relevant=then_value.decision_relevant,
            )
        )
    drift = _customer_drift(then.customer, now.customer)
    for then_line in then.lines:
        now_line = now.line(then_line.sku) or LineSnapshot(sku=then_line.sku, found=False)
        if then_line.sku in quantities:
            drift.extend(
                _line_drift(
                    then_line.sku, quantities[then_line.sku], then_line, now_line, tolerance
                )
            )
    return rows, drift


def quantities_of(doc: Any) -> dict[str, int]:
    """Total quantity per SKU of a :class:`PurchaseDocument`."""
    quantities: dict[str, int] = {}
    for line in doc.lines:
        if line.sku is not None:
            quantities[line.sku] = quantities.get(line.sku, 0) + line.quantity
    return quantities


# --------------------------------------------------------------------------
# expiry
# --------------------------------------------------------------------------


def expires_at(snapshot_at: str, ttl_hours: float) -> str:
    return (_parse(snapshot_at) + timedelta(hours=ttl_hours)).isoformat(timespec="seconds")


def is_expired(snapshot_at: str, ttl_hours: float, now: datetime | None = None) -> bool:
    """True when the proposal is older than ``ttl_hours``."""
    return (now or utcnow()) > _parse(snapshot_at) + timedelta(hours=ttl_hours)


def values_as_dict(
    values: list[SnapshotValue] | list[ValueComparison], attr: str = "value"
) -> dict:
    """Compact ``{key: value}`` map for the audit log."""
    return {v.key: getattr(v, attr) for v in values}
