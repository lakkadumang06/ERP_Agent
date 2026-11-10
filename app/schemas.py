"""Pydantic models shared by extraction, validation, the graph and the API.

The ``Field`` descriptions are not decoration: they are sent to the LLM as the
JSON schema for structured output, so they double as extraction instructions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentType = Literal["invoice", "purchase_order"]
Severity = Literal["warning", "error"]
ProcessStatus = Literal["saved", "needs_review", "needs_re_review", "rejected", "error"]
DriftSeverity = Literal["warn", "block"]


class OrderLine(BaseModel):
    """One line item of an invoice or purchase order."""

    sku: str | None = Field(
        default=None,
        description=(
            "Product code / article number (Art.-Nr.) exactly as printed, e.g. 'JWL-001'. "
            "null if the line shows no code. Never invent a code."
        ),
    )
    description: str = Field(description="Product name or description as printed.")
    quantity: int = Field(ge=1, description="Quantity ordered (Menge), as a number.")
    unit_price: float = Field(
        ge=0, description="Price per unit (Einzelpreis) as a number, without currency symbol."
    )
    line_total: float = Field(
        ge=0, description="Line amount as printed (Gesamt / Betrag), as a number."
    )

    @field_validator("sku")
    @classmethod
    def _normalise_sku(cls, value: str | None) -> str | None:
        """Upper-case and strip so 'jwl-001 ' matches 'JWL-001'."""
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned or None

    @property
    def computed_total(self) -> float:
        """Quantity multiplied by unit price, rounded to cents."""
        return round(self.quantity * self.unit_price, 2)


class PurchaseDocument(BaseModel):
    """Structured content of a purchase order (Bestellung) or an invoice (Rechnung)."""

    model_config = ConfigDict(extra="ignore")

    doc_type: DocumentType = Field(
        description="'invoice' for Invoice/Rechnung, 'purchase_order' for PO/Bestellung."
    )
    document_number: str = Field(description="Document, order or invoice number as printed.")
    document_date: str | None = Field(
        default=None, description="Document date as ISO YYYY-MM-DD, or null if absent."
    )
    customer_name: str = Field(
        description=(
            "Company name of the CUSTOMER buying the goods (the party placing the order "
            "or being billed), never the supplier Nova Trade Hub."
        )
    )
    currency: str = Field(default="EUR", description="ISO 4217 code, e.g. EUR.")
    lines: list[OrderLine] = Field(description="All line items in document order.")
    stated_total: float = Field(
        description="Grand total exactly as printed (Gesamtsumme / Total), as a number."
    )

    @field_validator("currency")
    @classmethod
    def _normalise_currency(cls, value: str) -> str:
        return value.strip().upper() or "EUR"

    @property
    def computed_total(self) -> float:
        """Sum of quantity multiplied by unit price over all lines."""
        return round(sum(line.computed_total for line in self.lines), 2)


class Issue(BaseModel):
    """One validation finding. ``code`` is stable for programs, ``message`` for humans."""

    code: str = Field(description="Machine-readable code, e.g. 'unknown_sku'.")
    message: str = Field(description="One short sentence a reviewer can act on.")
    severity: Severity = Field(default="error", description="'error' blocks saving.")


class ValidationResult(BaseModel):
    """Merged outcome of deterministic checks and the LLM tool-calling step."""

    is_valid: bool
    issues: list[Issue] = Field(default_factory=list)
    customer_id: int | None = Field(default=None, description="Matched ERP customer id.")


# --- approval re-check -----------------------------------------------------

SnapshotScalar = str | int | float | bool | None


class SnapshotValue(BaseModel):
    """One ERP value a proposal depends on, with its provenance.

    ``key`` is stable across snapshots (``"customer.credit_limit"``,
    ``"price:JWL-001"``), ``source`` names the table and field (or API) the
    value came from, and ``read_at`` is the UTC time it was read.
    """

    key: str
    label: str = Field(description="Short human label, e.g. 'Credit limit'.")
    sku: str | None = None
    value: SnapshotScalar
    source: str = Field(description="'table.field' or the API that supplied the value.")
    read_at: str = Field(description="UTC timestamp, ISO 8601.")
    decision_relevant: bool = Field(
        default=True,
        description=(
            "True for values in the proposal hash (price, customer data). False for "
            "informational values such as stock, which the save transaction checks itself."
        ),
    )


class LineSnapshot(BaseModel):
    """Catalogue price and stock of one SKU at a point in time."""

    sku: str
    found: bool = Field(description="False if the SKU was not in the catalogue.")
    unit_price_catalogue: float | None = None
    stock: int | None = None
    row_version: int | None = Field(default=None, description="products.row_version when read.")


class CustomerSnapshot(BaseModel):
    """The ERP customer the document was matched to (or not), with its master data."""

    found: bool
    id: int | None = None
    name: str | None = None
    status: str | None = None
    address: str | None = None
    credit_limit: float | None = None
    payment_terms: str | None = None
    row_version: int | None = Field(default=None, description="customers.row_version when read.")


class ErpSnapshot(BaseModel):
    """ERP values at one moment, for the customer and SKUs of one document.

    ``version`` counts the proposals made for one thread (1, 2, ...), and
    ``snapshot_hash`` fingerprints the decision-relevant ``values``. Approvals
    carry both, and the save transaction re-reads and compares against them.
    """

    snapshot_at: str = Field(description="UTC timestamp, ISO 8601.")
    version: int = Field(default=1, ge=1)
    snapshot_hash: str = Field(
        default="", description="Fingerprint of the decision-relevant values."
    )
    customer: CustomerSnapshot
    lines: list[LineSnapshot]
    values: list[SnapshotValue] = Field(default_factory=list)

    def line(self, sku: str) -> LineSnapshot | None:
        """Return the snapshot of ``sku`` or ``None`` if it was not recorded."""
        return next((line for line in self.lines if line.sku == sku), None)

    def value(self, key: str) -> SnapshotValue | None:
        """Return the recorded value with ``key`` or ``None``."""
        return next((v for v in self.values if v.key == key), None)


class Drift(BaseModel):
    """One difference between the proposal snapshot and the ERP now."""

    field: Literal["customer", "price", "stock", "sku"]
    sku: str | None = None
    value_at_proposal: SnapshotScalar
    value_now: SnapshotScalar
    severity: DriftSeverity = Field(description="'block' can never be saved; 'warn' needs a look.")
    message: str = Field(description="One sentence for the reviewer.")


class ValueComparison(BaseModel):
    """One row of the approval screen: a value at proposal time and now."""

    key: str
    label: str
    sku: str | None = None
    at_proposal: SnapshotScalar
    now: SnapshotScalar
    source: str
    read_at: str = Field(description="When the at-proposal value was read (UTC).")
    changed: bool
    decision_relevant: bool = True


class RecheckResult(BaseModel):
    """Outcome of ``compare_with_current``: then, now and what changed."""

    snapshot_at: str
    checked_at: str
    proposal_version: int
    proposal_hash: str
    expires_at: str
    expired: bool = Field(description="Older than PROPOSAL_TTL_HOURS: refresh before approving.")
    snapshot: ErpSnapshot
    current: ErpSnapshot
    values: list[ValueComparison] = Field(default_factory=list)
    drift: list[Drift] = Field(default_factory=list)
    stale: bool = Field(
        description="A decision-relevant value (price, customer) changed; refresh before approving."
    )
    blocked: bool = Field(description="True if any drift item has severity 'block'.")

    @property
    def needs_refresh(self) -> bool:
        """The approver must refresh the proposal before an approval can count."""
        return self.stale or self.blocked or self.expired

    def summary(self) -> str:
        """Short text for logs, e.g. ``"2 drift item(s): price JWL-001, stock JWL-004"``."""
        if not self.drift:
            return "no drift"
        parts = ", ".join(f"{d.field} {d.sku}" if d.sku else d.field for d in self.drift)
        return f"{len(self.drift)} drift item(s): {parts}"


class ProposalInfo(BaseModel):
    """Identity and lifecycle of the proposal the approver is looking at."""

    version: int
    hash: str
    created_at: str
    expires_at: str
    expired: bool = False
    stale_reason: str | None = Field(
        default=None,
        description="Why the previous version was marked stale (set on versions > 1).",
    )
    approved_by: str | None = None


class ProcessResult(BaseModel):
    """What the CLI and the API return for one processed document."""

    status: ProcessStatus
    order_id: int | None = None
    thread_id: str | None = None
    document: PurchaseDocument | None = None
    issues: list[Issue] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list, description="One entry per graph node that ran.")
    proposal: ProposalInfo | None = Field(
        default=None,
        description="Version, hash and expiry of the current proposal (once erp_check ran).",
    )
    recheck: RecheckResult | None = Field(
        default=None,
        description=(
            "Approval re-check: ERP values at proposal time vs now, plus the drift between "
            "them. Present while a run waits for review and after a review decision."
        ),
    )


class AgentVerdict(BaseModel):
    """What the validation agent reports after checking the ERP through tools."""

    customer_id: int | None = Field(default=None, description="Matched ERP customer id or null.")
    issues: list[Issue] = Field(
        default_factory=list,
        description=(
            "Problems you verified with a tool. Use codes: unknown_customer, unknown_sku, "
            "insufficient_stock, price_mismatch, description_mismatch, other."
        ),
    )
    summary: str = Field(description="One-sentence overall assessment.")


# --- API request payloads ----------------------------------------------------


class ProcessTextRequest(BaseModel):
    """JSON body for ``POST /process-document`` when sending raw text."""

    text: str = Field(min_length=1, description="Full text of the invoice / PO.")
    provider: str | None = Field(
        default=None,
        description="Override LLM_PROVIDER for this request (openai, anthropic, groq, ...).",
    )


class ProviderOut(BaseModel):
    """One entry of ``GET /providers``."""

    name: str
    label: str
    model: str
    free: bool
    configured: bool
    default: bool
    key_hint: str


class ReviewRequest(BaseModel):
    """Body for ``POST /documents/{thread_id}/review``."""

    approve: bool
    proposal_version: int | None = Field(
        default=None,
        description="Required for approve=true: the ``proposal.version`` the approver saw.",
    )
    proposal_hash: str | None = Field(
        default=None,
        description="Required for approve=true: the ``proposal.hash`` the approver saw.",
    )
    approver: str | None = Field(
        default=None, max_length=120, description="Who decides (stored in the audit log)."
    )


class RefreshRequest(BaseModel):
    """Body for ``POST /documents/{thread_id}/refresh``."""

    actor: str | None = Field(default=None, max_length=120)


class AuditEntryOut(BaseModel):
    """One row of ``GET /documents/{thread_id}/audit``."""

    id: int
    created_at: str
    thread_id: str
    proposal_version: int
    proposal_hash: str
    event: str
    actor: str | None
    reason: str | None
    order_id: int | None
    at_proposal: dict | None
    current: dict | None


class OrderLineOut(BaseModel):
    sku: str
    quantity: int
    unit_price: float


class ProductOut(BaseModel):
    """A catalogue entry as returned by ``GET /products``."""

    sku: str
    name: str
    category: str
    unit_price: float
    stock: int


class OrderOut(BaseModel):
    """A saved order as returned by ``GET /orders``."""

    id: int
    customer_id: int
    customer_name: str
    document_number: str | None
    total: float
    status: str
    thread_id: str | None = None
    proposal_version: int | None = None
    approved_by: str | None = None
    lines: list[OrderLineOut]
