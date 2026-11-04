"""Pydantic models shared by extraction, validation, the graph and the API.

The ``Field`` descriptions are not decoration: they are sent to the LLM as the
JSON schema for structured output, so they double as extraction instructions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentType = Literal["invoice", "purchase_order"]
Severity = Literal["warning", "error"]
ProcessStatus = Literal["saved", "needs_review", "rejected", "error"]


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


class ProcessResult(BaseModel):
    """What the CLI and the API return for one processed document."""

    status: ProcessStatus
    order_id: int | None = None
    thread_id: str | None = None
    document: PurchaseDocument | None = None
    issues: list[Issue] = Field(default_factory=list)
    log: list[str] = Field(default_factory=list, description="One entry per graph node that ran.")


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
    """Body for ``POST /review/{thread_id}``."""

    approve: bool


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
    lines: list[OrderLineOut]
