"""Unit tests for the Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import Issue, OrderLine, PurchaseDocument


def test_line_computed_total_rounds_to_cents() -> None:
    line = OrderLine(
        sku="SNK-001", description="Masala Peanuts", quantity=3, unit_price=1.2, line_total=3.6
    )
    assert line.computed_total == 3.6


def test_sku_is_normalised() -> None:
    assert (
        OrderLine(sku=" jwl-001 ", description="x", quantity=1, unit_price=1, line_total=1).sku
        == "JWL-001"
    )
    assert OrderLine(sku="   ", description="x", quantity=1, unit_price=1, line_total=1).sku is None


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        OrderLine(sku="JWL-001", description="x", quantity=0, unit_price=1, line_total=0)


def test_document_totals_and_currency(clean_doc: PurchaseDocument) -> None:
    assert clean_doc.computed_total == 495.0
    assert clean_doc.currency == "EUR"
    lower = clean_doc.model_copy(update={"currency": "eur "})
    assert PurchaseDocument.model_validate(lower.model_dump()).currency == "EUR"


def test_extra_fields_are_ignored(clean_doc: PurchaseDocument) -> None:
    data = clean_doc.model_dump() | {"hallucinated_field": 42}
    assert PurchaseDocument.model_validate(data).document_number == "PO-TEST-001"


def test_issue_defaults_to_error() -> None:
    assert Issue(code="unknown_sku", message="SKU X not found").severity == "error"
