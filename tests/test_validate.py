"""Unit tests for the deterministic checks in ``app/validate.py``."""

from __future__ import annotations

import pytest

from app.schemas import OrderLine, PurchaseDocument
from app.validate import (
    check_document_total,
    check_line_totals,
    check_required_fields,
    run_all_checks,
)


def make_doc(**overrides) -> PurchaseDocument:
    base = dict(
        doc_type="purchase_order",
        document_number="PO-2026-0113",
        document_date="2026-01-13",
        customer_name="Shree Ganesh Jewels",
        currency="EUR",
        lines=[
            OrderLine(
                sku="JWL-001",
                description="Bangle Set",
                quantity=10,
                unit_price=24.5,
                line_total=245.0,
            ),
            OrderLine(
                sku="JWL-004",
                description="Earrings",
                quantity=20,
                unit_price=12.5,
                line_total=250.0,
            ),
        ],
        stated_total=495.0,
    )
    return PurchaseDocument(**(base | overrides))


def codes(issues) -> list[str]:
    return [issue.code for issue in issues]


def test_clean_document_has_no_issues() -> None:
    assert run_all_checks(make_doc()) == []


def test_required_fields_missing() -> None:
    doc = make_doc(customer_name="  ", document_number="", lines=[])
    assert codes(check_required_fields(doc)) == [
        "missing_customer",
        "missing_document_number",
        "no_lines",
    ]


def test_required_fields_flags_line_without_sku() -> None:
    doc = make_doc(
        lines=[
            OrderLine(sku=None, description="Mystery", quantity=1, unit_price=5.0, line_total=5.0)
        ],
        stated_total=5.0,
    )
    issues = check_required_fields(doc)
    assert codes(issues) == ["missing_sku"]
    assert "Line 1" in issues[0].message


def test_line_total_mismatch_is_detected() -> None:
    doc = make_doc(
        lines=[
            OrderLine(
                sku="JWL-001", description="x", quantity=10, unit_price=24.5, line_total=250.0
            )
        ],
        stated_total=245.0,
    )
    issues = check_line_totals(doc)
    assert codes(issues) == ["line_total_mismatch"]
    assert "250.00" in issues[0].message and "245.00" in issues[0].message


def test_line_total_within_tolerance_passes() -> None:
    doc = make_doc(
        lines=[
            OrderLine(sku="SNK-001", description="x", quantity=3, unit_price=1.333, line_total=4.0)
        ],
        stated_total=4.0,
    )
    assert check_line_totals(doc) == []


@pytest.mark.parametrize(
    "stated, expected_codes", [(495.0, []), (495.009, []), (595.0, ["document_total_mismatch"])]
)
def test_document_total(stated: float, expected_codes: list[str]) -> None:
    assert codes(check_document_total(make_doc(stated_total=stated))) == expected_codes


def test_run_all_checks_combines_everything() -> None:
    doc = make_doc(
        customer_name="",
        lines=[OrderLine(sku=None, description="x", quantity=2, unit_price=10.0, line_total=25.0)],
        stated_total=99.0,
    )
    assert codes(run_all_checks(doc)) == [
        "missing_customer",
        "missing_sku",
        "line_total_mismatch",
        "document_total_mismatch",
    ]
    assert all(issue.severity == "error" for issue in run_all_checks(doc))
