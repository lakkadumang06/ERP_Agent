"""Deterministic validation of an extracted document. Pure Python, no LLM, no DB.

These checks are the safety net: they are exact, cheap and unit-tested, so a
wrong total or a missing field is caught even when the model has a bad day.
Database-backed checks (customer, SKU, stock, price) live in the graph's
validation step because they need the ERP tools.
"""

from __future__ import annotations

from app.schemas import Issue, PurchaseDocument

TOLERANCE = 0.01  # EUR; absorbs cent rounding


def _differs(a: float, b: float) -> bool:
    return abs(a - b) > TOLERANCE


def check_required_fields(doc: PurchaseDocument) -> list[Issue]:
    """Customer name, document number and at least one line must be present."""
    issues: list[Issue] = []
    if not doc.customer_name.strip():
        issues.append(Issue(code="missing_customer", message="Customer name is missing."))
    if not doc.document_number.strip():
        issues.append(Issue(code="missing_document_number", message="Document number is missing."))
    if not doc.lines:
        issues.append(Issue(code="no_lines", message="Document contains no line items."))
    for idx, line in enumerate(doc.lines, start=1):
        if line.sku is None:
            issues.append(
                Issue(code="missing_sku", message=f"Line {idx} ('{line.description}') has no SKU.")
            )
    return issues


def check_line_totals(doc: PurchaseDocument) -> list[Issue]:
    """Each printed line total must equal quantity x unit price."""
    issues: list[Issue] = []
    for idx, line in enumerate(doc.lines, start=1):
        if _differs(line.line_total, line.computed_total):
            issues.append(
                Issue(
                    code="line_total_mismatch",
                    message=(
                        f"Line {idx}: printed total {line.line_total:.2f} != "
                        f"{line.quantity} x {line.unit_price:.2f} = {line.computed_total:.2f}."
                    ),
                )
            )
    return issues


def check_document_total(doc: PurchaseDocument) -> list[Issue]:
    """The printed grand total must equal the sum of quantity x unit price."""
    if _differs(doc.stated_total, doc.computed_total):
        return [
            Issue(
                code="document_total_mismatch",
                message=(
                    f"Stated total {doc.stated_total:.2f} does not match the sum of lines "
                    f"{doc.computed_total:.2f}."
                ),
            )
        ]
    return []


def run_all_checks(doc: PurchaseDocument) -> list[Issue]:
    """Run every deterministic check and return the combined list of issues."""
    return [
        *check_required_fields(doc),
        *check_line_totals(doc),
        *check_document_total(doc),
    ]
