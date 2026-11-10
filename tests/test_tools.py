"""Tests for the ERP tools against a temporary seeded SQLite database (see conftest)."""

import pytest

from app.tools import AGENT_TOOLS, check_stock, create_order, find_customer, get_product_price


def test_find_customer_exact_and_fuzzy() -> None:
    exact = find_customer.invoke({"name": "Shree Ganesh Jewels"})
    assert exact["found"] and (exact["id"], exact["name"], exact["city"]) == (
        1,
        "Shree Ganesh Jewels",
        "Surat",
    )
    assert (exact["status"], exact["credit_limit"], exact["payment_terms"]) == (
        "active",
        15000.0,
        "net 30",
    )
    assert exact["row_version"] == 1 and "Surat" in exact["address"]
    assert (
        find_customer.invoke({"name": "Rhein Textil Handel GmbH"})["name"] == "Rhein Textil Handel"
    )
    assert (
        find_customer.invoke({"name": "Bayrische Feinkost"})["name"] == "Bayerische Feinkost GmbH"
    )


def test_find_customer_not_found_returns_candidates() -> None:
    result = find_customer.invoke({"name": "Golden Palm Trading LLC"})
    assert result["found"] is False
    assert len(result["candidates"]) == 3


def test_check_stock() -> None:
    assert check_stock.invoke({"sku": "txt-003", "quantity": 10}) == {
        "found": True,
        "sku": "TXT-003",
        "available": 35,
        "enough": True,
    }
    assert check_stock.invoke({"sku": "TXT-003", "quantity": 50})["enough"] is False
    assert check_stock.invoke({"sku": "TXT-099", "quantity": 1}) == {
        "found": False,
        "sku": "TXT-099",
    }


def test_get_product_price() -> None:
    assert get_product_price.invoke({"sku": "SNK-004"})["unit_price"] == 1.80
    assert get_product_price.invoke({"sku": "ZZZ-000"})["found"] is False


def test_create_order_reduces_stock_in_one_transaction() -> None:
    before = check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"]
    order_id = create_order.invoke(
        {
            "customer_id": 1,
            "lines": [
                {"sku": "JWL-001", "quantity": 2},
                {"sku": "JWL-004", "quantity": 1, "unit_price": 12.0},
            ],
            "document_number": "PO-X",
        }
    )
    assert isinstance(order_id, int)
    assert check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"] == before - 2


def test_create_order_is_atomic_on_unknown_sku() -> None:
    before = check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"]
    with pytest.raises(ValueError, match="NOPE-1 not found"):
        create_order.invoke(
            {
                "customer_id": 1,
                "lines": [{"sku": "JWL-001", "quantity": 2}, {"sku": "NOPE-1", "quantity": 1}],
            }
        )
    assert check_stock.invoke({"sku": "JWL-001", "quantity": 1})["available"] == before


def test_create_order_rejects_insufficient_stock_and_bad_customer() -> None:
    with pytest.raises(ValueError, match="insufficient stock"):
        create_order.invoke({"customer_id": 1, "lines": [{"sku": "JWL-006", "quantity": 999}]})
    with pytest.raises(ValueError, match="does not exist"):
        create_order.invoke({"customer_id": 999, "lines": [{"sku": "JWL-006", "quantity": 1}]})


def test_create_order_is_not_an_agent_tool() -> None:
    assert create_order not in AGENT_TOOLS
    assert {t.name for t in AGENT_TOOLS} == {"find_customer", "check_stock", "get_product_price"}
