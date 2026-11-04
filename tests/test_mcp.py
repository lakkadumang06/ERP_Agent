"""The MCP tools must expose the same shapes as the LangChain tools they wrap."""

from __future__ import annotations

import pytest

mcp = pytest.importorskip("mcp", reason="optional dependency 'mcp' not installed")

from app import mcp_server  # noqa: E402


def test_server_registers_the_three_read_only_tools() -> None:
    names = {tool.name for tool in mcp_server.server._tool_manager.list_tools()}
    assert names == {"erp_find_customer", "erp_check_stock", "erp_get_product_price"}


def test_tool_functions_return_expected_shapes() -> None:
    customer = mcp_server.erp_find_customer("Shree Ganesh Jewels")
    assert customer == {"found": True, "id": 1, "name": "Shree Ganesh Jewels", "city": "Surat"}
    assert mcp_server.erp_find_customer("Golden Palm Trading LLC")["found"] is False

    stock = mcp_server.erp_check_stock("JWL-006", 5)
    assert stock == {"found": True, "sku": "JWL-006", "available": 8, "enough": True}
    assert mcp_server.erp_check_stock("JWL-006", 50)["enough"] is False

    price = mcp_server.erp_get_product_price("snk-004")
    assert price == {"found": True, "sku": "SNK-004", "name": "Mukhwas Mix 100g", "unit_price": 1.8}
    assert mcp_server.erp_get_product_price("ZZZ-000") == {"found": False, "sku": "ZZZ-000"}
