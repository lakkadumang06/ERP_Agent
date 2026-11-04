"""Optional MCP server exposing the read-only ERP tools.

Run with ``python -m app.mcp_server`` (stdio transport). Any MCP client -
Claude Desktop, Claude Code, an IDE - can then call ``find_customer``,
``check_stock`` and ``get_product_price`` against the same SQLite database
the agent uses. ``create_order`` is intentionally not exposed: writes should
go through the validated LangGraph workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from app.config import configure_logging
from app.db import init_db
from app.tools import check_stock, find_customer, get_product_price

logger = logging.getLogger(__name__)

server = MCPServer(
    name="erp-agent",
    description="Read-only access to the ERP-Agent demo ERP: customers, stock and prices.",
)


@server.tool()
def erp_find_customer(name: str) -> dict[str, Any]:
    """Fuzzy-match a customer by company name; returns id, name, city, country."""
    return find_customer.invoke({"name": name})


@server.tool()
def erp_check_stock(sku: str, quantity: int) -> dict[str, Any]:
    """Check whether `quantity` units of `sku` are available (found, available, enough)."""
    return check_stock.invoke({"sku": sku, "quantity": quantity})


@server.tool()
def erp_get_product_price(sku: str) -> dict[str, Any]:
    """Return the current list price (EUR) of product `sku`."""
    return get_product_price.invoke({"sku": sku})


def main() -> None:
    configure_logging()
    init_db()
    logger.info("Starting ERP-Agent MCP server on stdio")
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
