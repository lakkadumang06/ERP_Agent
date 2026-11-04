"""Create the ERP database from scratch with fictional customers and products.

Run with ``python -m app.seed`` (or ``make seed``). The script DROPS and
recreates all tables, so you always start from the same known state.
"""

from __future__ import annotations

import logging

from sqlalchemy.engine import Engine

from app.config import configure_logging, get_settings
from app.db import Customer, Product, drop_db, get_engine, get_session, init_db

logger = logging.getLogger(__name__)

# (name, city, country): fictional businesses in Gujarat (IN) and Germany (DE).
CUSTOMERS: list[tuple[str, str, str]] = [
    ("Shree Ganesh Jewels", "Surat", "IN"),
    ("Spice Bite Snacks", "Ahmedabad", "IN"),
    ("Rhein Textil Handel", "Köln", "DE"),
    ("Bayerische Feinkost GmbH", "München", "DE"),
    ("Lotus Boutique", "Vadodara", "IN"),
    ("Rajkot Munchies", "Rajkot", "IN"),
    ("Nordlicht Schmuck", "Hamburg", "DE"),
    ("Hansa Heimtextil", "Bremen", "DE"),
    ("Saurashtra Silver Works", "Bhavnagar", "IN"),
    ("Schwaben Stoffe GmbH", "Stuttgart", "DE"),
]

# (sku, name, category, unit price in EUR, stock). Some stocks are deliberately low.
PRODUCTS: list[tuple[str, str, str, float, int]] = [
    # jewellery
    ("JWL-001", "Gold Plated Bangle Set", "jewellery", 24.50, 40),
    ("JWL-002", "Silver Anklet Pair", "jewellery", 18.00, 60),
    ("JWL-003", "Kundan Necklace Set", "jewellery", 69.00, 15),
    ("JWL-004", "Pearl Drop Earrings", "jewellery", 12.50, 80),
    ("JWL-005", "Oxidised Silver Ring", "jewellery", 6.90, 120),
    ("JWL-006", "Temple Jewellery Choker", "jewellery", 58.00, 8),
    ("JWL-007", "Brass Nose Pin", "jewellery", 2.20, 300),
    ("JWL-008", "Meenakari Jhumka", "jewellery", 14.00, 55),
    # FMCG snacks
    ("SNK-001", "Masala Peanuts 200g", "snacks", 1.20, 1200),
    ("SNK-002", "Roasted Chana 250g", "snacks", 1.50, 900),
    ("SNK-003", "Banana Chips 150g", "snacks", 1.35, 800),
    ("SNK-004", "Mukhwas Mix 100g", "snacks", 1.80, 1500),
    ("SNK-005", "Khakhra Methi 200g", "snacks", 2.10, 600),
    ("SNK-006", "Bhujia Sev 400g", "snacks", 2.60, 700),
    ("SNK-007", "Peanut Chikki 250g", "snacks", 1.95, 500),
    ("SNK-008", "Dry Fruit Mix 500g", "snacks", 9.80, 25),
    ("SNK-009", "Chai Masala 50g", "snacks", 1.60, 400),
    # textiles
    ("TXT-001", "Cotton Bed Sheet Double", "textile", 21.00, 150),
    ("TXT-002", "Linen Table Runner", "textile", 9.50, 90),
    ("TXT-003", "Handloom Cotton Saree", "textile", 39.00, 35),
    ("TXT-004", "Jute Tote Bag", "textile", 3.40, 400),
    ("TXT-005", "Block Print Cushion Cover", "textile", 4.80, 250),
    ("TXT-006", "Wool Throw Blanket", "textile", 32.00, 12),
    ("TXT-007", "Khadi Kurta Fabric (per metre)", "textile", 5.50, 500),
    ("TXT-008", "Embroidered Dupatta", "textile", 15.00, 70),
]


def seed(engine: Engine | None = None) -> tuple[int, int]:
    """Drop and recreate all tables, then insert the fixture data.

    Returns:
        ``(customer_count, product_count)``.
    """
    engine = engine or get_engine()
    drop_db(engine)
    init_db(engine)
    with get_session() as session:
        session.add_all(Customer(name=n, city=c, country=k) for n, c, k in CUSTOMERS)
        session.add_all(
            Product(sku=s, name=n, category=c, unit_price=p, stock=st)
            for s, n, c, p, st in PRODUCTS
        )
    logger.info("Seeded %d customers and %d products", len(CUSTOMERS), len(PRODUCTS))
    return len(CUSTOMERS), len(PRODUCTS)


def main() -> None:
    configure_logging()
    customers, products = seed()
    logger.info(
        "Database %s ready: %d customers, %d products.",
        get_settings().database_url,
        customers,
        products,
    )


if __name__ == "__main__":
    main()
