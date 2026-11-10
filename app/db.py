"""SQLAlchemy 2.x models and session management for the (tiny) ERP database.

SQLite is the default so the project runs with zero setup, but nothing here
is SQLite-specific: point ``DATABASE_URL`` at PostgreSQL and it just works.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Customer(Base):
    """A customer master record.

    ``row_version`` is SQLAlchemy's optimistic version counter: every ORM
    update of the row increments it, so a proposal can record which version
    of the customer it was built on.
    """

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    city: Mapped[str] = mapped_column(String(80), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    address: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    credit_limit: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    payment_terms: Mapped[str] = mapped_column(String(40), nullable=False, default="net 30")
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __mapper_args__ = {"version_id_col": row_version}

    orders: Mapped[list[Order]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __mapper_args__ = {"version_id_col": row_version}


class Order(Base):
    """A saved order.

    ``thread_id`` + ``proposal_version`` identify the approved proposal the
    order came from. The unique constraint is the last line of defence
    against a double approve creating two orders.
    """

    __tablename__ = "orders"
    __table_args__ = (UniqueConstraint("thread_id", "proposal_version", name="uq_orders_proposal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    document_number: Mapped[str | None] = mapped_column(String(60), nullable=True)
    total: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="confirmed")
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    proposal_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    customer: Mapped[Customer] = relationship(back_populates="orders")
    lines: Mapped[list[OrderLine]] = relationship(
        back_populates="order", cascade="all, delete-orphan"
    )


class OrderLine(Base):
    __tablename__ = "order_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)

    order: Mapped[Order] = relationship(back_populates="lines")
    product: Mapped[Product] = relationship()


class AuditLog(Base):
    """One event in the life of a proposal: proposed, approved, saved, stale, ...

    ``at_proposal`` and ``current`` hold the decision-relevant ERP values as
    JSON, so an auditor can see what the approver looked at and what the ERP
    said when the decision was applied.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    proposal_version: Mapped[int] = mapped_column(Integer, nullable=False)
    proposal_hash: Mapped[str] = mapped_column(String(32), nullable=False)
    event: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(120), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    order_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    at_proposal: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    current: Mapped[dict | None] = mapped_column(JSON, nullable=True)


# --- engine / session plumbing ------------------------------------------------

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine(url: str) -> Engine:
    parsed = make_url(url)
    connect_args = {"check_same_thread": False} if parsed.drivername.startswith("sqlite") else {}
    if parsed.drivername.startswith("sqlite") and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args)


def get_engine() -> Engine:
    """Return the process-wide engine, created from ``DATABASE_URL`` on first use."""
    global _engine, _session_factory
    if _engine is None:
        url = get_settings().database_url
        logger.info("Connecting to database %s", url)
        _engine = _build_engine(url)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def configure_engine(url: str) -> Engine:
    """Point the app at a different database (tests and evals use this)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = _build_engine(url)
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db(engine: Engine | None = None) -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(engine or get_engine())


def drop_db(engine: Engine | None = None) -> None:
    """Drop all tables (used by ``seed`` to start from scratch)."""
    Base.metadata.drop_all(engine or get_engine())


@contextmanager
def get_session() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on any exception."""
    get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
