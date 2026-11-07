"""Database tables, as SQLAlchemy classes.

`StoreRow` is a store and the hash of its API key.

`EventRow` is what is stored; `core.api.schemas.Event` is what arrives over HTTP. They
differ: a stored row also carries `store_id` (resolved from the API key, never sent by
the connector) and `received_at` (set by the database).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    String,
    UniqueConstraint,
    func,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StoreRow(Base):
    """A store the core serves. Platform-neutral: no shop domain or platform name here;
    connectors keep their own mapping from a platform's store to `store_id`."""

    __tablename__ = "stores"

    # The core's own name for the store; events and models are filed under it. Separate
    # from the key so a leaked key can be replaced without orphaning the store's events.
    store_id: Mapped[str] = mapped_column(String, primary_key=True)
    # SHA-256 of the store's API key; the key itself is never stored. Unique so a key
    # maps to exactly one store, and so the lookup on every request is indexed.
    api_key_hash: Mapped[str] = mapped_column(String, unique=True)
    # False once the store uninstalls: its key stops working, its events are kept.
    active: Mapped[bool] = mapped_column(Boolean, server_default=true())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    store_id: Mapped[str] = mapped_column(String)
    event_id: Mapped[str | None] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String)
    item_id: Mapped[str] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String)
    order_id: Mapped[str | None] = mapped_column(String)
    # When the event happened, as reported by the store. Evaluation splits on this.
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # When we received it, set by the database. Compared with `timestamp` to detect
    # store clock drift.
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # A repeated event_id from the same store is a duplicate. Rows without an
        # event_id are never treated as duplicates: PostgreSQL considers NULLs distinct.
        UniqueConstraint("store_id", "event_id", name="uq_events_store_event_id"),
        # Training reads "all events of one store in a time range".
        Index("ix_events_store_timestamp", "store_id", "timestamp"),
    )
