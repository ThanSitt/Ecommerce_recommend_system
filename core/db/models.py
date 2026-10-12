"""Database tables, as SQLAlchemy classes.

`EventRow` is what is stored; `core.api.schemas.Event` is what arrives over HTTP. They
differ: a stored row also carries `store_id` (resolved from the API key, never sent by
the connector) and `received_at` (set by the database).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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
