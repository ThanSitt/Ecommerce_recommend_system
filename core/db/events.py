"""Saving events to the database.

Kept out of the API route on purpose: the route only decides *whether* to save, this
decides *how*. If a queue is added in front of the database later, a worker calls this
same function and the route only enqueues.
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from core.api.schemas import Event
from core.db.models import EventRow


def save_events(session: Session, store_id: str, events: list[Event]) -> tuple[int, int]:
    """Insert events for one store. Returns (inserted, duplicates).

    A repeated (store_id, event_id) is skipped by the database, not raised as an error:
    webhooks retry on errors, so failing on a duplicate would cause a retry loop.

    Does not commit; the caller owns the transaction.
    """
    if not events:
        return 0, 0

    rows = [{"store_id": store_id, **event.model_dump()} for event in events]
    statement = (
        insert(EventRow.__table__)
        .on_conflict_do_nothing(constraint="uq_events_store_event_id")
        .returning(EventRow.id)
    )
    # Passing a list of rows lets SQLAlchemy split it into pages of 1,000 per INSERT,
    # which keeps each statement under PostgreSQL's 65,535-parameter limit.
    inserted = len(session.execute(statement, rows).all())
    return inserted, len(events) - inserted
