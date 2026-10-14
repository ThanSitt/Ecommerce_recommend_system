"""Database connection.

Usage:
    with get_session() as session:
        session.add(row)
        session.commit()

In API routes, use `session: Session = Depends(get_db)` instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """The connection pool — created once, on first use, from DATABASE_URL."""
    return create_engine(get_settings().database_url)


def get_session() -> Session:
    """A new unit of work. Use it in a `with` block so it is always closed."""
    return Session(get_engine())


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, closed when the request ends."""
    with get_session() as session:
        yield session
