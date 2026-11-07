"""Creating stores and their API keys.

A store's API key is shown once, when the store is created, and only its SHA-256 hash is
saved. A leaked database therefore does not leak working keys.
"""

from __future__ import annotations

import hashlib
import secrets

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db.models import StoreRow


def hash_api_key(api_key: str) -> str:
    """SHA-256 of a key, as hex. The same key always gives the same hash, so a store can
    be looked up by the hash of the key a request carries.

    SHA-256 rather than bcrypt: keys are long and random, so they can't be guessed from
    the hash, and a fast hash keeps the lookup a plain indexed query.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_store(session: Session) -> tuple[str, str]:
    """Create a store with a new API key. Returns (store_id, api_key).

    The plain key is returned so the caller can hand it over once; it is not saved.

    Does not commit; the caller owns the transaction.
    """
    store_id = f"st_{secrets.token_hex(8)}"
    api_key = f"rk_{secrets.token_urlsafe(32)}"
    session.add(StoreRow(store_id=store_id, api_key_hash=hash_api_key(api_key)))
    return store_id, api_key


def find_store_id(session: Session, api_key: str) -> str | None:
    """The store a key belongs to, or None if no active store has that key.

    Looked up by the key's hash, never by the key: the database does not have the key.
    An inactive (uninstalled) store's key gives None, the same as an unknown key.
    """
    return session.scalar(
        select(StoreRow.store_id).where(
            StoreRow.api_key_hash == hash_api_key(api_key),
            StoreRow.active.is_(True),
        )
    )
