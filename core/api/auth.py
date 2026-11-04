"""API key -> store.

Every request to the core carries `Authorization: Bearer <key>`. The store is resolved
from the key here, server-side, and never taken from the request body — so a connector
cannot write into another store's data.

Store keys are created by POST /v1/stores and looked up in the `stores` table by their
hash. The admin key (CORE_ADMIN_KEY) is separate and only creates stores.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core.config import Settings, get_settings
from core.db.session import get_db
from core.db.stores import find_store_id

# Parses "Authorization: Bearer <key>" and declares it as a security scheme, which is what
# gives /docs its "Authorize" button. auto_error=False so the 401 below is ours.
_bearer = HTTPBearer(auto_error=False)


def get_store_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    session: Session = Depends(get_db),
) -> str:
    """FastAPI dependency: the store making this request, or 401."""
    if credentials is None:
        raise _unauthorized("missing or malformed Authorization header")

    # No timing-safe comparison needed here: the lookup compares hashes, and an attacker
    # cannot steer the hash of their guess towards a real one character by character.
    store_id = find_store_id(session, credentials.credentials)
    if store_id is None:
        # Same answer for an unknown key and an uninstalled store's key, so a caller
        # cannot tell which keys once existed.
        raise _unauthorized("invalid API key")

    return store_id


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency: let the request through only if it carries the admin key.

    Guards endpoints a connector uses to manage stores (e.g. POST /v1/stores), so
    nobody else can create stores.
    """
    if credentials is None:
        raise _unauthorized("missing or malformed Authorization header")
    if not secrets.compare_digest(credentials.credentials, settings.core_admin_key):
        raise _unauthorized("invalid admin key")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )
