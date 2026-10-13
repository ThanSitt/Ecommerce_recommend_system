"""API key -> store.

Every request to the core carries `Authorization: Bearer <key>`. The store is resolved
from the key here, server-side, and never taken from the request body — so a connector
cannot write into another store's data.

Development: any key listed in CORE_API_KEYS belongs to the single store "dev".
Later this lookup moves to a stores table; endpoints using `get_store_id` don't change.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, status

from core.config import Settings, get_settings

DEV_STORE_ID = "dev"


def get_store_id(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """FastAPI dependency: the store making this request, or 401."""
    scheme, _, key = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not key:
        raise _unauthorized("missing or malformed Authorization header")

    valid_keys = [k.strip() for k in settings.core_api_keys.split(",") if k.strip()]
    # compare_digest takes the same time whether the key is almost right or totally
    # wrong, so response timing can't be used to guess a key character by character.
    if not any(secrets.compare_digest(key, valid) for valid in valid_keys):
        raise _unauthorized("invalid API key")

    return DEV_STORE_ID


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )
