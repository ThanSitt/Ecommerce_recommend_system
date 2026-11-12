"""POST /v1/stores — register a new store.

Called by a connector when a store installs it (from the platform's install hook).
Guarded by the admin key, which only the core and the connector know; a store's own API
key cannot create stores.

The request has no body: the core needs nothing about the store. Which platform it is on,
and its domain there, stay in the connector, which keeps its own mapping to `store_id`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from core.api.auth import require_admin
from core.api.schemas import StoreCreated
from core.db.session import get_db
from core.db.stores import create_store

router = APIRouter(prefix="/v1", tags=["stores"])


@router.post(
    "/stores",
    response_model=StoreCreated,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_admin)],
)
def register_store(
    response: Response,
    session: Session = Depends(get_db),
) -> StoreCreated:
    store_id, api_key = create_store(session)
    session.commit()
    # The answer carries a working key. Tell any proxy or client in between not to keep
    # a copy of it.
    response.headers["Cache-Control"] = "no-store"
    return StoreCreated(store_id=store_id, api_key=api_key)
