"""POST /v1/events — event ingestion.

Connectors and the dataset backfill both send batches here. The whole batch is validated
before anything is saved: if one event is invalid, the request fails with 422 and nothing
is stored, so a connector never has to work out which half of a batch went through.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from core.api.auth import get_store_id
from core.api.schemas import Event, IngestResponse
from core.db.events import save_events
from core.db.session import get_db

# Caps the size of one request. The backfill sends larger datasets in chunks below this.
MAX_BATCH_SIZE = 10_000

router = APIRouter(prefix="/v1", tags=["events"])


@router.post("/events", response_model=IngestResponse)
def ingest_events(
    events: Annotated[list[Event], Body(max_length=MAX_BATCH_SIZE)],
    store_id: str = Depends(get_store_id),
    session: Session = Depends(get_db),
) -> IngestResponse:
    inserted, duplicates = save_events(session, store_id, events)
    session.commit()
    return IngestResponse(received=len(events), inserted=inserted, duplicates=duplicates)
