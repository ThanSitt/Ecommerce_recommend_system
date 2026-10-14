"""Request and response shapes for the core API.

`Event` is the event contract from docs/decisions/0001-event-contract.md. Every connector
produces it; the core accepts nothing else.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class Event(BaseModel):
    # Unknown fields are dropped, not rejected, so a field can be added to the contract
    # without breaking connectors that are already deployed.
    model_config = ConfigDict(extra="ignore")

    event_id: str | None = Field(default=None, min_length=1)
    user_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    event_type: Literal["view", "cart", "purchase"]
    order_id: str | None = Field(default=None, min_length=1)
    # AwareDatetime rejects timestamps without a UTC offset: a naive time is ambiguous,
    # and would silently shift events across a time-based train/test split.
    timestamp: AwareDatetime

    @model_validator(mode="after")
    def order_id_only_on_purchases(self) -> Event:
        if self.order_id is not None and self.event_type != "purchase":
            raise ValueError("order_id is only allowed on purchase events")
        return self


class IngestResponse(BaseModel):
    # Informational. Connectors should rely only on the 2xx status: if a queue is added
    # in front of the database later, these counts are not known at response time.
    received: int
    inserted: int
    duplicates: int
