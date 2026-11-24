"""A store's events, from the core's database or a CSV export of it, in the standard shape.

The output has the same columns as `load_retailrocket` (EVENT_COLUMNS), so splitting,
filtering and the grid work the same on a real store as on the dataset. Only cleaning and
checking happens here; filtering runs after the time split, as for Retailrocket.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.data.retailrocket import EVENT_COLUMNS
from core.db.models import EventRow, StoreRow

EVENT_TYPES = ("view", "cart", "purchase")
# The events table's columns, in the database's order: what an export without a header row
# holds (see the migration that creates the table).
TABLE_COLUMNS = [
    "id", "store_id", "event_id", "user_id", "item_id", "event_type", "order_id",
    "timestamp", "received_at",
]
# How a spreadsheet program writes a long number it has rounded: 8313704921435 -> 8.3137E+12.
# The digits are gone, and different products end up with the same ID.
_ROUNDED_NUMBER = re.compile(r"\d(\.\d+)?[eE]\+\d+")


def load_store_events(session: Session, store_id: str) -> pd.DataFrame:
    """Every event of one store, oldest first."""
    query = (
        select(*[getattr(EventRow, column) for column in EVENT_COLUMNS])
        .where(EventRow.store_id == store_id)
        .order_by(EventRow.timestamp, EventRow.id)
    )
    return _standardise(pd.DataFrame(session.execute(query).all(), columns=EVENT_COLUMNS))


def load_events_export(path: Path | str, store_id: str) -> pd.DataFrame:
    """One store's events from a CSV export of the events table, oldest first.

    With a header row, as `\\copy events TO 'events.csv' CSV HEADER` writes, or without one,
    in the table's column order. Every value is read as text, so IDs stay exactly as stored.
    """
    raw = _read_export(path)
    events = raw.loc[raw["store_id"] == store_id, EVENT_COLUMNS].copy()
    # An empty cell is a missing order_id, as in the database.
    events["order_id"] = events["order_id"].where(events["order_id"] != "", None)
    return _standardise(events)


def export_store_ids(path: Path | str) -> list[str]:
    """The stores with events in an export, in ID order."""
    return sorted(_read_export(path)["store_id"].unique())


def _read_export(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if raw.empty:
        raise ValueError(f"{path} is empty")
    first_row = raw.iloc[0].tolist()
    if "store_id" in first_row:
        raw = raw.iloc[1:].set_axis(first_row, axis=1)
    elif raw.shape[1] == len(TABLE_COLUMNS):
        raw.columns = TABLE_COLUMNS
    else:
        raise ValueError(
            f"{path}: no header row, and {raw.shape[1]} columns instead of the events "
            f"table's {len(TABLE_COLUMNS)}"
        )
    missing = {"store_id", *EVENT_COLUMNS} - set(raw.columns)
    if missing:
        raise ValueError(f"{path} has no column {sorted(missing)}")
    return raw


def list_active_stores(session: Session) -> list[str]:
    """Every store that has not uninstalled, in ID order."""
    query = select(StoreRow.store_id).where(StoreRow.active.is_(True)).order_by(StoreRow.store_id)
    return list(session.scalars(query))


def _standardise(events: pd.DataFrame) -> pd.DataFrame:
    """Check the values, put timestamps in UTC and sort by time."""
    unknown = set(events["event_type"]) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"unknown event types: {sorted(unknown)}")

    for column in ("user_id", "item_id", "order_id"):
        values = events[column].dropna().astype(str)
        rounded = values[values.str.fullmatch(_ROUNDED_NUMBER)]
        if not rounded.empty:
            # Fail rather than train: every product would merge into a handful of IDs.
            raise ValueError(
                f"{column} values like {rounded.iloc[0]!r} look rounded by a spreadsheet "
                f"program, so the real IDs are lost. Export the events again and do not "
                f"open or save the file in a spreadsheet program."
            )

    events = events.copy()
    if pd.api.types.is_datetime64_any_dtype(events["timestamp"]):
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    else:
        # Exports of the table carry the offset ("+00"), as the contract requires.
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="ISO8601")
    return events.sort_values("timestamp", kind="stable").reset_index(drop=True)
