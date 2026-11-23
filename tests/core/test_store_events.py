"""Loading a store's events from the database or a CSV export of the events table."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.data.retailrocket import EVENT_COLUMNS
from core.data.store_events import (
    TABLE_COLUMNS,
    export_store_ids,
    list_active_stores,
    load_events_export,
    load_store_events,
)
from core.db.models import Base, EventRow, StoreRow

# Rows as Postgres exports them: no quotes, empty order_id, "+00" offsets. Out of time order.
ROWS = [
    "3,st_a,e3,u1,0042,purchase,o-1,2026-11-04 09:00:00+00,2026-11-04 09:00:01.5+00",
    "1,st_a,e1,u1,0042,view,,2026-11-04 08:00:00.250+00,2026-11-04 08:00:01+00",
    "2,st_b,e2,u9,1234,view,,2026-11-04 08:30:00+00,2026-11-04 08:30:01+00",
    "4,st_a,e4,u2,8313704921435,cart,,2026-11-04 10:00:00+02,2026-11-04 08:00:01+00",
]


def export(tmp_path, rows=ROWS, header=None):
    path = tmp_path / "events.csv"
    lines = ([header] if header else []) + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("header", [None, ",".join(TABLE_COLUMNS)])
def test_an_export_gives_one_stores_events_in_the_standard_shape(tmp_path, header):
    events = load_events_export(export(tmp_path, header=header), "st_a")

    assert list(events.columns) == EVENT_COLUMNS
    assert len(events) == 3  # st_b's event is left out
    # Oldest first, in UTC: 10:00+02 is 08:00 UTC, before 08:00:00.250.
    assert events["timestamp"].tolist() == [
        pd.Timestamp("2026-11-04 08:00:00", tz="UTC"),
        pd.Timestamp("2026-11-04 08:00:00.250", tz="UTC"),
        pd.Timestamp("2026-11-04 09:00:00", tz="UTC"),
    ]
    # IDs stay text: "0042" keeps its zeros, the long ID all its digits.
    assert events["item_id"].tolist() == ["8313704921435", "0042", "0042"]
    assert events["order_id"].isna().tolist() == [True, True, False]


def test_an_export_with_some_columns_in_another_order(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "event_type,timestamp,item_id,user_id,store_id,order_id\n"
        "view,2026-11-04T08:00:00Z,p1,u1,st_a,\n",
        encoding="utf-8",
    )
    events = load_events_export(path, "st_a")
    assert events.iloc[0][["user_id", "item_id", "event_type"]].tolist() == ["u1", "p1", "view"]


def test_the_stores_in_an_export(tmp_path):
    assert export_store_ids(export(tmp_path)) == ["st_a", "st_b"]


def test_ids_rounded_by_a_spreadsheet_are_refused(tmp_path):
    # What happened to the first demo export: every product became 8.3137E+12.
    rounded = [row.replace("8313704921435", "8.3137E+12") for row in ROWS]
    with pytest.raises(ValueError, match="spreadsheet"):
        load_events_export(export(tmp_path, rounded), "st_a")


def test_an_unknown_event_type_is_refused(tmp_path):
    rows = [ROWS[1].replace(",view,", ",wishlist,")]
    with pytest.raises(ValueError, match="wishlist"):
        load_events_export(export(tmp_path, rows), "st_a")


def test_an_export_without_a_header_must_have_the_tables_columns(tmp_path):
    with pytest.raises(ValueError, match="no header row"):
        load_events_export(export(tmp_path, ["st_a,u1,p1,view"]), "st_a")


# ---- From the database (SQLite in memory standing in for Postgres) ------------------------


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def utc(hour, minute=0):
    return datetime(2026, 11, 4, hour, minute, tzinfo=UTC)


def test_events_from_the_database(session):
    session.add_all([
        EventRow(id=1, store_id="st_a", user_id="u1", item_id="p2", event_type="cart",
                 timestamp=utc(9)),
        EventRow(id=2, store_id="st_a", user_id="u1", item_id="p1", event_type="view",
                 timestamp=utc(8)),
        EventRow(id=3, store_id="st_b", user_id="u9", item_id="p1", event_type="view",
                 timestamp=utc(7)),
        EventRow(id=4, store_id="st_a", user_id="u1", item_id="p2", event_type="purchase",
                 order_id="o-1", timestamp=utc(9, 30)),
    ])
    session.commit()

    events = load_store_events(session, "st_a")

    assert list(events.columns) == EVENT_COLUMNS
    assert events["item_id"].tolist() == ["p1", "p2", "p2"]
    assert events["timestamp"].tolist() == [
        pd.Timestamp(t) for t in (utc(8), utc(9), utc(9, 30))
    ]
    assert events["order_id"].iloc[2] == "o-1"


def test_a_store_without_events_gives_an_empty_table(session):
    events = load_store_events(session, "st_none")
    assert events.empty and list(events.columns) == EVENT_COLUMNS


def test_only_active_stores_are_listed(session):
    session.add_all([
        StoreRow(store_id="st_b", api_key_hash="h1"),
        StoreRow(store_id="st_a", api_key_hash="h2"),
        StoreRow(store_id="st_gone", api_key_hash="h3", active=False),
    ])
    session.commit()
    assert list_active_stores(session) == ["st_a", "st_b"]
