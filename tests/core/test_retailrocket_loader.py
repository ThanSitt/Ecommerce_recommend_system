"""The Retailrocket loader: raw CSV -> standard event shape."""

from pathlib import Path

import pandas as pd
import pytest

from core.data.retailrocket import EVENT_COLUMNS, load_retailrocket
from core.db.models import EventRow

FIXTURE = Path(__file__).parent.parent / "fixtures" / "events_sample.csv"

# A tiny file where every expected value can be checked by eye. Rows are deliberately out
# of time order, and the last row duplicates the first.
TINY_CSV = """timestamp,visitorid,event,itemid,transactionid
1433221332117,257597,view,355908,
1433221000000,111016,addtocart,318965,
1433222000000,111016,transaction,318965,4000
1433221332117,257597,view,355908,
"""


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(TINY_CSV)
    return load_retailrocket(path)


def test_columns_match_the_events_table(tiny):
    events, _ = tiny
    assert list(events.columns) == EVENT_COLUMNS
    # The same names as Efe's table, so a database loader can produce identical data.
    assert set(EVENT_COLUMNS) <= set(EventRow.__table__.columns.keys())


def test_event_names_are_mapped(tiny):
    events, _ = tiny
    assert set(events["event_type"]) == {"view", "cart", "purchase"}


def test_ids_are_strings_not_numbers(tiny):
    events, _ = tiny
    assert events["user_id"].tolist().count("111016") == 2
    assert "355908" in events["item_id"].tolist()


def test_order_id_is_not_turned_into_a_float(tiny):
    events, _ = tiny
    purchase = events[events["event_type"] == "purchase"].iloc[0]
    assert purchase["order_id"] == "4000"  # not "4000.0"


def test_order_id_only_on_purchases(tiny):
    events, _ = tiny
    not_purchases = events[events["event_type"] != "purchase"]
    assert not_purchases["order_id"].isna().all()


def test_timestamps_are_utc(tiny):
    events, _ = tiny
    assert str(events["timestamp"].dt.tz) == "UTC"
    # 1433221000000 ms after 1970-01-01, the earliest row in TINY_CSV.
    assert events["timestamp"].iloc[0] == pd.Timestamp("2015-06-02 04:56:40", tz="UTC")


def test_duplicates_are_dropped_and_counted(tiny):
    events, duplicates = tiny
    assert duplicates == 1
    assert len(events) == 3


def test_sorted_by_time(tiny):
    events, _ = tiny
    assert events["timestamp"].is_monotonic_increasing


def test_unknown_event_type_fails_loudly(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text("timestamp,visitorid,event,itemid,transactionid\n1,1,removefromcart,1,\n")
    with pytest.raises(ValueError, match="removefromcart"):
        load_retailrocket(path)


def test_loads_the_committed_fixture():
    events, _ = load_retailrocket(FIXTURE)
    assert len(events) > 10_000
    assert set(events["event_type"]) == {"view", "cart", "purchase"}
    assert events["timestamp"].is_monotonic_increasing
