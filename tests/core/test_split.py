"""Splitting events by time."""

import pandas as pd
import pytest

from core.data.split import split_by_time

VALIDATION_START = pd.Timestamp("2015-08-21", tz="UTC")
TEST_START = pd.Timestamp("2015-09-04", tz="UTC")


def events_at(*times: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(len(times))],
            "item_id": "i1",
            "event_type": "view",
            # ISO8601 accepts both "2015-08-21" and "2015-08-21 23:59:59" in one list.
            "timestamp": pd.to_datetime(list(times), utc=True, format="ISO8601"),
            "order_id": None,
        }
    )


def test_each_event_goes_to_the_right_period():
    events = events_at("2015-08-01", "2015-08-25", "2015-09-10")
    split = split_by_time(events, VALIDATION_START, TEST_START)

    assert split.train["user_id"].tolist() == ["u0"]
    assert split.validation["user_id"].tolist() == ["u1"]
    assert split.test["user_id"].tolist() == ["u2"]


def test_an_event_exactly_on_a_cutoff_belongs_to_the_later_period():
    events = events_at("2015-08-21 00:00:00", "2015-09-04 00:00:00")
    split = split_by_time(events, VALIDATION_START, TEST_START)

    assert len(split.train) == 0
    assert split.validation["user_id"].tolist() == ["u0"]
    assert split.test["user_id"].tolist() == ["u1"]


def test_no_event_is_lost_or_counted_twice():
    events = events_at("2015-05-03", "2015-08-20 23:59:59", "2015-08-21", "2015-09-03",
                       "2015-09-04", "2015-09-18")
    split = split_by_time(events, VALIDATION_START, TEST_START)

    assert len(split.train) + len(split.validation) + len(split.test) == len(events)


def test_every_train_event_is_before_every_test_event():
    events = events_at("2015-09-10", "2015-06-01", "2015-08-25", "2015-05-05")
    split = split_by_time(events, VALIDATION_START, TEST_START)

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_cutoffs_without_timezone_are_rejected():
    with pytest.raises(ValueError, match="timezone"):
        split_by_time(events_at("2015-08-01"), pd.Timestamp("2015-08-21"), TEST_START)


def test_cutoffs_in_the_wrong_order_are_rejected():
    with pytest.raises(ValueError, match="before"):
        split_by_time(events_at("2015-08-01"), TEST_START, VALIDATION_START)
