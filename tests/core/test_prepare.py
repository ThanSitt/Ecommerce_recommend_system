"""The steps together: load -> split -> filter -> grid."""

from pathlib import Path

import pandas as pd

from core.data.prepare import prepare_events
from core.data.retailrocket import load_retailrocket
from core.data.split import RETAILROCKET_TEST_START, RETAILROCKET_VALIDATION_START

FIXTURE = Path(__file__).parent.parent / "fixtures" / "events_sample.csv"

VALIDATION_START = pd.Timestamp("2015-08-21", tz="UTC")
TEST_START = pd.Timestamp("2015-09-04", tz="UTC")
TRAIN_DAY, VALIDATION_DAY, TEST_DAY = "2015-06-01", "2015-08-25", "2015-09-10"


def events_from(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """One view per (user, item, day) row."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _, _ in rows],
            "item_id": [i for _, i, _ in rows],
            "event_type": "view",
            "timestamp": pd.to_datetime([d for _, _, d in rows], utc=True),
            "order_id": None,
        }
    )


def prepare(events, **settings):
    return prepare_events(events, VALIDATION_START, TEST_START, **settings)


def test_grid_contains_only_the_training_period():
    events = events_from([
        ("a", "x", TRAIN_DAY),
        ("b", "y", VALIDATION_DAY),
        ("c", "z", TEST_DAY),
    ])
    prepared = prepare(events)

    assert prepared.grid.user_ids == ["a"]
    assert prepared.grid.item_ids == ["x"]
    assert prepared.validation["user_id"].tolist() == ["b"]
    assert prepared.test["user_id"].tolist() == ["c"]


def test_filtering_does_not_look_at_the_future():
    # In training, item y has 1 user, so with min 2 it must go. In validation it becomes
    # popular. If filtering looked at the whole file, y would wrongly survive.
    events = events_from([
        ("a", "x", TRAIN_DAY), ("b", "x", TRAIN_DAY), ("a", "y", TRAIN_DAY),
        ("c", "y", VALIDATION_DAY), ("d", "y", VALIDATION_DAY), ("e", "y", VALIDATION_DAY),
    ])
    prepared = prepare(events, min_users_per_item=2)

    assert "y" not in prepared.grid.item_index


def test_validation_and_test_are_not_filtered():
    # One-event users are removed from training, but kept in validation and test: the
    # evaluation decides what to do with them, not the filter.
    events = events_from([
        ("a", "x", TRAIN_DAY), ("a", "y", TRAIN_DAY), ("b", "x", TRAIN_DAY),
        ("q", "x", VALIDATION_DAY),
        ("r", "y", TEST_DAY),
    ])
    prepared = prepare(events, min_items_per_user=2)

    assert prepared.grid.user_ids == ["a"]
    assert prepared.validation["user_id"].tolist() == ["q"]
    assert prepared.test["user_id"].tolist() == ["r"]


def test_report_and_grid_agree():
    events = events_from([
        ("a", "x", TRAIN_DAY), ("a", "y", TRAIN_DAY), ("b", "x", TRAIN_DAY),
    ])
    prepared = prepare(events, min_items_per_user=2)

    assert prepared.filter_report.users_after == prepared.grid.n_users
    assert prepared.filter_report.items_after == prepared.grid.n_items
    assert len(prepared.train_events) == prepared.filter_report.events_after


def test_whole_pipeline_on_the_committed_fixture():
    # Looser than the Retailrocket settings: the fixture's 739 visitors are too few for
    # "items need 5 users", which leaves nothing (the small-store problem, Phase 5).
    events, _ = load_retailrocket(FIXTURE)
    prepared = prepare_events(
        events, RETAILROCKET_VALIDATION_START, RETAILROCKET_TEST_START,
        min_items_per_user=2, min_users_per_item=2,
    )

    assert prepared.grid.n_users > 0 and prepared.grid.n_items > 0
    assert prepared.grid.events_to < RETAILROCKET_VALIDATION_START
    assert len(prepared.validation) > 0 and len(prepared.test) > 0
    assert prepared.grid.matrix.getnnz(axis=1).min() >= 2
