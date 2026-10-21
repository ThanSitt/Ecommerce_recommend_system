"""Building the user x item grid."""

from pathlib import Path

import pandas as pd
import pytest

from core.data.filtering import filter_interactions
from core.data.interactions import DEFAULT_WEIGHTS, build_interactions
from core.data.retailrocket import load_retailrocket

FIXTURE = Path(__file__).parent.parent / "fixtures" / "events_sample.csv"


def events_from(rows: list[tuple[str, str, str]], day: str = "2015-06-01") -> pd.DataFrame:
    """One event per (user, item, event_type) row."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _, _ in rows],
            "item_id": [i for _, i, _ in rows],
            "event_type": [e for _, _, e in rows],
            "timestamp": pd.Timestamp(day, tz="UTC"),
            "order_id": None,
        }
    )


def cell(grid, user: str, item: str) -> float:
    return grid.matrix[grid.user_index[user], grid.item_index[item]]


def test_each_event_type_gets_its_weight():
    grid = build_interactions(
        events_from([("a", "x", "view"), ("a", "y", "cart"), ("a", "z", "purchase")])
    )
    assert cell(grid, "a", "x") == DEFAULT_WEIGHTS["view"]
    assert cell(grid, "a", "y") == DEFAULT_WEIGHTS["cart"]
    assert cell(grid, "a", "z") == DEFAULT_WEIGHTS["purchase"]


def test_strongest_event_wins_not_the_sum():
    grid = build_interactions(
        events_from([("a", "x", "view"), ("a", "x", "view"), ("a", "x", "purchase")])
    )
    assert cell(grid, "a", "x") == DEFAULT_WEIGHTS["purchase"]


def test_repeat_views_fill_one_cell_once():
    grid = build_interactions(events_from([("a", "x", "view")] * 20))
    assert grid.matrix.nnz == 1
    assert cell(grid, "a", "x") == DEFAULT_WEIGHTS["view"]


def test_untouched_cells_are_empty():
    grid = build_interactions(events_from([("a", "x", "view"), ("b", "y", "view")]))
    assert grid.matrix.shape == (2, 2)
    assert grid.matrix.nnz == 2
    assert cell(grid, "a", "y") == 0


def test_maps_go_both_ways():
    grid = build_interactions(events_from([("b", "y", "view"), ("a", "x", "cart")]))
    for user, row in grid.user_index.items():
        assert grid.user_ids[row] == user
    for item, col in grid.item_index.items():
        assert grid.item_ids[col] == item


def test_same_events_in_any_order_give_the_same_grid():
    rows = [("b", "y", "view"), ("a", "x", "cart"), ("c", "x", "purchase")]
    first = build_interactions(events_from(rows))
    second = build_interactions(events_from(list(reversed(rows))))

    assert first.user_ids == second.user_ids == ["a", "b", "c"]
    assert first.item_ids == second.item_ids == ["x", "y"]
    assert (first.matrix != second.matrix).nnz == 0


def test_custom_weights_are_used_and_kept():
    weights = {"view": 1.0, "cart": 2.0, "purchase": 10.0}
    grid = build_interactions(events_from([("a", "x", "purchase")]), weights)
    assert cell(grid, "a", "x") == 10.0
    assert grid.weights == weights


def test_time_window_is_recorded():
    events = pd.concat([
        events_from([("a", "x", "view")], day="2015-06-01"),
        events_from([("a", "y", "view")], day="2015-07-15"),
    ])
    grid = build_interactions(events)
    assert grid.events_from == pd.Timestamp("2015-06-01", tz="UTC")
    assert grid.events_to == pd.Timestamp("2015-07-15", tz="UTC")


def test_event_type_without_a_weight_fails_loudly():
    with pytest.raises(ValueError, match="no weight"):
        build_interactions(events_from([("a", "x", "wishlist")]))


def test_zero_weight_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        build_interactions(
            events_from([("a", "x", "view")]), {"view": 0.0, "cart": 3.0, "purchase": 5.0}
        )


def test_no_events_fails_loudly():
    with pytest.raises(ValueError, match="no events"):
        build_interactions(events_from([]))


def test_builds_from_the_committed_fixture():
    events, _ = load_retailrocket(FIXTURE)
    kept, report = filter_interactions(events, min_items_per_user=2, min_users_per_item=2)
    grid = build_interactions(kept)

    assert grid.n_users == report.users_after
    assert grid.n_items == report.items_after
    assert grid.matrix.nnz == kept[["user_id", "item_id"]].drop_duplicates().shape[0]
