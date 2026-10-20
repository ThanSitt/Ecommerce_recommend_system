"""Filtering quiet users, rare items and extreme accounts."""

import pandas as pd

from core.data.filtering import filter_interactions


def events_from(pairs: list[tuple[str, str]], event_type: str = "view") -> pd.DataFrame:
    """One event per (user, item) pair."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _ in pairs],
            "item_id": [i for _, i in pairs],
            "event_type": event_type,
            "timestamp": pd.Timestamp("2015-06-01", tz="UTC"),
            "order_id": None,
        }
    )


def test_no_limits_keeps_everything():
    events = events_from([("a", "x"), ("b", "y")])
    kept, report = filter_interactions(events)
    assert len(kept) == 2
    assert report.users_after == report.users_before == 2


def test_quiet_users_are_removed():
    # a touched 2 items, b touched 1.
    events = events_from([("a", "x"), ("a", "y"), ("b", "x")])
    kept, _ = filter_interactions(events, min_items_per_user=2)
    assert set(kept["user_id"]) == {"a"}


def test_rare_items_are_removed():
    # x was touched by 2 users, y by 1.
    events = events_from([("a", "x"), ("b", "x"), ("a", "y")])
    kept, _ = filter_interactions(events, min_users_per_item=2)
    assert set(kept["item_id"]) == {"x"}


def test_repeat_views_of_one_item_count_once():
    # 5 events, but all on the same item: one filled cell, so still a "1-item" user.
    events = events_from([("a", "x")] * 5)
    kept, _ = filter_interactions(events, min_items_per_user=2)
    assert kept.empty


def test_repeats_until_stable():
    # With min 2 for both: item z has 1 user, so it goes. That leaves c with only item y,
    # so c goes. That leaves y with only user b, so y goes. That leaves b with only x...
    # Only the a/b x/w block, where everyone has 2, survives.
    events = events_from([
        ("a", "x"), ("a", "w"),
        ("b", "x"), ("b", "w"), ("b", "y"),
        ("c", "y"), ("c", "z"),
    ])
    kept, report = filter_interactions(events, min_items_per_user=2, min_users_per_item=2)

    assert set(kept["user_id"]) == {"a", "b"}
    assert set(kept["item_id"]) == {"x", "w"}
    assert report.rounds > 1
    # Every survivor meets both rules.
    assert kept.groupby("user_id")["item_id"].nunique().min() >= 2
    assert kept.groupby("item_id")["user_id"].nunique().min() >= 2


def test_extreme_accounts_are_removed_by_event_count():
    events = pd.concat([
        events_from([("bot", "x")] * 10),
        events_from([("a", "x"), ("a", "y")]),
    ])
    kept, report = filter_interactions(events, max_events_per_user=5)
    assert "bot" not in set(kept["user_id"])
    assert report.heavy_users_removed == 1


def test_report_counts_purchases():
    events = pd.concat([
        events_from([("a", "x")], event_type="purchase"),
        events_from([("b", "y")], event_type="purchase"),
        events_from([("a", "y")]),
    ])
    _, report = filter_interactions(events, min_items_per_user=2)
    assert report.purchases_before == 2
    assert report.purchases_after == 1  # b is removed, and their purchase with them
