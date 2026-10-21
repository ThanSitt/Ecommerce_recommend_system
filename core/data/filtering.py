"""Remove users and items with too little activity to learn from.

Run on the **training period only**, after the time split. Filtering the whole dataset
would decide who to keep using future activity, which leaks the test period into training.

Validation and test events are never filtered here. Users and items the model never saw
are the evaluation's problem (Phase 2), not something to hide by deleting them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Retailrocket settings; see docs/decisions/0003.
# 2 is the smallest useful minimum: a user who touched one item says nothing about which
# items go together. Items touched by fewer than 5 users have no reliable similarity.
# Starting choice, to be checked against validation results in Phase 4.
RETAILROCKET_MIN_ITEMS_PER_USER = 2
RETAILROCKET_MIN_USERS_PER_ITEM = 5
# More than 1,000 events in the 110-day training period is ~9 a day, every day: 23 accounts
# that behave like resellers or bots, not shoppers, yet make 14.5% of all purchases.
RETAILROCKET_MAX_EVENTS_PER_USER = 1000


@dataclass
class FilterReport:
    """What the filter removed. Numbers for the thesis, not for the models."""

    users_before: int
    items_before: int
    events_before: int
    purchases_before: int
    heavy_users_removed: int
    users_after: int
    items_after: int
    events_after: int
    purchases_after: int
    rounds: int


def filter_interactions(
    events: pd.DataFrame,
    min_items_per_user: int = 1,
    min_users_per_item: int = 1,
    max_events_per_user: int | None = None,
) -> tuple[pd.DataFrame, FilterReport]:
    """Keep users who touched at least `min_items_per_user` different items, and items
    touched by at least `min_users_per_item` different users.

    Counts *different* items and users, not events: someone who viewed one product twenty
    times has twenty events but only one filled cell in the grid, and teaches the model
    nothing about which products go together.

    Repeats until nothing changes, because the two rules affect each other: removing a
    rare item can push a user below the minimum, and removing that user can push another
    item below its minimum. After this, every remaining user and item meets both rules.

    `max_events_per_user` removes extreme accounts first, by raw event count. None keeps
    everyone.
    """
    before = _counts(events)

    heavy_users_removed = 0
    if max_events_per_user is not None:
        per_user = events.groupby("user_id").size()
        heavy = per_user[per_user > max_events_per_user].index
        heavy_users_removed = len(heavy)
        events = events[~events["user_id"].isin(heavy)]

    rounds = 0
    while True:
        rounds += 1
        items_per_user = events.groupby("user_id")["item_id"].nunique()
        users_per_item = events.groupby("item_id")["user_id"].nunique()
        keep = events["user_id"].isin(
            items_per_user[items_per_user >= min_items_per_user].index
        ) & events["item_id"].isin(users_per_item[users_per_item >= min_users_per_item].index)
        if keep.all():
            break
        events = events[keep]

    events = events.reset_index(drop=True)
    after = _counts(events)
    report = FilterReport(
        users_before=before[0],
        items_before=before[1],
        events_before=before[2],
        purchases_before=before[3],
        heavy_users_removed=heavy_users_removed,
        users_after=after[0],
        items_after=after[1],
        events_after=after[2],
        purchases_after=after[3],
        rounds=rounds,
    )
    return events, report


def _counts(events: pd.DataFrame) -> tuple[int, int, int, int]:
    return (
        events["user_id"].nunique(),
        events["item_id"].nunique(),
        len(events),
        int((events["event_type"] == "purchase").sum()),
    )
