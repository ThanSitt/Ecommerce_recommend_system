"""Load -> split -> filter -> grid, in one call.

Every later phase (evaluation, tuning, training) starts from the same prepared data, so
the order of the steps is fixed here once instead of being retyped, and possibly
reordered, in each of them. The order is the point: split first, then filter using the
training period only (see core/data/filtering.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.data.filtering import (
    RETAILROCKET_MAX_EVENTS_PER_USER,
    RETAILROCKET_MIN_ITEMS_PER_USER,
    RETAILROCKET_MIN_USERS_PER_ITEM,
    FilterReport,
    filter_interactions,
)
from core.data.interactions import Interactions, build_interactions
from core.data.retailrocket import load_retailrocket
from core.data.split import (
    RETAILROCKET_TEST_START,
    RETAILROCKET_VALIDATION_START,
    split_by_time,
)


@dataclass
class PreparedData:
    """Everything a model needs to be trained and scored."""

    # What fit(data) receives.
    grid: Interactions
    # The filtered training events behind the grid, with timestamps kept, for models
    # that care about *when* (recently popular).
    train_events: pd.DataFrame
    # Not filtered: users and items the model never saw are the evaluation's problem.
    validation: pd.DataFrame
    # Sealed until Phase 6.
    test: pd.DataFrame
    filter_report: FilterReport
    duplicates_dropped: int = 0


def prepare_events(
    events: pd.DataFrame,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    min_items_per_user: int = 1,
    min_users_per_item: int = 1,
    max_events_per_user: int | None = None,
    weights: dict[str, float] | None = None,
) -> PreparedData:
    """Split, filter the training period, build the grid. Works on any events in the
    contract's columns, so the production pipeline can use it on a store's events."""
    split = split_by_time(events, validation_start, test_start)
    train_events, report = filter_interactions(
        split.train, min_items_per_user, min_users_per_item, max_events_per_user
    )
    return PreparedData(
        grid=build_interactions(train_events, weights),
        train_events=train_events,
        validation=split.validation,
        test=split.test,
        filter_report=report,
    )


def prepare_retailrocket(
    path: Path | str, weights: dict[str, float] | None = None
) -> PreparedData:
    """The Retailrocket evaluation data, with the settings recorded in
    docs/decisions/0003. Takes ~40 seconds on the full file, mostly the filter.

    Needs the full file: on a small sample "items need 5 users" leaves nothing, and
    building the grid fails. Use prepare_events with looser settings for samples."""
    events, duplicates = load_retailrocket(path)
    prepared = prepare_events(
        events,
        RETAILROCKET_VALIDATION_START,
        RETAILROCKET_TEST_START,
        min_items_per_user=RETAILROCKET_MIN_ITEMS_PER_USER,
        min_users_per_item=RETAILROCKET_MIN_USERS_PER_ITEM,
        max_events_per_user=RETAILROCKET_MAX_EVENTS_PER_USER,
        weights=weights,
    )
    prepared.duplicates_dropped = duplicates
    return prepared
