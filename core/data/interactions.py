"""Turn filtered training events into the user x item grid every model learns from.

This is the `data` that `Recommender.fit(data)` receives. It is built once and handed to
every model, so all six learn from exactly the same thing.

One row per user, one column per item. A cell holds how strongly that user showed
interest in that item; an empty cell means "never touched", not "disliked".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# How much each event says about interest. A cart is a stronger signal than a view, a
# purchase stronger than a cart. Starting choice, to be tuned on validation in Phase 4;
# see docs/decisions/0003.
DEFAULT_WEIGHTS = {"view": 1.0, "cart": 3.0, "purchase": 5.0}


@dataclass
class Interactions:
    """The grid plus the maps between real IDs and grid positions.

    Models only ever see positions (row 0, column 17). The maps turn those back into
    the IDs a store understands, which is why a saved model must keep them.
    """

    matrix: csr_matrix  # users x items, float32
    user_ids: list[str]  # row number -> user_id
    item_ids: list[str]  # column number -> item_id
    weights: dict[str, float]
    # The time window the events came from, for ModelMetadata.events_from / events_to.
    events_from: pd.Timestamp
    events_to: pd.Timestamp
    user_index: dict[str, int] = field(init=False, repr=False)  # user_id -> row number
    item_index: dict[str, int] = field(init=False, repr=False)  # item_id -> column number

    def __post_init__(self) -> None:
        self.user_index = {user: row for row, user in enumerate(self.user_ids)}
        self.item_index = {item: col for col, item in enumerate(self.item_ids)}

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def build_interactions(
    events: pd.DataFrame, weights: dict[str, float] | None = None
) -> Interactions:
    """Build the grid from events that have already been split and filtered.

    When a user has several events on the same item, the cell keeps the **strongest**
    one, not the sum: viewing a product twenty times is still "viewed", and summing
    would let a few obsessive clickers outweigh a purchase.

    Rows and columns are in sorted ID order, so the same events always give the same
    grid, whatever order they arrive in.
    """
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if events.empty:
        raise ValueError("no events to build a grid from")
    unknown = set(events["event_type"].unique()) - set(weights)
    if unknown:
        raise ValueError(f"no weight for event types: {sorted(unknown)}")
    # A zero would be stored as an empty cell, indistinguishable from "never touched".
    if any(w <= 0 for w in weights.values()):
        raise ValueError("weights must be positive")

    cells = (
        events.assign(weight=events["event_type"].map(weights))
        .groupby(["user_id", "item_id"], sort=False)["weight"]
        .max()
        .reset_index()
    )
    user_rows, user_ids = pd.factorize(cells["user_id"], sort=True)
    item_cols, item_ids = pd.factorize(cells["item_id"], sort=True)
    matrix = csr_matrix(
        (cells["weight"].to_numpy(dtype=np.float32), (user_rows, item_cols)),
        shape=(len(user_ids), len(item_ids)),
    )

    return Interactions(
        matrix=matrix,
        user_ids=user_ids.tolist(),
        item_ids=item_ids.tolist(),
        weights=weights,
        events_from=events["timestamp"].min(),
        events_to=events["timestamp"].max(),
    )
