"""The answer key: for each user, the products they actually touched in a later period.

Every model is marked against the same key, built once from the evaluation period and
independent of any model, so random, most-popular and ALS are compared on identical
answers. See docs/decisions/0004.

A product counts towards a user's answers only if a model could have recommended it:

  - the user is in the training grid (otherwise no model knows them),
  - the product is in the training grid (otherwise no model knows it),
  - the user had not touched it in training (models never recommend those).

Users left with no answers are not scored at all: they would score 0 for every model,
which says nothing about the models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.data.interactions import Interactions

# What counts as a hit. The main number uses any interaction; cart or purchase is
# reported second, with a warning, because far fewer users have one.
ANY_INTERACTION = ("view", "cart", "purchase")
CART_OR_PURCHASE = ("cart", "purchase")


@dataclass
class AnswerKey:
    """Who is scored and what counts as a hit for them."""

    # user_id -> the products that count as hits for that user
    relevant: dict[str, frozenset[str]]
    event_types: tuple[str, ...]
    # How many users there were at each stage, for the thesis.
    users_in_period: int
    known_users: int

    @property
    def scorable_users(self) -> int:
        return len(self.relevant)


def build_answer_key(
    grid: Interactions,
    future: pd.DataFrame,
    event_types: tuple[str, ...] = ANY_INTERACTION,
) -> AnswerKey:
    """Build the key from `future` events (validation, or test in Phase 6)."""
    users_in_period = future["user_id"].nunique()
    known = future[future["user_id"].isin(grid.user_index)]
    known_users = known["user_id"].nunique()

    events = known[
        known["event_type"].isin(event_types) & known["item_id"].isin(grid.item_index)
    ]
    if events.empty:
        # Checked separately: scipy returns a different type when indexing with nothing.
        hits = events
    else:
        rows = events["user_id"].map(grid.user_index).to_numpy()
        cols = events["item_id"].map(grid.item_index).to_numpy()
        # A filled cell in the training grid = the user already touched that item.
        seen_before = np.asarray(grid.matrix[rows, cols]).ravel() != 0
        hits = events[~seen_before]

    relevant = {
        user: frozenset(items) for user, items in hits.groupby("user_id")["item_id"]
    }
    return AnswerKey(
        relevant=relevant,
        event_types=tuple(event_types),
        users_in_period=users_in_period,
        known_users=known_users,
    )
