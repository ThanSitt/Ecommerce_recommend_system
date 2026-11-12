"""The answer keys: for each user, the products they actually touched in a later period.

Every model is marked against the same keys, built once from the evaluation period and
independent of any model, so random, most-popular and ALS are compared on identical
answers. There are two exams, one for each question a model can be asked:

  Personal (`recommend_for_user`, docs/decisions/0004): "what will this user touch next?"
  Scores users and products in the training grid, so its key changes with the filter.

  Product page (`similar_items`, docs/decisions/0005 and 0006): "this user is on this
  product's page, what will they touch after it?" The question the live endpoint asks. It
  uses no history, so it scores users no model has seen, and it covers the whole site,
  including pages the model cannot answer. Its key comes from the events alone, so it stays
  the same whatever the filter or model, and scores can be compared across both.

In both, a user gets no credit for being shown a product they touched in training.

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


@dataclass
class ProductPageKey:
    """Who is scored on the product page exam, on which page, and what counts as a hit."""

    # user_id -> the product whose page the model is asked about
    questions: dict[str, str]
    # user_id -> the products that count as hits for that user
    relevant: dict[str, frozenset[str]]
    event_types: tuple[str, ...]
    # How many users there were at each stage, for the thesis.
    users_in_period: int
    users_with_question: int

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
    hits = events[~_touched_in_training(grid, events)]

    return AnswerKey(
        relevant=_answers_per_user(hits),
        event_types=tuple(event_types),
        users_in_period=users_in_period,
        known_users=known_users,
    )


def build_product_page_key(
    train_all: pd.DataFrame,
    future: pd.DataFrame,
    event_types: tuple[str, ...] = ANY_INTERACTION,
) -> ProductPageKey:
    """Build the product page key from `future` events (validation, or test in Phase 6).

    `train_all` is every training event, before any filtering. No grid is involved, so the
    key is the same for every filter and every model.

    The question is the first product the user viewed, once per user, so every user counts
    the same as in the personal exam. It may be a product the model cannot answer: the site
    has those pages too, and the fallback serves them. A view is the question whichever
    `event_types` is, because a product page visit is a view.

    The answers are the products they first touched after that, among those seen in training
    (no setting could recommend a product nobody had touched) and not touched by that user in
    training. Retailrocket has no sessions, so "after" can mean days later.
    """
    ordered = future.sort_values("timestamp", kind="stable")
    views = ordered[ordered["event_type"] == "view"]
    first_views = views.drop_duplicates("user_id").set_index("user_id")

    events = ordered[ordered["user_id"].isin(first_views.index)]
    if events.empty:
        answers: dict[str, frozenset[str]] = {}
    else:
        asked_at = events["user_id"].map(first_views["timestamp"])
        # First touch, not any touch: this rules out the question's own product, and any
        # product the user had already touched before reaching the page.
        first_touch = events.groupby(["user_id", "item_id"])["timestamp"].transform("min")
        candidates = events[
            (first_touch > asked_at)
            & events["event_type"].isin(event_types)
            & events["item_id"].isin(train_all["item_id"])
        ]
        touched = train_all[["user_id", "item_id"]].drop_duplicates()
        marked = candidates.merge(touched, on=["user_id", "item_id"], how="left", indicator=True)
        answers = _answers_per_user(marked[marked["_merge"] == "left_only"])

    return ProductPageKey(
        questions={user: first_views.at[user, "item_id"] for user in answers},
        relevant=answers,
        event_types=tuple(event_types),
        users_in_period=future["user_id"].nunique(),
        users_with_question=len(first_views),
    )


def _touched_in_training(grid: Interactions, events: pd.DataFrame) -> np.ndarray:
    """For each event on a grid item: had this user already touched it in training?

    A filled cell in the training grid means yes. Users the grid doesn't know touched
    nothing it knows of.
    """
    touched = np.zeros(len(events), dtype=bool)
    known = events["user_id"].isin(grid.user_index).to_numpy()
    # Checked first: scipy returns a different type when indexing with nothing.
    if known.any():
        rows = events["user_id"][known].map(grid.user_index).to_numpy()
        cols = events["item_id"][known].map(grid.item_index).to_numpy()
        touched[known] = np.asarray(grid.matrix[rows, cols]).ravel() != 0
    return touched


def _answers_per_user(hits: pd.DataFrame) -> dict[str, frozenset[str]]:
    return {user: frozenset(items) for user, items in hits.groupby("user_id")["item_id"]}
