"""Building the answer key: who is scored, and what counts as a hit."""

import pandas as pd

from core.data.interactions import build_interactions
from core.evaluation.answer_key import CART_OR_PURCHASE, build_answer_key


def events_from(rows: list[tuple[str, str, str]], day: str) -> pd.DataFrame:
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


def train(rows):
    return build_interactions(events_from(rows, "2015-06-01"))


def later(rows):
    return events_from(rows, "2015-08-25")


# Training: a touched x, b touched y. The grid knows users a, b and items x, y.
GRID = train([("a", "x", "view"), ("b", "y", "view")])


def test_a_new_known_item_is_a_hit():
    key = build_answer_key(GRID, later([("a", "y", "view")]))
    assert key.relevant == {"a": frozenset({"y"})}


def test_unknown_users_are_not_scored():
    key = build_answer_key(GRID, later([("stranger", "x", "view")]))
    assert key.relevant == {}


def test_items_already_touched_in_training_do_not_count():
    # a goes back to x: real behaviour, but models are not allowed to recommend x to a.
    key = build_answer_key(GRID, later([("a", "x", "purchase"), ("a", "y", "view")]))
    assert key.relevant == {"a": frozenset({"y"})}


def test_items_not_in_the_grid_do_not_count():
    key = build_answer_key(GRID, later([("a", "brand-new", "view"), ("a", "y", "view")]))
    assert key.relevant == {"a": frozenset({"y"})}


def test_users_with_nothing_countable_are_left_out():
    # a only re-visits x; b only touches an unknown item. Neither can be scored.
    key = build_answer_key(GRID, later([("a", "x", "view"), ("b", "brand-new", "view")]))
    assert key.relevant == {}
    assert key.known_users == 2


def test_several_events_on_one_item_count_once():
    key = build_answer_key(
        GRID, later([("a", "y", "view"), ("a", "y", "cart"), ("a", "y", "purchase")])
    )
    assert key.relevant == {"a": frozenset({"y"})}


def test_cart_or_purchase_ignores_views():
    future = later([("a", "y", "view"), ("b", "x", "cart")])
    key = build_answer_key(GRID, future, CART_OR_PURCHASE)
    assert key.relevant == {"b": frozenset({"x"})}


def test_counts_at_each_stage():
    future = later([("a", "y", "view"), ("b", "y", "view"), ("stranger", "x", "view")])
    key = build_answer_key(GRID, future)
    assert key.users_in_period == 3  # a, b, stranger
    assert key.known_users == 2  # a, b
    assert key.scorable_users == 1  # only a: b already touched y in training


def test_no_future_events_gives_an_empty_key():
    key = build_answer_key(GRID, later([]))
    assert key.relevant == {}
    assert key.users_in_period == 0
