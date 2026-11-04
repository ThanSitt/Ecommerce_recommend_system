"""Building the answer key: who is scored, and what counts as a hit."""

import pandas as pd

from core.data.interactions import build_interactions
from core.evaluation.answer_key import (
    CART_OR_PURCHASE,
    build_answer_key,
    build_product_page_key,
)


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


# ---- The product page exam -----------------------------------------------------------


def timed(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """One event per (user, item, event_type, time of day) row, all in the later period."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _, _, _ in rows],
            "item_id": [i for _, i, _, _ in rows],
            "event_type": [e for _, _, e, _ in rows],
            "timestamp": pd.to_datetime([f"2015-08-25 {t}" for _, _, _, t in rows], utc=True),
            "order_id": None,
        }
    )


# Training: a touched lamp; b touched rug and vase. No one else is known.
SHOP = train([("a", "lamp", "view"), ("b", "rug", "view"), ("b", "vase", "view")])


def test_question_is_the_first_view_and_answers_come_after_it():
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"),
        ("anna", "rug", "view", "11:00"),
        ("anna", "vase", "purchase", "12:00"),
    ]))
    assert key.questions == {"anna": "lamp"}
    assert key.relevant == {"anna": frozenset({"rug", "vase"})}


def test_users_no_model_has_seen_are_scored():
    # The whole point of this exam: "anna" is not in the training grid.
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"), ("anna", "rug", "view", "11:00"),
    ]))
    assert "anna" not in SHOP.user_index
    assert key.scorable_users == 1


def test_a_product_the_model_does_not_know_cannot_be_the_question():
    key = build_product_page_key(SHOP, timed([
        ("anna", "brand-new", "view", "09:00"),
        ("anna", "lamp", "view", "10:00"),
        ("anna", "rug", "view", "11:00"),
    ]))
    assert key.questions == {"anna": "lamp"}


def test_a_product_the_model_does_not_know_cannot_be_an_answer():
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"), ("anna", "brand-new", "view", "11:00"),
    ]))
    assert key.relevant == {}


def test_going_back_to_the_question_product_does_not_count():
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"), ("anna", "lamp", "purchase", "11:00"),
    ]))
    assert key.relevant == {}


def test_products_touched_before_the_question_do_not_count():
    # Anna had the rug in her cart before reaching the lamp page.
    key = build_product_page_key(SHOP, timed([
        ("anna", "rug", "cart", "09:00"),
        ("anna", "lamp", "view", "10:00"),
        ("anna", "rug", "view", "11:00"),
        ("anna", "vase", "view", "12:00"),
    ]))
    assert key.relevant == {"anna": frozenset({"vase"})}


def test_products_touched_in_training_do_not_count():
    # b touched rug and vase in training, so only the lamp would be new, and it is b's question.
    key = build_product_page_key(SHOP, timed([
        ("b", "lamp", "view", "10:00"), ("b", "rug", "view", "11:00"),
    ]))
    assert key.relevant == {}


def test_a_cart_cannot_be_the_question():
    # A page visit is a view. The rug is the first one.
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "cart", "10:00"),
        ("anna", "rug", "view", "11:00"),
        ("anna", "vase", "view", "12:00"),
    ]))
    assert key.questions == {"anna": "rug"}


def test_cart_or_purchase_only_changes_the_answers():
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"),
        ("anna", "rug", "view", "11:00"),
        ("anna", "vase", "purchase", "12:00"),
    ]), CART_OR_PURCHASE)
    assert key.questions == {"anna": "lamp"}
    assert key.relevant == {"anna": frozenset({"vase"})}


def test_product_page_counts_at_each_stage():
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"), ("anna", "rug", "view", "11:00"),  # scored
        ("bo", "lamp", "view", "10:00"),  # one click: a question, nothing after it
        ("cy", "brand-new", "view", "10:00"),  # no question at all
    ]))
    assert key.users_in_period == 3
    assert key.users_with_question == 2
    assert key.scorable_users == 1


def test_only_scored_users_have_a_question():
    key = build_product_page_key(SHOP, timed([
        ("anna", "lamp", "view", "10:00"), ("anna", "rug", "view", "11:00"),
        ("bo", "lamp", "view", "10:00"),
    ]))
    assert set(key.questions) == set(key.relevant) == {"anna"}


def test_no_future_events_gives_an_empty_product_page_key():
    key = build_product_page_key(SHOP, timed([]))
    assert key.relevant == {}
    assert key.users_with_question == 0
