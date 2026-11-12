"""Scoring a model end to end."""

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.evaluation.answer_key import build_answer_key, build_product_page_key
from core.evaluation.run import PRODUCT_PAGE, evaluate, evaluate_product_page, main
from core.models.base import ColdStartError
from core.models.baselines import MostPopularRecommender


def events_from(pairs: list[tuple[str, str]], day: str) -> pd.DataFrame:
    """One view per (user, item) pair."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _ in pairs],
            "item_id": [i for _, i in pairs],
            "event_type": "view",
            "timestamp": pd.Timestamp(day, tz="UTC"),
            "order_id": None,
        }
    )


def views_at(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """One view per (user, item, time of day) row, in the later period."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _, _ in rows],
            "item_id": [i for _, i, _ in rows],
            "event_type": "view",
            "timestamp": pd.to_datetime([f"2015-08-25 {t}" for _, _, t in rows], utc=True),
            "order_id": None,
        }
    )


# Training: x touched by 3 users, y by 2, z by 1.
GRID = build_interactions(events_from(
    [("a", "x"), ("b", "x"), ("c", "x"), ("a", "y"), ("b", "y"), ("c", "z")],
    "2015-06-01",
))


def popular() -> MostPopularRecommender:
    model = MostPopularRecommender("test")
    model.fit(GRID)
    return model


def test_scores_worked_out_by_hand():
    # Later: a touches z, c touches y, b touches an unknown item (not scored).
    key = build_answer_key(
        GRID, events_from([("a", "z"), ("c", "y"), ("b", "new")], "2015-08-25")
    )

    # Most-popular minus what each has seen: a gets [z], c gets [y]. Both hit at #1.
    results = evaluate(popular(), key, "any")

    assert results.users == 2
    assert results.precision == pytest.approx(0.1)  # 1 right of 10, for both
    assert results.recall == pytest.approx(1.0)  # each found their only answer
    assert results.ndcg == pytest.approx(1.0)  # ... at the top
    assert results.coverage == pytest.approx(2 / 3)  # z and y shown, of x, y, z


def test_an_empty_answer_key_cannot_be_scored():
    grid = build_interactions(events_from([("a", "x")], "2015-06-01"))
    key = build_answer_key(grid, events_from([("stranger", "x")], "2015-08-25"))
    model = MostPopularRecommender("test")
    model.fit(grid)
    with pytest.raises(ValueError, match="no users"):
        evaluate(model, key, "any")


def test_a_user_the_model_cannot_answer_gets_the_fallback_list():
    key = build_answer_key(GRID, events_from([("a", "z"), ("c", "y")], "2015-08-25"))
    model = MagicMock()
    model.name = "item-to-item"
    model.metadata.n_items = 3

    def recommend_for_user(user, n):
        if user == "a":
            raise ColdStartError(user)
        return [("x", 1.0)]  # for c: not c's answer, a miss

    model.recommend_for_user.side_effect = recommend_for_user

    results = evaluate(model, key, "any", fallback=popular())

    assert results.fallback_share == pytest.approx(0.5)
    assert results.recall == pytest.approx(0.5)  # a found z through the fallback; c missed


def test_without_a_fallback_a_model_with_no_answer_is_an_error():
    key = build_answer_key(GRID, events_from([("a", "z")], "2015-08-25"))
    model = MagicMock()
    model.recommend_for_user.side_effect = ColdStartError("a")
    with pytest.raises(ColdStartError):
        evaluate(model, key, "any")


# ---- The product page exam -----------------------------------------------------------


# s and t are strangers. s: page x, then touches z. t: page y, then touches x.
PAGE_KEY = build_product_page_key(GRID.events, views_at([
    ("s", "x", "10:00"), ("s", "z", "11:00"),
    ("t", "y", "10:00"), ("t", "x", "11:00"),
]))


def test_product_page_scores_worked_out_by_hand():
    # Most-popular minus the page's own product: page x shows [y, z], page y shows [x, z].
    # s: z is at #2, worth 1/log2(3). t: x is at #1, worth 1.
    results = evaluate_product_page(popular(), popular(), PAGE_KEY, "any")

    assert results.exam == PRODUCT_PAGE
    assert results.users == 2
    assert results.recall == pytest.approx(1.0)
    assert results.ndcg == pytest.approx((1 / math.log2(3) + 1) / 2)
    assert results.precision == pytest.approx(0.1)
    assert results.coverage == pytest.approx(1.0)  # x, y and z were all shown
    assert results.fallback_share == 0


def test_a_page_the_model_cannot_answer_gets_the_fallback_list():
    # As on the live site: the model gives up on page x, so the fallback answers it.
    model = MagicMock()
    model.name = "item-to-item"
    model.metadata.n_items = 3

    def similar_items(item, n):
        if item == "x":
            raise ColdStartError(item)
        return [("z", 1.0)]  # for page y: not t's answer, a miss

    model.similar_items.side_effect = similar_items

    results = evaluate_product_page(model, popular(), PAGE_KEY, "any")

    assert results.model == "item-to-item"
    assert results.fallback_share == pytest.approx(0.5)  # s's page, not t's
    assert results.recall == pytest.approx(0.5)  # s found z through the fallback; t missed


def test_a_brand_new_product_page_counts_as_a_fallback_page():
    # r's first page is a product no model knows. The whole site is scored, so it still counts.
    key = build_product_page_key(GRID.events, views_at([
        ("r", "brand-new", "10:00"), ("r", "x", "11:00"),
    ]))
    model = MagicMock()
    model.name = "item-to-item"
    model.metadata.n_items = 3
    model.similar_items.side_effect = ColdStartError("brand-new")

    results = evaluate_product_page(model, popular(), key, "any")

    assert results.users == 1
    assert results.fallback_share == 1
    assert results.recall == 1  # most-popular's top product is x, r's answer


def test_an_empty_product_page_key_cannot_be_scored():
    key = build_product_page_key(GRID.events, views_at([("s", "x", "10:00")]))
    with pytest.raises(ValueError, match="no users"):
        evaluate_product_page(popular(), popular(), key, "any")


def test_command_prints_a_row_per_model_and_exam(capsys):
    main(["--dataset", "fixture", "--model", "popular", "random"])
    table = capsys.readouterr().out
    assert "| popular | product page | any |" in table
    assert "| random | product page | any |" in table
    assert "| popular | personal | any |" in table
    assert "| random | personal | any |" in table
    # The fixture is too small for a personal cart-or-purchase key: skipped, with a note.
    assert "skipping personal exam, hits='cart or purchase'" in table


def test_command_can_score_item_to_item(capsys):
    main(["--dataset", "fixture", "--model", "item-to-item"])
    table = capsys.readouterr().out
    assert "| item-to-item | product page | any |" in table
    assert "| item-to-item | personal | any |" in table
