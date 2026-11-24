"""The deployment gate, on a made-up store where the right answer is known: its shoppers
browse in themes, so a model that learned the themes must win (tests/core/store_data.py)."""

import numpy as np
import pandas as pd
import pytest

from core.data.filtering import filter_interactions
from core.data.interactions import build_interactions
from core.evaluation.answer_key import build_product_page_key
from core.models.base import ColdStartError
from core.models.baselines import RandomRecommender
from core.models.item_to_item import ItemToItemRecommender
from core.training.gate import GateSettings, check_structure, judge
from core.training.train import fit_fallback
from tests.core.store_data import START, themed_store

STORE = "st_test"
EVENTS = themed_store()
HOLDOUT_FROM = START + pd.Timedelta(days=21)
HISTORY = EVENTS[EVENTS["timestamp"] < HOLDOUT_FROM]
KEY = build_product_page_key(HISTORY, EVENTS[EVENTS["timestamp"] >= HOLDOUT_FROM])
CATALOGUE = sorted(HISTORY["item_id"].unique())
FILTERED = filter_interactions(HISTORY, 2, 3)[0]
BEST_SELLERS = fit_fallback(STORE, build_interactions(FILTERED))  # as the pipeline makes them


def trained(cls=ItemToItemRecommender, events=FILTERED, store=STORE):
    model = cls(store)
    model.fit(build_interactions(events))
    return model


GOOD = trained()


class Changed:
    """A trained model whose answers are changed on purpose by `change(model, item, n)`."""

    def __init__(self, model, change):
        self.model, self.change = model, change

    @property
    def metadata(self):
        return self.model.metadata

    def similar_items(self, item_id, n=10):
        return self.change(self.model, item_id, n)


def raise_error(error):
    def change(model, item, n):
        raise error
    return change


EVERY_OTHER_PRODUCT = set(CATALOGUE[1::2])


def odd_pages_cold(model, item, n):
    """Gives up on every other product, so the fallback answers half the pages."""
    if item in EVERY_OTHER_PRODUCT:
        raise ColdStartError(item)
    return model.similar_items(item, n)


def judged(new, **kwargs):
    return judge(new, BEST_SELLERS, KEY, CATALOGUE, STORE, **kwargs)


def test_the_made_up_store_has_enough_shoppers_to_judge():
    assert len(KEY.relevant) >= 100


# ---- Well-formed answers ---------------------------------------------------------------


def test_a_good_model_and_best_sellers_pass_the_structure_check():
    assert check_structure(GOOD, BEST_SELLERS, STORE, CATALOGUE) == []
    assert check_structure(BEST_SELLERS, BEST_SELLERS, STORE, CATALOGUE) == []


@pytest.mark.parametrize("change, problem", [
    (lambda m, item, n: m.similar_items(item, n)[:1] + m.similar_items(item, n),
     "the same product twice"),
    (lambda m, item, n: [(item, 99.0)] + m.similar_items(item, n - 1),
     "recommends the product being viewed"),
    (lambda m, item, n: [("another-stores-product", 99.0)] + m.similar_items(item, n - 1),
     "a product the store does not have"),
    (lambda m, item, n: m.similar_items(item, n)[::-1], "not in best-first order"),
    (lambda m, item, n: [(i, float("nan")) for i, _ in m.similar_items(item, n)],
     "not a finite number"),
    (lambda m, item, n: m.similar_items(item, n + 5) + BEST_SELLERS.similar_items(item, 10),
     "asked for"),
    (lambda m, item, n: [i for i, _ in m.similar_items(item, n)], "pairs"),
    (raise_error(RuntimeError("corrupt table")), "fails on the page"),
    (raise_error(ColdStartError("nothing")), "answers no product page"),
])
def test_badly_formed_answers_are_problems(change, problem):
    problems = check_structure(Changed(GOOD, change), BEST_SELLERS, STORE, CATALOGUE)
    assert any(problem in p for p in problems), problems


def test_a_model_trained_for_another_store_is_a_problem():
    other = trained(store="st_other")
    assert any("trained for store 'st_other'" in p
               for p in check_structure(other, BEST_SELLERS, STORE, CATALOGUE))


def test_a_fallback_with_no_answer_is_a_problem():
    # The fallback is what the API serves when the model gives up: it must always answer.
    silent = Changed(BEST_SELLERS, raise_error(ColdStartError("nothing")))
    assert any("fallback has no answer" in p
               for p in check_structure(GOOD, silent, STORE, CATALOGUE))


def test_badly_formed_answers_are_rejected_before_any_scoring():
    duplicates = Changed(GOOD, lambda m, item, n: m.similar_items(item, n)[:1] * 2)
    result = judged(duplicates)
    assert not result.accepted
    assert result.reason == "its answers are not well formed"
    assert result.vs_best_sellers is None


# ---- Better than best sellers ------------------------------------------------------------


def test_a_model_that_learned_the_themes_is_accepted():
    result = judged(GOOD)
    assert result.accepted
    assert result.vs_best_sellers.difference_range[0] > 0
    assert result.vs_live is None  # nothing live to compare with


def test_random_recommendations_are_rejected():
    # The deliberately bad model: well formed, but no better than best sellers.
    result = judged(trained(RandomRecommender))
    assert not result.accepted
    assert result.reason == "not measurably better than best sellers"


def test_a_model_trained_on_shuffled_product_ids_is_rejected():
    # Every answer is a real product of the store and well formed, but the links between
    # products are wrong. Only scoring can catch this.
    items = FILTERED["item_id"].unique()
    shuffled = dict(zip(items, np.random.default_rng(0).permutation(items), strict=True))
    model = trained(events=FILTERED.assign(item_id=FILTERED["item_id"].map(shuffled)))
    result = judged(model)
    assert check_structure(model, BEST_SELLERS, STORE, CATALOGUE) == []
    assert not result.accepted
    assert result.reason == "not measurably better than best sellers"


# ---- Not worse than the live model -------------------------------------------------------


def test_worse_than_the_live_model_is_rejected_even_when_better_than_best_sellers():
    half = Changed(GOOD, odd_pages_cold)
    result = judged(half, live=GOOD, live_fallback=BEST_SELLERS)
    assert result.vs_best_sellers.difference_range[0] > 0  # passes the first test
    assert result.vs_live.difference_range[1] < 0
    assert not result.accepted
    assert result.reason == "measurably worse than the live model"


def test_tolerance_lets_a_small_measurable_drop_through():
    half = Changed(GOOD, odd_pages_cold)
    drop = judged(half, live=GOOD, live_fallback=BEST_SELLERS).vs_live
    allowed = abs(drop.difference_range[1]) / drop.other + 0.01
    result = judged(half, live=GOOD, live_fallback=BEST_SELLERS,
                    settings=GateSettings(tolerance=allowed))
    assert result.accepted


def test_the_same_model_again_is_not_worse():
    result = judged(trained(), live=GOOD, live_fallback=BEST_SELLERS)
    assert result.accepted
    assert result.vs_live.difference == 0
    assert result.reason.endswith("not measurably worse than the live model")


def test_live_best_sellers_are_not_protected():
    # Beating best sellers is already the first test.
    result = judged(GOOD, live=BEST_SELLERS, live_fallback=BEST_SELLERS)
    assert result.accepted and result.vs_live is None


def test_a_live_model_is_scored_as_served_without_a_fallback():
    # No fallback file beside the live model: the API shows nothing on its cold pages.
    half = Changed(GOOD, odd_pages_cold)
    with_fallback = judged(GOOD, live=half, live_fallback=BEST_SELLERS).vs_live
    without = judged(GOOD, live=half, live_fallback=None).vs_live
    assert without.other < with_fallback.other


def test_a_broken_live_model_does_not_block_a_working_one():
    broken = Changed(GOOD, raise_error(RuntimeError("corrupt table")))
    result = judged(trained(), live=broken, live_fallback=BEST_SELLERS)
    assert result.accepted
    assert "corrupt table" in result.live_error


# ---- Enough shoppers to judge --------------------------------------------------------------


def test_too_few_held_out_shoppers_and_the_gate_does_not_judge():
    users = len(KEY.relevant)
    result = judged(GOOD, settings=GateSettings(min_users=users + 1))
    assert not result.accepted
    assert result.reason.startswith(f"only {users} held-out shoppers")
    assert result.vs_best_sellers is None


def test_the_metric_can_be_chosen():
    ndcg = judged(GOOD).vs_best_sellers
    recall = judged(GOOD, settings=GateSettings(metric="recall")).vs_best_sellers
    assert ndcg.new != recall.new
