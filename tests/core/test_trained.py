"""The shared top-n picker every trained model answers with."""

import numpy as np
import pytest

from core.models.trained import TrainedRecommender


class Picker(TrainedRecommender):
    """Just enough of a model to call `_top`."""

    def __init__(self, n_items: int):
        super().__init__("test")
        self._item_ids = [f"i{j}" for j in range(n_items)]

    def recommend_for_user(self, user_id, n=10, exclude_seen=True):
        raise NotImplementedError

    def similar_items(self, item_id, n=10):
        raise NotImplementedError


def sort_everything(model, scores, exclude, n):
    """The plain way: sort all products, best first, ties in product order."""
    scores = scores.astype(np.float64)
    scores[exclude] = -np.inf
    best = np.argsort(-scores, kind="stable")[:n]
    return [(model._item_ids[i], float(scores[i])) for i in best if np.isfinite(scores[i])]


@pytest.mark.parametrize("seed", range(5))
def test_same_answer_as_sorting_everything(seed):
    # Only the best n are sorted, for speed. Many ties (whole-number scores) and many
    # exclusions are where a shortcut would go wrong.
    rng = np.random.default_rng(seed)
    for case in range(400):
        size = int(rng.integers(1, 60))
        model = Picker(size)
        scores = rng.integers(0, 4, size).astype(float) if case % 2 else rng.random(size)
        exclude = rng.choice(size, int(rng.integers(0, size + 1)), replace=False)
        n = int(rng.integers(0, 70))
        assert model._top(scores, exclude, n) == sort_everything(model, scores, exclude, n)


def test_ties_keep_product_order():
    assert Picker(5)._top(np.array([1.0, 2.0, 2.0, 2.0, 0.0]), [], 2) == [("i1", 2.0), ("i2", 2.0)]


def test_excluded_products_never_appear_even_when_too_few_remain():
    assert Picker(3)._top(np.array([3.0, 2.0, 1.0]), [0, 2], 10) == [("i1", 2.0)]


def test_scores_passed_in_are_not_changed():
    scores = np.array([3.0, 2.0, 1.0])
    Picker(3)._top(scores, [0], 2)
    assert scores.tolist() == [3.0, 2.0, 1.0]
