"""The metrics, on examples small enough to work out by hand."""

import math

import pytest

from core.evaluation.metrics import (
    bootstrap_range,
    catalogue_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

TEN = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


def test_one_hit_out_of_two_answers():
    # b is right (position 2), z was never recommended.
    relevant = {"b", "z"}
    assert precision_at_k(TEN, relevant) == pytest.approx(1 / 10)
    assert recall_at_k(TEN, relevant) == pytest.approx(1 / 2)
    # found: 1/log2(3) = 0.631; best possible: 1/log2(2) + 1/log2(3) = 1.631
    assert ndcg_at_k(TEN, relevant) == pytest.approx(0.631 / 1.631, abs=1e-3)


def test_no_hits_scores_zero():
    relevant = {"z"}
    assert precision_at_k(TEN, relevant) == 0
    assert recall_at_k(TEN, relevant) == 0
    assert ndcg_at_k(TEN, relevant) == 0


def test_the_single_answer_at_the_top_is_perfect():
    relevant = {"a"}
    assert recall_at_k(TEN, relevant) == 1
    assert ndcg_at_k(TEN, relevant) == 1
    # Only 1 of 10 can ever be right for this user, so 0.1 is the best possible.
    assert precision_at_k(TEN, relevant) == pytest.approx(0.1)


def test_ndcg_rewards_a_hit_higher_up():
    top = ndcg_at_k(TEN, {"a"})  # position 1
    bottom = ndcg_at_k(TEN, {"j"})  # position 10
    assert top == 1
    assert bottom == pytest.approx(1 / math.log2(11))  # 0.289
    # Precision and recall don't care where the hit is.
    assert precision_at_k(TEN, {"a"}) == precision_at_k(TEN, {"j"})


def test_only_the_top_k_count():
    # d is at position 4, outside the top 3.
    assert precision_at_k(TEN, {"d"}, k=3) == 0
    assert precision_at_k(TEN, {"c"}, k=3) == pytest.approx(1 / 3)


def test_a_short_list_is_still_divided_by_k():
    assert precision_at_k(["a", "b"], {"a", "b"}) == pytest.approx(2 / 10)


def test_more_answers_than_k_can_still_score_a_perfect_ndcg():
    # 20 answers, all 10 recommendations right: as good as a list of 10 can be.
    relevant = set(TEN) | {f"x{i}" for i in range(10)}
    assert ndcg_at_k(TEN, relevant) == pytest.approx(1)
    assert recall_at_k(TEN, relevant) == pytest.approx(10 / 20)


def test_coverage_counts_different_products_shown_to_anyone():
    # a, b, c shown out of a catalogue of 10.
    assert catalogue_coverage([["a", "b"], ["b", "c"]], n_items=10) == pytest.approx(0.3)


def test_same_list_for_everyone_has_low_coverage():
    assert catalogue_coverage([["a", "b"]] * 1000, n_items=100) == pytest.approx(0.02)


def test_a_repeated_item_is_a_bug_not_two_hits():
    with pytest.raises(ValueError, match="twice"):
        precision_at_k(["a", "a"], {"a"})


def test_a_user_with_no_answers_cannot_be_scored():
    with pytest.raises(ValueError, match="right answer"):
        recall_at_k(TEN, set())


# ---- Bootstrap range ----------------------------------------------------------------


def test_range_contains_the_average():
    values = [0.0] * 90 + [1.0] * 10  # average 0.1
    low, high = bootstrap_range(values)
    assert low < 0.1 < high


def test_everyone_scoring_the_same_gives_no_range():
    assert bootstrap_range([0.5] * 50) == pytest.approx((0.5, 0.5))


def test_more_users_give_a_narrower_range():
    few = bootstrap_range([0.0] * 9 + [1.0])  # average 0.1 from 10 users
    many = bootstrap_range([0.0] * 900 + [1.0] * 100)  # average 0.1 from 1,000 users
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_range_is_the_same_on_every_run():
    values = [0.0] * 90 + [1.0] * 10
    assert bootstrap_range(values) == bootstrap_range(values)
