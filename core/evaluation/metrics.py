"""Scoring one model's recommendations against the answer key.

The per-user metrics compare one user's top-k list with that user's answers and give a
number between 0 and 1. A model's score is the average over all scored users. Coverage
looks at all users' lists at once.

  precision@k  of the k recommendations, what share were right?
  recall@k     of the user's right answers, what share were found?
  ndcg@k       like recall, but a hit at the top counts more than a hit at the bottom
  coverage     what share of the catalogue was recommended to anyone at all?

Recall and NDCG are the headline metrics. Precision is reported too, but a handful of
users with very many answers move it a lot (see docs/decisions/0004).

See docs/decisions/0004.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence, Set

import numpy as np


def precision_at_k(recommended: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    """Hits in the top k, divided by k.

    Always divided by k, even if the model returned fewer: a model that only dares to
    recommend 2 products should not score as if it had been right 2 times out of 2.
    """
    return _hits(recommended, relevant, k) / k


def recall_at_k(recommended: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    """Hits in the top k, divided by how many right answers the user has."""
    _check(relevant, k)
    return _hits(recommended, relevant, k) / len(relevant)


def ndcg_at_k(recommended: Sequence[str], relevant: Set[str], k: int = 10) -> float:
    """Rewards putting hits near the top of the list.

    A hit at position p is worth 1 / log2(p + 1): 1.0 at position 1, 0.63 at 2, 0.5 at
    3, ... 0.29 at 10. The total is divided by the best total possible for this user
    (all their answers at the top), so a perfect list scores 1 and no hits score 0.
    """
    _check(relevant, k)
    found = sum(
        1 / math.log2(position + 1)
        for position, item in enumerate(_top(recommended, k), start=1)
        if item in relevant
    )
    best = sum(1 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return found / best


def catalogue_coverage(recommendations: Iterable[Sequence[str]], n_items: int) -> float:
    """Different products recommended to anyone, divided by the catalogue size.

    Most-popular scores near 0 here: everyone gets the same 10 products. A model that
    scores well on accuracy *and* coverage is useful to more of the shop.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    shown = set()
    for recommended in recommendations:
        shown.update(recommended)
    return len(shown) / n_items


def bootstrap_range(
    values: Sequence[float], draws: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """The 95% range of an average, from re-drawing the users with replacement.

    Answers "how much could this score move if we had happened to get a different set of
    users?". With few users scoring hits, it moves a lot: two models whose ranges
    overlap cannot be told apart. Same seed, same range, so reruns match.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no values to draw from")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _hits(recommended: Sequence[str], relevant: Set[str], k: int) -> int:
    _check(relevant, k)
    return sum(1 for item in _top(recommended, k) if item in relevant)


def _top(recommended: Sequence[str], k: int) -> Sequence[str]:
    top = recommended[:k]
    # A repeated item would count as two hits. Models never return one, so a repeat
    # means a bug in the model, and is better caught here than averaged into a score.
    if len(set(top)) != len(top):
        raise ValueError("recommendations contain the same item twice")
    return top


def _check(relevant: Set[str], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    # The answer key never holds an empty set; users with no answers are not scored.
    if not relevant:
        raise ValueError("a scored user must have at least one right answer")
