"""The deployment gate: may a newly trained model replace the one a store is served?

The new model is judged on the store's newest events, which it never saw, by the product
page exam (docs/decisions/0005): each shopper is shown the model's list for the first
product they viewed, or the fallback's where the model has no answer, exactly as the API
serves it. Shopper by shopper, it is compared with two things:

  best sellers    the fallback alone. A model that cannot measurably beat it is not worth
                  serving: best sellers need no model.
  the live model  what the store is served now. The new one must not be measurably worse.

"Measurably" is the rule of docs/decisions/0004: the 95% range of the difference, from
re-drawing the shoppers, lies entirely on one side of zero. Before any scoring, the new
model's answers must be well formed. See docs/decisions/0009.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

from core.evaluation.answer_key import ProductPageKey
from core.evaluation.metrics import bootstrap_range, per_user_metric
from core.evaluation.run import product_page_lists
from core.models.base import ColdStartError, Recommender
from core.models.baselines import (
    MostPopularRecommender,
    RandomRecommender,
    RecentlyPopularRecommender,
)

# Live models with nothing to protect: any new model that beats best sellers replaces them.
BASELINES = frozenset(
    {RandomRecommender.name, MostPopularRecommender.name, RecentlyPopularRecommender.name}
)
# A product no store has, for checking that an unknown page gets the fallback, not a crash.
UNSEEN_PRODUCT = "__gate_check_unseen_product__"


@dataclass(frozen=True)
class GateSettings:
    # The per-shopper score the decision uses. NDCG@10 counts a hit near the top of the
    # list for more, and the top is what a shopper sees first.
    metric: str = "ndcg"
    k: int = 10
    # With fewer scored shoppers than this the gate does not judge: the range from
    # re-drawing a handful of shoppers is itself unreliable.
    min_users: int = 30
    # How far below the live model, as a share of its score, still counts as "not worse".
    # 0: any measurable drop rejects. 0.05: a measurable drop of up to 5% is tolerated.
    tolerance: float = 0.0
    # How many products' pages the structure check asks about, spread over the catalogue.
    pages_checked: int = 200


@dataclass
class Comparison:
    """The new model against one other, on the same shoppers."""

    against: str
    new: float  # average score of the new model
    other: float  # average score of the other
    difference: float  # new - other
    difference_range: tuple[float, float]  # 95% range of the difference


@dataclass
class GateResult:
    accepted: bool
    reason: str
    users: int = 0  # shoppers scored
    problems: list[str] = field(default_factory=list)  # from the structure check
    vs_best_sellers: Comparison | None = None
    vs_live: Comparison | None = None
    # Set when the live model could not be scored; the comparison with it is then skipped.
    live_error: str | None = None


def judge(
    new: Recommender,
    new_fallback: Recommender,
    key: ProductPageKey,
    catalogue: list[str],
    store_id: str,
    live: Recommender | None = None,
    live_fallback: Recommender | None = None,
    settings: GateSettings | None = None,
) -> GateResult:
    """Accept or reject `new` (served with `new_fallback`, which is also the best sellers
    it must beat). `key` holds the held-out shoppers; `catalogue` the store's products in
    the events `new` was trained on."""
    settings = settings or GateSettings()
    problems = check_structure(
        new, new_fallback, store_id, catalogue, settings.k, settings.pages_checked
    )
    if problems:
        return GateResult(False, "its answers are not well formed", problems=problems)

    users = sorted(key.relevant)
    if len(users) < max(1, settings.min_users):
        return GateResult(
            False,
            f"only {len(users)} held-out shoppers could be scored; "
            f"the gate needs {settings.min_users} to judge",
            users=len(users),
        )

    new_scores = _scores(new, new_fallback, key, users, settings)
    result = GateResult(True, "", users=len(users))
    result.vs_best_sellers = _compare(
        "best sellers", new_scores, _scores(new_fallback, new_fallback, key, users, settings)
    )
    if live is not None and live.metadata.algorithm not in BASELINES:
        try:
            live_scores = _scores(live, live_fallback or _NoFallback(), key, users, settings)
            result.vs_live = _compare(f"live {live.metadata.algorithm}", new_scores, live_scores)
        except Exception as error:  # a broken live model must not block a working one
            result.live_error = f"{type(error).__name__}: {error}"

    if result.vs_best_sellers.difference_range[0] <= 0:
        result.accepted = False
        result.reason = "not measurably better than best sellers"
    elif (
        result.vs_live is not None
        and result.vs_live.difference_range[1] < -settings.tolerance * result.vs_live.other
    ):
        result.accepted = False
        result.reason = "measurably worse than the live model"
    else:
        result.reason = "measurably better than best sellers" + (
            ", and not measurably worse than the live model" if result.vs_live else ""
        )
    return result


def check_structure(
    model: Recommender,
    fallback: Recommender,
    store_id: str,
    catalogue: list[str],
    k: int = 10,
    pages: int = 200,
) -> list[str]:
    """What is wrong with a model's answers, whatever its score. Empty if nothing.

    Asks the model and its fallback about up to `pages` products spread over `catalogue`,
    and about one product nobody has seen. A list must hold at most k different products
    of this store, best first, with finite scores, and never the product being viewed.
    """
    problems = []
    for name, recommender in [("model", model), ("fallback", fallback)]:
        try:
            owner = recommender.metadata.store_id
        except Exception as error:
            problems.append(f"{name} has no metadata ({type(error).__name__}: {error})")
            continue
        if owner != store_id:
            problems.append(f"{name} was trained for store {owner!r}, not {store_id!r}")

    known = set(catalogue)
    step = max(1, math.ceil(len(catalogue) / max(1, pages)))
    answered = 0
    for item in [*catalogue[::step], UNSEEN_PRODUCT]:
        for name, recommender in [("model", model), ("fallback", fallback)]:
            try:
                answer = recommender.similar_items(item, n=k)
            except ColdStartError:
                if name == "fallback":
                    problems.append(f"fallback has no answer for the page of {item!r}")
                continue
            except Exception as error:
                problems.append(f"{name} fails on the page of {item!r}: "
                                f"{type(error).__name__}: {error}")
                continue
            if name == "model" and item != UNSEEN_PRODUCT:
                answered += 1
            problems += [
                f"{name}, page of {item!r}: {problem}"
                for problem in _list_problems(answer, item, known, k)
            ]
        if len(problems) >= 10:  # plenty to reject on; keeps the report readable
            break
    if catalogue and answered == 0:
        problems.append("model answers no product page: every page would get the fallback")
    return problems


def _list_problems(answer, own: str, known: set[str], k: int) -> list[str]:
    if not isinstance(answer, list) or not all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str)
        and isinstance(pair[1], numbers.Real)
        for pair in answer
    ):
        return ["not a list of (product_id, score) pairs"]
    ids = [item for item, _ in answer]
    scores = [float(score) for _, score in answer]
    problems = []
    if len(ids) > k:
        problems.append(f"{len(ids)} products, {k} asked for")
    if len(set(ids)) != len(ids):
        problems.append("the same product twice")
    if own in ids:
        problems.append("recommends the product being viewed")
    unknown = [item for item in ids if item not in known]
    if unknown:
        problems.append(f"a product the store does not have: {unknown[0]!r}")
    if not all(math.isfinite(score) for score in scores):
        problems.append("a score that is not a finite number")
    elif any(a < b for a, b in zip(scores, scores[1:], strict=False)):
        problems.append("not in best-first order")
    return problems


def _scores(
    model: Recommender,
    fallback: Recommender,
    key: ProductPageKey,
    users: list[str],
    settings: GateSettings,
) -> np.ndarray:
    """Each shopper's score, in `users` order, so two models' scores pair up."""
    lists, _ = product_page_lists(model, fallback, key, settings.k)
    metric = per_user_metric(settings.metric)
    return np.array([metric(lists[user], key.relevant[user], settings.k) for user in users])


def _compare(against: str, new: np.ndarray, other: np.ndarray) -> Comparison:
    # Paired: the range is of the per-shopper differences, which is tighter than comparing
    # two separate ranges, because both models face the same shoppers.
    difference = new - other
    return Comparison(
        against=against,
        new=float(new.mean()),
        other=float(other.mean()),
        difference=float(difference.mean()),
        difference_range=bootstrap_range(difference),
    )


class _NoFallback:
    """What the API shows when a live model has no fallback file beside it: nothing."""

    def similar_items(self, item_id: str, n: int = 10) -> list[tuple[str, float]]:
        return []
