"""Score models on the validation period and print the results table.

    python -m core.evaluation.run --model popular
    python -m core.evaluation.run --model random popular
    python -m core.evaluation.run --model popular --dataset fixture

Each model is trained on the training grid, asked for 10 recommendations per scored
user, and marked against the answer key. Scores are averages over the scored users.

Every model sits two exams. The product page exam tests `similar_items`, what the live
endpoint serves, so it is the main number for the deployed system. The personal exam tests
`recommend_for_user`. Their scores cover different users, so they cannot be compared with
each other; compare models within one exam. See docs/decisions/0004 and 0005.

Only the validation period is ever scored here. The test period stays sealed until the
final run (Phase 6), so it cannot be opened by accident while tuning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from core.evaluation.answer_key import (
    ANY_INTERACTION,
    CART_OR_PURCHASE,
    AnswerKey,
    ProductPageKey,
    build_answer_key,
    build_product_page_key,
)
from core.evaluation.metrics import (
    bootstrap_range,
    catalogue_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from core.models.base import ColdStartError, Recommender
from core.training.train import MODELS, fit_fallback, prepare

HITS = {"any": ANY_INTERACTION, "cart or purchase": CART_OR_PURCHASE}
PERSONAL = "personal"
PRODUCT_PAGE = "product page"


@dataclass
class Results:
    """One row of the results table. Each `*_range` is the 95% bootstrap range."""

    model: str
    exam: str
    hits: str
    users: int
    precision: float
    precision_range: tuple[float, float]
    recall: float
    recall_range: tuple[float, float]
    ndcg: float
    ndcg_range: tuple[float, float]
    coverage: float
    # Share of scored users whose list came from the fallback, not the model. Part of the
    # score, as on the live site, so it has to be visible.
    fallback_share: float = 0.0


def evaluate(model: Recommender, key: AnswerKey, hits: str, k: int = 10) -> Results:
    """The personal exam: score `recommend_for_user` against an answer key."""
    if not key.relevant:
        raise ValueError("the answer key has no users to score")
    lists = {
        user: [item for item, _ in model.recommend_for_user(user, n=k)]
        for user in key.relevant
    }
    return _mark(model, PERSONAL, hits, lists, key.relevant, k)


def evaluate_product_page(
    model: Recommender,
    fallback: Recommender,
    key: ProductPageKey,
    hits: str,
    k: int = 10,
) -> Results:
    """The product page exam: score `similar_items` the way the endpoint serves it.

    When the model has no basis to answer, the fallback answers, exactly as on the live
    site, so what is measured is what is served.
    """
    if not key.relevant:
        raise ValueError("the answer key has no users to score")
    # The answer depends only on the product, and many users share a first product.
    pages: dict[str, list[str]] = {}
    gave_up: set[str] = set()
    for item in set(key.questions.values()):
        try:
            answer = model.similar_items(item, n=k)
        except ColdStartError:
            answer = fallback.similar_items(item, n=k)
            gave_up.add(item)
        pages[item] = [recommended for recommended, _ in answer]

    lists = {user: pages[key.questions[user]] for user in key.relevant}
    results = _mark(model, PRODUCT_PAGE, hits, lists, key.relevant, k)
    results.fallback_share = sum(key.questions[u] in gave_up for u in lists) / len(lists)
    return results


def _mark(
    model: Recommender,
    exam: str,
    hits: str,
    lists: dict[str, list[str]],
    relevant: dict[str, frozenset[str]],
    k: int,
) -> Results:
    """Mark each user's list against their answers and average over the users."""
    scores = {
        metric: [score(lists[u], relevant[u], k) for u in lists]
        for metric, score in [
            ("precision", precision_at_k), ("recall", recall_at_k), ("ndcg", ndcg_at_k),
        ]
    }
    return Results(
        model=model.name,
        exam=exam,
        hits=hits,
        users=len(lists),
        precision=_mean(scores["precision"]),
        precision_range=bootstrap_range(scores["precision"]),
        recall=_mean(scores["recall"]),
        recall_range=bootstrap_range(scores["recall"]),
        ndcg=_mean(scores["ndcg"]),
        ndcg_range=bootstrap_range(scores["ndcg"]),
        coverage=catalogue_coverage(lists.values(), model.metadata.n_items),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def run(dataset: str, models: list[str]) -> list[Results]:
    prepared = prepare(dataset)
    grid, validation = prepared.grid, prepared.validation
    page_keys = _with_users(PRODUCT_PAGE, {
        hits: build_product_page_key(grid, validation, types) for hits, types in HITS.items()
    })
    personal_keys = _with_users(PERSONAL, {
        hits: build_answer_key(grid, validation, types) for hits, types in HITS.items()
    })
    fallback = fit_fallback(dataset, grid)

    results = []
    for name in models:
        model = MODELS[name](store_id=dataset)
        model.fit(grid)
        for hits, key in page_keys.items():
            results.append(evaluate_product_page(model, fallback, key, hits))
        for hits, key in personal_keys.items():
            results.append(evaluate(model, key, hits))
    return results


def _with_users(exam: str, keys: dict) -> dict:
    """Drop keys with nobody to score, which happens on small samples, with a note."""
    for hits, key in keys.items():
        if not key.relevant:
            print(f"skipping {exam} exam, hits={hits!r}: no users to score")
    return {hits: key for hits, key in keys.items() if key.relevant}


def format_table(results: list[Results]) -> str:
    """Headline metrics first; each score with its 95% range in brackets."""
    lines = [
        "| model | exam | hits | users | Recall@10 | NDCG@10 | Precision@10 | coverage | fallback |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.model} | {r.exam} | {r.hits} | {r.users:,} "
            f"| {_with_range(r.recall, r.recall_range)} "
            f"| {_with_range(r.ndcg, r.ndcg_range)} "
            f"| {_with_range(r.precision, r.precision_range)} "
            f"| {r.coverage:.4f} | {r.fallback_share:.4f} |"
        )
    return "\n".join(lines)


def _with_range(value: float, bounds: tuple[float, float]) -> str:
    return f"{value:.4f} ({bounds[0]:.4f}-{bounds[1]:.4f})"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--dataset", choices=["fixture", "retailrocket"], default="retailrocket")
    args = parser.parse_args(argv)
    print(format_table(run(args.dataset, args.model)))


if __name__ == "__main__":
    main()
