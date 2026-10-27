"""Score models on the validation period and print the results table.

    python -m core.evaluation.run --model popular
    python -m core.evaluation.run --model random popular
    python -m core.evaluation.run --model popular --dataset fixture

Each model is trained on the training grid, asked for 10 recommendations per scored
user, and marked against the answer key. Scores are averages over the scored users.

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
    build_answer_key,
)
from core.evaluation.metrics import (
    bootstrap_range,
    catalogue_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from core.models.base import Recommender
from core.training.train import MODELS, prepare

HITS = {"any": ANY_INTERACTION, "cart or purchase": CART_OR_PURCHASE}


@dataclass
class Results:
    """One row of the results table. Each `*_range` is the 95% bootstrap range."""

    model: str
    hits: str
    users: int
    precision: float
    precision_range: tuple[float, float]
    recall: float
    recall_range: tuple[float, float]
    ndcg: float
    ndcg_range: tuple[float, float]
    coverage: float


def evaluate(model: Recommender, key: AnswerKey, hits: str, k: int = 10) -> Results:
    """Score a trained model against an answer key."""
    if not key.relevant:
        raise ValueError("the answer key has no users to score")
    lists = {
        user: [item for item, _ in model.recommend_for_user(user, n=k)]
        for user in key.relevant
    }
    scores = {
        metric: [score(lists[u], key.relevant[u], k) for u in lists]
        for metric, score in [
            ("precision", precision_at_k), ("recall", recall_at_k), ("ndcg", ndcg_at_k),
        ]
    }
    return Results(
        model=model.name,
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
    keys = {}
    for name, types in HITS.items():
        key = build_answer_key(prepared.grid, prepared.validation, types)
        if key.relevant:
            keys[name] = key
        else:
            # Happens on small samples: nobody's cart or purchase can count.
            print(f"skipping hits={name!r}: no users to score")
    results = []
    for name in models:
        model = MODELS[name](store_id=dataset)
        model.fit(prepared.grid)
        for hits, key in keys.items():
            results.append(evaluate(model, key, hits))
    return results


def format_table(results: list[Results]) -> str:
    """Headline metrics first; each score with its 95% range in brackets."""
    lines = [
        "| model | hits | users | Recall@10 | NDCG@10 | Precision@10 | coverage |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.model} | {r.hits} | {r.users:,} | {_with_range(r.recall, r.recall_range)} "
            f"| {_with_range(r.ndcg, r.ndcg_range)} "
            f"| {_with_range(r.precision, r.precision_range)} | {r.coverage:.4f} |"
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
