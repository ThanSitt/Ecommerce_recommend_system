"""Train a model and save it to a dated file, with its fallback beside it.

    python -m core.training.train --dataset fixture --model popular
    python -m core.training.train --dataset retailrocket --model random
    python -m core.training.train --dataset fixture --model popular --out models/current.joblib

`fixture` uses the small committed sample in tests/fixtures/, so a model file can be made
in a second without downloading Retailrocket. Its recommendations mean nothing; it is for
testing the API against a real file.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from core.data.interactions import Interactions
from core.data.prepare import PreparedData, prepare_events, prepare_retailrocket
from core.data.retailrocket import load_retailrocket
from core.data.split import RETAILROCKET_TEST_START, RETAILROCKET_VALIDATION_START
from core.models.base import Recommender
from core.models.baselines import (
    MostPopularRecommender,
    RandomRecommender,
    RecentlyPopularRecommender,
)
from core.models.item_to_item import ItemToItemRecommender
from core.models.loading import fallback_path

ROOT = Path(__file__).resolve().parents[2]
RETAILROCKET = ROOT / "data" / "raw" / "events.csv"
FIXTURE = ROOT / "tests" / "fixtures" / "events_sample.csv"

MODELS: dict[str, type[Recommender]] = {
    "item-to-item": ItemToItemRecommender,
    "random": RandomRecommender,
    "popular": MostPopularRecommender,
    "recent": RecentlyPopularRecommender,
}


def prepare(dataset: str) -> PreparedData:
    if dataset == "retailrocket":
        return prepare_retailrocket(RETAILROCKET)
    if dataset == "fixture":
        # Looser filter than Retailrocket, whose settings keep only 32 of the 739 visitors here.
        events, _ = load_retailrocket(FIXTURE)
        return prepare_events(
            events, RETAILROCKET_VALIDATION_START, RETAILROCKET_TEST_START,
            min_items_per_user=2, min_users_per_item=2,
        )
    raise ValueError(f"unknown dataset: {dataset}")


def fit_fallback(store_id: str, grid: Interactions) -> Recommender:
    """What the API serves when a model has no basis to answer.

    Recently popular, because it beats all-time most-popular on the product page exam
    (docs/decisions/0005). Trained on the same data as the model, so it only offers products
    the store had at that time. The evaluation uses this too, so what is scored is what is
    served.
    """
    fallback = RecentlyPopularRecommender(store_id=store_id)
    fallback.fit(grid)
    return fallback


def train(
    dataset: str, model: str, out: Path | None = None
) -> tuple[Recommender, Path, PreparedData]:
    """Prepare the data, fit the model, save it. Returns the model, where it went, and
    the data it was trained on."""
    prepared = prepare(dataset)
    recommender = MODELS[model](store_id=dataset)
    recommender.fit(prepared.grid)
    if out is None:
        # Versioned by training date (CLAUDE.md); the service loads one by path.
        out = ROOT / "models" / dataset / f"{model}-{date.today().isoformat()}.joblib"
    recommender.save(out)
    fit_fallback(dataset, prepared.grid).save(fallback_path(out))
    return recommender, out, prepared


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=["fixture", "retailrocket"], default="fixture")
    parser.add_argument("--model", choices=sorted(MODELS), default="popular")
    parser.add_argument("--out", type=Path, help="where to save (default: dated file)")
    args = parser.parse_args(argv)

    recommender, out, prepared = train(args.dataset, args.model, args.out)
    meta = recommender.metadata
    print(f"saved {meta.algorithm} model to {out}")
    print(f"saved fallback to {fallback_path(out)}")
    print(f"  {meta.n_users} users, {meta.n_items} items, "
          f"events {meta.events_from:%Y-%m-%d} -> {meta.events_to:%Y-%m-%d}")
    # Real IDs from the training data, to try against the API.
    user, item = prepared.grid.user_ids[0], prepared.grid.item_ids[0]
    print(f"  try user_id={user!r}: {recommender.recommend_for_user(user, n=3)}")
    print(f"  try item_id={item!r}: {recommender.similar_items(item, n=3)}")


if __name__ == "__main__":
    main()
