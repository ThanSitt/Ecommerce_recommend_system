"""Retrain a store's model, and replace the one it is served only if the gate accepts it.

    python -m core.training.retrain --store st_123                       # from the database
    python -m core.training.retrain --all                                # every active store
    python -m core.training.retrain --all --events events.csv            # from an export
    python -m core.training.retrain --store st_123 --model als

The newest `--holdout-days` of a store's events are held out. The new model is trained on
the events before them and judged on them by the deployment gate (core/training/gate.py).
What was judged is what is deployed: on acceptance the very same files are copied to the
store's `current.joblib` and `fallback.joblib`, which the API serves. Every run keeps its
files and a report of the decision, accepted or not:

    models/<store>/runs/<run>/model.joblib, fallback.joblib, report.json

A store with no model that qualifies is served best sellers, so it always gets something.
See docs/decisions/0009.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from core.data.filtering import (
    RETAILROCKET_MIN_ITEMS_PER_USER,
    RETAILROCKET_MIN_USERS_PER_ITEM,
    filter_interactions,
)
from core.data.interactions import build_interactions
from core.evaluation.answer_key import ANY_INTERACTION, build_product_page_key
from core.models.base import Recommender
from core.models.loading import (
    CURRENT_FILENAME,
    FALLBACK_FILENAME,
    fallback_path,
    load_model,
    store_dir,
)
from core.training.gate import BASELINES, GateResult, GateSettings, check_structure, judge
from core.training.train import MODELS, ROOT, fit_fallback

PROMOTED = "promoted"
KEPT = "kept the live model"
BEST_SELLERS = "serving best sellers"
NO_EVENTS = "nothing to do"

MODEL_FILENAME = "model.joblib"
BEST_SELLERS_FILENAME = "best-sellers.joblib"
REPORT_FILENAME = "report.json"


# For a store with no model live: no setting of it depends on the size of the catalogue,
# unlike ALS's 200 factors, which were tuned to Retailrocket's 50,000 products (0008).
FIRST_MODEL = "item-to-item"


@dataclass(frozen=True)
class RetrainSettings:
    # None: the algorithm of the store's live model, so a store keeps the kind of model that
    # passed the gate (a fresh item-to-item cannot replace a better ALS, so retraining it
    # every week would only let the ALS model age). FIRST_MODEL if nothing real is live.
    model: str | None = None
    # The newest week is the exam. The deployed model has not learned from it.
    holdout_days: float = 7.0
    # The Retailrocket filter (0003, 0006), not re-tuned per store.
    min_items_per_user: int = RETAILROCKET_MIN_ITEMS_PER_USER
    min_users_per_item: int = RETAILROCKET_MIN_USERS_PER_ITEM
    gate: GateSettings = field(default_factory=GateSettings)


@dataclass
class RunReport:
    """What one run did and why. Saved as report.json in the run's folder."""

    store_id: str
    run: str
    model: str
    decision: str = ""
    reason: str = ""
    events: int = 0
    events_from: datetime | None = None
    events_to: datetime | None = None
    # Events from here on were held out, and are what the gate scored on, unless the live
    # model had already seen some of them: then scoring starts after what it saw.
    holdout_from: datetime | None = None
    scored_from: datetime | None = None
    training_events: int = 0
    held_out_events: int = 0
    grid_users: int = 0
    grid_items: int = 0
    gate: GateResult | None = None
    live_before: dict | None = None
    live_error: str | None = None
    serving: dict | None = None
    folder: str = ""


def retrain_store(
    events: pd.DataFrame,
    store_id: str,
    models_dir: Path | str,
    settings: RetrainSettings | None = None,
    until: pd.Timestamp | None = None,
    run: str | None = None,
) -> RunReport:
    """One retraining run for one store. `events` are the store's events in the standard
    shape; `until` ignores later ones, to replay the past."""
    settings = settings or RetrainSettings()
    if settings.model is not None and settings.model not in MODELS:
        raise ValueError(f"unknown model {settings.model!r}; choose from {sorted(MODELS)}")
    folder = store_dir(models_dir, store_id)
    run = run or _new_run_name(folder)
    run_dir = folder / "runs" / run

    live, live_fallback, live_error = _load_live(folder, store_id)
    live_is_model = live is not None and live.metadata.algorithm not in BASELINES
    model_name = settings.model or (
        live.metadata.algorithm
        if live_is_model and live.metadata.algorithm in MODELS
        else FIRST_MODEL
    )
    report = RunReport(store_id=store_id, run=run, model=model_name, folder=str(run_dir))
    report.live_error = live_error
    report.live_before = _describe(live)

    if until is not None:
        events = events[events["timestamp"] <= until]
    report.events = len(events)
    if events.empty:
        report.decision, report.reason = NO_EVENTS, "the store has no events yet"
        _save_report(report, run_dir)
        return report
    report.events_from = events["timestamp"].min()
    report.events_to = events["timestamp"].max()

    # The clock is the newest event, not the time of the run, so the same events always
    # give the same decision.
    holdout_from = report.events_to - pd.Timedelta(days=settings.holdout_days)
    history = events[events["timestamp"] < holdout_from]
    report.holdout_from = holdout_from
    report.training_events = len(history)
    report.held_out_events = len(events) - len(history)

    # A live model that learned from part of the held-out events would be marked on answers
    # it has seen. Score only what came after them.
    scored_from = holdout_from
    if live_is_model:
        # Its fallback counts too: it learns from shoppers the filter removes, so it can
        # have seen later events than the model.
        live_saw_to = max(
            pd.Timestamp(m.metadata.events_to) for m in (live, live_fallback) if m is not None
        )
        if live_saw_to >= holdout_from:
            scored_from = live_saw_to + pd.Timedelta(microseconds=1)
    report.scored_from = scored_from

    gate, not_trained = None, ""
    if history.empty:
        not_trained = "no events before the held-out period to train on"
    else:
        kept, _ = filter_interactions(
            history, settings.min_items_per_user, settings.min_users_per_item
        )
        if kept.empty:
            not_trained = "too few shoppers who touched the same products to train a model"
        else:
            grid = build_interactions(kept)
            report.grid_users, report.grid_items = grid.n_users, grid.n_items
            model = MODELS[model_name](store_id=store_id)
            model.fit(grid)
            model.save(run_dir / MODEL_FILENAME)
            # From the same grid, as the evaluation in Phases 2 to 4 measured the pair.
            fit_fallback(store_id, grid).save(run_dir / FALLBACK_FILENAME)
            key = build_product_page_key(
                events[events["timestamp"] < scored_from],
                events[events["timestamp"] >= scored_from],
                ANY_INTERACTION,
            )
            # The files are judged, not the objects in memory: the API will load the files.
            gate = judge(
                new=load_model(run_dir / MODEL_FILENAME),
                new_fallback=load_model(run_dir / FALLBACK_FILENAME),
                key=key,
                catalogue=sorted(history["item_id"].unique()),
                store_id=store_id,
                live=live if live_is_model else None,
                live_fallback=live_fallback,
                settings=settings.gate,
            )
    report.gate = gate

    if gate is not None and gate.accepted:
        _promote(run_dir / MODEL_FILENAME, run_dir / FALLBACK_FILENAME, folder)
        report.decision, report.reason = PROMOTED, gate.reason
    else:
        why = not_trained or gate.reason
        if live_is_model:
            # A model that once beat best sellers is not swapped for them on a quiet week.
            report.decision = KEPT
            report.reason = f"new {model_name} model not used: {why}"
        else:
            _serve_best_sellers(events, store_id, run_dir, folder, settings)
            report.decision = BEST_SELLERS
            report.reason = f"no {model_name} model qualified: {why}"

    report.serving = _describe(load_model(folder / CURRENT_FILENAME))
    _save_report(report, run_dir)
    return report


def _serve_best_sellers(
    events: pd.DataFrame,
    store_id: str,
    run_dir: Path,
    folder: Path,
    settings: RetrainSettings,
) -> None:
    """Best sellers from every event, the held-out period included: they are the floor, so
    nothing judges them, and the newest week is what best sellers most need.

    Counted over the shoppers the filter keeps, like the fallback beside a model. Counting
    every shopper made Retailrocket's best sellers 180 MB in memory instead of 38 MB (0009).
    A store whose filter leaves nobody counts every shopper.
    """
    kept, _ = filter_interactions(
        events, settings.min_items_per_user, settings.min_users_per_item
    )
    best_sellers = fit_fallback(store_id, build_interactions(events if kept.empty else kept))
    path = run_dir / BEST_SELLERS_FILENAME
    best_sellers.save(path)
    saved = load_model(path)
    gate = settings.gate
    problems = check_structure(
        saved, saved, store_id, sorted(events["item_id"].unique()), gate.k, gate.pages_checked
    )
    if problems:  # never expected; refuse to serve rather than serve something broken
        raise RuntimeError(f"best sellers failed the structure check: {problems}")
    _promote(path, path, folder)


def _promote(model_file: Path, fallback_file: Path, folder: Path) -> None:
    """Copy a run's files to the ones the API serves. Each copy is written beside its
    target and then renamed over it, so a reader never sees half a file. The fallback goes
    first, so a new model never meets the old fallback."""
    current = folder / CURRENT_FILENAME
    for source, target in [(fallback_file, fallback_path(current)), (model_file, current)]:
        partial = target.with_name(target.name + ".partial")
        shutil.copyfile(source, partial)
        os.replace(partial, target)


def _load_live(
    folder: Path, store_id: str
) -> tuple[Recommender | None, Recommender | None, str | None]:
    """The store's live model and its fallback. A live file that cannot be used is
    reported and treated as absent: it must not stop a working model replacing it."""
    current = folder / CURRENT_FILENAME
    if not current.is_file():
        return None, None, None
    try:
        live = load_model(current)
        if live.metadata.store_id != store_id:
            raise ValueError(f"it was trained for store {live.metadata.store_id!r}")
    except Exception as error:
        return None, None, f"live model unusable: {type(error).__name__}: {error}"
    try:
        live_fallback = load_model(fallback_path(current))
    except Exception:
        live_fallback = None  # served without one, as the API would
    return live, live_fallback, None


def _describe(model: Recommender | None) -> dict | None:
    if model is None:
        return None
    meta = model.metadata
    return {
        "algorithm": meta.algorithm,
        "trained_at": meta.trained_at,
        "events_to": meta.events_to,
        "products": meta.n_items,
    }


def _new_run_name(folder: Path) -> str:
    """The run's UTC start time; Windows allows no colons in folder names."""
    name = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    candidate, number = name, 2
    while (folder / "runs" / candidate).exists():
        candidate, number = f"{name}-{number}", number + 1
    return candidate


def _save_report(report: RunReport, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / REPORT_FILENAME).write_text(
        json.dumps(asdict(report), indent=2, default=_json_value), encoding="utf-8"
    )


def _json_value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"cannot save {type(value).__name__} in a report")


def summary(report: RunReport) -> str:
    """A few lines for the terminal; the full numbers are in the report."""
    lines = [f"{report.store_id}: {report.decision} (run {report.run})", f"  {report.reason}"]
    gate = report.gate
    if gate is not None:
        for problem in gate.problems[:5]:
            lines.append(f"  problem: {problem}")
        for comparison in (gate.vs_best_sellers, gate.vs_live):
            if comparison is not None:
                low, high = comparison.difference_range
                lines.append(
                    f"  vs {comparison.against}: {comparison.new:.4f} against "
                    f"{comparison.other:.4f}, difference {comparison.difference:+.4f} "
                    f"(95% range {low:+.4f} to {high:+.4f}), {gate.users} shoppers"
                )
    if report.serving:
        lines.append(f"  serving: {report.serving['algorithm']}, trained on events up to "
                     f"{report.serving['events_to']:%Y-%m-%d %H:%M}")
    lines.append(f"  report: {Path(report.folder) / REPORT_FILENAME}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--store", nargs="+", help="store_id(s) to retrain")
    which.add_argument("--all", action="store_true", help="every store (active ones, or all "
                                                          "in --events)")
    parser.add_argument("--events", type=Path,
                        help="a CSV export of the events table, instead of the database")
    parser.add_argument("--model", choices=sorted(MODELS),
                        help=f"default: the algorithm of the store's live model, or "
                             f"{FIRST_MODEL} if none is live")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--holdout-days", type=float, default=RetrainSettings.holdout_days)
    parser.add_argument("--min-users", type=int, default=GateSettings.min_users,
                        help="held-out shoppers the gate needs before it judges")
    parser.add_argument("--tolerance", type=float, default=GateSettings.tolerance)
    parser.add_argument("--until", help="ignore events after this time, to replay the past "
                                        "(ISO 8601; no offset means UTC)")
    args = parser.parse_args(argv)

    settings = RetrainSettings(
        model=args.model,
        holdout_days=args.holdout_days,
        gate=GateSettings(min_users=args.min_users, tolerance=args.tolerance),
    )
    until = None
    if args.until:
        until = pd.Timestamp(args.until)
        until = until.tz_localize("UTC") if until.tzinfo is None else until.tz_convert("UTC")

    # Imported here: a run from an export needs no database settings.
    from core.data.store_events import (
        export_store_ids,
        list_active_stores,
        load_events_export,
        load_store_events,
    )

    if args.events:
        stores = export_store_ids(args.events) if args.all else args.store

        def load(store: str) -> pd.DataFrame:
            return load_events_export(args.events, store)
    else:
        from core.db.session import get_session

        with get_session() as session:
            stores = list_active_stores(session) if args.all else args.store

        def load(store: str) -> pd.DataFrame:
            with get_session() as session:
                return load_store_events(session, store)

    failed = 0
    for store in stores:
        try:
            report = retrain_store(load(store), store, args.models_dir, settings, until)
            print(summary(report))
        except Exception:  # one broken store must not stop the others
            failed += 1
            print(f"{store}: failed", file=sys.stderr)
            traceback.print_exc()
    # Rejecting a model is the gate working, not a failure. Only errors fail the job.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
