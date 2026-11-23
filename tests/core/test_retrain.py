"""Retraining a store end to end: train, judge, and replace the live model or not."""

import json

import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.models.base import ColdStartError
from core.models.item_to_item import ItemToItemRecommender
from core.models.loading import current_model_path, fallback_path, load_model, store_dir
from core.training.gate import GateSettings
from core.training.retrain import (
    BEST_SELLERS,
    KEPT,
    NO_EVENTS,
    PROMOTED,
    RetrainSettings,
    main,
    retrain_store,
)
from tests.core.store_data import START, themed_store

STORE = "st_test"
EVENTS = themed_store()  # 28 days, shoppers browse in themes
# The same store a week later.
NEXT_WEEK = pd.concat(
    [EVENTS, themed_store(shoppers=150, days=7, seed=1, start=START + pd.Timedelta(days=28))],
    ignore_index=True,
)
TINY = themed_store(shoppers=60)


def retrain(models, events=EVENTS, run="r1", **settings):
    return retrain_store(events, STORE, models, RetrainSettings(**settings), run=run)


def live(models):
    return load_model(current_model_path(models, STORE))


def live_bytes(models):
    current = current_model_path(models, STORE)
    return current.read_bytes(), fallback_path(current).read_bytes()


# ---- A new store ---------------------------------------------------------------------------


def test_a_good_first_model_is_promoted_and_served(tmp_path):
    report = retrain(tmp_path)

    assert report.decision == PROMOTED
    run_file = store_dir(tmp_path, STORE) / "runs/r1/model.joblib"
    model = live(tmp_path)
    assert model.metadata.algorithm == "item-to-item"
    # What was judged is exactly what is served.
    assert live_bytes(tmp_path)[0] == run_file.read_bytes()
    assert model.similar_items("t0-p0") == load_model(run_file).similar_items("t0-p0")
    fallback = load_model(fallback_path(current_model_path(tmp_path, STORE)))
    assert fallback.metadata.algorithm == "recent"


def test_the_served_model_never_saw_the_held_out_week(tmp_path):
    report = retrain(tmp_path)
    assert report.holdout_from == report.events_to - pd.Timedelta(days=7)
    assert live(tmp_path).metadata.events_to < report.holdout_from
    assert report.training_events + report.held_out_events == report.events == len(EVENTS)


def test_every_run_keeps_a_report(tmp_path):
    retrain(tmp_path)
    saved = json.loads((store_dir(tmp_path, STORE) / "runs/r1/report.json").read_text())
    assert saved["decision"] == PROMOTED
    assert saved["gate"]["vs_best_sellers"]["difference_range"][0] > 0
    assert saved["serving"]["algorithm"] == "item-to-item"


def test_a_tiny_store_is_served_best_sellers(tmp_path):
    report = retrain(tmp_path, TINY)

    assert report.decision == BEST_SELLERS
    assert "held-out shoppers could be scored" in report.reason
    served = live(tmp_path)
    assert served.metadata.algorithm == "recent"
    # Trained on the held-out period too: best sellers are not judged, and need it most.
    assert served.metadata.events_to > report.holdout_from
    # A product nobody has seen still gets a list, so the API never has nothing to show.
    assert len(served.similar_items("brand-new-product")) == 10


def test_a_store_with_only_this_weeks_events_is_served_best_sellers(tmp_path):
    report = retrain(tmp_path, themed_store(days=5))
    assert report.decision == BEST_SELLERS
    assert "no events before the held-out period" in report.reason


def test_a_store_with_no_events_gets_nothing(tmp_path):
    report = retrain(tmp_path, EVENTS.iloc[:0])
    assert report.decision == NO_EVENTS
    assert not current_model_path(tmp_path, STORE).exists()


# ---- Replacing a live model ----------------------------------------------------------------


def test_the_deliberately_bad_model_is_rejected_and_the_live_one_stays(tmp_path):
    retrain(tmp_path, run="good")
    before = live_bytes(tmp_path)

    report = retrain(tmp_path, NEXT_WEEK, run="bad", model="random")

    assert report.decision == KEPT
    assert report.reason == "new random model not used: not measurably better than best sellers"
    assert live_bytes(tmp_path) == before  # neither file touched
    assert (store_dir(tmp_path, STORE) / "runs/bad/model.joblib").exists()  # kept to inspect


def test_a_newer_good_model_replaces_the_live_one(tmp_path):
    retrain(tmp_path, run="week 4")
    report = retrain(tmp_path, NEXT_WEEK, run="week 5")

    assert report.decision == PROMOTED
    assert report.gate.vs_live is not None
    assert report.live_before["events_to"] < report.serving["events_to"]
    # The old model was trained before this week's held-out events: nothing to trim.
    assert report.scored_from == report.holdout_from


def test_a_store_starts_with_item_to_item(tmp_path):
    assert retrain(tmp_path).model == "item-to-item"
    assert retrain(tmp_path / "tiny", TINY).model == "item-to-item"


def test_a_store_keeps_the_kind_of_model_that_is_live(tmp_path):
    # On Retailrocket a fresh item-to-item could not replace a live ALS model (0009), so
    # retraining item-to-item every week would only let the ALS model age.
    assert retrain(tmp_path, model="als", run="als").decision == PROMOTED
    report = retrain(tmp_path, NEXT_WEEK, run="next week")
    assert report.model == "als"
    assert live(tmp_path).metadata.events_to > report.live_before["events_to"]


def test_a_store_on_best_sellers_tries_item_to_item_again(tmp_path):
    retrain(tmp_path, TINY, run="small")
    assert retrain(tmp_path, EVENTS, run="grown").model == "item-to-item"


def test_a_quiet_week_does_not_swap_a_model_for_best_sellers(tmp_path):
    retrain(tmp_path)
    before = live_bytes(tmp_path)
    report = retrain(tmp_path, NEXT_WEEK, run="r2", gate=GateSettings(min_users=10_000))
    assert report.decision == KEPT
    assert live_bytes(tmp_path) == before


def test_live_best_sellers_give_way_to_a_model_once_there_is_enough_data(tmp_path):
    retrain(tmp_path, TINY, run="small")
    report = retrain(tmp_path, EVENTS, run="grown")
    assert report.live_before["algorithm"] == "recent"
    assert report.decision == PROMOTED


def test_live_best_sellers_are_refreshed_when_no_model_qualifies(tmp_path):
    first = retrain(tmp_path, TINY, run="first")
    newer = pd.concat([TINY, themed_store(shoppers=10, days=2, seed=3,
                                          start=START + pd.Timedelta(days=28))])
    report = retrain(tmp_path, newer, run="second")
    assert report.decision == BEST_SELLERS
    assert live(tmp_path).metadata.events_to > first.events_to


def test_held_out_events_the_live_model_saw_are_not_scored(tmp_path):
    # A live model trained up to one day before the end has seen six of the seven
    # held-out days. Scoring starts after what it saw, so it is not marked on its own lessons.
    first = retrain(tmp_path, holdout_days=1, run="short", gate=GateSettings(min_users=5))
    assert first.decision == PROMOTED
    seen_to = max(live(tmp_path).metadata.events_to,
                  load_model(fallback_path(current_model_path(tmp_path, STORE))).metadata.events_to)

    report = retrain(tmp_path, run="normal")

    assert report.scored_from > report.holdout_from
    assert report.scored_from == pd.Timestamp(seen_to) + pd.Timedelta(microseconds=1)


def test_a_broken_live_file_does_not_stop_a_working_model(tmp_path):
    current = current_model_path(tmp_path, STORE)
    current.parent.mkdir(parents=True)
    current.write_bytes(b"not a model")

    report = retrain(tmp_path)

    assert report.decision == PROMOTED
    assert report.live_error.startswith("live model unusable")


def test_a_live_model_of_another_store_is_not_used(tmp_path):
    other = ItemToItemRecommender("st_other")
    other.fit(build_interactions(EVENTS))
    other.save(current_model_path(tmp_path, STORE))

    report = retrain(tmp_path)

    assert "trained for store 'st_other'" in report.live_error
    assert live(tmp_path).metadata.store_id == STORE


# ---- Repeatable, replayable, safe ----------------------------------------------------------


def test_same_events_same_decision_and_same_answers(tmp_path):
    first = retrain(tmp_path / "a")
    second = retrain(tmp_path / "b")
    assert first.decision == second.decision
    assert first.gate.vs_best_sellers == second.gate.vs_best_sellers
    a, b = live(tmp_path / "a"), live(tmp_path / "b")
    assert a.similar_items("t1-p2") == b.similar_items("t1-p2")


def test_later_events_are_ignored_when_replaying_the_past(tmp_path):
    until = START + pd.Timedelta(days=20)
    report = retrain_store(NEXT_WEEK, STORE, tmp_path, until=until, run="r1")
    assert report.events_to <= until
    assert report.events == (NEXT_WEEK["timestamp"] <= until).sum()


def test_runs_without_a_name_get_separate_folders(tmp_path):
    first = retrain_store(TINY, STORE, tmp_path)
    second = retrain_store(TINY, STORE, tmp_path)
    assert first.run != second.run
    assert len(list((store_dir(tmp_path, STORE) / "runs").iterdir())) == 2


@pytest.mark.parametrize("store", ["../escape", "st/x", "", "st x"])
def test_a_store_id_cannot_leave_the_models_folder(tmp_path, store):
    with pytest.raises(ValueError, match="store_id"):
        retrain_store(TINY, store, tmp_path)


def test_an_unknown_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown model"):
        retrain(tmp_path, model="deep-learning")


# ---- The command -----------------------------------------------------------------------------


def write_export(path, stores: dict[str, pd.DataFrame]):
    """An export of the events table, with a header, as Postgres writes it."""
    rows = [frame.assign(store_id=store) for store, frame in stores.items()]
    table = pd.concat(rows, ignore_index=True)
    table.insert(0, "id", range(1, len(table) + 1))
    table["event_id"] = "e" + table["id"].astype(str)
    table["received_at"] = table["timestamp"]
    table = table[["id", "store_id", "event_id", "user_id", "item_id", "event_type", "order_id",
                   "timestamp", "received_at"]]
    table.to_csv(path, index=False)
    return path


def test_command_retrains_every_store_in_an_export(tmp_path, capsys):
    export = write_export(tmp_path / "events.csv", {"st_big": EVENTS, "st_small": TINY})

    code = main(["--all", "--events", str(export), "--models-dir", str(tmp_path / "models")])

    out = capsys.readouterr().out
    assert code == 0
    assert "st_big: promoted" in out
    assert "st_small: serving best sellers" in out
    for store in ["st_big", "st_small"]:
        assert current_model_path(tmp_path / "models", store).is_file()


def test_command_fails_only_the_broken_store(tmp_path, capsys):
    rounded = TINY.assign(item_id="8.3137E+12")
    export = write_export(tmp_path / "events.csv", {"st_big": EVENTS, "st_rounded": rounded})

    code = main(["--all", "--events", str(export), "--models-dir", str(tmp_path / "models")])

    captured = capsys.readouterr()
    assert code == 1
    assert "st_big: promoted" in captured.out
    assert "st_rounded: failed" in captured.err
    assert not current_model_path(tmp_path / "models", "st_rounded").exists()


def test_command_settings_reach_the_gate(tmp_path, capsys):
    export = write_export(tmp_path / "events.csv", {"st_big": EVENTS})
    main(["--store", "st_big", "--events", str(export), "--models-dir", str(tmp_path),
          "--min-users", "100000", "--holdout-days", "3", "--model", "als"])
    out = capsys.readouterr().out
    assert "no als model qualified: only" in out
    assert "the gate needs 100000" in out


def test_unknown_products_still_raise_cold_start_on_a_promoted_model(tmp_path):
    retrain(tmp_path)
    with pytest.raises(ColdStartError):
        live(tmp_path).similar_items("brand-new-product")
