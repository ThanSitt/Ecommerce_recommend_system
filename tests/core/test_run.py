"""Scoring a model end to end."""

import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.evaluation.answer_key import build_answer_key
from core.evaluation.run import evaluate, main
from core.models.baselines import MostPopularRecommender


def events_from(pairs: list[tuple[str, str]], day: str) -> pd.DataFrame:
    """One view per (user, item) pair."""
    return pd.DataFrame(
        {
            "user_id": [u for u, _ in pairs],
            "item_id": [i for _, i in pairs],
            "event_type": "view",
            "timestamp": pd.Timestamp(day, tz="UTC"),
            "order_id": None,
        }
    )


def test_scores_worked_out_by_hand():
    # Training: x touched by 3 users, y by 2, z by 1.
    grid = build_interactions(events_from(
        [("a", "x"), ("b", "x"), ("c", "x"), ("a", "y"), ("b", "y"), ("c", "z")],
        "2015-06-01",
    ))
    # Later: a touches z, c touches y, b touches an unknown item (not scored).
    key = build_answer_key(
        grid, events_from([("a", "z"), ("c", "y"), ("b", "new")], "2015-08-25")
    )
    model = MostPopularRecommender("test")
    model.fit(grid)

    # Most-popular minus what each has seen: a gets [z], c gets [y]. Both hit at #1.
    results = evaluate(model, key, "any")

    assert results.users == 2
    assert results.precision == pytest.approx(0.1)  # 1 right of 10, for both
    assert results.recall == pytest.approx(1.0)  # each found their only answer
    assert results.ndcg == pytest.approx(1.0)  # ... at the top
    assert results.coverage == pytest.approx(2 / 3)  # z and y shown, of x, y, z


def test_an_empty_answer_key_cannot_be_scored():
    grid = build_interactions(events_from([("a", "x")], "2015-06-01"))
    key = build_answer_key(grid, events_from([("stranger", "x")], "2015-08-25"))
    model = MostPopularRecommender("test")
    model.fit(grid)
    with pytest.raises(ValueError, match="no users"):
        evaluate(model, key, "any")


def test_command_prints_a_row_per_model(capsys):
    main(["--dataset", "fixture", "--model", "popular", "random"])
    table = capsys.readouterr().out
    assert "| popular | any |" in table
    assert "| random | any |" in table
    # The fixture is too small for a cart-or-purchase key: skipped, with a note.
    assert "skipping hits='cart or purchase'" in table
