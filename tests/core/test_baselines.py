"""The baselines, and saving and loading them."""

import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.models.baselines import (
    MostPopularRecommender,
    RandomRecommender,
    RecentlyPopularRecommender,
)
from core.models.loading import fallback_path, load_model
from core.training.train import train


def grid_from(pairs: list[tuple[str, str]]):
    """One view per (user, item) pair."""
    return build_interactions(
        pd.DataFrame(
            {
                "user_id": [u for u, _ in pairs],
                "item_id": [i for _, i in pairs],
                "event_type": "view",
                "timestamp": pd.Timestamp("2015-06-01", tz="UTC"),
                "order_id": None,
            }
        )
    )


# x touched by 3 users, y by 2, z by 1.
GRID = grid_from([("a", "x"), ("b", "x"), ("c", "x"), ("a", "y"), ("b", "y"), ("c", "z")])


def trained(model_class, **kwargs):
    model = model_class(store_id="test", **kwargs)
    model.fit(GRID)
    return model


def ids(recs):
    return [item for item, _ in recs]


# ---- Most-popular -------------------------------------------------------------------


def test_popular_ranks_by_number_of_users():
    recs = trained(MostPopularRecommender).recommend_for_user("new-user")
    assert recs == [("x", 3.0), ("y", 2.0), ("z", 1.0)]


def test_popular_excludes_what_the_user_already_touched():
    assert ids(trained(MostPopularRecommender).recommend_for_user("a")) == ["z"]


def test_popular_can_include_seen_items_when_asked():
    recs = trained(MostPopularRecommender).recommend_for_user("a", exclude_seen=False)
    assert ids(recs) == ["x", "y", "z"]


def test_popular_similar_items_leaves_out_the_item_itself():
    assert ids(trained(MostPopularRecommender).similar_items("x")) == ["y", "z"]


def test_n_limits_the_list():
    assert ids(trained(MostPopularRecommender).recommend_for_user("new-user", n=2)) == ["x", "y"]


# ---- Recently popular ---------------------------------------------------------------


def grid_with_days(rows: list[tuple[str, str, str]]):
    """One view per (user, item, day) row."""
    return build_interactions(
        pd.DataFrame(
            {
                "user_id": [u for u, _, _ in rows],
                "item_id": [i for _, i, _ in rows],
                "event_type": "view",
                "timestamp": pd.to_datetime([d for _, _, d in rows], utc=True),
                "order_id": None,
            }
        )
    )


# "old" was touched by 3 users in May. "new" by 2 users in the last week of training.
TRENDING = grid_with_days([
    ("a", "old", "2015-05-10"), ("b", "old", "2015-05-11"), ("c", "old", "2015-05-12"),
    ("d", "new", "2015-08-15"), ("e", "new", "2015-08-20"),
])


def test_recent_ranks_by_users_inside_the_window_only():
    model = RecentlyPopularRecommender("test")
    model.fit(TRENDING)
    assert ids(model.recommend_for_user("someone-new")) == ["new", "old"]


def test_most_popular_would_rank_the_same_products_the_other_way():
    # The difference between the two baselines, on the same data.
    model = MostPopularRecommender("test")
    model.fit(TRENDING)
    assert ids(model.recommend_for_user("someone-new")) == ["old", "new"]


def test_recent_counts_different_users_not_events():
    grid = grid_with_days([
        ("a", "x", "2015-08-19"), ("a", "x", "2015-08-20"), ("a", "x", "2015-08-20"),
        ("b", "y", "2015-08-19"), ("c", "y", "2015-08-20"),
    ])
    model = RecentlyPopularRecommender("test")
    model.fit(grid)
    assert ids(model.recommend_for_user("someone-new")) == ["y", "x"]


def test_window_is_counted_back_from_the_last_event():
    # Last event 08-20, so a 14-day window starts on 08-06: that day is in, 08-05 is out.
    grid = grid_with_days([
        ("a", "edge", "2015-08-06"), ("b", "out", "2015-08-05"), ("c", "last", "2015-08-20"),
    ])
    model = RecentlyPopularRecommender("test")
    model.fit(grid)
    scores = dict(model.recommend_for_user("someone-new"))
    # The whole number is users in the window; the fraction only breaks ties.
    assert {item: int(score) for item, score in scores.items()} == {
        "edge": 1, "last": 1, "out": 0,
    }


def test_recent_window_is_a_setting_and_is_recorded():
    model = RecentlyPopularRecommender("test", days=3)
    model.fit(TRENDING)
    assert model.metadata.hyperparameters == {"days": 3}
    assert model.metadata.algorithm == "recent"
    # 08-15 is now outside the window: only e's touch counts.
    scores = dict(model.recommend_for_user("someone-new"))
    assert int(scores["new"]) == 1
    assert int(scores["old"]) == 0


def test_ties_go_to_the_product_with_more_users_overall():
    # One user each in the window. On ID order alone, apple would come first.
    grid = grid_with_days([
        ("a", "zebra", "2015-05-10"), ("b", "zebra", "2015-05-11"),
        ("c", "zebra", "2015-08-20"), ("d", "apple", "2015-08-20"),
    ])
    model = RecentlyPopularRecommender("test")
    model.fit(grid)
    assert ids(model.recommend_for_user("someone-new")) == ["zebra", "apple"]


def test_a_quiet_store_still_gets_a_sensible_list():
    # Almost nothing in the window: the rest of the list falls back on all-time popularity.
    grid = grid_with_days([
        ("a", "zebra", "2015-05-10"), ("b", "zebra", "2015-05-11"), ("c", "zebra", "2015-05-12"),
        ("d", "apple", "2015-05-13"),
        ("e", "new", "2015-08-20"),
    ])
    model = RecentlyPopularRecommender("test")
    model.fit(grid)
    assert ids(model.recommend_for_user("someone-new")) == ["new", "zebra", "apple"]


def test_breaking_a_tie_never_beats_having_more_recent_users():
    # big is far ahead all-time, but small has more users in the window.
    grid = grid_with_days(
        [(f"old{i}", "big", "2015-05-10") for i in range(50)]
        + [("a", "big", "2015-08-20"), ("b", "small", "2015-08-20"), ("c", "small", "2015-08-20")]
    )
    model = RecentlyPopularRecommender("test")
    model.fit(grid)
    assert ids(model.recommend_for_user("someone-new")) == ["small", "big"]


def test_recent_excludes_what_the_user_already_touched():
    model = RecentlyPopularRecommender("test")
    model.fit(TRENDING)
    assert ids(model.recommend_for_user("d")) == ["old"]


# ---- Random -------------------------------------------------------------------------


def test_random_is_repeatable_for_the_same_user():
    model = trained(RandomRecommender)
    assert model.recommend_for_user("new-user") == model.recommend_for_user("new-user")


def test_random_excludes_what_the_user_already_touched():
    assert ids(trained(RandomRecommender).recommend_for_user("a")) == ["z"]


def test_random_differs_between_seeds():
    # 50 items, so two seeds giving the same order by chance is practically impossible.
    grid = grid_from([("a", f"item{i:02d}") for i in range(50)])
    first, second = RandomRecommender("test", seed=1), RandomRecommender("test", seed=2)
    first.fit(grid)
    second.fit(grid)
    assert first.recommend_for_user("new-user") != second.recommend_for_user("new-user")


# ---- Both ---------------------------------------------------------------------------


ALL_BASELINES = [MostPopularRecommender, RandomRecommender, RecentlyPopularRecommender]


@pytest.mark.parametrize("model_class", ALL_BASELINES)
def test_unknown_user_and_item_still_get_an_answer(model_class):
    model = trained(model_class)
    assert len(model.recommend_for_user("never-seen")) == 3
    assert len(model.similar_items("never-seen")) == 3


@pytest.mark.parametrize("model_class", ALL_BASELINES)
def test_save_and_load_give_the_same_answers(model_class, tmp_path):
    model = trained(model_class)
    path = tmp_path / "model.joblib"
    model.save(path)

    for loaded in (model_class.load(path), load_model(path)):
        assert loaded.recommend_for_user("a") == model.recommend_for_user("a")
        assert loaded.similar_items("x") == model.similar_items("x")
        assert loaded.metadata == model.metadata


def test_metadata_describes_the_model():
    meta = trained(RandomRecommender, seed=7).metadata
    assert meta.algorithm == "random"
    assert meta.store_id == "test"
    assert (meta.n_users, meta.n_items) == (3, 3)
    assert meta.hyperparameters == {"seed": 7}


def test_loading_the_wrong_class_fails(tmp_path):
    path = tmp_path / "model.joblib"
    trained(RandomRecommender).save(path)
    with pytest.raises(TypeError):
        MostPopularRecommender.load(path)


def test_untrained_model_cannot_be_saved(tmp_path):
    with pytest.raises(RuntimeError):
        MostPopularRecommender("test").save(tmp_path / "model.joblib")


def test_training_command_makes_a_loadable_file(tmp_path):
    _, path, _ = train("fixture", "popular", tmp_path / "popular.joblib")
    model = load_model(path)
    assert model.metadata.algorithm == "popular"
    assert len(model.recommend_for_user("someone-new")) == 10


def test_training_command_can_make_a_recent_model(tmp_path):
    _, path, _ = train("fixture", "recent", tmp_path / "recent.joblib")
    model = load_model(path)
    assert model.metadata.algorithm == "recent"
    assert model.metadata.hyperparameters == {"days": 14}


def test_training_saves_a_recently_popular_fallback_beside_the_model(tmp_path):
    trained, path, _ = train("fixture", "random", tmp_path / "current.joblib")
    fallback = load_model(tmp_path / "fallback.joblib")

    assert path == tmp_path / "current.joblib"
    assert fallback_path(path) == tmp_path / "fallback.joblib"
    assert trained.metadata.algorithm == "random"
    assert fallback.metadata.algorithm == "recent"
    # Same training data, same store.
    assert fallback.metadata.store_id == trained.metadata.store_id
    assert fallback.metadata.events_to == trained.metadata.events_to
