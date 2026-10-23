"""Random and most-popular, and saving and loading them."""

import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.models.baselines import MostPopularRecommender, RandomRecommender
from core.models.loading import load_model
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


@pytest.mark.parametrize("model_class", [MostPopularRecommender, RandomRecommender])
def test_unknown_user_and_item_still_get_an_answer(model_class):
    model = trained(model_class)
    assert len(model.recommend_for_user("never-seen")) == 3
    assert len(model.similar_items("never-seen")) == 3


@pytest.mark.parametrize("model_class", [MostPopularRecommender, RandomRecommender])
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
