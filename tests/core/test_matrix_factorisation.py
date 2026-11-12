"""ALS and BPR. The maths is the library's, so these check what is ours: that the saved
tables answer exactly as the library would, that the answers follow the interface's rules,
and that the files work without the library."""

import numpy as np
import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.models.base import ColdStartError
from core.models.loading import fallback_path, load_model
from core.models.matrix_factorisation import ALSRecommender, BPRRecommender
from core.training.train import train

BOTH = [ALSRecommender, BPRRecommender]
# Small and fast, for tiny grids. The real defaults are for Retailrocket's 50,000 products.
# ALS needs very few factors here: with more numbers than the grid has patterns it learns
# noise (4 factors on the two-worlds grid puts b0 second on a0's page).
SMALL = {ALSRecommender: {"factors": 2, "regularisation": 1.0, "alpha": 10.0},
         BPRRecommender: {"factors": 4, "learning_rate": 0.05, "iterations": 200}}


def grid_from(rows: list[tuple[str, str, str]]):
    """One event per (user, item, event_type) row."""
    return build_interactions(
        pd.DataFrame(
            {
                "user_id": [u for u, _, _ in rows],
                "item_id": [i for _, i, _ in rows],
                "event_type": [e for _, _, e in rows],
                "timestamp": pd.Timestamp("2015-06-01", tz="UTC"),
                "order_id": None,
            }
        )
    )


def two_worlds(shoppers_per_world: int = 15):
    """Two groups of shoppers who never touch the other group's products: everyone in a
    world touches 3 of its 4 products. Any model worth having keeps the worlds apart."""
    rows = []
    for world in ["a", "b"]:
        products = [f"{world}{i}" for i in range(4)]
        for s in range(shoppers_per_world):
            skip = s % 4
            rows += [(f"{world}-shopper-{s}", p, "view") for i, p in enumerate(products) if i != skip]
    return grid_from(rows)


def random_grid(seed: int = 0, users: int = 60, items: int = 25):
    rng = np.random.default_rng(seed)
    rows = {
        (f"u{u}", f"i{i}", rng.choice(["view", "cart", "purchase"]))
        for u in range(users)
        for i in rng.choice(items, size=rng.integers(2, 8), replace=False)
    }
    return grid_from(sorted(rows))


def trained(cls, grid=None, **settings):
    model = cls("test", **{**SMALL[cls], **settings})
    model.fit(two_worlds() if grid is None else grid)
    return model


def ids(recs):
    return [item for item, _ in recs]


# ---- Answers exactly as the library would ---------------------------------------------


def library_model(cls, grid, **settings):
    """The library's own model, trained the same way as ours."""
    from implicit.cpu.als import AlternatingLeastSquares
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    from threadpoolctl import threadpool_limits

    s = {**SMALL[cls], **settings}
    with threadpool_limits(1, "blas"):
        if cls is ALSRecommender:
            model = AlternatingLeastSquares(
                factors=s["factors"], regularization=s["regularisation"], alpha=s["alpha"],
                random_state=0,
            )
        else:
            model = BayesianPersonalizedRanking(
                factors=s["factors"], learning_rate=s["learning_rate"],
                iterations=s["iterations"], random_state=0, num_threads=1,
            )
        model.fit(grid.matrix, show_progress=False)
    return model


@pytest.mark.parametrize("cls", BOTH)
def test_similar_items_match_the_library(cls):
    grid = random_grid()
    ours, theirs = trained(cls, grid), library_model(cls, grid)
    for row, item in enumerate(grid.item_ids[:10]):
        their_rows, their_scores = theirs.similar_items(row, N=6)
        # The library lists the product itself first; ours never does.
        expected = [(grid.item_ids[r], s) for r, s in zip(their_rows, their_scores) if r != row]
        recs = ours.similar_items(item, n=5)
        assert ids(recs) == [i for i, _ in expected][:5]
        assert [s for _, s in recs] == pytest.approx([s for _, s in expected][:5], rel=1e-4)


@pytest.mark.parametrize("cls", BOTH)
def test_recommendations_match_the_library(cls):
    grid = random_grid()
    ours, theirs = trained(cls, grid), library_model(cls, grid)
    for row, user in enumerate(grid.user_ids[:10]):
        their_rows, their_scores = theirs.recommend(
            row, grid.matrix[row], N=5, filter_already_liked_items=True
        )
        recs = ours.recommend_for_user(user, n=5)
        assert ids(recs) == [grid.item_ids[r] for r in their_rows]
        assert [s for _, s in recs] == pytest.approx(list(their_scores), rel=1e-4)


# ---- The interface's rules ------------------------------------------------------------


@pytest.mark.parametrize("cls", BOTH)
def test_similar_products_come_from_the_same_world(cls):
    model = trained(cls)
    for item in ["a0", "b2"]:
        assert all(other[0] == item[0] for other in ids(model.similar_items(item, n=3)))


@pytest.mark.parametrize("cls", BOTH)
def test_recommends_the_missing_product_of_the_users_world(cls):
    # a-shopper-0 skipped a0 and touched a1, a2, a3: a0 is the obvious next product.
    assert ids(trained(cls).recommend_for_user("a-shopper-0"))[0] == "a0"


@pytest.mark.parametrize("cls", BOTH)
def test_a_product_is_never_its_own_recommendation(cls):
    model = trained(cls)
    for item in ["a0", "a1", "b0", "b3"]:
        assert item not in ids(model.similar_items(item))


@pytest.mark.parametrize("cls", BOTH)
def test_never_recommends_what_the_user_touched(cls):
    recs = trained(cls).recommend_for_user("a-shopper-1")
    assert not {"a0", "a2", "a3"} & set(ids(recs))


@pytest.mark.parametrize("cls", BOTH)
def test_can_include_seen_items_when_asked(cls):
    recs = trained(cls).recommend_for_user("a-shopper-1", exclude_seen=False)
    assert {"a0", "a2", "a3"} <= set(ids(recs))


@pytest.mark.parametrize("cls", BOTH)
def test_n_limits_the_list(cls):
    model = trained(cls)
    assert len(model.similar_items("a0", n=2)) == 2
    assert len(model.recommend_for_user("a-shopper-0", n=3)) == 3
    # Asking for more than there is gives every other product, once.
    assert sorted(ids(model.similar_items("a0", n=50))) == ["a1", "a2", "a3", "b0", "b1", "b2", "b3"]


@pytest.mark.parametrize("cls", BOTH)
def test_unknown_user_or_product_raises_cold_start(cls):
    model = trained(cls)
    with pytest.raises(ColdStartError):
        model.recommend_for_user("never-seen")
    with pytest.raises(ColdStartError):
        model.similar_items("never-seen")


# ---- What each model learns from ------------------------------------------------------


def test_als_uses_the_grids_weights_and_bpr_does_not():
    mixed = random_grid()  # views, carts and purchases
    # The same cells, but every one a view.
    all_views = grid_from([(u, mixed.item_ids[i], "view")
                           for u, row in zip(mixed.user_ids, mixed.matrix.tolil().rows)
                           for i in row])
    assert all_views.matrix.nnz == mixed.matrix.nnz
    als = [trained(ALSRecommender, g)._item_factors for g in (mixed, all_views)]
    bpr = [trained(BPRRecommender, g)._item_factors for g in (mixed, all_views)]
    assert not np.allclose(als[0], als[1])
    assert np.array_equal(bpr[0], bpr[1])


def test_bpr_adds_a_popularity_number_per_product():
    model = trained(BPRRecommender)
    assert model._item_factors.shape == (8, 4 + 1)
    assert np.allclose(model._user_factors[:, -1], 1)


# ---- Repeatable, saved, and served without the library ---------------------------------


@pytest.mark.parametrize("cls", BOTH)
def test_same_data_and_seed_give_the_same_answers(cls):
    grid = random_grid()
    first, second = trained(cls, grid), trained(cls, grid)
    assert np.array_equal(first._item_factors, second._item_factors)
    assert first.similar_items("i3") == second.similar_items("i3")


@pytest.mark.parametrize("cls", BOTH)
def test_the_seed_is_used(cls):
    grid = random_grid()
    assert not np.array_equal(
        trained(cls, grid, seed=0)._item_factors, trained(cls, grid, seed=1)._item_factors
    )


@pytest.mark.parametrize("cls", BOTH)
def test_save_and_load_give_the_same_answers(cls, tmp_path):
    model = trained(cls)
    path = tmp_path / "model.joblib"
    model.save(path)

    for loaded in (cls.load(path), load_model(path)):
        assert loaded.similar_items("a0") == model.similar_items("a0")
        assert loaded.recommend_for_user("b-shopper-2") == model.recommend_for_user("b-shopper-2")
        assert loaded.metadata == model.metadata


@pytest.mark.parametrize("cls", BOTH)
def test_saved_model_holds_only_plain_tables_not_the_library(cls):
    model = trained(cls)
    assert not any("implicit" in type(value).__module__ for value in vars(model).values())
    assert model._item_factors.dtype == np.float32


def test_metadata_describes_the_model():
    als = trained(ALSRecommender).metadata
    assert als.algorithm == "als"
    assert als.hyperparameters == {
        "factors": 2, "regularisation": 1.0, "iterations": 15, "seed": 0, "alpha": 10.0,
    }
    assert (als.n_users, als.n_items) == (30, 8)
    bpr = trained(BPRRecommender).metadata
    assert bpr.algorithm == "bpr"
    assert bpr.hyperparameters == {
        "factors": 4, "regularisation": 0.01, "iterations": 200, "seed": 0,
        "learning_rate": 0.05,
    }


def test_defaults_are_the_settings_chosen_on_validation():
    # docs/decisions/0008. Changing them changes every reported number.
    als, bpr = ALSRecommender("test"), BPRRecommender("test")
    assert (als.factors, als.regularisation, als.alpha, als.iterations) == (200, 10.0, 200.0, 15)
    assert (bpr.factors, bpr.regularisation, bpr.learning_rate, bpr.iterations) == (
        100, 0.01, 0.1, 100,
    )


@pytest.mark.parametrize("name", ["als", "bpr"])
def test_training_command_makes_a_loadable_file_with_a_fallback(name, tmp_path):
    _, path, prepared = train("fixture", name, tmp_path / "current.joblib")
    model = load_model(path)

    assert model.metadata.algorithm == name
    assert load_model(fallback_path(path)).metadata.algorithm == "recent"
    assert len(model.similar_items(prepared.grid.item_ids[0], n=3)) == 3
    assert len(model.recommend_for_user(prepared.grid.user_ids[0], n=3)) == 3
