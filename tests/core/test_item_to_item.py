"""Item-to-item, on grids small enough to work the similarity out by hand."""

import math

import pandas as pd
import pytest

from core.data.interactions import build_interactions
from core.models.base import ColdStartError
from core.models.item_to_item import ItemToItemRecommender
from core.models.loading import fallback_path, load_model
from core.training.train import train


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


# x is touched by a, b, c. y by a, b. z by c. And separately, w and v by d.
# 5 products; everyone touched 2 of them. BM25 gives each shared user this much weight:
#   how_rare(user) = log(5) - log(1 + 2)
#   weight(product) = (1.2 + 1) / (1.2 * length(product) + 1) * how_rare(user)
#   length(product) = 0.25 + 0.75 * users(product) / average users per product (8 / 5)
# And a similarity adds up weight(first) * weight(second) over the users they share.
HOW_RARE = math.log(5) - math.log(3)


def weight(users_of_product: int) -> float:
    length = 0.25 + 0.75 * users_of_product / (8 / 5)
    return (1.2 + 1) / (1.2 * length + 1) * HOW_RARE


X_AND_Y = 2 * weight(3) * weight(2)  # share a and b
X_AND_Z = 1 * weight(3) * weight(1)  # share c
GRID = grid_from([
    ("a", "x", "view"), ("a", "y", "view"),
    ("b", "x", "view"), ("b", "y", "view"),
    ("c", "x", "view"), ("c", "z", "view"),
    ("d", "w", "view"), ("d", "v", "view"),
])


def trained(grid=GRID, **settings) -> ItemToItemRecommender:
    model = ItemToItemRecommender("test", **settings)
    model.fit(grid)
    return model


def ids(recs):
    return [item for item, _ in recs]


# ---- similar_items: what the product page asks ---------------------------------------


def test_similarity_worked_out_by_hand():
    recs = trained().similar_items("x")
    assert ids(recs) == ["y", "z"]
    assert recs[0][1] == pytest.approx(X_AND_Y)
    assert recs[1][1] == pytest.approx(X_AND_Z)


def test_a_product_is_never_its_own_recommendation():
    for item in ["x", "y", "z", "w", "v"]:
        assert item not in ids(trained().similar_items(item))


def test_products_with_no_shared_users_are_not_similar():
    # y and z share nobody, and w and v are a separate world. Baselines would show them all.
    assert ids(trained().similar_items("y")) == ["x"]
    assert ids(trained().similar_items("w")) == ["v"]


def filler_products(count: int) -> list[tuple[str, str, str]]:
    """Unrelated products, so a tiny grid is a catalogue where touching 2 is touching few."""
    return [(f"filler-user-{i}", f"filler-{i}", "view") for i in range(count)]


def test_best_sellers_do_not_appear_next_to_everything():
    # "hit" is touched by 20 users, one of whom also touched "a-pair". "b-pair" is touched
    # only by a user who touched "a-pair". Each shares 1 user, but sharing 1 of 20 is weak.
    rows = [(f"u{i}", "hit", "view") for i in range(20)]
    rows += [("u0", "a-pair", "view")]
    rows += [("p", "a-pair", "view"), ("p", "b-pair", "view")]
    rows += filler_products(10)
    assert ids(trained(grid_from(rows)).similar_items("a-pair")) == ["b-pair", "hit"]


def test_a_shopper_who_touched_many_products_counts_less():
    # The reason for choosing BM25. m and n are linked by a shopper who touched only those two.
    # p and q are linked by one who touched 30 products. Each pair has one user, and nobody
    # else touched them. With cosine both pairs would score the same.
    rows = [("focused", "m", "view"), ("focused", "n", "view")]
    rows += [("browser", "p", "view"), ("browser", "q", "view")]
    rows += [("browser", f"other-{i}", "view") for i in range(28)]
    rows += filler_products(30)
    model = trained(grid_from(rows))

    assert dict(model.similar_items("m"))["n"] > dict(model.similar_items("p", n=30))["q"]


def test_n_limits_the_list():
    assert ids(trained().similar_items("x", n=1)) == ["y"]


def test_each_product_keeps_its_own_strongest_neighbours():
    # x's strongest is y and z is weaker, so a limit of 1 shows the rows are per product: x
    # keeps y, while z has only x and still keeps it.
    model = trained(neighbours=1)
    assert ids(model.similar_items("x"))[0] == "y"
    assert ids(model.similar_items("y")) == ["x"]
    assert ids(model.similar_items("z")) == ["x"]


def test_neighbours_setting_limits_the_table():
    # One above the setting at most: the library asks for one extra to allow for the product
    # itself, which isn't always among the strongest.
    rows = [("many", f"p{i}", "view") for i in range(30)] + filler_products(30)
    model = trained(grid_from(rows), neighbours=5)
    assert model._similarity.getnnz(axis=1).max() <= 6


def test_unknown_product_raises_cold_start():
    with pytest.raises(ColdStartError):
        trained().similar_items("never-seen")


def test_known_product_with_no_neighbours_raises_cold_start():
    # e touched only "lonely", so nothing shares a user with it.
    grid = grid_from([("a", "x", "view"), ("b", "x", "view"), ("e", "lonely", "view")])
    with pytest.raises(ColdStartError):
        trained(grid).similar_items("lonely")


# ---- recommend_for_user --------------------------------------------------------------


def test_recommends_neighbours_of_what_the_user_touched():
    # a touched x and y. x votes for z. Nothing else is new to a.
    recs = trained().recommend_for_user("a")
    assert ids(recs) == ["z"]
    assert recs[0][1] == pytest.approx(X_AND_Z)


def test_never_recommends_what_the_user_touched():
    assert not {"x", "z"} & set(ids(trained().recommend_for_user("c")))


def test_can_include_seen_items_when_asked():
    recs = trained().recommend_for_user("a", exclude_seen=False)
    assert set(ids(recs)) == {"x", "y", "z"}


def test_stronger_interest_gives_a_stronger_vote():
    # p bought m and only viewed n: the purchase (5) counts five times as much as the view (1).
    rows = [
        ("p", "m", "purchase"), ("p", "n", "view"),
        ("q", "m", "view"), ("q", "from-n-side", "view"),
        ("r", "n", "view"), ("r", "z-from-m", "view"),
        ("s", "m", "view"), ("s", "z-from-m", "view"), ("s", "from-n-side", "view"),
    ]
    model = trained(grid_from(rows))
    similarity = {item: dict(model.similar_items(item)) for item in ["m", "n"]}
    recs = dict(model.recommend_for_user("p"))
    for candidate, score in recs.items():
        expected = 5 * similarity["m"].get(candidate, 0) + 1 * similarity["n"].get(candidate, 0)
        assert score == pytest.approx(expected)


def test_unknown_user_raises_cold_start():
    with pytest.raises(ColdStartError):
        trained().recommend_for_user("never-seen")


def test_user_with_nothing_new_to_see_raises_cold_start():
    # d touched w and v, which are only related to each other.
    with pytest.raises(ColdStartError):
        trained().recommend_for_user("d")


# ---- Saving, and the rest of the interface -------------------------------------------


def test_same_data_gives_the_same_answers():
    assert trained().similar_items("x") == trained().similar_items("x")
    assert trained().recommend_for_user("c") == trained().recommend_for_user("c")


def test_save_and_load_give_the_same_answers(tmp_path):
    model = trained()
    path = tmp_path / "model.joblib"
    model.save(path)

    for loaded in (ItemToItemRecommender.load(path), load_model(path)):
        assert loaded.similar_items("x") == model.similar_items("x")
        assert loaded.recommend_for_user("c") == model.recommend_for_user("c")
        assert loaded.metadata == model.metadata


def test_saved_model_holds_only_plain_tables_not_the_library():
    # Serving should not depend on the training library's own objects.
    model = trained()
    assert not any("implicit" in type(value).__module__ for value in vars(model).values())


def test_metadata_describes_the_model():
    meta = trained(neighbours=7).metadata
    assert meta.algorithm == "item-to-item"
    assert meta.hyperparameters == {"neighbours": 7, "k1": 1.2, "b": 0.75}
    assert (meta.n_users, meta.n_items) == (4, 5)


def test_training_command_makes_a_loadable_file_with_a_fallback(tmp_path):
    _, path, prepared = train("fixture", "item-to-item", tmp_path / "current.joblib")
    model = load_model(path)
    fallback = load_model(fallback_path(path))

    assert model.metadata.algorithm == "item-to-item"
    assert fallback.metadata.algorithm == "recent"
    # On the fixture's filter every user touched 2+ products, so every product has a neighbour.
    assert len(model.similar_items(prepared.grid.item_ids[0], n=3)) >= 1
