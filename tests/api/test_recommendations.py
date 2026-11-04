"""GET /v1/recommendations, tested with a fake model.

The model is replaced by a fake that returns fixed answers, so these tests cover the route
itself: auth, parameters and response shape. What the model answers is tested in
tests/core.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from core.api import auth
from core.api.main import app
from core.api.model import get_fallback, get_model
from core.config import Settings, get_settings
from core.db.session import get_db
from core.models.base import ColdStartError
from core.training.train import train

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.similar_items.return_value = [("257", 3.0), ("88", 2.0), ("12", 1.0)]
    return model


@pytest.fixture
def client(fake_model, monkeypatch):
    # Stands in for the stores table: only "test-key" belongs to a store.
    monkeypatch.setattr(
        auth, "find_store_id", lambda session, key: "st_test" if key == "test-key" else None
    )
    app.dependency_overrides[get_db] = lambda: MagicMock()
    app.dependency_overrides[get_model] = lambda: fake_model
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_returns_item_ids_best_first(client, fake_model):
    response = client.get("/v1/recommendations?item_id=100&n=3", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"items": ["257", "88", "12"]}
    fake_model.similar_items.assert_called_once_with("100", n=3)


def test_n_defaults_to_10(client, fake_model):
    client.get("/v1/recommendations?item_id=100", headers=AUTH)

    fake_model.similar_items.assert_called_once_with("100", n=10)


def test_missing_key_is_rejected(client):
    response = client.get("/v1/recommendations?item_id=100")

    assert response.status_code == 401


def test_item_id_is_required(client):
    response = client.get("/v1/recommendations", headers=AUTH)

    assert response.status_code == 422


@pytest.mark.parametrize("n", [0, 51])
def test_n_out_of_range_is_rejected(client, n):
    response = client.get(f"/v1/recommendations?item_id=100&n={n}", headers=AUTH)

    assert response.status_code == 422


def test_missing_model_file_gives_503(tmp_path):
    settings = Settings(
        _env_file=None, database_url="unused", core_admin_key="unused",
        model_path=str(tmp_path / "missing.joblib"),
    )

    with pytest.raises(HTTPException) as error:
        get_model(settings)

    assert error.value.status_code == 503


# ---- Fallback: the model has no basis to answer -------------------------------------


@pytest.fixture
def cold_model(fake_model):
    fake_model.similar_items.side_effect = ColdStartError("unknown item")
    return fake_model


def test_unknown_product_gets_the_fallback_list(client, cold_model):
    fallback = MagicMock()
    fallback.similar_items.return_value = [("9", 700.0), ("8", 600.0)]
    app.dependency_overrides[get_fallback] = lambda: fallback

    response = client.get("/v1/recommendations?item_id=unknown&n=2", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"items": ["9", "8"]}
    fallback.similar_items.assert_called_once_with("unknown", n=2)


def test_no_fallback_file_gives_an_empty_list_not_an_error(client, cold_model):
    app.dependency_overrides[get_fallback] = lambda: None

    response = client.get("/v1/recommendations?item_id=unknown", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_fallback_is_not_used_when_the_model_answers(client):
    fallback = MagicMock()
    app.dependency_overrides[get_fallback] = lambda: fallback

    client.get("/v1/recommendations?item_id=100", headers=AUTH)

    fallback.similar_items.assert_not_called()


def test_files_written_by_training_serve_the_fallback(client, cold_model, tmp_path):
    # No fake fallback here: the real file that training writes, read the way the API does.
    trained, path, _ = train("fixture", "random", tmp_path / "current.joblib")
    settings = Settings(
        _env_file=None, database_url="unused", core_admin_key="unused", model_path=str(path)
    )
    app.dependency_overrides[get_settings] = lambda: settings

    response = client.get("/v1/recommendations?item_id=unknown&n=3", headers=AUTH)

    assert response.status_code == 200
    assert len(response.json()["items"]) == 3
    assert trained.name == "random"  # the main model was random; the answer is the fallback
    fallback = get_fallback(settings)
    assert fallback.name == "recent"
    assert response.json()["items"] == [item for item, _ in fallback.similar_items("unknown", n=3)]


def test_missing_fallback_file_gives_none(tmp_path):
    settings = Settings(
        _env_file=None, database_url="unused", core_admin_key="unused",
        model_path=str(tmp_path / "current.joblib"),
    )
    assert get_fallback(settings) is None
