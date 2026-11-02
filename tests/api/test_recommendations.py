"""GET /v1/recommendations, tested with a fake model.

The model is replaced by a fake that returns fixed answers, so these tests cover the route
itself: auth, parameters and response shape. What the model answers is tested in
tests/core.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from core.api.main import app
from core.api.model import get_model
from core.config import Settings, get_settings

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def fake_model():
    model = MagicMock()
    model.similar_items.return_value = [("257", 3.0), ("88", 2.0), ("12", 1.0)]
    return model


@pytest.fixture
def client(fake_model):
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, database_url="unused", core_api_keys="test-key"
    )
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
        _env_file=None, database_url="unused", core_api_keys="k",
        model_path=str(tmp_path / "missing.joblib"),
    )

    with pytest.raises(HTTPException) as error:
        get_model(settings)

    assert error.value.status_code == 503
