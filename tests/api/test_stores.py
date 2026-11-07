"""POST /v1/stores, tested without a database.

The session is a fake, so these tests cover the route itself: the admin check, that the
store is saved and committed, and the response. What `create_store` saves is tested in
tests/core/test_stores.py.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from core.api import auth
from core.api.main import app
from core.config import Settings, get_settings
from core.db.models import StoreRow
from core.db.session import get_db

ADMIN = {"Authorization": "Bearer admin-key"}


@pytest.fixture
def session():
    return MagicMock()


@pytest.fixture
def client(session, monkeypatch):
    # "store-key" is a real store's key, so the store-key case below really tests that a
    # store's key is refused here, not just that an unknown key is.
    monkeypatch.setattr(
        auth, "find_store_id", lambda s, key: "st_x" if key == "store-key" else None
    )
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        database_url="unused",
        core_admin_key="admin-key",
    )
    app.dependency_overrides[get_db] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_admin_key_creates_a_store(client, session):
    response = client.post("/v1/stores", headers=ADMIN)

    assert response.status_code == 201
    body = response.json()
    assert body["store_id"].startswith("st_")
    assert body["api_key"].startswith("rk_")

    # One store row added, then committed.
    (row,), _ = session.add.call_args
    assert isinstance(row, StoreRow)
    assert row.store_id == body["store_id"]
    session.commit.assert_called_once()


def test_response_is_not_cached(client):
    # It carries a working key.
    response = client.post("/v1/stores", headers=ADMIN)

    assert response.headers["Cache-Control"] == "no-store"


def test_each_call_creates_a_different_store(client):
    first = client.post("/v1/stores", headers=ADMIN).json()
    second = client.post("/v1/stores", headers=ADMIN).json()

    assert first["store_id"] != second["store_id"]
    assert first["api_key"] != second["api_key"]


@pytest.mark.parametrize(
    "headers",
    [
        {},  # no key
        {"Authorization": "Bearer wrong-key"},
        # A store's own key can send events, but must not create stores.
        {"Authorization": "Bearer store-key"},
    ],
    ids=["missing", "wrong", "store-key"],
)
def test_without_admin_key_nothing_is_created(client, session, headers):
    response = client.post("/v1/stores", headers=headers)

    assert response.status_code == 401
    session.add.assert_not_called()
    session.commit.assert_not_called()
