from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core.api import auth
from core.api.auth import get_store_id
from core.db.session import get_db

# A throwaway app with one protected endpoint, so auth is tested on its own,
# independently of any real endpoint.
app = FastAPI()


@app.get("/protected")
def protected(store_id: str = Depends(get_store_id)):
    return {"store_id": store_id}


# Stands in for the stores table: key -> store. An uninstalled store's key is simply not
# in here, which is what find_store_id reports for it (tested in tests/core/test_stores.py).
STORES = {"key-a": "st_a", "key-b": "st_b"}


@pytest.fixture(autouse=True)
def fake_stores(monkeypatch):
    monkeypatch.setattr(auth, "find_store_id", lambda session, key: STORES.get(key))
    app.dependency_overrides[get_db] = lambda: MagicMock()
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


@pytest.mark.parametrize("key, store_id", [("key-a", "st_a"), ("key-b", "st_b")])
def test_each_key_resolves_to_its_own_store(key, store_id):
    response = client.get("/protected", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    assert response.json() == {"store_id": store_id}


def test_scheme_is_case_insensitive():
    response = client.get("/protected", headers={"Authorization": "bearer key-a"})

    assert response.status_code == 200


def test_missing_header_is_rejected():
    response = client.get("/protected")

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [
        "Bearer wrong-key",
        "Bearer ",
        "key-a",  # no scheme
        "Basic key-a",  # wrong scheme
    ],
)
def test_bad_header_is_rejected(header):
    response = client.get("/protected", headers={"Authorization": header})

    assert response.status_code == 401
