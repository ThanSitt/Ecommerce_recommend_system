import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core.api.auth import get_store_id
from core.config import Settings, get_settings

# A throwaway app with one protected endpoint, so auth is tested on its own,
# independently of any real endpoint.
app = FastAPI()


@app.get("/protected")
def protected(store_id: str = Depends(get_store_id)):
    return {"store_id": store_id}


app.dependency_overrides[get_settings] = lambda: Settings(
    _env_file=None,
    database_url="unused",
    core_api_keys="key-a, key-b",
)

client = TestClient(app)


@pytest.mark.parametrize("key", ["key-a", "key-b"])
def test_valid_key_resolves_to_store(key):
    response = client.get("/protected", headers={"Authorization": f"Bearer {key}"})

    assert response.status_code == 200
    assert response.json() == {"store_id": "dev"}


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
        "Bearer key-a, key-b",  # the whole env value is not a key
    ],
)
def test_bad_header_is_rejected(header):
    response = client.get("/protected", headers={"Authorization": header})

    assert response.status_code == 401
