"""POST /v1/events, tested without a database.

`save_events` is replaced by a fake that records what it was given, so these tests cover
the route itself: auth, validation, batch limit and response. Saving to PostgreSQL is
save_events' job and is tested separately against a real database.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from core.api import auth
from core.api.main import app
from core.api.routes import events as events_route
from core.db.session import get_db

AUTH = {"Authorization": "Bearer test-key"}
STORE_ID = "st_test"


def make_event(**overrides):
    event = {
        "event_id": "e1",
        "user_id": "u1",
        "item_id": "i1",
        "event_type": "view",
        "timestamp": "2026-10-13T12:00:00Z",
    }
    event.update(overrides)
    return event


@pytest.fixture
def fake_save(monkeypatch):
    """Replaces save_events; by default reports every event as newly inserted."""
    fake = MagicMock(side_effect=lambda session, store_id, events: (len(events), 0))
    monkeypatch.setattr(events_route, "save_events", fake)
    return fake


@pytest.fixture
def client(monkeypatch):
    # Stands in for the stores table: only "test-key" belongs to a store.
    monkeypatch.setattr(
        auth, "find_store_id", lambda session, key: STORE_ID if key == "test-key" else None
    )
    app.dependency_overrides[get_db] = lambda: MagicMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_valid_batch_is_saved(client, fake_save):
    batch = [make_event(event_id="e1"), make_event(event_id="e2")]

    response = client.post("/v1/events", json=batch, headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"received": 2, "inserted": 2, "duplicates": 0}
    fake_save.assert_called_once()


def test_store_comes_from_the_key_not_the_body(client, fake_save):
    client.post("/v1/events", json=[make_event(store_id="someone-else")], headers=AUTH)

    _, store_id, saved = fake_save.call_args.args
    assert store_id == STORE_ID
    assert not hasattr(saved[0], "store_id")


def test_duplicates_still_return_200(client, fake_save):
    fake_save.side_effect = lambda session, store_id, events: (0, len(events))

    response = client.post("/v1/events", json=[make_event()], headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"received": 1, "inserted": 0, "duplicates": 1}


def test_empty_batch_is_accepted(client, fake_save):
    response = client.post("/v1/events", json=[], headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"received": 0, "inserted": 0, "duplicates": 0}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-key"}])
def test_missing_or_wrong_key_is_rejected(client, fake_save, headers):
    response = client.post("/v1/events", json=[make_event()], headers=headers)

    assert response.status_code == 401
    fake_save.assert_not_called()


def test_one_invalid_event_rejects_the_whole_batch(client, fake_save):
    batch = [make_event(event_id="ok"), make_event(event_id="bad", event_type="like")]

    response = client.post("/v1/events", json=batch, headers=AUTH)

    assert response.status_code == 422
    fake_save.assert_not_called()


def test_single_object_instead_of_array_is_rejected(client, fake_save):
    response = client.post("/v1/events", json=make_event(), headers=AUTH)

    assert response.status_code == 422
    fake_save.assert_not_called()


def test_batch_over_the_limit_is_rejected(client, fake_save):
    batch = [make_event(event_id=str(i)) for i in range(events_route.MAX_BATCH_SIZE + 1)]

    response = client.post("/v1/events", json=batch, headers=AUTH)

    assert response.status_code == 422
    fake_save.assert_not_called()


def test_batch_at_the_limit_is_accepted(client, fake_save):
    batch = [make_event(event_id=str(i)) for i in range(events_route.MAX_BATCH_SIZE)]

    response = client.post("/v1/events", json=batch, headers=AUTH)

    assert response.status_code == 200
