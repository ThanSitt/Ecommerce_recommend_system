from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.api.schemas import Event

VALID = {
    "user_id": "v-8812",
    "item_id": "123",
    "event_type": "view",
    "timestamp": "2026-10-03T09:14:22Z",
}


def make(**changes):
    """A valid event with some fields changed; a value of None removes the field."""
    data = {**VALID, **changes}
    return {k: v for k, v in data.items() if v is not None}


def test_minimal_event_is_valid():
    event = Event.model_validate(VALID)

    assert event.user_id == "v-8812"
    assert event.event_id is None
    assert event.order_id is None


def test_timestamp_is_parsed_to_aware_datetime():
    event = Event.model_validate(VALID)

    assert event.timestamp == datetime(2026, 10, 3, 9, 14, 22, tzinfo=timezone.utc)


def test_timestamp_with_offset_is_valid():
    event = Event.model_validate(make(timestamp="2026-10-03T12:14:22+03:00"))

    assert event.timestamp == datetime(2026, 10, 3, 9, 14, 22, tzinfo=timezone.utc)


def test_timestamp_without_timezone_is_rejected():
    with pytest.raises(ValidationError):
        Event.model_validate(make(timestamp="2026-10-03T09:14:22"))


@pytest.mark.parametrize("field", ["user_id", "item_id", "event_type", "timestamp"])
def test_required_field_missing_is_rejected(field):
    with pytest.raises(ValidationError):
        Event.model_validate(make(**{field: None}))


@pytest.mark.parametrize("field", ["user_id", "item_id"])
def test_empty_id_is_rejected(field):
    with pytest.raises(ValidationError):
        Event.model_validate(make(**{field: ""}))


@pytest.mark.parametrize("event_type", ["view", "cart", "purchase"])
def test_known_event_types_are_valid(event_type):
    assert Event.model_validate(make(event_type=event_type)).event_type == event_type


@pytest.mark.parametrize("event_type", ["addtocart", "transaction", "VIEW", ""])
def test_unknown_event_type_is_rejected(event_type):
    with pytest.raises(ValidationError):
        Event.model_validate(make(event_type=event_type))


def test_purchase_with_order_id_is_valid():
    event = Event.model_validate(make(event_type="purchase", order_id="1001"))

    assert event.order_id == "1001"


@pytest.mark.parametrize("event_type", ["view", "cart"])
def test_order_id_on_non_purchase_is_rejected(event_type):
    with pytest.raises(ValidationError, match="order_id"):
        Event.model_validate(make(event_type=event_type, order_id="1001"))


def test_unknown_fields_are_ignored():
    event = Event.model_validate(make(price=30, store="should-not-matter"))

    assert not hasattr(event, "price")
    assert "store" not in event.model_dump()
