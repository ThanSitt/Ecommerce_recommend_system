"""Checks the table definition itself; no database needed."""

from core.db.models import EventRow

table = EventRow.__table__


def test_table_name():
    assert table.name == "events"


def test_required_and_optional_columns():
    required = {"store_id", "user_id", "item_id", "event_type", "timestamp", "received_at"}
    optional = {"event_id", "order_id"}

    for name in required:
        assert table.c[name].nullable is False, name
    for name in optional:
        assert table.c[name].nullable is True, name


def test_timestamps_are_timezone_aware():
    assert table.c.timestamp.type.timezone is True
    assert table.c.received_at.type.timezone is True


def test_received_at_is_set_by_the_database():
    assert table.c.received_at.server_default is not None


def test_event_id_is_unique_per_store():
    unique = {
        tuple(c.name for c in constraint.columns)
        for constraint in table.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }
    assert ("store_id", "event_id") in unique
