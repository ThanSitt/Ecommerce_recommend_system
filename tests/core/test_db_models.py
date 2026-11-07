"""Checks the table definition itself; no database needed."""

from core.db.models import EventRow, StoreRow

table = EventRow.__table__
stores = StoreRow.__table__


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


def test_stores_table_name():
    assert stores.name == "stores"


def test_store_id_is_the_primary_key():
    assert [c.name for c in stores.primary_key.columns] == ["store_id"]


def test_store_columns_are_all_required():
    for name in ("store_id", "api_key_hash", "active", "created_at"):
        assert stores.c[name].nullable is False, name


def test_api_key_hash_is_unique():
    # One key must never resolve to two stores.
    assert stores.c.api_key_hash.unique is True


def test_store_is_active_by_default():
    assert stores.c.active.server_default is not None


def test_store_created_at_is_timezone_aware_and_set_by_the_database():
    assert stores.c.created_at.type.timezone is True
    assert stores.c.created_at.server_default is not None


def test_stores_table_has_no_platform_columns():
    # Platform identity (e.g. a shop domain) belongs to the connector, not the core.
    assert set(stores.c.keys()) == {"store_id", "api_key_hash", "active", "created_at"}
