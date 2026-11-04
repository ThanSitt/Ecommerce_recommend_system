"""create_store and hash_api_key; no database needed (the session is a fake)."""

import hashlib
from unittest.mock import MagicMock

from core.db.models import StoreRow
from core.db.stores import create_store, hash_api_key


def test_hash_is_sha256_hex():
    assert hash_api_key("rk_abc") == hashlib.sha256(b"rk_abc").hexdigest()


def test_same_key_gives_same_hash():
    # The lookup on every request depends on this.
    assert hash_api_key("rk_abc") == hash_api_key("rk_abc")


def test_create_store_saves_the_hash_not_the_key():
    session = MagicMock()

    store_id, api_key = create_store(session)

    (row,), _ = session.add.call_args
    assert isinstance(row, StoreRow)
    assert row.store_id == store_id
    assert row.api_key_hash == hash_api_key(api_key)
    assert api_key not in row.api_key_hash


def test_create_store_does_not_commit():
    # The caller owns the transaction.
    session = MagicMock()

    create_store(session)

    session.commit.assert_not_called()


def test_ids_and_keys_have_their_prefixes():
    store_id, api_key = create_store(MagicMock())

    assert store_id.startswith("st_")
    assert api_key.startswith("rk_")


def test_two_stores_get_different_ids_and_keys():
    first = create_store(MagicMock())
    second = create_store(MagicMock())

    assert first[0] != second[0]
    assert first[1] != second[1]
