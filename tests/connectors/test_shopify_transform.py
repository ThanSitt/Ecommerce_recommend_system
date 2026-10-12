"""Unit tests for the Shopify pixel -> event contract transform.

Inputs are real pixel payloads captured from a dev store (tests/fixtures/shopify/).
"""

from __future__ import annotations

import json
from pathlib import Path

from connectors.shopify.app.transform import transform

FIXTURES = Path(__file__).parent.parent / "fixtures" / "shopify"

CONTRACT_FIELDS = {"event_id", "user_id", "item_id", "event_type", "order_id", "timestamp"}
CLIENT_ID = "6dce103d-b085-4ec9-9f84-3c937440e610"


def load_fixture(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


# ---- view ------------------------------------------------------------------------


def test_view_gives_one_view_event():
    result = transform(load_fixture("pixel_product_viewed.json"))

    assert result == [
        {
            "event_id": "sh-0e7ae0b5-1162-407E-307E-A93A03559637",
            "user_id": CLIENT_ID,
            "item_id": "8313705169069",
            "event_type": "view",
            "timestamp": "2026-10-05T23:51:35.903Z",
        }
    ]


def test_view_uses_product_id_not_variant_id():
    event = load_fixture("pixel_product_viewed.json")

    result = transform(event)

    assert result[0]["item_id"] == event["data"]["productVariant"]["product"]["id"]
    assert result[0]["item_id"] != event["data"]["productVariant"]["id"]


# ---- cart ------------------------------------------------------------------------


def test_cart_gives_one_cart_event():
    result = transform(load_fixture("pixel_product_added_to_cart.json"))

    assert result == [
        {
            "event_id": "sh-0e855d35-DE35-4C26-C5D5-6AE11BD7FEDB",
            "user_id": CLIENT_ID,
            "item_id": "8313705300141",
            "event_type": "cart",
            "timestamp": "2026-10-06T00:02:57.421Z",
        }
    ]


def test_cart_uses_product_id_not_variant_id():
    event = load_fixture("pixel_product_added_to_cart.json")

    result = transform(event)

    merchandise = event["data"]["cartLine"]["merchandise"]
    assert result[0]["item_id"] == merchandise["product"]["id"]
    assert result[0]["item_id"] != merchandise["id"]


# ---- purchase --------------------------------------------------------------------


def test_checkout_gives_one_purchase_per_product():
    result = transform(load_fixture("pixel_checkout_completed.json"))

    assert len(result) == 3
    assert all(e["event_type"] == "purchase" for e in result)
    assert [e["item_id"] for e in result] == ["8313705300141", "8313705169069", "8313705398445"]


def test_checkout_events_share_one_order_id():
    result = transform(load_fixture("pixel_checkout_completed.json"))

    assert {e["order_id"] for e in result} == {"18804001407149"}


def test_checkout_event_ids_are_unique_and_built_from_order_and_variant():
    result = transform(load_fixture("pixel_checkout_completed.json"))

    assert [e["event_id"] for e in result] == [
        "purchase:18804001407149:44906800480429",
        "purchase:18804001407149:44906800349357",
        "purchase:18804001407149:44906800578733",
    ]


def test_two_variants_of_same_product_get_same_item_but_different_event_ids():
    # Not in the captured order, so build it: the shirt bought in two sizes.
    event = load_fixture("pixel_checkout_completed.json")
    shirt = event["data"]["checkout"]["lineItems"][0]
    other_size = json.loads(json.dumps(shirt))
    other_size["variant"]["id"] = "99999999999999"
    event["data"]["checkout"]["lineItems"] = [shirt, other_size]

    result = transform(event)

    assert result[0]["item_id"] == result[1]["item_id"]
    assert result[0]["event_id"] != result[1]["event_id"]


# ---- rules for every event type --------------------------------------------------


def test_same_visitor_has_same_user_id_across_all_event_types():
    names = [
        "pixel_product_viewed.json",
        "pixel_product_added_to_cart.json",
        "pixel_checkout_completed.json",
    ]
    user_ids = {e["user_id"] for name in names for e in transform(load_fixture(name))}

    assert user_ids == {CLIENT_ID}


def test_output_contains_only_contract_fields():
    # The checkout payload carries email, address and prices; none of it may leak.
    for name in [
        "pixel_product_viewed.json",
        "pixel_product_added_to_cart.json",
        "pixel_checkout_completed.json",
    ]:
        for event in transform(load_fixture(name)):
            assert set(event) <= CONTRACT_FIELDS, name


def test_order_id_only_on_purchases():
    view = transform(load_fixture("pixel_product_viewed.json"))
    cart = transform(load_fixture("pixel_product_added_to_cart.json"))

    assert "order_id" not in view[0]
    assert "order_id" not in cart[0]


def test_unknown_event_type_is_ignored():
    event = load_fixture("pixel_product_viewed.json")
    event["name"] = "page_viewed"

    assert transform(event) == []
