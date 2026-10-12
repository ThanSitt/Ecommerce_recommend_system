"""Shopify Web Pixel event -> standard event contract.

Field paths come from the captured payloads in tests/fixtures/shopify/ and the mapping
table in docs/investigations/shopify-event-sources.md.
"""

from __future__ import annotations

from typing import Any


def transform(pixel_event: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn one Shopify pixel event into a list of contract events.

    A list because one checkout becomes one event per product. Event types we don't
    handle return an empty list instead of raising.
    """
    name = pixel_event.get("name")

    if name == "product_viewed":
        return _product_viewed(pixel_event)
    if name == "product_added_to_cart":
        return _product_added_to_cart(pixel_event)
    if name == "checkout_completed":
        return _checkout_completed(pixel_event)

    return []


def _product_viewed(event: dict[str, Any]) -> list[dict[str, Any]]:
    product_id = event["data"]["productVariant"]["product"]["id"]

    return [
        {
            "event_id": event["id"],
            # clientId is Shopify's anonymous browser ID, never the customer ID.
            "user_id": event["clientId"],
            # The product, not the variant (productVariant.id).
            "item_id": product_id,
            "event_type": "view",
            "timestamp": event["timestamp"],
        }
    ]


def _product_added_to_cart(event: dict[str, Any]) -> list[dict[str, Any]]:
    # `merchandise` is Shopify's name for the variant; the product sits inside it.
    product_id = event["data"]["cartLine"]["merchandise"]["product"]["id"]

    return [
        {
            "event_id": event["id"],
            "user_id": event["clientId"],
            "item_id": product_id,
            "event_type": "cart",
            "timestamp": event["timestamp"],
        }
    ]


def _checkout_completed(event: dict[str, Any]) -> list[dict[str, Any]]:
    checkout = event["data"]["checkout"]
    order_id = checkout["order"]["id"]

    # One purchase event per product in the order, all sharing the same order_id.
    events = []
    for line in checkout["lineItems"]:
        variant_id = line["variant"]["id"]
        events.append(
            {
                # Built, not copied: Shopify's `id` is one per checkout, but we emit one
                # event per line. Variant (not product) keeps two variants of the same
                # product in one order from colliding.
                "event_id": f"purchase:{order_id}:{variant_id}",
                "user_id": event["clientId"],
                "item_id": line["variant"]["product"]["id"],
                "event_type": "purchase",
                "order_id": order_id,
                "timestamp": event["timestamp"],
            }
        )
    return events
