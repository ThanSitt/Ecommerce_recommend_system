"""Shopify connector server.

The Web Pixel in the shopper's browser POSTs each event here.

Run locally from the repo root:
    uvicorn connectors.shopify.app.main:app --port 8001 --reload
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from connectors.shopify.app.transform import transform

logger = logging.getLogger("shopify_connector")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Shopify connector")

# The pixel runs in the shopper's browser on the store's domain, so its requests are
# cross-origin. Without this the browser blocks them before they reach us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/pixel")
async def receive_pixel_event(request: Request) -> dict[str, str]:
    # The pixel sends JSON as text/plain: a "simple" request the browser sends without
    # a CORS preflight. FastAPI only auto-parses application/json, so parse it here.
    try:
        pixel_event: dict[str, Any] = json.loads(await request.body())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="body is not valid JSON") from exc

    events = transform(pixel_event)

    # TODO: forward `events` to the core's POST /v1/events once it exists.
    logger.info(
        "pixel event received: %s -> %d contract event(s)\nproducts: %s\n%s",
        pixel_event.get("name"),
        len(events),
        _debug_product_titles(pixel_event),
        json.dumps(events, indent=2),
    )
    return {"status": "received"}


def _debug_product_titles(pixel_event: dict[str, Any]) -> list[str]:
    """Product names, for reading the log only. Deliberately not part of the contract
    events: the core never needs them."""
    data = pixel_event.get("data", {})
    name = pixel_event.get("name")
    if name == "product_viewed":
        return [data["productVariant"]["product"]["title"]]
    if name == "product_added_to_cart":
        return [data["cartLine"]["merchandise"]["product"]["title"]]
    if name == "checkout_completed":
        return [line["title"] for line in data["checkout"]["lineItems"]]
    return []
