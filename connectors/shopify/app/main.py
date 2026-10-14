"""Shopify connector server.

The Web Pixel in the shopper's browser POSTs each event here.

Run locally from the repo root:
    uvicorn connectors.shopify.app.main:app --port 8001 --reload
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from connectors.shopify.app.config import get_settings
from connectors.shopify.app.core_client import send_events
from connectors.shopify.app.transform import transform

logger = logging.getLogger("shopify_connector")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # One HTTP client for the app's lifetime, so connections to the core are reused
    # instead of opened and closed on every pixel event.
    async with httpx.AsyncClient() as client:
        app.state.core_client = client
        yield


app = FastAPI(title="Shopify connector", lifespan=lifespan)

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

    logger.info(
        "pixel event received: %s -> %d contract event(s)\nproducts: %s\n%s",
        pixel_event.get("name"),
        len(events),
        _debug_product_titles(pixel_event),
        json.dumps(events, indent=2),
    )
    await send_events(request.app.state.core_client, get_settings(), events)
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
