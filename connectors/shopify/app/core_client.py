"""Sends contract events to the core's POST /v1/events.

The connector talks to the core over HTTP like any other client. It never imports from
core/, so the core stays unaware that Shopify exists.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from connectors.shopify.app.config import Settings

logger = logging.getLogger("shopify_connector")

# A slow or hung core must not hold the connector's requests open indefinitely.
TIMEOUT_SECONDS = 5.0


async def send_events(
    client: httpx.AsyncClient, settings: Settings, events: list[dict[str, Any]]
) -> bool:
    """POST events to the core. Returns True on success; failures are logged, not raised.

    Not raised because the caller is the pixel, which cannot do anything useful with an
    error. Events sent while the core is unreachable are lost for now; retrying them is
    a later step.
    """
    if not events:
        return True

    try:
        response = await client.post(
            f"{settings.core_api_url}/v1/events",
            json=events,
            headers={"Authorization": f"Bearer {settings.core_api_key}"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "core rejected %d event(s): %s %s",
            len(events),
            exc.response.status_code,
            exc.response.text,
        )
        return False
    except httpx.HTTPError as exc:
        logger.error("could not reach core, %d event(s) lost: %r", len(events), exc)
        return False

    logger.info("core accepted %d event(s): %s", len(events), response.text)
    return True
