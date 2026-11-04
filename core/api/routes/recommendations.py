"""GET /v1/recommendations — products to show next to a product.

The connector sends the product being viewed and gets back product IDs, best first.
Only IDs: names, prices and images belong to the store, so the connector looks them up.

A product the model knows nothing about gets the most-popular list instead, so a shopper
sees something rather than an error.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from core.api.auth import get_store_id
from core.api.model import get_fallback, get_model
from core.models.base import ColdStartError, Recommender

MAX_N = 50

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["recommendations"])


class RecommendationsResponse(BaseModel):
    items: list[str]


@router.get("/recommendations", response_model=RecommendationsResponse)
def recommendations(
    item_id: str,
    n: int = Query(10, ge=1, le=MAX_N),
    store_id: str = Depends(get_store_id),
    model: Recommender = Depends(get_model),
    fallback: Recommender | None = Depends(get_fallback),
) -> RecommendationsResponse:
    try:
        results = model.similar_items(item_id, n=n)
    except ColdStartError:
        if fallback is None:
            # Nothing to show, but no reason to fail the page. Logged because it means
            # the fallback file is missing, which training should have written.
            logger.warning("no fallback file; returning no recommendations")
            results = []
        else:
            results = fallback.similar_items(item_id, n=n)
    return RecommendationsResponse(items=[item for item, _score in results])
