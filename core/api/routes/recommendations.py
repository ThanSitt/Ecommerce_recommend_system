"""GET /v1/recommendations — products to show next to a product.

The connector sends the product being viewed and gets back product IDs, best first.
Only IDs: names, prices and images belong to the store, so the connector looks them up.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from core.api.auth import get_store_id
from core.api.model import get_model
from core.models.base import Recommender

MAX_N = 50

router = APIRouter(prefix="/v1", tags=["recommendations"])


class RecommendationsResponse(BaseModel):
    items: list[str]


@router.get("/recommendations", response_model=RecommendationsResponse)
def recommendations(
    item_id: str,
    n: int = Query(10, ge=1, le=MAX_N),
    store_id: str = Depends(get_store_id),
    model: Recommender = Depends(get_model),
) -> RecommendationsResponse:
    results = model.similar_items(item_id, n=n)
    return RecommendationsResponse(items=[item for item, _score in results])
