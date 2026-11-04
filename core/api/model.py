"""The model the API answers from.

The file at MODEL_PATH is read once, on the first request that needs it, and kept in
memory; every later request reuses it. A new model file is picked up on restart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import Depends, HTTPException, status

from core.config import Settings, get_settings
from core.models.base import Recommender
from core.models.loading import fallback_path, load_model


@lru_cache
def _load(path: str) -> Recommender:
    return load_model(path)


def get_model(settings: Settings = Depends(get_settings)) -> Recommender:
    """FastAPI dependency: the loaded model, or 503 if no model file exists yet."""
    path = Path(settings.model_path)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no model file yet; run core.training.train",
        )
    return _load(str(path))


def get_fallback(settings: Settings = Depends(get_settings)) -> Recommender | None:
    """FastAPI dependency: the fallback model saved beside the main one, for when the main
    one has no basis to answer. None if there is no such file."""
    path = fallback_path(settings.model_path)
    if not path.is_file():
        return None
    return _load(str(path))
