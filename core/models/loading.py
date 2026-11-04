"""Load any saved model without knowing which algorithm it is.

The API reads whatever file MODEL_PATH points to; the file itself records which class it
holds, so the service never needs to know whether it is serving most-popular or ALS.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from core.models.base import Recommender

FALLBACK_FILENAME = "fallback.joblib"


def fallback_path(model_path: Path | str) -> Path:
    """Where a model's fallback lives: beside it, in the same folder.

    Training writes it here and the API reads it from here, so the two cannot disagree,
    and giving each store its own model folder later gives it its own fallback too.
    """
    return Path(model_path).with_name(FALLBACK_FILENAME)


def load_model(path: Path | str) -> Recommender:
    """Read a model written by `Recommender.save`.

    Only load files this project produced: like any pickle-based format, loading runs
    code stored in the file.
    """
    model = joblib.load(path)
    if not isinstance(model, Recommender):
        raise TypeError(f"{path} does not hold a recommender model")
    return model
