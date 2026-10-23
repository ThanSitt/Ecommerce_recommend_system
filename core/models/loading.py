"""Load any saved model without knowing which algorithm it is.

The API reads whatever file MODEL_PATH points to; the file itself records which class it
holds, so the service never needs to know whether it is serving most-popular or ALS.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from core.models.base import Recommender


def load_model(path: Path | str) -> Recommender:
    """Read a model written by `Recommender.save`.

    Only load files this project produced: like any pickle-based format, loading runs
    code stored in the file.
    """
    model = joblib.load(path)
    if not isinstance(model, Recommender):
        raise TypeError(f"{path} does not hold a recommender model")
    return model
