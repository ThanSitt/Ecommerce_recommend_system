"""Load any saved model without knowing which algorithm it is, and where each store's is.

The API reads whatever file it is pointed at; the file itself records which class it
holds, so the service never needs to know whether it is serving most-popular or ALS.

Each store has its own folder of models (docs/decisions/0009):

    <models dir>/<store_id>/current.joblib      what the API serves for this store
    <models dir>/<store_id>/fallback.joblib     its fallback
    <models dir>/<store_id>/runs/<run>/...      every retraining run, kept or not, and why
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib

from core.models.base import Recommender

FALLBACK_FILENAME = "fallback.joblib"
CURRENT_FILENAME = "current.joblib"
# Store IDs are made by the core (`st_` + hex), but a folder name is built from them, so
# anything that could climb out of the models folder ("..", "/") is refused.
_STORE_ID = re.compile(r"[A-Za-z0-9_-]+")


def store_dir(models_dir: Path | str, store_id: str) -> Path:
    """A store's model folder."""
    if not _STORE_ID.fullmatch(store_id):
        raise ValueError(f"not a valid store_id for a folder name: {store_id!r}")
    return Path(models_dir) / store_id


def current_model_path(models_dir: Path | str, store_id: str) -> Path:
    """The model the API serves for a store. Only the retraining pipeline replaces it, and
    only when the deployment gate accepts a new model. Its fallback is beside it."""
    return store_dir(models_dir, store_id) / CURRENT_FILENAME


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
