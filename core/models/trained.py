"""What every trained model shares: the ID maps, what each user touched, metadata, saving.

A subclass says what to learn, in `_fit`, and how to answer `recommend_for_user` and
`similar_items`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from core.data.interactions import Interactions
from core.models.base import ModelMetadata, Recommender


class TrainedRecommender(Recommender):
    def __init__(self, store_id: str):
        self.store_id = store_id
        self._metadata: ModelMetadata | None = None

    # ---- Frozen ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> Recommender:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"{path} holds a {type(model).__name__}, not a {cls.__name__}")
        return model

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise RuntimeError("model is not trained yet; call fit first")
        return self._metadata

    # ---- Internal ----------------------------------------------------------------

    def fit(self, data: Interactions) -> None:
        self._item_ids = data.item_ids
        self._item_index = data.item_index
        self._user_index = data.user_index
        # Kept to know what each user already touched (row = user).
        self._seen = data.matrix
        self._fit(data)
        self._metadata = ModelMetadata(
            store_id=self.store_id,
            algorithm=self.name,
            trained_at=datetime.now(timezone.utc),
            events_from=data.events_from.to_pydatetime(),
            events_to=data.events_to.to_pydatetime(),
            n_users=data.n_users,
            n_items=data.n_items,
            hyperparameters=self._hyperparameters(),
        )

    def save(self, path: Path) -> None:
        if self._metadata is None:
            raise RuntimeError("model is not trained yet; call fit first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def _fit(self, data: Interactions) -> None:
        """Anything the subclass learns. Nothing by default."""

    def _hyperparameters(self) -> dict:
        return {}

    def _seen_by(self, user_id: str) -> np.ndarray:
        row = self._user_index.get(user_id)
        if row is None:
            return np.array([], dtype=int)
        return self._seen.indices[self._seen.indptr[row] : self._seen.indptr[row + 1]]

    def _top(self, scores: np.ndarray, exclude, n: int) -> list[tuple[str, float]]:
        scores = scores.astype(np.float64)  # a copy, so excluding never changes the model
        scores[exclude] = -np.inf
        candidates = np.arange(len(scores))
        if 0 < n < len(scores):
            # Only the best n need sorting. Everything scoring at least the n-th best score
            # stays a candidate, ties included, so the answer is the same as sorting all.
            nth_best = np.partition(scores, len(scores) - n)[len(scores) - n]
            candidates = np.flatnonzero(scores >= nth_best)
        # Stable sort: ties keep grid order (sorted item ID), so answers never flicker.
        best = candidates[np.argsort(-scores[candidates], kind="stable")[:n]]
        return [(self._item_ids[i], float(scores[i])) for i in best if np.isfinite(scores[i])]
