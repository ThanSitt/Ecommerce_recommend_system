"""The baselines every real model must beat: random, most-popular and recently popular.

Random is the floor: a model that cannot beat it has learned nothing. Most-popular is the
baseline that matters. Recently popular is the same idea over a short window, which makes
it a harder baseline to beat, and it is also the fallback the API serves when a real model
has no basis to answer.

None needs a user's history, so none raises ColdStartError: they answer for any user and
any item, known or not.
"""

from __future__ import annotations

import zlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.data.interactions import Interactions
from core.models.base import ModelMetadata, Recommender


class _Baseline(Recommender):
    """What both baselines share: remembering what each user touched, metadata, saving.

    A subclass only says how to score every item, via `_scores`.
    """

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

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        exclude = self._seen_by(user_id) if exclude_seen else []
        return self._top(self._scores(f"user:{user_id}"), exclude, n)

    def similar_items(self, item_id: str, n: int = 10) -> list[tuple[str, float]]:
        # Baselines know nothing about similarity: the answer is their usual list,
        # minus the item being looked at.
        exclude = [self._item_index[item_id]] if item_id in self._item_index else []
        return self._top(self._scores(f"item:{item_id}"), exclude, n)

    # ---- Internal ----------------------------------------------------------------

    def fit(self, data: Interactions) -> None:
        self._item_ids = data.item_ids
        self._item_index = data.item_index
        self._user_index = data.user_index
        # Kept only to know what each user already touched (row = user).
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

    def _scores(self, key: str) -> np.ndarray:
        """One score per item, in grid column order. `key` names who is asking."""
        raise NotImplementedError

    def _seen_by(self, user_id: str) -> np.ndarray:
        row = self._user_index.get(user_id)
        if row is None:
            return np.array([], dtype=int)
        return self._seen.indices[self._seen.indptr[row] : self._seen.indptr[row + 1]]

    def _top(self, scores: np.ndarray, exclude, n: int) -> list[tuple[str, float]]:
        scores = scores.astype(np.float64)  # a copy, so excluding never changes the model
        scores[exclude] = -np.inf
        # Stable sort: ties keep grid order (sorted item ID), so answers never flicker.
        best = np.argsort(-scores, kind="stable")[:n]
        return [(self._item_ids[i], float(scores[i])) for i in best if np.isfinite(scores[i])]


class MostPopularRecommender(_Baseline):
    """The same best-seller list for everyone: items ranked by how many different users
    touched them in training. Score = that number of users."""

    name = "popular"

    def _fit(self, data: Interactions) -> None:
        # Filled cells per column = different users per item, the same count the filter
        # uses. Counting users, not events, stops one obsessive viewer making an item
        # "popular".
        self._popularity = data.matrix.getnnz(axis=0).astype(np.float64)

    def _scores(self, key: str) -> np.ndarray:
        return self._popularity


class RecentlyPopularRecommender(MostPopularRecommender):
    """Most-popular, counting only the last `days` days of the training period.

    What people buy changes over time, so the last two weeks may predict the next two
    better than the last four months do. The window is fixed, not tuned: a baseline tuned
    until it scores well is just another model. 14 days matches the length of the
    evaluation periods, and was chosen before seeing any results.

    Score = different users in the window, plus a fraction below 1 that breaks ties by
    all-time popularity. A quiet store may have few events in the window, and this keeps
    its list sensible rather than in ID order.
    """

    name = "recent"

    def __init__(self, store_id: str, days: int = 14):
        super().__init__(store_id)
        self.days = days

    def _hyperparameters(self) -> dict:
        return {"days": self.days}

    def _fit(self, data: Interactions) -> None:
        super()._fit(data)
        all_time = self._popularity
        cutoff = data.events_to - pd.Timedelta(days=self.days)
        recent = data.events[data.events["timestamp"] >= cutoff]
        users_per_item = recent.groupby("item_id")["user_id"].nunique()
        in_window = users_per_item.reindex(data.item_ids, fill_value=0).to_numpy(
            dtype=np.float64
        )
        self._popularity = in_window + all_time / (all_time.max() + 1)


class RandomRecommender(_Baseline):
    """Random items. The floor any model must clear.

    Random but repeatable: the same user (or item) always gets the same answer for the
    same seed, so API responses and evaluation reruns don't change between calls.
    """

    name = "random"

    def __init__(self, store_id: str, seed: int = 0):
        super().__init__(store_id)
        self.seed = seed

    def _hyperparameters(self) -> dict:
        return {"seed": self.seed}

    def _scores(self, key: str) -> np.ndarray:
        # crc32, not hash(): Python's hash() of a string changes every run.
        rng = np.random.default_rng([self.seed, zlib.crc32(key.encode())])
        return rng.random(len(self._item_ids))
