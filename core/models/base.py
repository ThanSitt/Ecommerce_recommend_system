"""The load-and-predict interface

Every recommender in this package implements `Recommender`, so the evaluation harness
and the API can both treat all of them identically without knowing which algorithm
they hold.

Two halves, with different stability guarantees:

  Frozen — Person B's API calls these, so changing them breaks their code:
      load(), recommend_for_user(), similar_items(), metadata

  Internal — only Person A calls these, so they can change freely:
      fit(), save()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ModelMetadata:
    """What a saved model knows about itself.

    Needed because an artefact on disk is otherwise anonymous: the service has to be
    able to report which model it is serving, and the deployment gate has to compare a
    candidate against the model it would replace.
    """

    store_id: str
    algorithm: str
    trained_at: datetime
    # The event window the model was trained on — both bounds, so it is possible to
    # tell later whether a model predates some change in the data.
    events_from: datetime
    events_to: datetime
    n_users: int
    n_items: int
    hyperparameters: dict = field(default_factory=dict)


class ColdStartError(Exception):
    """Raised when the model has no basis to answer.

    An unknown user, or a user whose history was filtered out of training. The API
    catches this and falls back to most-popular rather than returning an empty list —
    a shopper should always see something.
    """


class Recommender(ABC):
    """Base class every recommendation algorithm implements.

    Implementations hold wildly different things internally — a sorted list, a
    similarity table, two matrices of factors — but all of them answer the same
    questions, which is what lets one evaluation loop measure all of them.
    """

    #: Short identifier used in results tables and in saved metadata, e.g. "als".
    name: str = "base"

    # ---- Frozen: Person B's API depends on these -------------------------------

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Recommender:
        """Read a model previously written by `save`, ready to answer questions.

        Must restore the item/user ID mappings along with everything else — without
        them the model's internal indices cannot be turned back into real IDs.
        """

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """What this model is and when it was trained."""

    @abstractmethod
    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        """Up to `n` (item_id, score) pairs for this user, best score first.

        `exclude_seen` drops items the user already interacted with, and defaults to
        true because that is what the evaluation assumes — recommending something
        already bought is not a hit.

        Scores are comparable within one call for ordering, and mean nothing across
        different algorithms. Never compare an ALS score to an item-to-item score.

        Raises ColdStartError when the user is unknown to the model.
        """

    @abstractmethod
    def similar_items(self, item_id: str, n: int = 10) -> list[tuple[str, float]]:
        """Up to `n` (item_id, score) pairs similar to `item_id`, best first.

        Separate from `recommend_for_user` because most real traffic has no usable
        history: the common question is "a shopper is looking at this product, what
        else?", which needs no user at all.

        Raises ColdStartError when the item is unknown to the model.
        """

    # ---- Internal: only the training pipeline calls these ----------------------

    @abstractmethod
    def fit(self, data) -> None:
        """Learn from past behaviour.

        `data` is the prepared interaction set produced by `core/data` — the sparse
        user x item matrix plus the ID mappings. Its concrete type is defined in
        Phase 1; models receive it already built so that the preparation happens once
        rather than six times.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write the trained model to `path`.

        Must include the ID mappings and the metadata, not just the learned values.
        """
