"""ALS and BPR: describe every user and every product by a short list of numbers.

Both are matrix factorisation. The grid of who touched what is mostly empty; these models
learn, for each user and each product, a list of `factors` numbers (hidden "tastes") such
that a user's numbers multiplied with a product's numbers predict how likely the user is
to touch it. Nobody says what each number means; the model works that out from the data.

They differ only in what they learn from:

  - ALS (alternating least squares) tries to predict every cell of the grid, filled or
    empty, trusting the filled ones more. The grid's weights (view 1, cart 3, purchase 5)
    say how much more.
  - BPR (Bayesian personalised ranking) only learns to rank a product the user touched
    above one they didn't. It ignores the weights.

The learning is a library call (`implicit`). What is saved is two plain tables of numbers,
so serving needs no library. Answering is shared:

  - recommend_for_user: the user's numbers times every product's numbers;
  - similar_items: products whose numbers point the same way (cosine similarity), as the
    library does it.

See docs/decisions/0008.
"""

from __future__ import annotations

import numpy as np

from core.data.interactions import Interactions
from core.models.base import ColdStartError
from core.models.trained import TrainedRecommender


class _FactorModel(TrainedRecommender):
    """A subclass says how to learn the two tables, in `_learn`."""

    def __init__(
        self,
        store_id: str,
        factors: int,
        regularisation: float,
        iterations: int,
        seed: int,
    ):
        super().__init__(store_id)
        self.factors = factors
        self.regularisation = regularisation
        self.iterations = iterations
        self.seed = seed

    # ---- Frozen ------------------------------------------------------------------

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        row = self._user_index.get(user_id)
        if row is None:
            raise ColdStartError(f"unknown user: {user_id}")
        scores = self._item_factors @ self._user_factors[row]
        exclude = self._seen_by(user_id) if exclude_seen else []
        return self._top(scores, exclude, n)

    def similar_items(self, item_id: str, n: int = 10) -> list[tuple[str, float]]:
        row = self._item_index.get(item_id)
        if row is None:
            raise ColdStartError(f"unknown item: {item_id}")
        # Cosine: how closely the two lists of numbers point the same way, whatever their
        # size. A best seller's numbers are large, which would put it next to everything.
        scores = self._item_factors @ self._item_factors[row]
        scores = scores / (self._item_norms * self._item_norms[row])
        return self._top(scores, [row], n)

    # ---- Internal ----------------------------------------------------------------

    def _hyperparameters(self) -> dict:
        return {
            "factors": self.factors,
            "regularisation": self.regularisation,
            "iterations": self.iterations,
            "seed": self.seed,
        }

    def _fit(self, data: Interactions) -> None:
        # Imported here, not at the top: only training needs these. The library first, as
        # importing it loads another copy of the maths library the limit below must cover.
        import implicit  # noqa: F401
        from threadpoolctl import threadpool_limits

        # The library runs its own threads; the maths library running more threads inside
        # each of them slows training down (the library warns about this).
        with threadpool_limits(1, "blas"):
            model = self._learn(data)
        # Plain copies: the saved model must not hold the library's objects.
        self._user_factors = np.array(model.user_factors, dtype=np.float32)
        self._item_factors = np.array(model.item_factors, dtype=np.float32)
        norms = np.linalg.norm(self._item_factors, axis=1)
        norms[norms == 0] = 1e-10  # as the library does: no division by zero
        self._item_norms = norms

    def _learn(self, data: Interactions):
        """Fit the library's model on `data` and return it."""
        raise NotImplementedError


class ALSRecommender(_FactorModel):
    """Alternating least squares, for implicit feedback (Hu, Koren and Volinsky, 2008).

    Every empty cell counts as "probably not interested", with low confidence; a filled
    cell counts as "interested", with confidence `alpha` times the grid's weight. So a
    purchase is trusted 5 times as much as a view.
    """

    name = "als"

    # Defaults chosen on the Retailrocket validation period (docs/decisions/0008). The
    # library's own (100 factors, regularisation 0.01, alpha 1) scored a quarter as well:
    # with alpha 1 a purchase barely counts more than an empty cell.
    def __init__(
        self,
        store_id: str,
        factors: int = 200,
        regularisation: float = 10.0,
        alpha: float = 200.0,
        iterations: int = 15,
        seed: int = 0,
    ):
        super().__init__(store_id, factors, regularisation, iterations, seed)
        self.alpha = alpha

    def _hyperparameters(self) -> dict:
        return {**super()._hyperparameters(), "alpha": self.alpha}

    def _learn(self, data: Interactions):
        from implicit.cpu.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularisation,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.seed,
        )
        model.fit(data.matrix, show_progress=False)
        return model


class BPRRecommender(_FactorModel):
    """Bayesian personalised ranking (Rendle et al., 2009).

    Learns by repeatedly picking a user, a product they touched and one they didn't, and
    nudging the numbers so the first scores higher. Treats every filled cell the same:
    the library ignores the grid's weights.

    The library adds one more number per product, a popularity bias (and a fixed 1 for
    every user), so its tables are `factors + 1` wide.

    Trains on one thread: with several, the nudges happen in a different order each run
    and the results change between runs.
    """

    name = "bpr"

    # Defaults chosen on the Retailrocket validation period (docs/decisions/0008). Only the
    # learning rate differs from the library's (0.01), which learned too slowly in 100 rounds.
    def __init__(
        self,
        store_id: str,
        factors: int = 100,
        regularisation: float = 0.01,
        learning_rate: float = 0.1,
        iterations: int = 100,
        seed: int = 0,
    ):
        super().__init__(store_id, factors, regularisation, iterations, seed)
        self.learning_rate = learning_rate

    def _hyperparameters(self) -> dict:
        return {**super()._hyperparameters(), "learning_rate": self.learning_rate}

    def _learn(self, data: Interactions):
        from implicit.cpu.bpr import BayesianPersonalizedRanking

        model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularisation,
            iterations=self.iterations,
            random_state=self.seed,
            num_threads=1,
        )
        model.fit(data.matrix, show_progress=False)
        return model
