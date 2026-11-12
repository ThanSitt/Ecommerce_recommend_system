"""Item-to-item: products touched by the same users count as similar.

The first model here that gives really related products, and the one built for the product
page: `similar_items` needs only the product being viewed, so it works for visitors the model
has never seen. It reads no names or pictures, only who touched what.

The similarity itself is a library call. Everything after training is one table of numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import diags

from core.data.interactions import Interactions
from core.models.base import ColdStartError
from core.models.trained import TrainedRecommender

# BM25's two settings: the library's defaults, not tuned. Passed explicitly so that a new
# version of the library cannot change the results unnoticed.
BM25_K1 = 1.2
BM25_B = 0.75


class ItemToItemRecommender(TrainedRecommender):
    """Keeps each product's `neighbours` most similar products, by BM25 similarity.

    Two products are similar when the same users touched both. BM25 decides how much each
    shared user counts:

      - a user who touched few products counts more than one who touched thousands, because
        what the second one touched says little about which products go together;
      - a product touched by many users is scaled down, which stops best sellers appearing
        next to everything.

    On Retailrocket this beat plain cosine similarity at every setting tried, and removed the
    need to delete very active accounts. See docs/decisions/0006.

    Unlike the baselines, this model can have no answer, and then raises ColdStartError so
    that the API serves the fallback.
    """

    name = "item-to-item"

    def __init__(self, store_id: str, neighbours: int = 50):
        super().__init__(store_id)
        self.neighbours = neighbours

    # ---- Frozen ------------------------------------------------------------------

    def recommend_for_user(
        self, user_id: str, n: int = 10, exclude_seen: bool = True
    ) -> list[tuple[str, float]]:
        row = self._user_index.get(user_id)
        if row is None:
            raise ColdStartError(f"unknown user: {user_id}")
        # Each product the user touched votes for its neighbours, by how strong the user's
        # interest was (the grid's weights) and how similar the neighbour is.
        scores = (self._seen[row] @ self._similarity).toarray().ravel()
        exclude = self._seen_by(user_id) if exclude_seen else np.array([], dtype=int)
        unrelated = np.flatnonzero(scores <= 0)
        results = self._top(scores, np.concatenate([unrelated, exclude]), n)
        if not results:
            raise ColdStartError(f"nothing related to user {user_id}")
        return results

    def similar_items(self, item_id: str, n: int = 10) -> list[tuple[str, float]]:
        row = self._item_index.get(item_id)
        if row is None:
            raise ColdStartError(f"unknown item: {item_id}")
        neighbours = self._similarity[row]
        if neighbours.nnz == 0:
            raise ColdStartError(f"nothing related to item {item_id}")
        # Stable sort over this product's neighbours only: ties keep sorted item ID order.
        best = np.argsort(-neighbours.data, kind="stable")[:n]
        return [
            (self._item_ids[neighbours.indices[i]], float(neighbours.data[i])) for i in best
        ]

    # ---- Internal ----------------------------------------------------------------

    def _hyperparameters(self) -> dict:
        return {"neighbours": self.neighbours, "k1": BM25_K1, "b": BM25_B}

    def _fit(self, data: Interactions) -> None:
        # Imported here, not at the top: only training needs the library. Serving a saved
        # model needs just the table below.
        from implicit.nearest_neighbours import BM25Recommender
        from implicit.utils import ParameterWarning

        # One extra, because the library usually counts a product among its own neighbours.
        model = BM25Recommender(K=self.neighbours + 1, K1=BM25_K1, B=BM25_B)
        with warnings.catch_warnings():
            # The library passes itself a matrix in the wrong sparse format. Harmless.
            warnings.simplefilter("ignore", ParameterWarning)
            model.fit(data.matrix, show_progress=False)

        # Row i holds the neighbours of item i. A product is not its own recommendation.
        similarity = model.similarity
        similarity = (similarity - diags(similarity.diagonal(), format="csr")).tocsr()
        similarity.eliminate_zeros()
        similarity.sort_indices()
        self._similarity = similarity
