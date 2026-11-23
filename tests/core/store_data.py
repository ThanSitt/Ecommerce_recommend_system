"""A made-up store for testing the retraining pipeline: shoppers browse in themes.

Each shopper visits once, picks a theme (snowboards, tea towels...) and views 2 to 4 of its
products a few minutes apart. Some themes and some products are more popular than others.
So the right answer is known: a model that learned the themes must beat best sellers, and
best sellers must beat random. Not data for any reported number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.retailrocket import EVENT_COLUMNS

START = pd.Timestamp("2026-09-01", tz="UTC")


def themed_store(
    shoppers: int = 600,
    days: int = 28,
    themes: int = 8,
    per_theme: int = 6,
    seed: int = 0,
    start: pd.Timestamp = START,
    prefix: str = "",
) -> pd.DataFrame:
    """Events in the standard shape, oldest first. Product IDs are `<prefix>t<theme>-p<n>`."""
    rng = np.random.default_rng(seed)
    theme_weights = _decreasing(themes)
    product_weights = _decreasing(per_theme)
    rows = []
    for shopper in range(shoppers):
        theme = rng.choice(themes, p=theme_weights)
        # Whole milliseconds, like real event times.
        visit = start + pd.Timedelta(milliseconds=int(rng.integers(0, days * 86_400_000)))
        products = rng.choice(
            per_theme, size=int(rng.integers(2, 5)), replace=False, p=product_weights
        )
        for step, product in enumerate(products):
            at = visit + pd.Timedelta(minutes=5 * step)
            item = f"{prefix}t{theme}-p{product}"
            rows.append((f"shopper-{seed}-{shopper}", item, "view", at, None))
            if rng.random() < 0.2:
                rows.append((f"shopper-{seed}-{shopper}", item, "cart", at, None))
    events = pd.DataFrame(rows, columns=["user_id", "item_id", "event_type", "timestamp",
                                         "order_id"])
    return events[EVENT_COLUMNS].sort_values("timestamp", kind="stable").reset_index(drop=True)


def _decreasing(n: int) -> np.ndarray:
    weights = 1 / np.arange(1, n + 1)
    return weights / weights.sum()
