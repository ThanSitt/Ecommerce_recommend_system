"""Load the Retailrocket dataset into the standard event shape.

The output has the same column names as the core's `events` table (core/db/models.py), so
everything downstream — splitting, filtering, building the grid — works the same whether
the events came from this file or from a real store.

Only cleaning happens here. Filtering (dropping quiet users and items, extreme users)
is *not* done here: it must use the training period only, so it runs after the time split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The standard event columns, in this order. Matches EventRow in core/db/models.py.
# Not included: store_id (one dataset, no stores), event_id (Retailrocket has none, and
# it is optional in the contract), received_at (set by the server, not by the source).
EVENT_COLUMNS = ["user_id", "item_id", "event_type", "timestamp", "order_id"]

# Retailrocket's event names -> the contract's.
EVENT_NAMES = {
    "view": "view",
    "addtocart": "cart",
    "transaction": "purchase",
}


def load_retailrocket(path: str | Path) -> tuple[pd.DataFrame, int]:
    """Read events.csv and return (events, number_of_duplicates_dropped).

    Events are sorted by time, oldest first.
    """
    raw = pd.read_csv(
        path,
        # IDs are read as text, not numbers. Shopify sends IDs as strings, so both sources
        # must agree, or "355908" and 355908 become two different items. Reading as text
        # also stops pandas turning transactionid into a float ("4000.0"), which it does
        # because the column has empty cells.
        dtype={"visitorid": str, "itemid": str, "transactionid": str},
    )

    unknown = set(raw["event"].unique()) - set(EVENT_NAMES)
    if unknown:
        # Fail loudly rather than silently dropping rows: a new event type means the
        # mapping above needs a decision.
        raise ValueError(f"unknown event types in {path}: {sorted(unknown)}")

    events = pd.DataFrame(
        {
            "user_id": raw["visitorid"],
            "item_id": raw["itemid"],
            "event_type": raw["event"].map(EVENT_NAMES),
            # Milliseconds since 1970, in UTC — the contract requires a timezone.
            "timestamp": pd.to_datetime(raw["timestamp"], unit="ms", utc=True),
            # Only set on purchases, as the contract requires; empty everywhere else.
            "order_id": raw["transactionid"],
        },
        columns=EVENT_COLUMNS,
    )

    before = len(events)
    events = events.drop_duplicates()
    duplicates = before - len(events)

    # The file is not in time order. Everything after this relies on it being sorted:
    # the split cuts by time, and "what came before" must mean earlier rows.
    # A stable sort keeps same-millisecond events in their original order.
    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)

    return events, duplicates
