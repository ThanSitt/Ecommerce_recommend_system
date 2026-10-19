"""Split events into train / validation / test by time.

Models learn from the past and are tested on the future, exactly as in a real store. A
random split would let a model learn from events that happen after the ones it is tested
on, which inflates every metric — the most common error in this kind of work.

The split is generic: two cut-off dates, any events. The Retailrocket dates are below;
the production pipeline can use the same function with dates relative to "now".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Chosen in notebooks/01_explore_retailrocket.ipynb. The data runs 2015-05-03 -> 2015-09-18
# with no gaps, so the last two weeks are the test and the two before them validation.
RETAILROCKET_VALIDATION_START = pd.Timestamp("2015-08-21", tz="UTC")
RETAILROCKET_TEST_START = pd.Timestamp("2015-09-04", tz="UTC")


@dataclass
class Split:
    """The three periods. Named fields, not a tuple, so they cannot be mixed up."""

    train: pd.DataFrame
    validation: pd.DataFrame
    # Sealed: used for tuning nothing. Opened once, in Phase 6, for the final numbers.
    test: pd.DataFrame


def split_by_time(
    events: pd.DataFrame, validation_start: pd.Timestamp, test_start: pd.Timestamp
) -> Split:
    """Cut events at two dates.

        train:       timestamp <  validation_start
        validation:  validation_start <= timestamp < test_start
        test:        timestamp >= test_start

    Every event lands in exactly one period: an event exactly on a cut-off date belongs
    to the later period.
    """
    for name, cutoff in [("validation_start", validation_start), ("test_start", test_start)]:
        # Same reason the contract requires a timezone: a date without one is ambiguous,
        # and a few hours' shift moves events across the cut.
        if cutoff.tzinfo is None:
            raise ValueError(f"{name} must have a timezone, e.g. pd.Timestamp(..., tz='UTC')")
    if not validation_start < test_start:
        raise ValueError("validation_start must be before test_start")

    time = events["timestamp"]
    return Split(
        train=events[time < validation_start].reset_index(drop=True),
        validation=events[(time >= validation_start) & (time < test_start)].reset_index(
            drop=True
        ),
        test=events[time >= test_start].reset_index(drop=True),
    )
