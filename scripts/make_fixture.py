"""Build the committed test fixture from data/raw/events.csv.

Tests and CI run against a small sample so they need neither the 90 MB dataset nor
Kaggle credentials. This script is committed so the fixture is reproducible: delete
the fixture, rerun, get the same file.

Sampling is by *visitor*, not by row. Taking the first N rows of events.csv would cut
every user's history at an arbitrary timestamp, which makes any test of time-based
splitting or "exclude already-seen items" meaningless. Here, whole histories are kept
for a sample of visitors, deliberately over-representing purchasers because
transactions are only ~0.8% of events and a fixture without them tests nothing about
purchases.

    python scripts/make_fixture.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SOURCE = Path("data/raw/events.csv")
CATEGORY_SOURCE = Path("data/raw/category_tree.csv")
OUT_DIR = Path("tests/fixtures")
OUT = OUT_DIR / "events_sample.csv"
CATEGORY_OUT = OUT_DIR / "category_tree_sample.csv"

N_PURCHASERS = 150
N_BROWSERS = 600
SEED_MODULUS = 7  # deterministic, cheap stand-in for random sampling


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purchasers", type=int, default=N_PURCHASERS)
    parser.add_argument("--browsers", type=int, default=N_BROWSERS)
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"{SOURCE} not found. Run scripts/download_data.py first.")
        return 1

    # Pass 1: find visitors who bought something, and a spread of visitors who did not.
    purchasers: set[str] = set()
    browsers: set[str] = set()

    with SOURCE.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            visitor = row["visitorid"]
            if row["event"] == "transaction":
                if len(purchasers) < args.purchasers:
                    purchasers.add(visitor)
            elif (
                len(browsers) < args.browsers
                and visitor not in purchasers
                and int(visitor) % SEED_MODULUS == 0
            ):
                browsers.add(visitor)
            if len(purchasers) >= args.purchasers and len(browsers) >= args.browsers:
                break

    keep = purchasers | browsers
    if not keep:
        print("No visitors selected — is events.csv empty?")
        return 1

    # Pass 2: write every event belonging to the selected visitors, in file order
    # (which is chronological, so the fixture stays valid for time-split tests).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    rows = 0

    with (
        SOURCE.open(newline="", encoding="utf-8") as src,
        OUT.open("w", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        for row in reader:
            if row["visitorid"] in keep:
                writer.writerow(row)
                counts[row["event"]] = counts.get(row["event"], 0) + 1
                rows += 1

    if CATEGORY_SOURCE.exists():
        CATEGORY_OUT.write_bytes(CATEGORY_SOURCE.read_bytes())

    print(f"Wrote {OUT} — {rows} rows, {len(keep)} visitors, {OUT.stat().st_size / 1e3:.0f} kB")
    for event, count in sorted(counts.items()):
        print(f"  {event:<12} {count:>6}")
    if CATEGORY_OUT.exists():
        print(f"Wrote {CATEGORY_OUT} — {CATEGORY_OUT.stat().st_size / 1e3:.0f} kB (full tree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
