# 0003 — Preparing Retailrocket for training and evaluation

**Status:** Accepted, 2026-10-21. The filter thresholds and weights are starting values,
to be revisited against validation results in Phase 4.
**Decided by:** Person A
**Affects:** `core/data`, and every accuracy number reported in the thesis

## Context

Every reported accuracy number comes from Retailrocket, so how the raw file becomes
training and evaluation data decides what those numbers mean. The dataset runs from
2015-05-03 to 2015-09-18 with steady activity and no gaps: 2,755,641 events after
removing 460 exact duplicates, of which only about 0.8% are purchases. It is extremely
sparse — the median visitor has one event — so the raw data cannot be handed to a model
as it is.

Four choices follow, each of which materially changes the results: where to cut time,
which users and items to keep, what to do with extreme accounts, and how to turn event
types into numbers. They are made once, in `core/data/prepare.py`, so every phase uses
the same data in the same order.

## Decision

| Setting | Value | Code |
| --- | --- | --- |
| Training period | before 2015-08-21 (110 days) | `core/data/split.py` |
| Validation period | 2015-08-21 → 2015-09-04 (14 days) | `core/data/split.py` |
| Test period | from 2015-09-04 (14 days), sealed until the final run | `core/data/split.py` |
| Heavy-account cap | drop users with more than 1,000 training events | `core/data/filtering.py` |
| Minimum per user | 2 different items | `core/data/filtering.py` |
| Minimum per item | 5 different users | `core/data/filtering.py` |
| Event weights | view 1, cart 3, purchase 5 | `core/data/interactions.py` |
| Repeated events on one item | keep the strongest, not the sum | `core/data/interactions.py` |

The order is fixed: **split, then filter the training period only, then build the grid.**
Validation and test events are never filtered.

## Why each part

### Split by time, into three periods

Models learn from the past and are scored on the future, as they are in a real store. A
random split lets a model learn from events that happen after the ones it is scored on,
which inflates every metric without any visible error.

Validation is used for all tuning; test is opened once, for the final numbers, so that it
remains an honest estimate. Two weeks per evaluation period gives each about 2,000
purchases (validation 2,063, test 1,932), against roughly 1,100 per week overall, while
leaving 110 days and 18,462 purchases for training.

**Considered instead:** a random split (rejected for the reason above); a single
train/test split with tuning on test (rejected: tuning on the test period turns it into a
second validation set and its numbers stop meaning anything).

### Filter after splitting, on training data only

Filtering decides who and what to keep based on activity. Done on the whole file, it
would keep a product because it became popular in the test period — information the
model could not have had at training time. Nothing crashes; the numbers simply come out
too good. A test (`tests/core/test_prepare.py`) checks that an item popular only after
the cut is still removed.

Validation and test are not filtered, because removing hard cases from the evaluation
would hide them. How unseen users and items are scored is an evaluation decision, made in
Phase 2.

### Count different items and users, repeat until stable

The minimums count *distinct* items per user and *distinct* users per item, not events.
A visitor who viewed one product twenty times fills one cell of the grid and says
nothing about which products go together.

The two rules affect each other — removing a rare item can push a user below the minimum,
and removing that user can push another item below its own — so the filter repeats until
nothing changes (18 rounds on Retailrocket). Afterwards every remaining user and item
meets both rules.

### Minimums of 2 items per user and 5 users per item

Two is the smallest useful minimum for a user: a single item gives no information about
which items go together. For items, fewer than five users give no reliable similarity.

The thresholds trade data quality against the number of people the model can be scored
on. Survivors in the training period, with the heavy-account cap applied; "scorable" is
the validation users and buyers still present after filtering, i.e. the people a
personalised model can be evaluated on:

| users ≥ | items ≥ | users | items | events | purchases | val users scorable | val buyers scorable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1,151,456 | 214,337 | 2,214,021 | 15,781 | 14,229 | 186 |
| 2 | 2 | 207,214 | 73,609 | 1,038,103 | 11,704 | 5,600 | 106 |
| **2** | **5** | **164,828** | **32,222** | **843,621** | **10,289** | **4,450** | **97** |
| 3 | 3 | 72,480 | 36,282 | 623,172 | 8,958 | 2,705 | 76 |
| 3 | 5 | 60,951 | 21,972 | 522,298 | 7,566 | 2,265 | 75 |
| 5 | 5 | 16,875 | 12,899 | 263,086 | 4,604 | 957 | 53 |
| 5 | 10 | 7,340 | 3,028 | 109,856 | 1,821 | 472 | 35 |

Before filtering, the training period has 1,151,479 users, 215,409 items, 2,259,563
events and 18,462 purchases.

The chosen row removes most one-click visitors and rare products while keeping about
4,450 validation users to score. Stricter settings give cleaner data but leave too few
people for stable numbers: at 5 / 10, a handful of lucky recommendations would move the
result.

**Considered instead:** no filtering (the grid is dominated by one-event visitors who
cannot be learned from); stricter thresholds such as 5 / 5, common in published work on
denser datasets (rejected here because Retailrocket is sparse enough that they leave
under a thousand scorable users).

### Remove accounts with more than 1,000 training events

1,000 events in 110 days is about nine a day, every day. Twenty-three accounts exceed it;
the largest has 7,757 events and 559 purchases. Together they make about 2% of training
events but 14.5% of training purchases. They behave like resellers, staff or bots rather
than shoppers, and left in, a few accounts would dominate what the model learns about
purchases.

They are removed by raw event count, before the minimums are applied. The effect at
2 / 5:

| cap | users | items | events | purchases | val users scorable | val buyers scorable |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 164,828 | 32,222 | 843,621 | 10,289 | 4,450 | 97 |
| none | 165,952 | 33,317 | 891,275 | 13,002 | 4,497 | 104 |

Keeping them adds back only about 1,100 users but about 2,700 purchases, which confirms
how concentrated those purchases are.

**Considered instead:** keeping them (rejected for the reason above); down-weighting them
rather than removing them (more complex, and the cap removes 23 users, not a population
worth modelling).

### Weights view 1, cart 3, purchase 5

The grid holds one number per user and item, and the event types are not equally
informative: 2.7% of viewers add to cart and 0.8% buy, so a cart or purchase is a much
stronger signal of interest than a view. Weights that increase with commitment encode
that. The values are a starting point, exposed as a setting (`DEFAULT_WEIGHTS`) and tuned
on validation in Phase 4.

Weights must be positive: a weight of zero would be stored as an empty cell, which is
indistinguishable from "never touched".

**Considered instead:** using purchases only (rejected: about 10,000 purchase cells
remain after filtering, far too few to learn from); treating all events equally
(rejected: discards the strongest signal in the data).

### Keep the strongest event per cell, not the sum

When a user has several events on one item, the cell keeps the highest weight. A visitor
who viewed a product twenty times and then bought it scores 5, not 25. Summing would let
repeated clicking outweigh a purchase, and lets a few compulsive viewers dominate an
item.

**Considered instead:** summing, which treats repeated views as growing interest. It is
a reasonable alternative and may be tried in Phase 4 if tuning time allows.

## Consequences

- The training grid is 164,828 users × 32,222 items with 578,143 filled cells — **0.011%
  full**, about one cell in 9,000. By strongest event, 548,461 cells are views, 20,014
  carts and 9,668 purchases. Stored sparsely it takes 5 MB; a full table would need about
  21 GB.
- **Most evaluation users are unknown to the model.** Only 7.6% of test-period users
  (10,955 of 143,689) appear in training at all, and only 137 of 1,073 test buyers. Items
  are much better covered: 91.6% of items bought in the test period were seen in training.
  This confirms the finding in 0001 that item-based recommendation, not user history,
  carries most real traffic, and it forces a decision in Phase 2 on what counts as a hit
  and how unseen users are scored.
- **Small stores cannot use these settings.** On the 739-visitor test fixture, "items need
  5 users" removes everything and the grid cannot be built. A small store is in the same
  position, so the production pipeline (Phase 5) needs per-store settings or a
  minimum-data rule, with the most-popular fallback below it.
- Every setting is a named constant or parameter, and `prepare_retailrocket` applies all
  of them in one call, so the results table can always be traced back to the settings
  recorded here.
