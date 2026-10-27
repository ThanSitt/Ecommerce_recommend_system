# 0004 — How models are scored

**Status:** Accepted, 2026-10-27
**Decided by:** Person A
**Affects:** `core/evaluation`, every results table in the thesis, and the deployment gate

## Context

Every model is compared on the same question: given a user's past, which products will
they touch next? The model sees the training period (before 2015-08-21, see 0003) and
recommends 10 products; its list is compared with what the user actually did in the
validation period (2015-08-21 to 2015-09-04). The test period is never scored during
development.

Retailrocket makes this harder than it sounds. Of the 141,559 users active in
validation, only 4,450 (3%) appear in the training grid at all, and only 58 of those
bought something new that a model could have recommended. How a "hit" is defined
therefore decides whether there is anything to measure.

The scoring was built and checked on the two baselines, random and most-popular, before
any real model, so that the first real model gets a number that can be trusted.

## Decision

| Choice | Value |
| --- | --- |
| What counts as a hit | **any interaction** (view, cart or purchase) with the product |
| Second, reported with a warning | cart or purchase only |
| A product can be a hit only if | it is in the training grid **and** the user had not touched it in training |
| Who is scored | users in the training grid with at least one possible hit |
| List length | 10 |
| Headline metrics | **Recall@10, NDCG@10** |
| Also reported | Precision@10, catalogue coverage |
| Model score | average over scored users, with a **95% bootstrap range** |
| Period scored | validation only; test is opened once, at the end |

Code: `core/evaluation/answer_key.py` (who is scored, what counts),
`core/evaluation/metrics.py` (the metrics and the range), `core/evaluation/run.py` (the
results table: `python -m core.evaluation.run`).

## Why each part

### Any interaction counts as a hit

How many validation users can be scored under each definition:

| A hit is... | users scored |
| --- | --- |
| any interaction | 2,632 |
| cart or purchase | 141 |
| purchase only | 58 |

Purchases are what a store cares about most, but with 58 users a single lucky
recommendation moves the score by almost 2 percentage points, and the result says more
about chance than about the model. A view still shows that a recommendation would have
interested the user. Any interaction is therefore the main number; cart-or-purchase is
reported beside it as the number closer to business value, with its user count shown.

**Considered instead:** purchase only (too few users); weighting hits by event type
(adds a second set of weights to justify, on top of the training weights in 0003, without
solving the small-numbers problem).

### Only products a model could have recommended count

Three rules decide what goes into a user's answers:

1. **The user must be in the training grid.** No model can personalise for someone it has
   never seen. These users receive the most-popular fallback in production.
2. **The product must be in the training grid.** A model cannot recommend a product it
   does not know.
3. **The user must not have touched it in training.** Every model excludes products the
   user already interacted with, so these could never be recommended.

Users left with no answers are not scored at all, since they would score 0 for every
model and say nothing about the models. How 141,559 validation users become 2,632:

| Stage | Users | Why the others drop out |
| --- | --- | --- |
| active in validation | 141,559 | |
| in the training grid | 4,450 | new visitors, or removed by the filter (0003) |
| at least one possible hit | 2,632 | 686 only revisited products they had already touched; 1,057 only touched products outside the grid; 75 both |

The answer key is built once per period and is independent of any model, so every model
is marked against identical answers.

**What this hides, deliberately:** the 686 users who came back to products they already
knew show real behaviour (returning to consider a purchase) that no model here is
allowed to recommend. The 1,057 who touched only products outside the grid are a cost of
the "items need 5 users" filter in 0003.

**Considered instead:** scoring all 141,559 users, with unknown users counted as 0
(every model's score would be dominated by the 97% no model can serve, hiding the
differences between models); scoring unknown users with the most-popular list (measures
the fallback, not the model; it can be reported separately).

### Recall@10 and NDCG@10 as headline metrics

| Metric | Question it answers |
| --- | --- |
| Recall@10 | of the user's answers, what share did the 10 recommendations find? |
| NDCG@10 | the same, but a hit at the top of the list counts more than a hit at the bottom |
| Precision@10 | of the 10 recommendations, what share were right? |
| Coverage | what share of the catalogue was recommended to anyone? |

NDCG weights a hit at position p by 1 / log2(p + 1): 1.0 at position 1, 0.63 at 2, 0.5
at 3, down to 0.29 at 10. It is divided by the best value possible for that user, so a
perfect list scores 1. Precision always divides by 10, even if a model returns fewer, so
a model is not rewarded for returning short lists.

Precision is reported but not used as a headline because a handful of users move it
heavily. Answers per scored user have a median of 1, 90% of users have at most 4 and 99%
at most 21, but 11 users have more than 50. Those 11 produce 10 of most-popular's 47
hits:

| Most-popular | Precision@10 | Recall@10 |
| --- | --- | --- |
| all 2,632 users | 0.00179 | 0.00790 |
| without the 11 users with > 50 answers | 0.00141 (−21%) | 0.00791 |

Precision credits each of their hits in full; recall divides by their many answers, so
their influence stays small. All users are kept: choosing who to score based on how they
behave in the evaluation period would make the results depend on that choice.

Because 64% of scored users have exactly one answer, their Precision@10 can be at most
0.1 and their Recall@10 is either 0 or 1. Absolute scores are therefore small for every
model; the comparison between models is what carries meaning.

Coverage is reported because accuracy alone favours showing everyone the same products:
most-popular reaches 0.05% of the catalogue, random 56%.

### A 95% range on every score

Only 41 of the 2,632 users receive a hit from most-popular, so the averages rest on very
few people. Each score is reported with a bootstrap range: the scored users are drawn
again, with replacement, 1,000 times, and the middle 95% of the resulting averages is
the range. It answers "how much could this number move with a different sample of
users?". The draw uses a fixed seed, so reruns give identical ranges.

**Rule for comparisons:** one model is only reported as better than another if their
ranges do not overlap. A difference inside the ranges is reported as no measurable
difference. The deployment gate in Phase 5 uses the same idea, so that a model is not
promoted on a difference produced by chance.

**Considered instead:** a statistical significance test per pair of models (more
machinery, same conclusion at this sample size, and harder to explain in a results
table); no ranges (on this data, a reader could not tell real differences from noise).

## Results on the baselines (validation)

| Model | Hits | Users | Recall@10 | NDCG@10 | Precision@10 | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| popular | any | 2,632 | 0.0079 (0.0049–0.0110) | 0.0037 (0.0024–0.0051) | 0.0018 (0.0012–0.0024) | 0.0005 |
| popular | cart or purchase | 141 | 0.0124 (0.0000–0.0301) | 0.0055 (0.0000–0.0122) | 0.0021 (0.0000–0.0050) | 0.0005 |
| random | any | 2,632 | 0.0010 (0.0001–0.0024) | 0.0007 (0.0000–0.0017) | 0.0002 (0.0000–0.0003) | 0.5570 |
| random | cart or purchase | 141 | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | 0.0429 |

Checks run before accepting these numbers:

- **Repeatable.** Running most-popular twice, and random twice with the same seed, gives
  identical results.
- **Random is near zero.** Across five seeds, random's Recall@10 stays between 0.00005
  and 0.00104.
- **Most-popular beats random.** Most-popular finds 47 hits for 41 users, random 4 for 4.
  On "any interaction" the ranges do not overlap, so the difference is real.
- **No leakage.** No recommendation list contains a product the user had already touched.
- **Most-popular is sensible.** The top product was touched by 710 training users, the
  tenth by 358.

On cart-or-purchase, most-popular's range reaches 0 and overlaps random's: with 141 users,
the two cannot be told apart. This is the reason that definition is secondary.

## Consequences

- Absolute numbers are small for every model on this dataset and must be read as
  comparisons, never as "the model is right 0.8% of the time" in a commercial sense.
- Results cover about 3% of validation users. The thesis states this alongside every
  table: the other 97% receive the most-popular fallback, which is what most real
  traffic would see.
- `similar_items` is not measured by this method. Measuring it for users unknown to the
  model, using the first product they viewed, is planned for Phase 3, when item-to-item
  exists.
- The test period has no code path in the evaluation runner. Adding one is a deliberate
  step in Phase 6.
