# 0005 — The fallback, and scoring what the site actually serves

**Status:** Accepted by Person A, 2026-11-04. The fallback changes `core/api`, which Person
B owns, so that part is to be confirmed with them. The fallback was first all-time
most-popular and is now recently popular, for the reason given below. 0006 replaced this exam's
key with one that covers the whole site and does not depend on the filter. The exam and results
below describe the first version; 0006 has the current ones.
**Decided by:** Person A
**Affects:** `core/api`, `core/training`, `core/evaluation`, the deployment gate, and every
results table in the thesis. Extends 0004.

## Context

Connecting the models to the service exposed two gaps.

**A model that had no answer crashed the request.** `GET /v1/recommendations` calls
`similar_items`. The interface in `core/models/base.py` says a model raises `ColdStartError`
when it has no basis to answer, and that the API then falls back to most-popular, but
neither the catch nor the list existed. The route returned a 500. Nothing had shown this
because the baselines never raise it; item-to-item and ALS will, on any product they do not
know.

**The evaluation measured a different method from the one served.** The endpoint takes a
product and calls `similar_items`. It has no `user_id`, so `recommend_for_user` is never
used. 0004 scores only `recommend_for_user`. That breaks the principle stated in 0001: the
system that was measured is the system that was demonstrated.

## Decision

| Choice | Value |
| --- | --- |
| When a model raises `ColdStartError` | the API serves the recently popular list instead |
| Where the list comes from | `fallback.joblib`, a recently popular model written by training beside the main model, from the same data |
| If that file is missing | an empty list and a logged warning, not an error |
| New exam: product page | scores `similar_items` the way the endpoint uses it |
| The question | the first product the user viewed that the model knows, once per user |
| The answers | known products the user first touched after that, not touched in training |
| Main number for the deployed system, and for the gate | the product page exam |
| The personal exam from 0004 | kept, as a second table |

Code: `core/models/loading.py` (`fallback_path`), `core/training/train.py`,
`core/api/routes/recommendations.py`, `core/evaluation/answer_key.py`
(`build_product_page_key`), `core/evaluation/run.py` (`evaluate_product_page`).

## Why each part

### A fallback file beside the model

A popularity list needs no history, so it can answer for any product. Training it on the same
data as the main model means it only offers products the store had at that time, and saving
it in the same folder means that giving each store its own model folder later gives it its
own fallback too, with no further change.

**Considered instead:** putting the list inside every model file (every algorithm would have
to carry a second model, and giving up is the API's concern in the interface, not the
model's); computing it from the events table at request time (a second definition of
"popular" that could disagree with the one evaluated, and a database query on every request
that needs it).

### Recently popular, not all-time most-popular

Recently popular counts different users over the last 14 days of training, instead of all
110. Scored on the product page exam, which is the kind of page the fallback serves:

| Validation, any interaction | Recall@10 | NDCG@10 |
| --- | --- | --- |
| most-popular | 0.0101 (0.0087–0.0116) | 0.0049 (0.0043–0.0057) |
| recently popular | 0.0141 (0.0125–0.0160) | 0.0098 (0.0087–0.0112) |

The ranges do not overlap, so the difference is real. What is popular does move: only 5 of
the two lists' top 10 products are shared. On cart-or-purchase, and on the personal exam, the
two cannot be told apart. Nowhere is most-popular measurably better. The exam only asks about
products the model knows, while the fallback answers the others, so this is the best evidence
available rather than a direct measurement of fallback pages.

The window was fixed at 14 days before any results were seen, because it matches the length
of the evaluation periods, and it is not tuned. 7 and 28 days were run only to check that the
result does not hinge on the number: they score a Recall@10 of 0.0148 and 0.0127.

Ties go to the product more users touched overall. A quiet store may have few events in the
window, and this keeps its list sensible rather than in ID order. It never outranks a product
with more users in the window, and it leaves every Retailrocket score above unchanged.

**Considered instead:** keeping most-popular (simpler, but it scored measurably worse, and the
fallback answers about 4 in 10 product views); choosing the window that scores best (a baseline tuned until it scores well
is just another model).

### Score the product page, and keep the personal exam

Both exams use the same metrics, ranges and training data. Only the question differs:

| | Personal (0004) | Product page (this record) |
| --- | --- | --- |
| Method tested | `recommend_for_user` | `similar_items` |
| The model is given | the user's history in training | one product |
| Who can be scored | users in the training grid | anyone, known or not |
| Used by the live site | no | yes |

The personal exam remains useful: it shows how good personal recommendations would be for
known users, and it would justify adding a user endpoint later. It is simply not what the
site serves. Scores cover different users and cannot be compared across exams; models are
compared within one exam.

The product page exam also answers a weakness recorded in 0004, that results rested on very
few people:

| Stage | Users |
| --- | --- |
| active in validation | 141,559 |
| viewed a product the model knows (have a question) | 82,234 |
| touched another known product afterwards (scored) | 14,921 |
| of those, unknown to the model as users | 13,847 |

**Considered instead:** changing the site to serve `recommend_for_user` (it would need a
shopper ID in the storefront request, and only 3% of validation users are known to the model
anyway); scoring every consecutive pair of products (very active users would dominate, and
scores could no longer be averaged per user as in the personal exam); scoring questions on
products the model does not know (every model would give the same fallback list, which says
nothing about the models).

### No credit for products the user already touched

The evaluation rules exclude items a user has already interacted with, but `similar_items`
receives no user, so it cannot remove them from its list, and neither does the live page.
They are removed from the answers instead: showing one simply counts as a miss. This is the
cautious choice, since a model can never gain from it. The same applies to the question's own
product and to anything touched before reaching the page.

### The fallback counts towards the score

If a model gives up on a page, the exam uses the fallback's list, exactly as the site would,
so what is measured is what is served. Because that could flatter a model that gives up
often, the results table shows the share of scored users answered by the fallback.

## Results on the baselines (validation)

| Model | Exam | Hits | Users | Recall@10 | NDCG@10 | Precision@10 | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| popular | product page | any | 14,921 | 0.0101 (0.0087–0.0116) | 0.0049 (0.0043–0.0057) | 0.0018 (0.0016–0.0021) | 0.0003 |
| popular | product page | cart or purchase | 976 | 0.0136 (0.0071–0.0211) | 0.0061 (0.0035–0.0092) | 0.0026 (0.0014–0.0038) | 0.0003 |
| random | product page | any | 14,921 | 0.0002 (0.0000–0.0005) | 0.0001 (0.0000–0.0002) | 0.0000 (0.0000–0.0001) | 0.9175 |
| random | product page | cart or purchase | 976 | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | 0.2346 |

Baselines never give up, so the fallback share is 0 in every row. Personal exam results are
unchanged from 0004.

Checks run before accepting these numbers:

- **Repeatable.** Scoring most-popular twice gives identical results.
- **Random is near zero.** Across five seeds, Recall@10 stays between 0.0002 and 0.0005.
- **Most-popular beats random.** 275 hits for 243 users, against 6 for 6. The ranges do not
  overlap, on cart-or-purchase as well, which the personal exam's 141 users could not show.
- **A page never recommends itself.** No list contains its own question product.
- **A few heavy users matter less.** 22 users have more than 50 answers and produce 28 of the
  275 hits. Without them Precision@10 falls by about 10% (21% in the personal exam), and
  Recall@10 is unchanged, so Recall@10 and NDCG@10 stay the headline metrics.

## Consequences

- **The filter, not new products, causes most fallback pages.** Only 61% of validation views
  are on products in the training grid. 94.3% are on products seen in training before
  filtering, so 33.3% of all views would get the fallback because of the filter in 0003, and
  only 5.7% because the product is genuinely new. Some of that cannot be recovered, since a
  product nobody viewed alongside another has no neighbours to offer. The thresholds in 0003
  were judged on the personal exam, so they are to be checked again on this one.
- **Comparing filter settings needs care.** Both answer keys depend on the grid, so changing
  the filter changes who is scored. Scores from different settings are not directly
  comparable.
- **The deployment gate should use this exam**: it measures what is live, and its ranges are
  tighter. The ranges cover about half the width of the personal exam's.
- A new model is picked up on restart, so promoting one has to trigger a restart.
- The endpoint still returns 503 if the main model file is missing, even when a fallback
  exists. Whether a small store with no trainable model should be served from the fallback
  alone is decided with the small-store rule in Phase 5.

## Limits

- Retailrocket has no sessions, so "after" can mean days later, not the same visit.
- The exam covers about 11% of validation users. One-click visitors are served, but cannot be
  scored, because they did nothing afterwards.
- Questions are only asked about products the model knows, so the score describes those
  pages. The other 39% of views get best sellers.
- Like the personal exam, this is an offline estimate. These shoppers were not shown these
  recommendations, so a hit means they touched the product anyway, not that a recommendation
  caused it.
