# 0006 — Item-to-item, and the settings it needs

**Status:** Accepted by Person A, 2026-11-05. Changes the minimum users per item from 0003,
and replaces the product page exam's key from 0005.
**Decided by:** Person A
**Affects:** `core/models`, `core/data`, `core/evaluation`, the deployment gate, and every
results table in the thesis

## Context

Item-to-item is the first model that gives related products, and the one built for the
product page: `similar_items` needs only the product being viewed, so it works for visitors
no model has seen. It came before ALS because it is cheap, explainable, and answers the
question the live endpoint asks.

The first version used cosine similarity with the filter from 0003. Two things it showed
prompted this record:

- A third of validation views were on products the filter had removed, so the site would show
  those visitors best sellers although the product had been seen in training.
- The product page exam from 0005 could not say what to do about that. It only asked about
  products in the training grid, so changing the filter changed who was scored and what
  counted as an answer. Scores from different settings could not be compared.

## Decision

| Choice | Value |
| --- | --- |
| Similarity | BM25, by library call, with the library's defaults (K1 1.2, B 0.75), not tuned |
| Neighbours kept per product | 50, not tuned, since product pages only show 10 |
| Minimum users per item | 3 (was 5) |
| Minimum items per user | 2 (unchanged) |
| Heavy-account cap | 1,000 (unchanged), to be reviewed once ALS is tested |
| Product page exam | one fixed key, covering the whole site, that no filter or model can change |

Code: `core/models/item_to_item.py`, `core/models/trained.py` (saving, loading and metadata,
shared with later models), `core/data/filtering.py`, `core/evaluation/answer_key.py`
(`build_product_page_key`).

## Why each part

### A fixed exam, so that settings can be compared

The key is now built from events alone, with no reference to the grid:

- **Question:** the first product each user viewed in the period, known to the model or not.
  The site has those pages too, and the fallback serves them.
- **Answers:** products the user first touched after that, among those seen in training before
  any filtering, and not touched by that user in training.

Every setting therefore faces the same users, questions and answers. A stricter filter pays
for what it removes, because a product it dropped can still be an answer it cannot give. The
key also covers the limit recorded in 0005, that the score only described pages the model
could answer. It is the number the deployment gate needs too, since an old and a new model
have different grids.

| Stage | Users |
| --- | --- |
| active in validation | 141,559 |
| viewed a product (have a question) | 141,109 |
| touched another product afterwards (scored) | 22,602 |
| of those, unknown to the model as users | 20,941 |

The personal exam from 0004 still depends on the grid, because it can only score users the
model knows. Its scores cannot be compared across filter settings.

### BM25 rather than cosine

Both call two products similar when the same users touched both, and both correct for
popularity, so that best sellers do not appear next to everything. They differ in how much
each shared user counts. In cosine, every user counts the same. In BM25, a user counts less
the more products they touched: an account that touched hundreds of products says little about
whether any two of them belong together. BM25 comes from search engines, where a word found
in every document tells you little about any of them.

Whole-site Recall@10 and NDCG@10 on validation, with filter 2 / 3 / 1000:

| Similarity | Recall@10 | NDCG@10 | Cart-or-purchase Recall@10 | Slots on the top 100 products |
| --- | --- | --- | --- | --- |
| cosine | 0.1578 (0.1533–0.1620) | 0.1044 (0.1014–0.1074) | 0.1314 (0.1159–0.1503) | 0.5% |
| TF-IDF | 0.2116 (0.2065–0.2162) | 0.1414 (0.1376–0.1448) | 0.1967 (0.1777–0.2191) | 0.7% |
| BM25 | 0.2161 (0.2112–0.2210) | 0.1440 (0.1401–0.1476) | 0.1950 (0.1745–0.2162) | 0.4% |

BM25 and TF-IDF cannot be told apart. BM25 was chosen for its slightly lower share of best
sellers and because it needs the same effort, one library call. Its two settings are the
library's defaults, passed explicitly so that a library update cannot change results
unnoticed.

**Considered instead:** cosine with a lower cap (see below); tuning BM25's settings (more
tuning on a single validation period, for a model that is already far ahead of the baselines).

### Keep the cap for now, although this model no longer needs it

The cap in 0003 was chosen by reasoning about 23 accounts, and it did matter for cosine:
without it, Recall@10 falls from 0.1503 to 0.1270 at filter 2 / 5. Lowering it kept helping,
with no natural stopping point, and each step deleted more real buyers:

| Cap (filter 2 / 3) | Accounts removed | Share of training purchases | Cosine Recall@10 | BM25 Recall@10 |
| --- | --- | --- | --- | --- |
| none | 0 | 0% | | 0.2154 (0.2105–0.2205) |
| 1,000 | 23 | 14.5% | 0.1578 (0.1533–0.1620) | 0.2161 (0.2112–0.2210) |
| 200 | 165 | 25.7% | 0.1850 (0.1804–0.1897) | 0.2154 (0.2104–0.2205) |
| 100 | 356 | 29.0% | 0.1923 (0.1877–0.1969) | |
| 50 | 1,039 | 33.4% | 0.1972 (0.1923–0.2017) | |

That pattern suggested the cap was a blunt fix for the real problem, very active accounts
counting too much, which BM25 addresses directly. With BM25 the cap makes no measurable
difference, and every BM25 setting above beats every cosine setting. The cap stays only because
the grid is shared with ALS, which has not been tested yet. If ALS does not need it either, it
should go: it is a rule tuned to one dataset, and removes 14.5% of training purchases.

### Three users per product, not five

BM25 Recall@10 on validation, at cap 1,000:

| Users ≥ / items ≥ | Recall@10 | Views the model can answer | Users in the grid | Products in the grid |
| --- | --- | --- | --- | --- |
| 2 / 5 (before) | 0.2089 (0.2037–0.2134) | 61% | 164,828 | 32,222 |
| **2 / 3** | **0.2161 (0.2112–0.2210)** | **71%** | **189,012** | **50,978** |
| 2 / 2 | 0.2108 (0.2057–0.2158) | 78% | 207,214 | 73,609 |
| 2 / 1 | 0.1878 (0.1831–0.1926) | 87% | 233,882 | 138,573 |
| 1 / 5 | 0.2168 (0.2118–0.2216) | 78% | 988,360 | 70,275 |
| 1 / 2 | 0.1998 (0.1950–0.2045) | 85% | 1,102,737 | 136,911 |

Below 3 the score falls: more pages get an answer, but rare products have unreliable
neighbours. The gain over 2 / 5 is modest and the ranges overlap slightly under BM25. A paired
comparison under cosine, on the same users, was clear (+0.0075, range +0.0052 to +0.0099). What
it certainly buys is 10 more product views in every 100 answered by the model.

**Considered instead:** 1 / 5, which scores the same and answers more pages (rejected: it keeps
five times as many users, most of them one-click visitors who link nothing, makes the model file
52 MB instead of 31 MB, and leaves more pages with fewer than 10 products); keeping 2 / 5 (it
answers fewer pages for no better score).

## Results with the chosen settings (validation)

| Model | Exam | Hits | Users | Recall@10 | NDCG@10 | Coverage | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| item-to-item | product page | any | 22,602 | 0.2161 (0.2112–0.2210) | 0.1440 (0.1401–0.1476) | 76.7% | 23.2% |
| recently popular | product page | any | 22,602 | 0.0096 (0.0084–0.0107) | 0.0066 (0.0058–0.0074) | 0.02% | |
| most-popular | product page | any | 22,602 | 0.0070 (0.0060–0.0080) | 0.0034 (0.0029–0.0038) | 0.02% | |
| random | product page | any | 22,602 | 0.0001 (0.0000–0.0002) | 0.0001 (0.0000–0.0001) | 94.6% | |
| item-to-item | product page | cart or purchase | 1,310 | 0.1950 (0.1745–0.2162) | 0.1231 (0.1093–0.1378) | 15.7% | 16.6% |
| recently popular | product page | cart or purchase | 1,310 | 0.0088 (0.0046–0.0140) | 0.0062 (0.0033–0.0100) | 0.02% | |
| most-popular | product page | cart or purchase | 1,310 | 0.0101 (0.0055–0.0159) | 0.0044 (0.0026–0.0068) | 0.02% | |
| item-to-item | personal | any | 3,351 | 0.0658 (0.0580–0.0731) | 0.0430 (0.0375–0.0484) | 40.0% | 0% |
| recently popular | personal | any | 3,351 | 0.0085 (0.0059–0.0116) | 0.0050 (0.0034–0.0067) | 0.04% | |
| most-popular | personal | any | 3,351 | 0.0072 (0.0048–0.0102) | 0.0033 (0.0023–0.0046) | 0.03% | |
| item-to-item | personal | cart or purchase | 170 | 0.0661 (0.0344–0.1015) | 0.0423 (0.0209–0.0655) | 3.2% | 0% |
| recently popular | personal | cart or purchase | 170 | 0.0160 (0.0012–0.0348) | 0.0081 (0.0012–0.0165) | 0.04% | |

Item-to-item is ahead by non-overlapping ranges on every row but the last, where the ranges just
touch and 170 users are too few to separate them. On pages the model can answer, Recall@10 is
0.2804, against 0.0030 on the pages the fallback serves.

A gap this large could be a bug, so these checks were run to try to disprove it:

- **Scrambled product IDs.** With product IDs shuffled before training, Recall@10 falls to
  0.0009 (0.0006–0.0012). The same code and users give nothing without the real links between
  products, so the score does not come from the scoring. It is slightly above random because
  the fallback still answers about a quarter of pages.
- **No future data.** Every training event is before 2015-08-21, and no test event is used.
- **Strangers.** Users unknown to the model score 0.2238, so user history cannot explain it.
  Known users score 0.1191, a harder task, since products they already touched do not count.
- **Repeatable.** Training twice gives an identical table and identical results.
- **No outliers.** 28 users have more than 50 answers and produce 13 of the 7,339 hits.

Cost on Retailrocket: trains in about a second, a 31 MB file, about 0.1 ms per request. The
saved file holds plain tables, so serving does not need the training library.

## Consequences

- **The bar for ALS and BPR is item-to-item, not the baselines**: above 0.2210 on the product
  page exam, or an explanation of why not. They must also answer `similar_items`.
- **ALS should be tested with and without the cap**, and the cap removed if it does not help.
- **The fallback now serves about 3 in 10 product views** (71% are answered by the model), down
  from about 4 in 10.
- **The gate should use this exam.** It is unchanged by retraining with a different grid.
- Scoring all models now takes about 6 minutes, because the exam asks about many more pages.
  Scoring one model with `--model` is much quicker.

## Limits

- **It predicts browsing, not persuasion.** 86% of hits happen within an hour of the question
  (64% of all answers do). These shoppers were not shown any recommendation, so a hit means
  they touched the product anyway. The score does not mean that a quarter of recommendations
  work, or that sales would rise. Only a live test could show that.
- **It may partly copy the shop's own layout.** If the Retailrocket shop already linked related
  products, shoppers would move between them anyway, and the model would learn those links. The
  data is anonymised, so this cannot be checked.
- **Slightly optimistic.** About 25 settings were tried on validation, so this period has been
  used for choosing. The sealed test gives the honest number.
- **Tuning stops here.** Neighbours, BM25's settings and the event weights were not tuned. The
  thesis is about the system around the model, and this model is already far ahead of the
  baselines.
- **Small catalogues.** BM25 gives a user no weight once they have touched nearly every product
  in the catalogue. In a 3-product shop, someone who touched 2 counts for nothing. The 2 / 3
  filter also keeps only 32 of the 739 visitors in the test sample. Both belong to the small-store
  rule in Phase 5.
