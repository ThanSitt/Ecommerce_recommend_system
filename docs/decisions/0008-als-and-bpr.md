# 0008 — ALS and BPR, and removing the heavy-account cap

**Status:** Accepted by Person A, 2026-11-13. Removes the heavy-account cap from 0003 and 0006.
**Decided by:** Person A
**Affects:** `core/models`, `core/data`, every results table in the thesis, and which model the
site should serve

## Context

Phase 4 adds the two matrix factorisation models planned from the start. 0006 set the bar:
item-to-item, Recall@10 0.2161 (0.2112–0.2210) on the product page exam, measured with the
1,000-event cap. Both new models must also answer `similar_items`, since that is what the live
endpoint asks. 0006 also asked for ALS to be tested with and without the cap, and the cap
removed if ALS does not need it either.

Matrix factorisation describes every user and every product by a short list of numbers
("factors"), learned so that a user's numbers multiplied with a product's numbers predict
whether the user touches it. Nobody says what each number means. The two models differ in
what they learn from:

- **ALS** (alternating least squares, Hu, Koren and Volinsky 2008) tries to predict every cell
  of the grid, filled or empty. A filled cell is trusted `alpha` times its weight (view 1, cart
  3, purchase 5); an empty one is trusted a little, as "probably not interested".
- **BPR** (Bayesian personalised ranking, Rendle et al. 2009) only learns to rank a product the
  user touched above one they did not. It ignores the weights.

## Decision

| Choice | Value |
| --- | --- |
| ALS | 200 factors, regularisation 10, alpha 200, 15 iterations, seed 0 |
| BPR | 100 factors, learning rate 0.1, regularisation 0.01, 100 iterations, seed 0, one thread |
| `similar_items` for both | cosine similarity between product factors, as the library does it |
| `recommend_for_user` for both | user factors times product factors, seen products removed |
| Heavy-account cap | removed (was 1,000 events) |
| Filter | 2 items per user, 3 users per product (unchanged) |
| Event weights | view 1, cart 3, purchase 5 (unchanged, now tested) |
| Best model on Retailrocket | ALS, on the product page exam |

Code: `core/models/matrix_factorisation.py` (`ALSRecommender`, `BPRRecommender`, sharing
`_FactorModel`), `core/models/trained.py` (a faster top-10, below), `core/data/filtering.py`
and `core/data/prepare.py` (cap removed). `--model als` and `--model bpr` in both commands.

The learning is a library call (`implicit`); what is saved is two plain tables of numbers, so
serving needs no library. A test checks that the saved tables give exactly the library's own
answers.

## Why each part

### ALS: the confidence setting matters most

With the library's defaults (100 factors, regularisation 0.01, alpha 1), ALS scored 0.0524,
a quarter of item-to-item. With alpha 1 a purchase is barely trusted more than an empty cell,
and there are 50,000 empty cells for every filled one in a user's row. Raising alpha was by far
the largest gain; factors and regularisation then added a little.

Whole-site Recall@10 on validation, product page exam, any interaction. All at 15 iterations,
with the cap (as measured before removing it):

| Factors | Regularisation | Alpha | Recall@10 | NDCG@10 |
| --- | --- | --- | --- | --- |
| 100 | 0.01 | 1 (library default) | 0.0524 (0.0497–0.0548) | 0.0317 |
| 100 | 0.01 | 10 | 0.1381 (0.1337–0.1423) | 0.0828 |
| 100 | 0.01 | 40 | 0.1788 (0.1740–0.1832) | 0.1117 |
| 100 | 0.01 | 100 | 0.1986 (0.1939–0.2033) | 0.1272 |
| 100 | 0.01 | 200 | 0.2099 (0.2050–0.2150) | 0.1356 |
| 100 | 0.01 | 500 | 0.2104 (0.2053–0.2153) | 0.1385 |
| 50 | 0.01 | 200 | 0.1831 (0.1784–0.1882) | 0.1167 |
| 200 | 0.01 | 200 | 0.2112 (0.2061–0.2162) | 0.1394 |
| 100 | 0.1 | 200 | 0.2190 (0.2142–0.2241) | 0.1419 |
| 100 | 1 | 200 | 0.2185 (0.2137–0.2234) | 0.1406 |
| 100 | 10 | 200 | 0.2198 (0.2149–0.2252) | 0.1412 |
| 100 | 100 | 200 | 0.2186 (0.2139–0.2236) | 0.1399 |
| 100 | 10 | 500 | 0.2250 (0.2198–0.2300) | 0.1463 |
| 100 | 10 | 200, 30 iterations | 0.2119 (0.2071–0.2169) | 0.1359 |
| **200** | **10** | **200** | **0.2334 (0.2281–0.2384)** | **0.1510** |
| 200 | 10 | 500 | 0.2300 (0.2248–0.2349) | 0.1486 |
| 200 | 100 | 200 | 0.2366 (0.2316–0.2415) | 0.1524 |
| 200 | 10 | 200, 10 iterations | 0.2367 (0.2313–0.2419) | 0.1533 |

The last four rows cannot be told apart. The chosen setting is the middle of that plateau, with
the library's default number of iterations, rather than the single highest number: picking the
top of a group of equal scores picks noise. More iterations did not help.

Different random starting points (seeds 0, 1, 2) give 0.2334, 0.2342 and 0.2336, so the result
does not depend on the seed.

**Considered instead:** 400 factors (not tried: the file would double to about 400 MB for a
model already ahead, and the thesis is about the system around the model); tuning on the
personal exam (rejected: the product page is what the site asks).

### BPR: a faster learning rate, and one thread

With the library's defaults BPR scored 0.1056. It learned too slowly in 100 rounds: raising the
learning rate helped, nothing else did. Without the cap:

| Factors | Learning rate | Regularisation | Iterations | Recall@10 |
| --- | --- | --- | --- | --- |
| 100 | 0.01 (default) | 0.01 | 100 | 0.1056 (0.1017–0.1093) (with cap) |
| 100 | 0.05 | 0.01 | 100 | 0.1668 (0.1626–0.1712) |
| 50 | 0.05 | 0.01 | 100 | 0.1599 (0.1558–0.1640) |
| **100** | **0.1** | **0.01** | **100** | **0.1786 (0.1740–0.1833)** |
| 100 | 0.2 | 0.01 | 100 | 0.1821 (0.1773–0.1866) |
| 100 | 0.1 | 0.001 | 100 | 0.1569 (0.1524–0.1612) |
| 100 | 0.1 | 0.1 | 100 | 0.1592 (0.1545–0.1634) |
| 200 | 0.1 | 0.01 | 100 | 0.1786 (0.1740–0.1830) |
| 100 | 0.1 | 0.01 | 200 | 0.1821 (0.1774–0.1865) |

BPR levels off around 0.18, well below ALS and item-to-item. The likely reason is how it
learns: from sampled pairs, one touched and one untouched product at a time, rather than from
every cell of the grid as ALS does. It also ignores the weights, but that is not the reason:
ALS with every weight 1 scores the same as with them (below). Tuning stopped once more factors
and more rounds stopped helping.

BPR trains on one thread. With several, the updates happen in a different order every run,
and the same data gives different models; the gate in Phase 5 has to compare models that only
differ in their data. One thread costs about 90 seconds of training on Retailrocket.

### `similar_items`: cosine between product factors

Two products are similar when their lists of numbers point the same way. Cosine ignores the
lists' size, which grows with popularity; a raw product of the two lists would put best
sellers next to everything. This is what the library's own `similar_items` does, and a test
checks that ours gives the same answers.

For BPR the library adds one more number per product, a popularity bias (and a fixed 1 for
every user). It is kept in the cosine, as the library does.

### Removing the cap

ALS, chosen setting, product page exam:

| Cap | Recall@10 | NDCG@10 |
| --- | --- | --- |
| 1,000 | 0.2334 (0.2281–0.2384) | 0.1510 |
| none | 0.2346 (0.2298–0.2395) | 0.1512 |

Item-to-item: 0.2161 with the cap, 0.2154 without (0006). Neither model is affected, so the cap
was removed, as 0006 planned. It was a rule tuned to one dataset that deleted 23 accounts and
14.5% of training purchases. `filter_interactions` keeps the option (`max_events_per_user`),
unused for Retailrocket.

### The event weights

0003 left the weights (view 1, cart 3, purchase 5) to be checked in Phase 4. ALS is the
model that uses them most directly, as confidence, so it was trained again with every event
weighted 1:

| Weights | ALS Recall@10 |
| --- | --- |
| view 1, cart 3, purchase 5 | 0.2346 |
| all 1 | 0.2358 |

Paired on the same users, the difference is −0.0013 (−0.0028 to +0.0003): none. Carts and
purchases are 5.3% of filled cells, too few to move the model either way, and with alpha
200 every filled cell is already trusted far more than an empty one. The weights stay as they
are: changing them would gain nothing measurable and change every other result. The finding
is that, on this data, *whether* a user touched a product carries the information, not how.

**Considered instead:** tuning the weights further (nothing to tune when "all equal" scores
the same); summing repeated events instead of keeping the strongest (an open question from 0003,
not tried for the same reason).

## Results (validation, no cap)

| Model | Exam | Hits | Users | Recall@10 | NDCG@10 | Coverage | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **ALS** | product page | any | 22,602 | **0.2346 (0.2298–0.2395)** | **0.1512 (0.1478–0.1546)** | 65.3% | 22.7% |
| item-to-item | product page | any | 22,602 | 0.2154 (0.2105–0.2205) | 0.1435 (0.1397–0.1471) | 76.1% | 22.7% |
| BPR | product page | any | 22,602 | 0.1786 (0.1740–0.1833) | 0.1156 (0.1122–0.1189) | 65.9% | 22.7% |
| recently popular | product page | any | 22,602 | 0.0096 (0.0084–0.0107) | 0.0066 (0.0058–0.0074) | 0.02% | |
| most-popular | product page | any | 22,602 | 0.0070 (0.0060–0.0080) | 0.0034 (0.0029–0.0038) | 0.02% | |
| random | product page | any | 22,602 | 0.0001 (0.0000–0.0003) | 0.0001 (0.0000–0.0001) | 94.2% | |
| ALS | product page | cart or purchase | 1,310 | 0.2134 (0.1926–0.2360) | 0.1344 (0.1200–0.1496) | 14.5% | 15.8% |
| item-to-item | product page | cart or purchase | 1,310 | 0.1925 (0.1723–0.2134) | 0.1219 (0.1084–0.1361) | 15.6% | 15.8% |
| BPR | product page | cart or purchase | 1,310 | 0.1620 (0.1433–0.1805) | 0.1014 (0.0887–0.1147) | 14.5% | 15.8% |
| recently popular | product page | cart or purchase | 1,310 | 0.0088 (0.0046–0.0140) | 0.0061 (0.0032–0.0099) | 0.02% | |
| most-popular | product page | cart or purchase | 1,310 | 0.0101 (0.0055–0.0159) | 0.0044 (0.0026–0.0068) | 0.02% | |
| ALS | personal | any | 3,383 | 0.0712 (0.0633–0.0796) | 0.0423 (0.0374–0.0475) | 17.9% | 0% |
| item-to-item | personal | any | 3,383 | 0.0656 (0.0574–0.0738) | 0.0429 (0.0375–0.0484) | 39.5% | 0% |
| BPR | personal | any | 3,383 | 0.0423 (0.0365–0.0486) | 0.0256 (0.0218–0.0297) | 32.4% | 0% |
| recently popular | personal | any | 3,383 | 0.0084 (0.0061–0.0114) | 0.0052 (0.0037–0.0069) | 0.06% | |
| most-popular | personal | any | 3,383 | 0.0072 (0.0048–0.0099) | 0.0035 (0.0024–0.0046) | 0.04% | |
| ALS | personal | cart or purchase | 178 | 0.0741 (0.0399–0.1101) | 0.0382 (0.0205–0.0569) | 2.8% | 0% |
| item-to-item | personal | cart or purchase | 178 | 0.0612 (0.0298–0.0949) | 0.0391 (0.0186–0.0638) | 3.2% | 0% |
| BPR | personal | cart or purchase | 178 | 0.0544 (0.0264–0.0850) | 0.0286 (0.0129–0.0467) | 3.2% | 0% |
| recently popular | personal | cart or purchase | 178 | 0.0155 (0.0015–0.0339) | 0.0099 (0.0028–0.0187) | 0.06% | |

Random scores 0.0001 or less on every row. Each model is scored with the recently popular
fallback answering where it cannot, as on the live site.

- **Product page, any interaction (the main number): ALS beats item-to-item**, on Recall@10
  and on NDCG@10, by ranges that do not overlap.
- **Everywhere else, ALS and item-to-item cannot be told apart.** Cart or purchase has only
  1,310 scored users, the personal exam 3,383.
- **BPR is third.** It beats every baseline everywhere, and loses to both ALS and item-to-item
  on the main number.
- **Coverage:** ALS shows 65% of products somewhere, item-to-item 76%. ALS spreads its pages
  over fewer products.

### Checks

ALS's lead is small, so these checks tried to disprove it, as 0006 did for item-to-item:

- **Paired comparison.** The same 22,602 users, the same pages. ALS minus item-to-item:
  Recall@10 +0.0191 (+0.0160 to +0.0225), NDCG@10 +0.0077 (+0.0057 to +0.0097). Small but
  clear. 29.5% of users get at least one hit from ALS, 27.3% from item-to-item.
- **Different, not just better.** On pages both models answer, they share 4.8 of the 10
  products. 33.2% of users would get a hit from at least one of the two.
- **Scrambled product IDs.** Trained on a grid whose products carry the wrong IDs, ALS falls to
  0.0007, and to 0.0001 on the pages it answers: random's level. The score comes from real
  links between products, not from the scoring.
- **Strangers.** Users unknown to the model score 0.2426 (item-to-item 0.2232); known users
  0.1338 (0.1189). ALS is ahead in both groups, so the lead is not from user history.
- **Model pages and fallback pages.** 0.3026 on the pages ALS answers, 0.0029 on the pages the
  fallback answers (22.7% of users). Same split as item-to-item, since both use the same grid.
- **Repeatable.** Same data and seed, same model (tested). Seeds 0, 1, 2 differ by 0.0008.
- **Browsing, again.** 85.7% of ALS's hits come within an hour of the question, 93.7% within a
  day: the same pattern as item-to-item.
- **No drift to best sellers.** On the pages each model answers, as users saw them, the 100
  most popular products fill 1.07% of ALS's slots, 1.04% of item-to-item's and 1.02% of BPR's.
  (Counted per user, so not comparable with the 0.4% per page in 0006.)

### Cost on Retailrocket

| | ALS | item-to-item | BPR |
| --- | --- | --- | --- |
| Training | ~20 s | ~1 s | ~90 s (one thread) |
| Model file | 203 MB | 32 MB | 107 MB |
| Loading | 1.0 s | 0.6 s | 0.6 s |
| Product page request | 3.0 ms | 0.05 ms | 4.4 ms |
| Personal request | 2.8 ms | 1.6 ms | 4.1 ms |

All well inside what a product page can wait for. ALS and BPR score every product on each
request; item-to-item looks up a stored list.

Most of ALS's file is the users' numbers (189,507 users × 200), which only the personal
endpoint needs; the product page needs only the products' 52,000 × 200. If the personal
endpoint is never served, Phase 5 could leave the users' numbers out and save about three
quarters of the file. Not done now: it changes the handover interface.

## Consequences

- **ALS is the best model on Retailrocket, but not by much.** Item-to-item is close behind,
  far cheaper (about a second to train, a smaller file), and easier to explain. Which one a
  store gets is the per-store setting planned for Phase 5; for a store the size of Retailrocket,
  ALS.
- **To serve ALS, retrain with `--model als`.** No change to the API: the file says which
  model it holds, and the fallback is written beside it as before.
- **The settings are tuned to a 50,000-product catalogue.** On a tiny test grid, 4 factors
  already learned noise (two separate groups of products got mixed); 2 were right. A small
  store needs far fewer factors, or item-to-item. This belongs to the small-store rule in
  Phase 5.
- **The gate can compare any two of these models**, since the product page exam does not depend
  on the grid or the model.
- **A faster top-10.** Every model now finds the 10th-best score first and sorts only the
  products at or above it, instead of sorting all 50,000. The answers are identical, ties
  included (tested against a full sort), and scoring all six models takes about 6 minutes,
  as long as four took before.

## Limits

- **Optimistic, like 0006.** 18 ALS and 9 BPR settings were tried on validation, and about 25
  for item-to-item. ALS's lead (about 0.02) is small enough that choosing settings on this same
  period could account for part of it, so the sealed test may not repeat it. The test period
  decides in Phase 6.
- **It predicts browsing, not persuasion**, as recorded in 0006, and that holds for ALS too
  (see Checks). A hit means the shopper touched the product anyway; nobody was shown it.
- **BPR was tuned less than ALS** (9 settings against 18), and only in the library's form.
  Variants exist that sample pairs differently, but that is model work beyond this thesis.
- **The personal exam cannot separate the models.** It is reported, not used for choosing.
