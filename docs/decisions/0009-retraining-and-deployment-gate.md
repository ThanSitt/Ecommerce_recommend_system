# 0009 — Retraining each store, and the deployment gate

**Status:** Accepted by Person A, 2026-11-23. Where a store's model file lives changes what
`core/api` loads, and the scheduled job belongs to `infra/` and `.github/workflows/`, both
Person B's: those parts are to be agreed with them (see Consequences).
**Decided by:** Person A
**Affects:** `core/training`, `core/data`, `core/models/loading.py`, which file `core/api` serves
for a store, and the scheduled retraining job

## Context

Until now one command trained one model on Retailrocket and wrote one file, served to every
store. Phase 5 turns this into the pipeline the thesis is about: each store's model trained on
its own events, on a schedule, and put live only if it passes a deployment gate. CLAUDE.md sets
the gate's job: before replacing the live model, a new one "must beat a configured threshold
relative to the current one on a held-out set, and produce structurally valid output", because
a model can load and still be much worse than what it replaces.

Earlier records left questions for this phase:

- 0003: a small store cannot use the Retailrocket filter; it can leave nothing to train on.
- 0004: a difference counts only when the 95% ranges say so. The gate should use the same idea.
- 0005: the API answers 503 if a store has no model file. Should a store with no trainable
  model get best sellers alone?
- 0005, 0006, 0008: the gate should use the product page exam, which measures what the site
  serves and can compare any two models.
- 0008: ALS's settings were tuned on 50,000 products; on a tiny grid they learn noise. Which
  model does a store get?
- `.env.example` held `GATE_MIN_RELATIVE_IMPROVEMENT=0.0`, from the project skeleton: a rule
  on two averages, with nothing reading it.

## Decision

| Choice | Value |
| --- | --- |
| Events | the store's rows of the core's `events` table (or a CSV export of it) |
| Held-out period | the store's newest 7 days; "now" is the newest event, not the clock |
| Training | every event before the held-out period, Retailrocket's filter (2 products per shopper, 3 shoppers per product) |
| Model | the kind that is live; item-to-item if none is; `--model` overrides |
| Best sellers | recently popular (0005), from the same filtered grid as the model; from every event if the filter leaves nothing |
| Exam | the product page exam on the held-out events, any interaction, NDCG@10 per shopper, the model with its fallback as the API serves them |
| 1. Well formed | every checked page: at most 10 different products of this store, best first, finite scores, never the product itself; the fallback always answers; the model answers some pages |
| 2. Enough shoppers | at least 30 scored; fewer and the gate does not judge |
| 3. Beats best sellers | the 95% range of the paired difference (new − best sellers) lies above 0 |
| 4. Not worse than live | the 95% range of the paired difference (new − live) does not lie entirely below 0 (tolerance 0) |
| Accepted | the judged files are copied to `current.joblib` and `fallback.joblib` |
| Rejected | a live model stays; a store with no real model is served best sellers from all its events |
| Files | `models/<store_id>/runs/<run>/` keeps every run's files and `report.json`; `models/<store_id>/current.joblib` and `fallback.joblib` are served |

Code: `core/training/gate.py` (`judge`, `check_structure`), `core/training/retrain.py`
(`retrain_store`, the command), `core/data/store_events.py` (loading a store's events),
`core/models/loading.py` (`store_dir`, `current_model_path`).

```bash
python -m core.training.retrain --all                        # every active store, from the database
python -m core.training.retrain --all --events events.csv    # every store in an export
python -m core.training.retrain --store st_123 --model als   # one store, another model
```

### Two questions, not one

The gate asks whether the new model is better than **best sellers**, and whether it is worse
than **the live model**. A threshold against the live model alone, as CLAUDE.md first puts it,
has two gaps. On a store's first run there is no live model, so nothing would be checked. And if
a weak model ever went live, every later one would only have to match it. Best sellers are the
floor every result in this thesis is measured against (most-popular and recently popular are
mandatory baselines), so a model that cannot beat them is not worth serving: best sellers need
no model.

### "Not worse", not "better"

The live model is protected against a measurable drop, but a new model does not have to be
measurably better than it. Demanding a gain every week would freeze the first model that went
live: a week's more data seldom makes a difference the ranges can show, yet retraining is the
only way new products and new habits reach the recommendations. A model that is as good as the
live one and a week fresher is worth having.

The tolerance (`--tolerance`, default 0) loosens this: 0.05 lets through a measurable drop of up
to 5% of the live model's score. 0 is the default because no drop has a reason to be accepted.

### Measurable, as in 0004

Both comparisons are **paired**: both models face the same held-out shoppers, and the range is
that of their per-shopper differences, from re-drawing the shoppers 1,000 times. This is
tighter than comparing two separate ranges, because the variation between shoppers cancels
out. It replaces the skeleton's `GATE_MIN_RELATIVE_IMPROVEMENT=0.0`, which compared two averages:
with a few dozen shoppers, a new model as good as the old one would have been rejected about
half the time, by chance.

NDCG@10 rather than Recall@10: it counts a hit near the top of the list for more, and the top is
what a shopper sees first. One metric, chosen in advance, so the gate cannot pass a model on
whichever metric happens to favour it.

### Enough shoppers to judge

Below 30 scored shoppers the gate does not judge: re-drawing from a handful of shoppers gives a
range that cannot be trusted either. 30 is a common rule of thumb, not tuned. A store that
small gets best sellers (or keeps its live model) until it has more shoppers.

### Well formed

Before scoring, the gate asks the new model and its fallback about up to 200 products spread
over the catalogue, and about one product nobody has seen. Each answer must be a list of at most
10 different products this store has, best first, with finite scores, never the product being
viewed. A product the model cannot answer must give `ColdStartError` (so the API serves the
fallback), not a crash; the fallback must answer every page; the model must answer some. A model
trained for another store is refused. These problems would not always lower a score (a repeated
product is still a hit), so scoring alone would not catch them.

The gate judges the saved files loaded back, not the objects in memory: the API loads files.

### What is judged is what is deployed

The deployed model is the one trained without the held-out week, so the store's newest week is
not in it. The alternative is to retrain on every event once the gate has passed, which is
fresher but deploys a file nobody judged: a broken week of data (a bot, a connector sending
duplicates) would go live unchecked, in exactly the week it happened. The thesis's claim is that
a model reaches shoppers only after passing the gate, so the judged files are the deployed ones.

The cost is a week's delay: a product first sold this week appears in recommendations, and in
the best sellers beside them, from next week's run on.

### When no model qualifies

- **A store with a live model keeps it.** It once beat best sellers; one week in which a new
  model does not is no reason to drop to best sellers.
- **A store without one is served best sellers**, trained on all its events, the newest week
  included. Best sellers are the floor, so nothing judges them, and freshness is what they need
  most. Live best sellers are refreshed on every such run, and give way to a model as soon as
  one passes.

So every store with at least one event has a file to serve, and the 503 that 0005 left open
does not happen once a store has been through one run. A store whose events all fall in the
last week is served best sellers: there is nothing before the held-out week to train on.

### The newest event is "now"

The held-out period is the 7 days before the store's newest event, not before the time of the
run. The same events give the same decision whenever the run happens, a run can be replayed for
any past date (`--until`), and a store that stopped sending events is not judged on an empty
week.

If the live model was trained on part of the held-out period (by hand, or by a run with a shorter
period), it would be marked on answers it learned from. Scoring then starts after the newest event
the live model or its fallback saw; the report records both dates.

### Best sellers from the filtered grid

Best sellers count the shoppers the filter keeps, as in Phases 2 to 4, so the model and its
fallback are the pair those phases measured. A store whose filter leaves nobody counts every
shopper instead, so it still gets a list.

Counting every shopper was tried first. The filter exists to learn which products go together,
and popularity needs no pairs, so it seemed the better count. But the model file keeps a row for
every shopper it counted (to leave out what a shopper has already seen), and on Retailrocket that
is everyone. Retailrocket's training period, scored on validation (product page exam, any
interaction):

| Best sellers counted over | Shoppers in the file | File | In memory | Alone: Recall@10 | Alone: NDCG@10 | Item-to-item with them: Recall@10 |
| --- | --- | --- | --- | --- | --- | --- |
| the filtered grid | 189,507 | 10 MB | 38 MB | 0.0096 (0.0084–0.0107) | 0.0066 | 0.2154 |
| every shopper | 1,151,479 | 41 MB | 180 MB | 0.0078 (0.0067–0.0087) | 0.0043 | 0.2153 |

Counting every shopper costs nearly five times the memory, loads about five times slower, and
gives no better list (the two share 8 of their top 10 products). So the filtered grid it is.

### Which model a store gets

A store with no model live starts with item-to-item: about a second to train, no setting that
depends on the size of the catalogue, and close to ALS on Retailrocket (0008). After that, each
run trains **the kind of model that is live**, unless `--model` says otherwise. So trying ALS on
a store is one run with `--model als`; if it passes, the store stays on ALS.

Retraining a fixed default every week does not work, as the replay below found: once ALS was
live, a fresh item-to-item model was measurably worse than it and rejected, so every weekly
item-to-item run would have been rejected while the ALS model aged. Whatever the model, the gate
decides whether it is served, so a model that suits the store badly (0008's ALS settings on a
tiny grid) is caught by rule 3 or 4 rather than by a rule about store size.

## Retailrocket replayed through the pipeline

Retailrocket's events up to the sealed test period, run through `retrain_store` as one store,
once a week. Weeks 1 to 4 retrain item-to-item. Week 5 tries four candidates in turn: two
deliberately bad ones against the live item-to-item model, then ALS, then item-to-item against
the ALS model just promoted. Each week's held-out period is the 7 days before the run;
NDCG@10, product page exam; in brackets the 95% range of the paired difference. The test period
was never loaded.

| Run (held out) | Candidate | Shoppers | Candidate | Best sellers | Live | Candidate − best sellers | Candidate − live | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (Jul 31–Aug 7) | item-to-item | 11,694 | 0.1448 | 0.0032 | none | +0.1416 (+0.1367 to +0.1463) | | promoted |
| 2 (Aug 7–14) | item-to-item | 10,489 | 0.1410 | 0.0066 | 0.1361 | +0.1345 (+0.1292 to +0.1395) | +0.0050 (+0.0033 to +0.0067) | promoted |
| 3 (Aug 14–21) | item-to-item | 11,261 | 0.1455 | 0.0058 | 0.1368 | +0.1397 (+0.1342 to +0.1450) | +0.0087 (+0.0071 to +0.0104) | promoted |
| 4 (Aug 21–28) | item-to-item | 11,060 | 0.1525 | 0.0068 | 0.1424 | +0.1457 (+0.1404 to +0.1509) | +0.0101 (+0.0083 to +0.0120) | promoted |
| 5 (Aug 28–Sep 4) | random | 11,213 | 0.0001 | 0.0070 | 0.1399 | −0.0069 (−0.0081 to −0.0057) | −0.1397 (−0.1450 to −0.1347) | rejected, rule 3 |
| 5 | ALS, 2 factors | 11,213 | 0.0030 | 0.0070 | 0.1399 | −0.0040 (−0.0053 to −0.0028) | −0.1369 (−0.1421 to −0.1320) | rejected, rule 3 |
| 5 | ALS | 11,213 | 0.1535 | 0.0070 | 0.1399 | +0.1465 (+0.1411 to +0.1517) | +0.0137 (+0.0105 to +0.0166) | promoted |
| 5 | item-to-item | 11,213 | 0.1478 | 0.0070 | 0.1535 (ALS) | +0.1407 (+0.1358 to +0.1463) | −0.0058 (−0.0085 to −0.0028) | rejected, rule 4 |

- **Every rule acted on real data.** Weekly retraining passed every week. Random recommendations
  and a misconfigured ALS were stopped by rule 3. Item-to-item, twenty times better than best
  sellers, was stopped by rule 4, because the store had something better live.
- **The misconfigured ALS is the case the gate exists for.** With 2 factors instead of 200 it
  trains, saves, loads and answers every page in the right form, so it passes every structural
  check; it is only worse than best sellers, which only scoring shows.
- **A week's data is worth having.** Each retrained item-to-item beat the week-old one by
  0.005 to 0.010, with ranges clear of zero. On a store this size even "measurably better" would
  have let them through. On a small store the same gain sits inside the range, which is why the
  rule is "not measurably worse".
- **Week 5: ALS ahead of item-to-item by 0.0137**, the same finding as 0008. It is not new
  evidence: weeks 4 and 5 fall in the validation period the models were tuned on. These rows
  show the gate working, not new accuracy results.
- **Cost:** an item-to-item run takes about a minute on 2.5 million events, most of it the
  filter; an ALS run about two. A run's folder is 36 to 44 MB for item-to-item, 226 MB for ALS.

## The demo stores

Person B's mock events (two stores, 500 events each over 12 days, about 90 shoppers) go through
the same command. Both are served best sellers: item-to-item is not measurably better than best
sellers in either store, and in one it is measurably worse (NDCG@10 0.295 against 0.380, difference
−0.085, range −0.137 to −0.033, 49 shoppers). The mock shoppers do not browse in themes, so there is nothing to learn, and the gate
says so. A demo meant to show a model winning needs mock shoppers who stay within themes.
Demo data, so these numbers never appear among the thesis's results.

## Tests

`tests/core/test_gate.py`, `test_retrain.py` and `test_store_events.py` (64 tests) use a made-up
store whose shoppers browse in themes (`tests/core/store_data.py`), so the right decision is
known. Among them:

- **The deliberately bad model is rejected and the live files are untouched**: random
  recommendations after a good model; the live model and fallback stay byte for byte.
- A model trained on shuffled product IDs, well formed but with the wrong links, is rejected by
  scoring (rule 3), which is the only check that can catch it.
- A model that gives up on half its pages beats best sellers but is measurably worse than the
  live model, and is rejected by rule 4.
- Each structural problem (a repeated product, the product itself, another store's product,
  wrong order, NaN scores, too many products, a crash, no answers at all, another store's model,
  a fallback with no answer) is caught.
- The same model retrained is accepted (difference exactly 0); a store keeps the kind of model
  that is live; live best sellers give way to a model; a tiny store and a store with only this week's events get best sellers; a quiet week keeps
  the live model; a corrupt live file or another store's model does not block a working one.
- The served file is byte for byte the judged one, and never saw the held-out week.
- Spreadsheet-rounded IDs (`8.3137E+12`, as in the first demo export) are refused when loading.
- The command retrains every store in an export, and one broken store fails the job without
  stopping the others.

## Considered instead

- **Accept only a measurably better model.** Freezes the first model, as above.
- **Compare averages with a margin** (the skeleton's setting). Chance decides when the store is
  small, as above.
- **Retrain on everything after passing.** Deploys a file nobody judged, as above.
- **A random held-out sample of events.** Leaks the future into training (CLAUDE.md); the held-out
  period is the newest week, as the evaluation's split is by time.
- **Pick the best model per store automatically** (train item-to-item, ALS and BPR, serve the
  winner). Choosing the winner on the same week the gate judges it on would flatter it, and ALS
  costs far more to train for every store on every run. Possible later with a separate week for
  choosing.
- **A pointer file or symbolic link to the live run instead of copies.** Symbolic links need
  administrator rights on Windows, and a pointer would change what the API reads. A copy renamed
  over the old file is just as atomic, and the API reads a `.joblib` as before.
- **Gate settings in `.env`.** The gate runs in the retraining job, not in the API, so its
  settings are the command's flags; `.env.example` now says so.

## Consequences

For Person B (core/api, infra, workflows):

- **The API should load `current_model_path(models_dir, store_id)`** and its `fallback_path`, both
  in `core/models/loading.py`, instead of one `MODEL_PATH` for all stores. A store with no folder
  yet has had no run.
- **A new model must be picked up without a restart.** The API caches the loaded model by path,
  so it needs to reload when the file changes (keyed on the file's modification time, say).
- **The job needs write access to the models folder.** `docker-compose.prod.yml` mounts it
  read-only into `core`, rightly. The job can use the same image with the folder mounted
  writable, e.g. a separate service run on a schedule:
  `python -m core.training.retrain --all --models-dir /models`. The image runs as uid 10001, which
  must be able to write there.
- **Reading the database needs `DATABASE_URL`**, and `get_settings` also requires
  `CORE_ADMIN_KEY` to be set, though the job does not use it.
- A run's exit code is 1 only if a store failed with an error. A rejected model is the gate
  working, not a failure.

Also:

- **Runs are never deleted.** On a Retailrocket-sized store an item-to-item run's folder is about
  40 MB and an ALS run's about 230 MB, plus the served copies. A rule for keeping the last few runs
  is needed before weekly runs pile up.
- **The reports are the thesis's evidence for the pipeline**: each run's decision, numbers and
  reason, in one file.

## Limits

- **One week, one metric.** A model better on some other measure or another week could be rejected.
- **30 shoppers is a rule of thumb.** A range from 30 re-drawn shoppers is still rough.
- **A week's delay**, as above.
- **The live model is never demoted.** If it slowly falls below best sellers while no new model
  passes, nothing notices. Each run could also score the live model against best sellers; not
  done.
- **It judges browsing.** As in 0006 and 0008, a hit means the shopper touched the product anyway;
  the gate rewards predicting that, not persuading anyone.
