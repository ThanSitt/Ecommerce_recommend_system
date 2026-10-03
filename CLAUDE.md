# CLAUDE.md

Context for working in this repository.

## What this is

A self-hosted, platform-independent product recommendation service for online stores. Stores send customer activity (views, cart additions, purchases) and request recommendations back. Built as a bachelor's thesis by two students over 12 weeks.

The thesis is about the **system around the model**, not the model itself. The recommendation algorithm is a library call; the contribution is the service, the connectors, the automated retraining and deployment, and the evaluation. Weight effort accordingly — do not gold-plate the model at the expense of the pipeline.

## The one architectural rule

**The core service must never contain platform-specific code.** No Shopify, WooCommerce, or any other platform name should appear anywhere under `core/`. Platforms are handled entirely by connectors, which translate platform data into the standard event format and call the core service's API like any other client.

This is not a style preference. Platform independence is a central claim of the thesis and is evaluated directly by measuring how much work the second connector required and whether it forced changes to the core. If platform logic leaks into the core, the claim fails and so does part of the evaluation.

When tempted to add a platform-specific field or special case to the core, extend the connector instead, or change the standard event format for everyone.

## The event contract

Everything depends on this. All connectors produce it; the core service accepts nothing else.

```json
{
  "event_id": "string, optional — connector-generated, for idempotency",
  "user_id": "string",
  "item_id": "string",
  "event_type": "view" | "cart" | "purchase",
  "order_id": "string, optional — groups a multi-item purchase",
  "timestamp": "ISO 8601, UTC offset required"
}
```

Rules that travel with the schema:

- **Store identity is never in the payload.** The connector authenticates with an API key; the server resolves key → store and tags every event itself. A connector therefore cannot write into another store's data, and `item_id` cannot collide between stores.
- `user_id` is an anonymous identifier, never a name or email. The system requires no personal data — keep it that way.
- A repeated `event_id` (unique per store) is **discarded with a success response**. Webhooks retry on any non-2xx, so returning an error for a duplicate causes a retry loop.
- `order_id` appears only on `purchase` events.
- Unknown fields are ignored, not rejected, so a field can be added without breaking a deployed connector.
- Ingestion always takes an array, even for one event. The dataset backfill and live webhooks use the same path.
- The endpoint is versioned in the path: `POST /v1/events`.
- The server records its own `received_at` alongside the supplied `timestamp`. Evaluation uses `timestamp`; `received_at` exists to detect store clock drift, which would otherwise corrupt time-based splits invisibly.

Both optional fields are genuinely optional: a connector that cannot supply them still works, and the Retailrocket loader omits both.

Changing this schema affects both halves of the project and both connectors. Treat changes as a deliberate joint decision, not a local edit. The reasoning is in `docs/decisions/0001-event-contract.md`.

## Structure

```
core/              # Platform-independent. The thesis deliverable.
  api/             # FastAPI service: event ingestion, recommendation endpoints
  models/          # Recommendation algorithms (baselines, item-to-item, ALS)
  training/        # Training pipeline, model serialisation
  evaluation/      # Metrics, splits, baseline comparison
  data/            # Loaders that map raw datasets into the event schema
connectors/
  shopify/         # Shopify custom app (OAuth, webhooks, storefront display)
  woocommerce/     # Second platform connector
infra/
  docker/          # Dockerfiles, docker-compose.yml
  monitoring/      # Prometheus and Grafana configuration
.github/workflows/ # CI/CD: test, build, deploy; scheduled retraining
notebooks/         # Exploration only. Nothing here is production code.
tests/
docs/              # Architecture notes, decision records, API documentation
```

## Stack

Python 3.11, FastAPI, PostgreSQL, Redis, the `implicit` library for ALS and BPR, Docker and Docker Compose, GitHub Actions, Prometheus and Grafana, pytest, Locust for load testing. Deployed to a single Azure VM.

Deliberately not used: Kubernetes (complexity beyond this project's scale), Airflow (GitHub Actions scheduled jobs are sufficient). These were considered and rejected; the reasoning belongs in the thesis, so do not quietly introduce them.

## Data rules

**Never commit datasets or model files.** Both are large. `data/` and `models/` artefacts stay gitignored; document how to obtain them instead.

Two datasets, two purposes, never mixed:

- **Retailrocket** (Kaggle) — real behaviour, all three event types, anonymised item IDs. This is what every reported accuracy number comes from.
- **Online Retail II** (UCI) or generated store data — real product names, used to populate test stores for the demo. Accuracy measured on this is meaningless and must never appear in results.

Retailrocket is extremely sparse: ~2.76M events across ~1.4M visitors, of which only ~22.5k are purchases. Aggressive filtering (minimum interactions per user and per item) is required, and the filtering choices materially affect results — so they belong in the thesis, not buried in a script.

## Evaluation rules

- **Split by time, never randomly.** A random split leaks future information and inflates every metric. This is the most common error in this kind of work.
- **Baselines are mandatory**: random and most-popular. A model that cannot beat most-popular is a finding worth reporting, not a bug to hide.
- Exclude items a user has already interacted with from scored recommendations.
- Core metrics: Precision@10, Recall@10, NDCG@10. Catalogue coverage is a useful addition.
- Hold out a final test period that is never touched during tuning.

## Division of work

**Person A** owns `core/models`, `core/training`, `core/evaluation`, `core/data`.
**Person B** owns `core/api`, `connectors/`, `infra/`, `.github/workflows/`.

The handover between them is the saved model file plus a documented load-and-predict interface. Keep that interface stable; changing it blocks the other person.

## Commands

```bash
docker compose -f infra/docker/docker-compose.yml up   # full stack locally
pytest                                                 # tests
python -m core.training.train --dataset retailrocket   # train
python -m core.evaluation.run --model als              # evaluate
```

## Conventions

- Write the test with the code, not afterwards. CI runs pytest and blocks deployment on failure.
- Record architectural decisions in `docs/decisions/` as short notes: what was chosen, what else was considered, why. These become the thesis's justification chapters directly — writing them later from memory wastes time and loses detail.
- Keep notebooks out of the production path. Explore there, then move anything that works into `core/`.
- Model files are versioned by training date. The service loads the current one by path; it does not pick versions itself.

## Model deployment gate

A newly trained model is not accepted automatically. Before replacing the live model it must beat a configured threshold relative to the current one on a held-out set, and produce structurally valid output. A model can be technically loadable and still much worse than what it replaces — catching that is the point of the gate, and it is one of the more interesting parts of the thesis, so keep it meaningful rather than a formality.
