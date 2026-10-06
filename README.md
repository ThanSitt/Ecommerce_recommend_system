# Ecommerce Recommendation Service

A self-hosted, platform-independent product recommendation service for online stores.
Stores send customer activity (views, cart additions, purchases) and request
recommendations back.

Bachelor's thesis project. The contribution is the **system around the model** — the
service, the connectors, automated retraining and deployment, and the evaluation.

## Architecture

```
store (Shopify / WooCommerce)
        │  platform-specific
        ▼
   connector  ──── standard event JSON ────▶  core API  ──▶  PostgreSQL
                                                 │              │
                                             Redis cache    training pipeline
                                                 │              │
                                                 ▼              ▼
                                           recommendations   model artefact
                                                             (deployment gate)
```

`core/` is platform-independent and contains no platform names. Platforms are handled
entirely by `connectors/`, which translate platform data into the standard event format
and call the core API like any other client. See [CLAUDE.md](CLAUDE.md).

### Event contract

Everything depends on this. All connectors produce it; the core service accepts nothing
else. Agreed in [docs/decisions/0001-event-contract.md](docs/decisions/0001-event-contract.md),
which records why each field is there.

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

```http
POST /v1/events
Authorization: Bearer <store api key>

[ { "user_id": "v-8812", "item_id": "123", "event_type": "view",
    "timestamp": "2026-10-03T09:14:22Z" } ]
```

Rules a connector must honour:

- **The store is never named in the payload.** The connector authenticates with an API
  key; the server resolves key → store and tags every event itself. A connector
  therefore cannot write into another store's data, and `item_id` cannot collide
  between stores.
- `user_id` is an anonymous identifier, never a name or email — and it must be **stable
  across all event types** from that connector. A platform that identifies browsers and
  buyers differently (Shopify does) must reconcile them in the connector, or user
  histories break at the purchase.
- `item_id` is the **product**, not the variant. Connectors collapse size/colour
  variants onto the product.
- A repeated `event_id` is discarded and the response is still a success. Webhooks
  retry on any non-2xx, so rejecting a duplicate with an error causes a retry loop.
- `order_id` appears only on `purchase` events.
- Ingestion always takes an array, even for a single event — the same path serves the
  dataset backfill and live webhooks.
- Unknown fields are ignored, so a field can be added without breaking a deployed
  connector.

The server records its own `received_at` in addition to `timestamp`. Evaluation uses
`timestamp`; `received_at` exists to detect store clock drift, which would otherwise
corrupt time-based splits invisibly.

### Where events come from

Neither platform delivers all three event types through one mechanism.

| Event      | Shopify                        | WooCommerce                      |
| ---------- | ------------------------------ | -------------------------------- |
| `view`     | Web Pixel (browser)            | plugin hook (server)             |
| `cart`     | Web Pixel, or cart webhook¹    | `woocommerce_add_to_cart`        |
| `purchase` | `orders/create` webhook        | `woocommerce_thankyou` / webhook |

¹ Cart webhooks deliver whole cart state, so the connector must diff to find what was
added.

Shopify has no view webhook — browsing never reaches its backend in a subscribable
form, so views require client-side code. WooCommerce runs inside the site and can
capture all three server-side, but its REST webhooks dispatch via WP-Cron and so arrive
late on low-traffic stores.

## Layout

| Path                 | Contents                                              |
| -------------------- | ----------------------------------------------------- |
| `core/api/`          | FastAPI service: event ingestion, recommendations     |
| `core/models/`       | Recommendation algorithms (baselines, item-to-item, ALS) |
| `core/training/`     | Training pipeline, serialisation, deployment gate     |
| `core/evaluation/`   | Metrics, time-based splits, baseline comparison       |
| `core/data/`         | Loaders mapping raw datasets into the event schema    |
| `connectors/`        | Shopify and WooCommerce connectors                    |
| `infra/`             | Dockerfiles, compose, Prometheus, Grafana             |
| `.github/workflows/` | CI/CD and scheduled retraining                        |
| `notebooks/`         | Exploration only — never the production path          |
| `docs/`              | Architecture notes, decision records, API docs        |

## Status

Skeleton only — the directory tree and project config are in place, no code written yet.
Directories are held by `.gitkeep` files; delete each one as real files land.

## Getting started

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on Unix
pip install -r requirements-dev.txt
cp .env.example .env        # then fill in values
```

## Commands (target interface)

```bash
docker compose -f infra/docker/docker-compose.yml up   # full stack locally
pytest                                                 # tests
python -m core.training.train --dataset retailrocket   # train
python -m core.evaluation.run --model als              # evaluate
```

## Data setup

Datasets are not committed — they are large and `data/` is gitignored.

1. Download the Retailrocket dataset:
   https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
2. Place `events.csv` and `category_tree.csv` in `data/raw/`
   (`item_properties_*.csv` are not needed)

**Retailrocket** produces every reported accuracy number. **Online Retail II** (UCI) or
generated data populate demo stores only — accuracy measured on it is meaningless and
must never appear in results.

Note that `events.csv` uses the event names `view`, `addtocart`, `transaction` and
millisecond timestamps. Mapping those to the service's event contract belongs in
`core/data/`, so the raw vocabulary never reaches the core service.

### Test fixtures

`tests/fixtures/` holds a committed sample (710 kB) so tests and CI need neither the
90 MB download nor Kaggle credentials. Regenerate with:

```bash
python scripts/make_fixture.py
```

It samples whole visitor histories rather than the first N rows — a row-prefix sample
truncates every user's history at an arbitrary timestamp, which makes tests of
time-based splitting meaningless — and over-represents purchasers, since transactions
are only 0.8% of events.

## Division of work

- **Person A** — `core/models`, `core/training`, `core/evaluation`, `core/data`
- **Person B** — `core/api`, `connectors/`, `infra/`, `.github/workflows/`

The handover is the saved model file plus a documented load-and-predict interface, to
live in `core/models/`. Once defined, keep it stable — changing it blocks the other
person.
