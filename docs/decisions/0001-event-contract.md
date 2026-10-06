# 0001 — The event contract

**Status:** Accepted, 2026-10-03
**Decided by:** Person A and B
**Affects:** `core/api`, `core/data`, both connectors — every component in the system

## Context

The event contract is the single interface between connectors and the core service. All
connectors produce it; the core accepts nothing else. It is what makes the core
platform-independent, so it is also what the thesis's platform-independence claim rests
on. Getting it wrong is expensive: a change means a coordinated redeploy of the core and
both connectors.

The first draft had four fields — `user_id`, `item_id`, `event_type`, `timestamp`.
Reviewing it against the Retailrocket dataset and against how platform webhooks actually
behave surfaced three gaps, all of which are invisible when working with the dataset and
only appear once real stores are connected.

## Decision

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

With these rules:

1. Store identity is resolved server-side from the API key, never sent in the payload.
2. A repeated `event_id` is discarded and the response is still a success.
3. `order_id` appears only on `purchase` events.
4. Unknown fields are ignored, not rejected.
5. Ingestion always takes an array, even for a single event.
6. The endpoint is versioned in the path: `POST /v1/events`.
7. The server records `received_at` itself, in addition to the supplied `timestamp`.

## Why each part

### The four original fields are kept unchanged

They map onto Retailrocket losslessly: `visitorid` → `user_id`, `itemid` → `item_id`,
`event` → `event_type` (renaming `addtocart` → `cart` and `transaction` → `purchase`),
and the millisecond epoch timestamp → ISO 8601. No contract field goes unfilled and no
dataset field is distorted. The mapping lives in `core/data/`, so the dataset's own
vocabulary never reaches the core service.

### `order_id`, optional

Retailrocket has a `transactionid` column that the four-field contract discarded. In the
dataset, 2,710 of 17,672 orders (15%) contain more than one item, the largest containing
31. Without a grouping field, a three-item basket is indistinguishable from three
separate purchases made minutes apart, which rules out co-purchase recommendations
("frequently bought together") — one of the most commercially recognisable
recommendation types.

The grouping is also available for free at the connector boundary: a Shopify order
webhook arrives as one order containing line items, so flattening it loses information
the connector already held.

**Considered instead:** omitting it, on the grounds that the project's scope is
item-to-item on views and ALS, neither of which needs baskets. Rejected because the field
is optional and costs nothing, while retrofitting it later would mean reprocessing
ingested data that no longer carries the grouping.

### `event_id`, optional, with duplicates returning success

Webhook delivery is at-least-once. Shopify re-delivers if the endpoint times out or
returns any non-2xx status. Without an event identity, a re-delivered order silently
inflates that item's purchase count — and purchases are only 0.8% of Retailrocket events,
so a small number of duplicates measurably distorts the most-popular baseline. The source
dataset already contains 460 rows that are exact duplicates on
(visitor, item, event, millisecond), so the problem is not hypothetical even before
webhooks are involved.

Returning success rather than a conflict status is the critical detail: an error response
makes the platform retry, so rejecting a duplicate with `409` produces a retry loop that
never terminates.

**Considered instead:** deduplicating server-side on
(store, user, item, event_type, timestamp). Rejected because it is a guess — two genuine
views of the same product within the same second are legitimate — and because it pushes
the cost onto every read path rather than resolving it once at ingestion.

### Store identity from the API key, not the payload

The service is multi-store by design, and `item_id: "123"` denotes a different product in
every shop. Without scoping, two stores' catalogues merge into one model and
recommendations cross-contaminate. This is undetectable while evaluating on Retailrocket,
which is a single store, and appears for the first time in the demo — where a Shopify
store and a WooCommerce store must run against one core service simultaneously, since
that is the only way to demonstrate platform independence.

Resolving the store from the API key means a misconfigured or compromised connector
cannot write into another store's data, and keeps the payload minimal.

**Considered instead:** a `store_id` field in the event. Rejected because a client-supplied
tenant identifier is spoofable, and because it would be repeated on every one of millions
of records to carry information the authenticated connection already establishes.

### `timestamp` requires a UTC offset

ISO 8601 permits a local time with no zone, e.g. `2015-06-02T05:02:12`. Stores are in
different timezones and the evaluation splits by time, so zone-less timestamps would
misorder events across stores with no visible symptom. Time-ordering errors are the most
common methodological fault in this kind of work, so the contract rejects anything
ambiguous rather than guessing a zone.

### `received_at`, recorded server-side

A store's clock can be wrong, and a connector sending systematically skewed timestamps
would corrupt the time-based splits silently — producing plausible-looking metrics derived
from a mis-ordered history. Evaluation continues to use `timestamp`, which is when the
event actually happened; `received_at` exists so the skew can be detected. It is not part
of the contract, because connectors neither send nor see it.

### Arrays always, and a versioned path

The Retailrocket backfill is 2.7M events while demo stores send one at a time. A single
endpoint accepting a list serves both, avoiding a second bulk-ingestion path later. Path
versioning costs nothing now and is what allows the contract to change later without a
simultaneous redeploy of the core and both connectors.

### Unknown fields ignored

A deployed connector keeps working when the contract gains a field. This is Pydantic's
default behaviour, recorded here so it is a decision rather than an accident.

## Consequences

- Both connectors must generate an `event_id` per event and supply `order_id` on
  purchases. Neither is required for a connector to function, so a first version can skip
  both.
- The core needs a store table, API key issuance, and a uniqueness constraint on
  (store, `event_id`).
- `core/data/` loaders emit this format with both optional fields absent, so dataset
  ingestion and live webhook ingestion share one code path. This matters for the thesis:
  the system that was measured is the system that was demonstrated.

## Open, deliberately

`user_id` is kept as the primary subject of an event, but the dataset shows 71% of
visitors produce exactly one event and 91% complete all activity within 24 hours. User
histories are therefore too short for personalisation in the large majority of traffic,
and the dominant serving path is likely to be item-to-item rather than user-based. This
does not change the contract — `user_id` works as an anonymous visitor identifier — but it
shapes which recommendation endpoints matter, which is decided separately.
