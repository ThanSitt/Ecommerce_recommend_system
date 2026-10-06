# Shopify event sources — investigation

**Owner:** Person B
**Goal:** decide where the Shopify connector gets each event in the contract from, and how.
**Output:** a filled-in mapping table (section 4) → becomes `docs/decisions/0002-shopify-event-sources.md`.

---

## 1. What we need

Every event the connector sends to `POST /v1/events` must look like this
(see `docs/decisions/0001-event-contract.md`):

| Field | Meaning | Why the system needs it |
|---|---|---|
| `user_id` | anonymous visitor ID | links one person's events together — item-to-item and ALS depend on it |
| `item_id` | product ID | the thing being recommended |
| `event_type` | `view` / `cart` / `purchase` | purchases weigh more than views |
| `timestamp` | when it happened, with timezone | time-based evaluation split, "recently popular" |
| `event_id` *(optional)* | unique ID of this event | duplicates are dropped by the core |
| `order_id` *(optional, purchases only)* | Shopify order ID | groups items bought together |

The same data feeds every algorithm (baselines, item-to-item, ALS). Only the algorithm
changes — so this investigation does not depend on which model wins.

---

## 2. The three Shopify sources

| | Web Pixels | Webhooks | Admin API |
|---|---|---|---|
| What it is | JS running in the shopper's browser, sends us events | Shopify's server POSTs to our connector when something happens | we query Shopify whenever we want |
| `view` | `product_viewed` | — none | — none |
| `cart` | `product_added_to_cart` | `carts/update` (no visitor ID) | — |
| `purchase` | `checkout_completed` | `orders/create` | `orders` query (history only) |
| Visitor ID | `clientId` — same across all pixel events | customer ID, or nothing for guests | customer ID, or nothing |
| Reliability | can be blocked (ad-blockers, cookie consent) | reliable, Shopify retries | reliable |
| Use for | **live events** | maybe purchase backup | **backfill + product catalogue** |

*Written from memory — every cell must be verified in section 5.*

---

## 3. The duplicate problem: same purchase from two sources

If a purchase comes from **both** the pixel and the webhook, how do we know it is the same
purchase?

### How matching works

Both sources carry the **Shopify order ID** and the **product ID**. The connector builds the
`event_id` from those, the same way in both handlers:

```
event_id = "purchase:{order_id}:{variant_id}"
```

- Pixel `checkout_completed` for order 1001, variant 55 → `purchase:1001:55`
- Webhook `orders/create` for order 1001, variant 55 → `purchase:1001:55`

(Variant, not product: two variants of the same product in one order must not collide.)

The core already drops a repeated `event_id` (unique per store), so the second one is
discarded. **No extra logic in the core.** The trick is that the ID is *built from data both
sources share*, not randomly generated.

> Views and carts only come from the pixel, so they cannot be duplicated across sources.
> For them, the pixel's own event `id` is enough as `event_id`.

### The catch: `user_id` differs

| | Pixel | Webhook |
|---|---|---|
| `user_id` | `clientId` (e.g. `abc-123`) | customer ID or **nothing** |

Whichever arrives first is kept. If the webhook wins, the purchase gets a different
`user_id` than that visitor's views — the model can no longer link "viewed A → bought A",
and for guest checkouts there is no user at all.

### Options

| Option | How | Pro | Con |
|---|---|---|---|
| **A. Pixel only** | all three event types from the pixel | one source, one `user_id`, no duplicates, simplest | loses purchases when the pixel is blocked |
| **B. Pixel + webhook backup** | webhook handler waits a few minutes; forwards only if the pixel did not already send that order | catches blocked purchases | connector must store which orders it has seen; webhook purchase may have wrong/no `user_id` |
| **C. Pixel + webhook, linked** | store `checkout token → clientId` from the pixel; webhook looks it up to get the right `user_id` | reliable *and* correct `user_id` | most work; only works if the token appears in both — verify |

**Leaning:** start with **A**. Count how many orders the webhook sees vs. how many
purchases the pixel sends — that number tells us whether B/C is worth it, and it is a
result for the thesis.

---

## 4. Mapping table (fill in)

Real captured payloads live in `tests/fixtures/shopify/` — check there before trusting this table.

| Contract field | `view` — pixel `product_viewed` ✅ | `cart` — pixel `product_added_to_cart` ✅ | `purchase` — pixel `checkout_completed` ✅ |
|---|---|---|---|
| `user_id` | `clientId` | `clientId` | `clientId` |
| `item_id` | `data.productVariant.product.id` | `data.cartLine.merchandise.product.id` | `data.checkout.lineItems[].variant.product.id` — **one event per line item** |
| `timestamp` | `timestamp` | `timestamp` | `timestamp` |
| `event_id` | `id` | `id` | built: `purchase:{order.id}:{variant.id}` (see section 3) |
| `order_id` | — | — | `data.checkout.order.id` |

### Findings so far

**Pixel `product_viewed`** — captured 2026-10-06 from a dev store via a custom pixel → webhook.site.
Fixture: `tests/fixtures/shopify/pixel_product_viewed.json`

- `clientId` present (UUID, e.g. `6dce103d-…`). Still to check: is it the same in cart and checkout events, and across page reloads?
- Product ID and variant ID are both present, as **plain numeric strings** (`"8313705169069"`), not GIDs → no normalisation needed for pixel events. (Webhooks / Admin API may use other formats — check.)
- A product without options still has one variant, titled `"Default Title"`.
- `timestamp` is UTC with `Z` (`2026-10-05T23:51:35.903Z`) → valid for the contract as-is.
- `id` (`sh-…`) is unique per event → usable as `event_id`.
- Payload also carries title, price, vendor, URL, image — not needed for the contract, but available if a catalogue feature is added later.
- `context` contains page URL, referrer, browser language and user agent — not needed; the connector must **not** forward these (no personal data in the core).

**Pixel `product_added_to_cart`** — captured 2026-10-06, same browser session.
Fixture: `tests/fixtures/shopify/pixel_product_added_to_cart.json`

- ✅ **Same `clientId` as the view event** (`6dce103d-…`) → view and cart link to the same visitor.
- Product ID sits at a **different path** than in `product_viewed`: `data.cartLine.merchandise.product.id` (`merchandise` = the variant). The connector needs one extractor per event type.
- IDs again plain numeric strings; `timestamp` again UTC `Z`; `id` again unique `sh-…`.
- Payload has `quantity`. The contract has no quantity — adding 3 of an item becomes one `cart` event. Fine for implicit feedback; note it as a deliberate choice.
- `seq` is `1` here too → `seq` counts per page load, not per visitor. Not usable for ordering; use `timestamp`.

**Pixel `checkout_completed`** — captured 2026-10-06, same browser session, 3-item order paid via Shopify Payments test mode.
Fixture: `tests/fixtures/shopify/pixel_checkout_completed.json` (email replaced with `test-buyer@example.com`; address fields are the fake test input)

- ✅ **Same `clientId` again** (`6dce103d-…`) → **view, cart and purchase all link to one visitor.** The pixel alone delivers a consistent `user_id` across all three event types.
- ✅ **Order ID present**: `data.checkout.order.id` = `"18804001407149"` (plain numeric).
- ✅ **All line items present**: 3 products in `lineItems[]`. One order → 3 `purchase` events sharing one `order_id`.
- **Also present, useful for linking to webhooks later (option C):**
  - `data.checkout.token` (checkout token)
  - `data.checkout.order.customer.id` (Shopify customer ID — what webhooks / Admin API use)
  → the pixel event links `clientId` ↔ `order.id` ↔ `customer.id` ↔ checkout `token` in one payload.
- `lineItems[].id` equals the **variant ID**, not a unique line ID.
- `event_id` must be **built**, not taken from `id`: the pixel's `id` is one per checkout, but we emit one event per line item, and a webhook for the same order must produce the same IDs. Use `purchase:{order.id}:{variant.id}` — variant rather than product, so two variants of the same product in one order do not collide and get dropped as duplicates.
- The event `id` format differs from the storefront events (plain UUID here, `sh-…` there) → treat it as an opaque string.
- `quantity: 2` on the shirt → still one `purchase` event (no quantity in contract, same as cart).
- ⚠️ **This payload contains personal data**: email, billing and shipping address, phone, marketing consent. The connector must extract only the contract fields and **never forward or log the rest**. `customer.id` must not be used as `user_id` either (it identifies a real customer account; `clientId` is the anonymous one).

Decisions to record next to the table:

- [ ] `item_id` = product ID or variant ID? (Retailrocket items are product-level → probably product ID)
- [ ] Do the IDs come as plain numbers (`1001`) or GIDs (`gid://shopify/Order/1001`)? Pick one form and normalise in the connector.
- [ ] Option A, B or C for purchases?

---

## 5. To-do checklist

### Web Pixels
- [ ] Read the standard events reference: `product_viewed`, `product_added_to_cart`, `checkout_completed`
- [ ] Does every event have `clientId`, `id` and `timestamp`? Does the timestamp include a timezone?
- [ ] Where is the product ID in each event? Product, variant, or both?
- [ ] `checkout_completed`: does it contain the **order ID** and **all line items**, *and* the `clientId`?
- [ ] Does it contain a checkout **token** (needed for option C)?
- [ ] Is `clientId` stable for a visitor across page loads and days?
- [ ] Cookie consent: what happens when a visitor declines? (EU stores)
- [ ] How does the pixel send data to our connector? (`fetch` / `sendBeacon` from the sandbox; any URL restrictions?)

### Webhooks
- [ ] `orders/create` vs `orders/paid` — which one, and why?
- [ ] Which fields: order ID, line items with product ID, `created_at`, customer, checkout/cart token?
- [ ] For a guest checkout: is there any visitor identifier at all?
- [ ] `carts/update`: is there any visitor identifier? (probably not → rule out for `cart`)
- [ ] HMAC verification and retry behaviour (how often, how long)
- [ ] Is a webhook payload delivered more than once? (should be — this is why `event_id` exists)

### Admin API
- [ ] `orders` query: how far back without extra permission? (~60 days; `read_all_orders` for older)
- [ ] Do backfilled orders have the same IDs as the webhook/pixel, so `event_id` matches?
- [ ] Backfilled orders have no `clientId` — accept that history has customer IDs only?
- [ ] `products` query + `products/update` / `products/delete` webhooks: note what exists (needed later to hide deleted / out-of-stock items, not part of the contract yet)

### Wrap-up
- [ ] Fill in the mapping table (section 4)
- [ ] Choose option A / B / C, with reasoning
- [ ] Write `docs/decisions/0002-shopify-event-sources.md`: what was chosen, what else was considered, why
- [ ] Share with Person A — the `user_id` and `item_id` choices affect the models

---

## 6. Open questions for Person A

- Are purchases from history (customer ID, no `clientId`) useful for training, or do they only pollute the user matrix?
- Product ID vs variant ID — any preference from the modelling side?
