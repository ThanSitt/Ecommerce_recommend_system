# 0007 — Store registration and API keys

**Status:** Accepted, 2026-11-04
**Decided by:** Person B (reviewed by Person A)
**Affects:** `core/api`, `core/db`, `connectors/shopify/shpfy-rec-app`

## Context

Contract rule 1 (0001) says the store is resolved from the API key, never sent in the
payload. Until now that was a placeholder: the core accepted any key listed in the
`CORE_API_KEYS` environment variable and mapped every one of them to a single store,
`"dev"`. One connector, one hand-copied key, one store.

Platform independence is demonstrated by running several stores — on different platforms —
against one core at the same time. That needs a real answer to three questions: how a store
comes into existence, how it gets its key, and how a request is tied to exactly one store.

## Decision

1. **A store is created by the connector, through the core's API.** When a Shopify shop
   installs the app, the app calls `POST /v1/stores`. The request has no body: the core
   needs nothing about the store, so no platform information reaches it.
2. **The core creates the store's ID and key itself** and returns both once
   (`201 { store_id, api_key }`, with `Cache-Control: no-store`). It saves only the SHA-256
   hash of the key.
3. **Creating stores requires a separate admin key** (`CORE_ADMIN_KEY`), shared only by the
   core and the connector's server. It can create stores and nothing else.
4. **Every other request is tied to a store by its key.** The core hashes the presented key,
   looks it up in the `stores` table, and uses that row's `store_id`. An unknown key and the
   key of an inactive store get the same `401`.
5. **The connector keeps the mapping from platform to core.** The Shopify app stores each
   shop's domain, `store_id` and key in its own `Shop` table. The core's `stores` table has
   no platform columns.
6. **The Shopify pixel is an app extension**, switched on per shop by the app, rather than
   code each merchant pastes into their store's admin.

## Why each part

### The store comes from the key, never from the request body

A `store_id` is an identifier, not a secret: it appears in logs, error messages, support
conversations and, eventually, dashboards. If the core trusted a `store_id` in the body, any
holder of any valid key — including the owner of a legitimately installed shop — could write
into another store by naming it. With the store derived from the key, knowing another
store's ID gives nothing; writing into it requires that store's key, which exists only in
that shop's row of the connector's database.

The ID and the key are separate columns so a leaked key can be replaced without renaming the
store or orphaning its events.

**Considered instead:** a `store_id` field in the event. Rejected in 0001 for the same
reason; recorded again here because it is the property this design exists to provide.

### The core issues keys; only their hashes are stored

Keys are created with `secrets.token_urlsafe(32)` (32 random bytes) on the core's server and
travel once, server to server, to the connector. A copy of the core's database therefore
contains no working keys.

SHA-256 rather than a slow password hash (bcrypt, argon2): those exist to protect short,
human-chosen passwords from guessing. A 32-byte random key cannot be guessed from its hash,
and a fast hash keeps the lookup on every request a single indexed query. For the same
reason the lookup needs no constant-time comparison: an attacker cannot steer the hash of a
guess towards a stored hash one character at a time.

**Considered instead:** keeping the environment-variable key list. Rejected because it
supports one store per manually copied key, cannot express which store a key belongs to,
and requires a restart to add a store.

### A separate admin key for creating stores

Store keys and the admin key do different jobs. A store key sends events and gets
recommendations for one store; the admin key only creates stores. If the admin key leaks,
the damage is limited to new, empty stores — no endpoint returns an existing store's key, so
existing data cannot be read or written with it. It is rotated by changing it in the core's
and the connector's environment.

### Registration on install, in the connector

The Shopify app registers a shop in its `afterAuth` hook, which runs on install and on every
token renewal. The `Shop` table decides whether the shop is new: if a row exists, nothing
is sent to the core.

`Shop` is separate from the Shopify library's `Session` table because the library rewrites
`Session` on every token renewal and deletes it on uninstall. Keeping the core mapping in a
table we own means a reinstall keeps the same `store_id` and its event history.

Registration never throws. If the core is unreachable or answers with an error, nothing is
saved and the merchant's login continues; the next `afterAuth` retries. Two concurrent
installs of the same shop can both create a core store; the second `Shop` insert fails on
the primary key and its store is left unused. This is rare and harmless, and preferable to
failing a login.

### The pixel as an app extension

A pasted custom pixel has to be added by hand in each shop's admin, cannot be updated
centrally, and has no per-shop configuration. As a web pixel app extension, the code is
deployed with the app and runs on every shop that installs it. The app switches it on per
shop (`webPixelCreate`) when the merchant opens the app, which Shopify does right after
install.

The pixel sends to the app's server, whose address it reads from a pixel setting, `appUrl`.
The value is the same for every shop. It is a setting rather than part of the code because
the address changes: on every restart in development (a new tunnel), and on a server move in
production. The app compares and updates it each time the app is opened, so a change does
not require redeploying the pixel.

Privacy: the pixel declares that it needs **analytics** consent only — recording views,
carts and purchases to build recommendations. It declares no marketing or preferences
purpose and `sale_of_data = "disabled"`: the data goes only to this service, for that store.
Shopify's defaults (marketing and sale of data required) would drop events from every
visitor who declines marketing, a purpose the system does not have.

### Which shop sent an event

The widget's requests come through Shopify's App Proxy, which adds the shop's domain and
signs the request, so the shop is trusted. Pixel requests go directly to the app and carry
the shop's permanent domain (`init.data.shop.myshopifyDomain`) in the body. That value can be
forged — the endpoint must be public for browsers to reach it. The worst case is fake events
in a shop that has installed the app; events naming an unknown shop are ignored. This is
inherent to client-side tracking and is a limitation, not a defect.

**Considered instead:** sending pixel events through the App Proxy too, which would sign the
shop and remove the `appUrl` setting. Postponed until it is verified that a pixel, which runs
in Shopify's sandbox and also on checkout pages, can reach the proxy.

## Consequences

- `CORE_API_KEYS` is removed. Events stored earlier under the `"dev"` store belong to no store.
- Each connector needs `CORE_ADMIN_KEY` and its own store table mapping its platform's
  stores to core `store_id`s and keys. A second connector (WooCommerce) follows the same
  pattern without changes to the core.
- The connector's `Shop` table holds every store's key, so it must move from SQLite to a
  protected database before deployment, and traffic between connector and core must use
  HTTPS.
- The Python connector that received pixel events is replaced by the `/pixel` route in the
  Shopify app.

## Not done yet

- **Uninstall:** a shop that uninstalls should have its store set inactive, and a reinstall
  should reactivate the same store.
- **Key rotation:** replacing a single store's key is possible by design but has no endpoint.
- **Recommendations per store:** events are stored per store, but training and serving still
  use one model for all stores. Decided separately.
