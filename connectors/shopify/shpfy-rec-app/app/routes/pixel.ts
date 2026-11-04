import type { ActionFunctionArgs } from "react-router";
import prisma from "../db.server";
import { sendEvents } from "../core-client.server";
import {
  transform,
  type LineItem,
  type PixelEvent,
} from "../transform.server";

// POST /pixel — the Web Pixel in the shopper's browser sends each event here.
// Ported from the Python connector (connectors/shopify/app/main.py).
//
// Body: { shop: "store-a.myshopify.com", event: <pixel event> }
//
// An `action` runs only on the server, so the shop's core API key never reaches the
// browser.
//
// Public on purpose: browsers must reach it. So `shop` can be forged; the worst a forger
// can do is send events into a shop that installed the app (a known limitation, see
// docs/investigations/shopify-event-sources.md). Unknown shops are ignored.

// The pixel runs on the store's domain, so its requests are cross-origin. Without this
// header the browser blocks them before they reach us.
const CORS = { "Access-Control-Allow-Origin": "*" };

// Always 200: the pixel cannot do anything useful with an error, and every failure is
// logged here instead.
const ok = () => Response.json({ status: "received" }, { headers: CORS });

export const action = async ({ request }: ActionFunctionArgs) => {
  // The pixel sends JSON as text/plain, a "simple" request the browser sends without a
  // CORS pre-check. Answer a pre-check anyway, in case a browser sends one.
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        ...CORS,
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Content-Type",
      },
    });
  }

  let body: { shop?: unknown; event?: PixelEvent };
  try {
    body = JSON.parse(await request.text());
  } catch {
    return Response.json(
      { error: "body is not valid JSON" },
      { status: 400, headers: CORS },
    );
  }

  const { shop, event } = body;
  if (typeof shop !== "string" || !event) {
    console.error("pixel request without shop or event; ignored");
    return ok();
  }

  const registered = await prisma.shop.findUnique({ where: { shop } });
  if (!registered) {
    console.error(`pixel event from unknown shop ${shop}; ignored`);
    return ok();
  }

  let events;
  try {
    events = transform(event);
  } catch (error) {
    // A payload missing a field we expect (e.g. Shopify changed its format).
    console.error(`could not transform ${event.name} from ${shop}`, error);
    return ok();
  }

  // Contract events only: the raw pixel event can contain personal data.
  console.log(
    `pixel event from ${shop}: ${event.name} -> ${events.length} contract event(s)\n` +
      `products: ${JSON.stringify(productTitles(event))}\n` +
      JSON.stringify(events, null, 2),
  );

  await sendEvents(registered.coreApiKey, events);
  return ok();
};

/** Product names, for reading the log only. Deliberately not part of the contract
 * events: the core never needs them. */
function productTitles(event: PixelEvent): (string | undefined)[] {
  const data = event.data;
  switch (event.name) {
    case "product_viewed":
      return [data.productVariant.product.title];
    case "product_added_to_cart":
      return [data.cartLine.merchandise.product.title];
    case "checkout_completed":
      return data.checkout.lineItems.map((line: LineItem) => line.title);
    default:
      return [];
  }
}
