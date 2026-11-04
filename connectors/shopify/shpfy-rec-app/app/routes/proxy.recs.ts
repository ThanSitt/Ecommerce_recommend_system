import type { LoaderFunctionArgs } from "react-router";
import prisma from "../db.server";
import { authenticate } from "../shopify.server";

// GET shop.com/apps/recs?item_id=123 arrives here through the App Proxy.
// The shop's core API key is added here, on the server, so the browser never sees it.
export const loader = async ({ request }: LoaderFunctionArgs) => {
  // Rejects requests whose signature doesn't match (not forwarded by Shopify).
  await authenticate.public.appProxy(request);

  const params = new URL(request.url).searchParams;
  const itemId = params.get("item_id");
  if (!itemId) {
    return Response.json({ error: "item_id is required" }, { status: 400 });
  }

  // Shopify adds `shop` to every proxied request, and it is covered by the signature
  // checked above, so it can be trusted (unlike a shop name sent by the browser).
  const shop = params.get("shop");
  const registered = shop
    ? await prisma.shop.findUnique({ where: { shop } })
    : null;
  if (!registered) {
    // Installed, but registering with the core failed (it retries on the next
    // afterAuth). The visitor just sees no recommendations.
    console.error(`no core store for ${shop}; showing no recommendations`);
    return Response.json({ items: [] });
  }

  const url = new URL("/v1/recommendations", process.env.CORE_API_URL);
  url.searchParams.set("item_id", itemId);

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${registered.coreApiKey}` },
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) {
    console.error(`Core API returned ${response.status} for item ${itemId}`);
    return Response.json({ items: [] }, { status: 502 });
  }

  return Response.json(await response.json());
};
