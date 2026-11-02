import type { LoaderFunctionArgs } from "react-router";
import { authenticate } from "../shopify.server";

// GET shop.com/apps/recs?item_id=123 arrives here through the App Proxy.
// The Core API key is added here, on the server, so the browser never sees it.
export const loader = async ({ request }: LoaderFunctionArgs) => {
  // Rejects requests whose signature doesn't match (not forwarded by Shopify).
  await authenticate.public.appProxy(request);

  const itemId = new URL(request.url).searchParams.get("item_id");
  if (!itemId) {
    return Response.json({ error: "item_id is required" }, { status: 400 });
  }

  const url = new URL("/v1/recommendations", process.env.CORE_API_URL);
  url.searchParams.set("item_id", itemId);

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${process.env.CORE_API_KEY}` },
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) {
    console.error(`Core API returned ${response.status} for item ${itemId}`);
    return Response.json({ items: [] }, { status: 502 });
  }

  return Response.json(await response.json());
};
