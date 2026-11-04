import { Prisma } from "@prisma/client";
import type { AdminApiContext } from "@shopify/shopify-app-react-router/server";
import prisma from "./db.server";

/**
 * Makes sure a shop has a store in the core, and remembers that store's id and key.
 *
 * Called from afterAuth, which runs on install and again on every token renewal, so
 * most calls find the Shop row and stop there.
 *
 * Never throws: a failure here must not break the merchant's login. Without a Shop row
 * the next afterAuth simply tries again.
 */
export async function registerShop(shop: string): Promise<void> {
  const existing = await prisma.shop.findUnique({ where: { shop } });
  if (existing) {
    console.log(`${shop} already registered as ${existing.coreStoreId}`);
    return;
  }
  console.log(`${shop} is new: register it with the core`);

  let response: Response;
  try {
    // The admin key proves to the core that this request comes from this app.
    response = await fetch(new URL("/v1/stores", process.env.CORE_API_URL), {
      method: "POST",
      headers: { Authorization: `Bearer ${process.env.CORE_ADMIN_KEY}` },
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    // Core down, unreachable or too slow.
    console.error(`registering ${shop} failed: core not reachable`, error);
    return;
  }

  if (response.status !== 201) {
    console.error(`registering ${shop} failed: core answered ${response.status}`);
    return;
  }

  // The only time the core hands out this key; it keeps just its hash.
  const { store_id, api_key } = (await response.json()) as {
    store_id: string;
    api_key: string;
  };

  try {
    await prisma.shop.create({
      data: { shop, coreStoreId: store_id, coreApiKey: api_key },
    });
  } catch (error) {
    // P2002 = unique constraint: another afterAuth for the same shop got here first.
    // Its store wins; the one just created stays unused in the core. Rare and harmless.
    if (
      error instanceof Prisma.PrismaClientKnownRequestError &&
      error.code === "P2002"
    ) {
      console.warn(`${shop} was registered concurrently; ${store_id} left unused`);
      return;
    }
    console.error(`registering ${shop} failed: could not save Shop row`, error);
    return;
  }

  // Never log the key itself.
  console.log(`registered ${shop} as ${store_id}`);
}

/**
 * Switches on our pixel (extensions/rec-pixel) for the shop this admin session belongs
 * to, and keeps its `appUrl` setting pointing at this app's current address.
 *
 * Called each time the app is opened in the admin (routes/app._index.tsx). Shopify opens
 * the app right after install, so a new shop gets the pixel immediately. In production
 * the address never changes, so after the first call this does nothing. In development
 * it changes on every `npm run dev`; opening the app once updates it.
 *
 * Never throws: a failure here must not break the admin page. It is logged and retried
 * the next time the app is opened.
 */
export async function activatePixel(admin: AdminApiContext): Promise<void> {
  const appUrl = process.env.SHOPIFY_APP_URL;
  const settings = JSON.stringify({ appUrl });

  try {
    const pixel = await findPixel(admin);
    if (pixel && JSON.parse(pixel.settings).appUrl === appUrl) return; // already current

    const response = pixel
      ? await admin.graphql(
          `#graphql
          mutation updatePixel($id: ID!, $webPixel: WebPixelInput!) {
            webPixelUpdate(id: $id, webPixel: $webPixel) {
              userErrors { field message code }
            }
          }`,
          { variables: { id: pixel.id, webPixel: { settings } } },
        )
      : await admin.graphql(
          `#graphql
          mutation createPixel($webPixel: WebPixelInput!) {
            webPixelCreate(webPixel: $webPixel) {
              userErrors { field message code }
            }
          }`,
          { variables: { webPixel: { settings } } },
        );

    const { data } = await response.json();
    const errors = (data?.webPixelUpdate ?? data?.webPixelCreate)?.userErrors ?? [];
    if (errors.length > 0) {
      console.error("activating pixel failed", errors);
      return;
    }
    console.log(`pixel ${pixel ? "updated" : "activated"}: ${settings}`);
  } catch (error) {
    console.error("activating pixel failed", error);
  }
}

/** The app's existing pixel on this shop, or null if there is none yet. Shopify's docs
 * don't say whether "none yet" comes back as null or as an error, so both mean null. */
async function findPixel(
  admin: AdminApiContext,
): Promise<{ id: string; settings: string } | null> {
  try {
    const response = await admin.graphql(`#graphql
      query appPixel { webPixel { id settings } }`);
    return (await response.json()).data?.webPixel ?? null;
  } catch {
    return null;
  }
}
