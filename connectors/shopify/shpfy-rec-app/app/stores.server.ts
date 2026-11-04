import { Prisma } from "@prisma/client";
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
