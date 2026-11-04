// Sends contract events to the core's POST /v1/events.
//
// Ported from connectors/shopify/app/core_client.py. The app talks to the core over HTTP
// like any other client, so the core stays unaware that Shopify exists.

import type { ContractEvent } from "./transform.server";

// A slow or hung core must not hold the pixel's request open indefinitely.
const TIMEOUT_MS = 5000;

/**
 * POST events to the core with one store's key. Returns true on success.
 *
 * Failures are logged, not thrown: the caller is the pixel, which cannot do anything
 * useful with an error. Events sent while the core is unreachable are lost for now;
 * retrying them is a later step.
 */
export async function sendEvents(
  coreApiKey: string,
  events: ContractEvent[],
): Promise<boolean> {
  if (events.length === 0) return true;

  let response: Response;
  try {
    response = await fetch(new URL("/v1/events", process.env.CORE_API_URL), {
      method: "POST",
      headers: {
        Authorization: `Bearer ${coreApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(events),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (error) {
    console.error(`could not reach core, ${events.length} event(s) lost`, error);
    return false;
  }

  const body = await response.text();
  if (!response.ok) {
    console.error(
      `core rejected ${events.length} event(s): ${response.status} ${body}`,
    );
    return false;
  }

  console.log(`core accepted ${events.length} event(s): ${body}`);
  return true;
}
