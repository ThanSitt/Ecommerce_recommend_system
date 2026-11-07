import { register } from "@shopify/web-pixels-extension";

// Runs in the shopper's browser, in Shopify's sandbox, on every store that installed the
// app. Sends the three events we use to our app's /pixel route (app/routes/pixel.ts),
// which forwards them to the core with that shop's key.
register(({ analytics, init, settings }) => {
  // Set per shop by the app (activatePixel in app/stores.server.ts).
  const url = `${settings.appUrl}/pixel`;
  // Shopify's permanent name for the store; the app finds the shop's key by it.
  const shop = init.data.shop.myshopifyDomain;

  function send(event: unknown) {
    fetch(url, {
      method: "POST",
      // text/plain: the browser sends it without a CORS pre-check.
      headers: { "Content-Type": "text/plain" },
      body: JSON.stringify({ shop, event }),
      // Lets the request finish even if the shopper navigates away right after.
      keepalive: true,
    });
  }

  analytics.subscribe("product_viewed", send);
  analytics.subscribe("product_added_to_cart", send);
  analytics.subscribe("checkout_completed", send);
});
