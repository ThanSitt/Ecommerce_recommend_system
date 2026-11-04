// Shopify Web Pixel event -> standard event contract (docs/decisions/0001-event-contract.md).
//
// Ported from connectors/shopify/app/transform.py. Field paths come from the captured
// payloads in tests/fixtures/shopify/ and the mapping table in
// docs/investigations/shopify-event-sources.md.
//
// `.server.ts`: never bundled for the browser; the build fails if client code imports it.

export type ContractEvent = {
  event_id: string;
  user_id: string;
  item_id: string;
  event_type: "view" | "cart" | "purchase";
  order_id?: string;
  timestamp: string;
};

// Only the fields we read. The real payload carries much more, including personal data
// on checkout (email, address, phone), which must never be forwarded or logged.
type Product = { id: string; title?: string };

export type LineItem = {
  title?: string;
  variant: { id: string; product: Product };
};

// Each event type carries only its own key (a product_viewed has `productVariant`, and
// so on). Reading a missing one throws, and the caller (routes/pixel.ts) catches that.
type PixelData = {
  productVariant: { product: Product };
  cartLine: { merchandise: { product: Product } };
  checkout: { order: { id: string }; lineItems: LineItem[] };
};

export type PixelEvent = {
  id: string;
  name: string;
  clientId: string;
  timestamp: string;
  data: PixelData;
};

/**
 * Turns one pixel event into a list of contract events: a list because one checkout
 * becomes one event per product. Event types we don't handle give an empty list.
 */
export function transform(event: PixelEvent): ContractEvent[] {
  switch (event.name) {
    case "product_viewed":
      return [
        {
          event_id: event.id,
          // clientId is Shopify's anonymous browser ID, never the customer ID.
          user_id: event.clientId,
          // The product, not the variant (productVariant.id).
          item_id: event.data.productVariant.product.id,
          event_type: "view",
          timestamp: event.timestamp,
        },
      ];

    case "product_added_to_cart":
      return [
        {
          event_id: event.id,
          user_id: event.clientId,
          // `merchandise` is Shopify's name for the variant; the product sits inside it.
          item_id: event.data.cartLine.merchandise.product.id,
          event_type: "cart",
          timestamp: event.timestamp,
        },
      ];

    case "checkout_completed": {
      const checkout = event.data.checkout;
      const orderId: string = checkout.order.id;
      // One purchase event per product in the order, all sharing the same order_id.
      return checkout.lineItems.map((line) => ({
        // Built, not copied: Shopify's `id` is one per checkout, but we emit one event
        // per line. Variant (not product) keeps two variants of the same product in one
        // order from colliding.
        event_id: `purchase:${orderId}:${line.variant.id}`,
        user_id: event.clientId,
        item_id: line.variant.product.id,
        event_type: "purchase" as const,
        order_id: orderId,
        timestamp: event.timestamp,
      }));
    }

    default:
      return [];
  }
}
