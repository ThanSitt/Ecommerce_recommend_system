import type { HeadersFunction, LoaderFunctionArgs } from "react-router";
import { authenticate } from "../shopify.server";
import { activatePixel } from "../stores.server";
import { boundary } from "@shopify/shopify-app-react-router/server";

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { admin } = await authenticate.admin(request);
  // Switch the pixel on for this shop, or point it at the current app URL.
  await activatePixel(admin);

  return null;
};

export default function Index() {
  return (
    <s-page heading="Product Recommendations">
      <s-section heading="Recommendations are active">
        <s-paragraph>
          This app learns from what shoppers do in your store (products
          viewed, added to cart, and bought) and shows each shopper products
          they&apos;re likely to want next.
        </s-paragraph>
      </s-section>

      <s-section slot="aside" heading="How it works">
        <s-unordered-list>
          <s-list-item>Shopper activity is collected automatically.</s-list-item>
          <s-list-item>
            No names or emails are used, only anonymous IDs.
          </s-list-item>
          <s-list-item>
            Recommendations update as new activity comes in.
          </s-list-item>
        </s-unordered-list>
      </s-section>
    </s-page>
  );
}

export const headers: HeadersFunction = (headersArgs) => {
  return boundary.headers(headersArgs);
};
