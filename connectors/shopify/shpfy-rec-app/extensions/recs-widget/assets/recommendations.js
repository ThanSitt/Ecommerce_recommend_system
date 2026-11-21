// Fills every Recommendations block on the page. The proxy path (/apps/recs in
// production) is on the shop's own domain: Shopify forwards it, signed, to our app
// (app/routes/proxy.recs.ts). It comes from the block's settings, because the dev and
// prod apps each own a different path on the same store.
for (const box of document.querySelectorAll(".recs-widget")) {
  const path = box.dataset.proxyPath || "/apps/recs";
  fetch(`${path}?item_id=${encodeURIComponent(box.dataset.itemId)}`)
    .then((res) => res.json())
    .then((data) => {
      box.textContent = "Recommended: " + data.items.join(", ");
    })
    .catch(() => {
      // A visitor never sees an error; the box just stays empty.
      box.textContent = "";
    });
}
