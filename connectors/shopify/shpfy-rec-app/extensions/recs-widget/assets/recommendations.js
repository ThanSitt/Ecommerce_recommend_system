// Fills every Recommendations block on the page. /apps/recs is the shop's own domain:
// Shopify forwards it, signed, to our app (app/routes/proxy.recs.ts).
for (const box of document.querySelectorAll(".recs-widget")) {
  fetch(`/apps/recs?item_id=${encodeURIComponent(box.dataset.itemId)}`)
    .then((res) => res.json())
    .then((data) => {
      box.textContent = "Recommended: " + data.items.join(", jj");
    })
    .catch(() => {
      // A visitor never sees an error; the box just stays empty.
      box.textContent = "";
    });
}
