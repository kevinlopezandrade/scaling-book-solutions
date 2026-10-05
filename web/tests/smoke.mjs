import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { load } from "cheerio";

import { chapters } from "./helpers.mjs";

const dist = new URL("../dist/", import.meta.url);
const theme = new URL("../../../web/assets/theme/", import.meta.url);
const origin = "https://site.invalid";
const base = new URL("/posts/scaling-book-solutions/", origin);

for (const file of JSON.parse(readFileSync(new URL("assets.json", theme), "utf8"))) {
  assert(existsSync(new URL(file, theme)), `Shared theme asset is missing: ${file}`);
}

function readPage(url) {
  return load(readFileSync(url, "utf8"));
}

const home = readPage(new URL("index.html", dist));
assert.deepEqual(
  home(".chapter-list a").map((_, link) => home(link).attr("href")).get(),
  chapters.map((chapter) => chapter.route),
  "Homepage links must list every chapter in order.",
);

for (const route of ["index.html", ...chapters.map((chapter) => chapter.route)]) {
  const $ = readPage(new URL(route, dist));

  const ids = $("[id]").map((_, element) => $(element).attr("id")).get();
  assert.equal(new Set(ids).size, ids.length, `${route}: duplicate element IDs.`);

  if (route !== "index.html") {
    const sections = $("article h3").toArray();
    assert.equal($("merror").length, 0, `${route}: MathML contains an error.`);

    assert.deepEqual(
      $(".toc nav a").map((_, link) => $(link).attr("href")).get(),
      sections.map((heading) => `#${$(heading).attr("id")}`),
      `${route}: contents links must cover every section in order.`,
    );
  }

  // Resolve deployed URLs against either this build or the shared article theme.
  for (const element of $("a[href], link[href], img[src], script[src]").toArray()) {
    const value = $(element).attr("href") ?? $(element).attr("src");
    assert(value.trim(), `${route}: empty link or asset URL.`);
    const target = new URL(value, new URL(route, base));
    if (target.origin !== origin) continue;
    if (target.pathname.endsWith("/")) target.pathname += "index.html";

    const shared = target.pathname.startsWith("/assets/theme/");
    const root = shared ? theme : dist;
    const prefix = shared ? "/assets/theme/" : base.pathname;
    assert(target.pathname.startsWith(prefix), `${route}: unexpected local URL: ${value}`);
    const file = new URL(target.pathname.slice(prefix.length), root);
    assert(file.href.startsWith(root.href), `${route}: reference escapes its directory: ${value}`);
    assert(existsSync(file), `${route}: missing local file: ${value}`);

    if (target.hash) {
      const id = decodeURIComponent(target.hash.slice(1));
      const targetPage = readPage(file);
      assert(
        targetPage("[id]").toArray().some((element) => element.attribs.id === id),
        `${route}: missing link target: ${value}`,
      );
    }
  }
}

console.log(`HTML checks passed for the homepage and ${chapters.length} chapters.`);
