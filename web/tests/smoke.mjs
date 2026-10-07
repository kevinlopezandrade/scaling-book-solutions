import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { load } from "cheerio";

import { chapters } from "./helpers.mjs";

const dist = new URL("../dist/", import.meta.url);
const website = new URL("../../../web/", import.meta.url);
const theme = new URL("assets/theme/", website);
const origin = "https://site.invalid";
const base = new URL("/posts/scaling-book-solutions/", origin);

const mathFont = readFileSync(new URL("assets/fonts/NewCMMath-Book.woff2", dist));
assert.equal(mathFont.subarray(0, 4).toString("ascii"), "wOF2", "Missing or invalid bundled math font.");
for (const license of ["GUST-FONT-LICENSE.txt", "LPPL-1.3c.txt"]) {
  assert(existsSync(new URL(`assets/fonts/${license}`, dist)), `Math font license is missing: ${license}`);
}

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
    const resources = $("article > .chapter-resources");
    assert.equal(resources.length, 1, `${route}: expected one chapter resource block.`);
    assert($("article").children().first().hasClass("chapter-resources"), `${route}: resources must precede the solutions.`);
    const chapter = chapters.find((chapter) => chapter.route === route);
    assert.equal(
      resources.find("a").first().attr("href"),
      `https://github.com/kevinlopezandrade/scaling-book-solutions/blob/main/chapter-${chapter.number}/c${chapter.number}-handwritten.pdf`,
      `${route}: the first resource must link to this chapter's handwritten solutions.`,
    );

    const sections = $("article h3").toArray();
    assert.equal($("merror").length, 0, `${route}: MathML contains an error.`);
    const preload = $('link[rel="preload"][as="font"]');
    assert.equal(preload.length, 1, `${route}: expected a math font preload.`);
    assert.equal(preload.attr("href"), "../assets/fonts/NewCMMath-Book.woff2");
    assert.equal(preload.attr("crossorigin"), "anonymous");

    assert.deepEqual(
      $(".toc nav a").map((_, link) => $(link).attr("href")).get(),
      sections.map((heading) => `#${$(heading).attr("id")}`),
      `${route}: contents links must cover every section in order.`,
    );
  }

  // Resolve deployed URLs against either this build or the main website.
  for (const element of $("a[href], link[href], img[src], script[src]").toArray()) {
    const value = $(element).attr("href") ?? $(element).attr("src");
    assert(value.trim(), `${route}: empty link or asset URL.`);
    const target = new URL(value, new URL(route, base));
    if (target.origin !== origin) continue;
    if (target.pathname.endsWith("/")) target.pathname += "index.html";

    const project = target.pathname.startsWith(base.pathname);
    const root = project ? dist : website;
    const prefix = project ? base.pathname : "/";
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

console.log(`HTML and math-font checks passed for the homepage and ${chapters.length} chapters.`);
