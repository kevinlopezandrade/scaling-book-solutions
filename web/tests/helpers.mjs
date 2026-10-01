import assert from "node:assert/strict";
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT_DIR = fileURLToPath(new URL("../../", import.meta.url));

export const chapters = readdirSync(ROOT_DIR, { withFileTypes: true })
  .filter((entry) => entry.isDirectory() && entry.name.startsWith("chapter-"))
  .map(({ name }) => {
    const number = Number(name.slice("chapter-".length));
    assert(Number.isInteger(number) && number > 0, `Invalid chapter directory: ${name}`);

    const source = join(name, `c${number}-transcribed.typ`);
    assert(existsSync(join(ROOT_DIR, source)), `Missing source: ${source}`);

    return { number, source, route: `${name}/index.html` };
  })
  .sort((left, right) => left.number - right.number);

assert(chapters.length > 0, "No chapter sources found.");
