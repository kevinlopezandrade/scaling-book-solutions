import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { chapters, ROOT_DIR } from "./helpers.mjs";

const outputDirectory = mkdtempSync(join(tmpdir(), "scaling-book-pdfs-"));

try {
  for (const chapter of chapters) {
    execFileSync("typst", [
      "compile",
      "--root", ROOT_DIR,
      chapter.source,
      join(outputDirectory, `chapter-${chapter.number}.pdf`),
    ], { cwd: ROOT_DIR, stdio: "inherit" });
  }
} finally {
  rmSync(outputDirectory, { recursive: true, force: true });
}

console.log(`PDF compilation passed for ${chapters.length} chapters.`);
