import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";

const expectedVersion = "0.15.0";
const output = execFileSync("typst", ["--version"], { encoding: "utf8" }).trim();
const [command, version] = output.split(" ");

assert.equal(command, "typst", `Unexpected version output: ${output}`);
assert.equal(version, expectedVersion, `This project requires Typst ${expectedVersion}; found ${version}.`);
