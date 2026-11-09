// Fail when a built native module does not load.
//
// A release build can link a module that the dynamic loader then rejects, so
// `build:release` loads it right after building and calls one cheap export.
//
//   node scripts/check-native-load.mjs              every index.*.node in caliche_js
//   node scripts/check-native-load.mjs FILE.node... the given modules

import { readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const packageDir = join(dirname(fileURLToPath(import.meta.url)), "..");

const args = process.argv.slice(2);
const modules =
  args.length > 0
    ? args.map((file) => resolve(file))
    : readdirSync(packageDir)
        .filter((file) => /^index\..+\.node$/.test(file))
        .map((file) => join(packageDir, file));

if (modules.length === 0) {
  console.error("No native module found to check.");
  process.exit(1);
}

let failed = false;
for (const file of modules) {
  try {
    const id = require(file).newEntryId();
    if (typeof id !== "string" || id.length === 0) {
      throw new Error(`newEntryId() returned ${JSON.stringify(id)}`);
    }
    console.log(`Native module loads: ${file}`);
  } catch (error) {
    failed = true;
    console.error(`Native module does not load: ${file}`);
    console.error(error instanceof Error ? error.message : String(error));
  }
}
process.exit(failed ? 1 : 0);
