/**
 * Entry point for running web_scraper as a module.
 * bun run examples/web_scraper/cli-main.ts
 */

import { main } from "./cli";

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
