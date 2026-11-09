import { describe, expect, test } from "bun:test";

// Path to a native module built with `--features panic-probe`. No package
// script builds one, so the test is skipped unless this is set.
const probePath = process.env.CALICHE_PANIC_PROBE_NODE;

describe("Rust panic in a native export", () => {
  test.skipIf(!probePath)("an async export rejects the promise and keeps the process alive", async () => {
    const binding = require(probePath as string) as { _panicProbe: () => Promise<void> };
    await expect(binding._panicProbe()).rejects.toThrow();
    expect(await Promise.resolve("still running")).toBe("still running");
  });

  test.skipIf(!probePath)("a sync export throws a JS error and keeps the process alive", () => {
    const binding = require(probePath as string) as { _panicProbeSync: () => void };
    expect(() => binding._panicProbeSync()).toThrow("caliche sync panic probe");
    expect("still running").toBe("still running");
  });
});
