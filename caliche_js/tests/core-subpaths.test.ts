import { describe, expect, test } from "bun:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

describe("@caliche/core subpaths", () => {
  test("pure env resolver subpath imports without native setup", async () => {
    const envResolve = await import("@caliche/core/runtime/env-resolve");

    expect(envResolve.resolveProviderFromEnv({ provider: "anthropic" })).toBe("anthropic");
    expect(envResolve.resolveModelFromEnv("openai", { model: "gpt-test" })).toBe("gpt-test");
  });

  test("pure streaming and transport subpaths support import and require", async () => {
    const events = await import("@caliche/core/streaming/events");
    const sse = require("@caliche/core/transport/sse");

    expect(typeof events.createInitialAgentStreamState).toBe("function");
    expect(typeof sse.encodeSseEvent).toBe("function");
  });

  test("contracts package remains available", async () => {
    const contracts = await import("@caliche/contracts");

    expect(contracts.parseSlashIntent("/plan")).toEqual({ intent: "plan" });
    expect(require("@caliche/contracts").parsePermissionMode("accept-all")).toBe("accept-all");
  });
});
