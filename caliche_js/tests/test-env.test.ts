import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { liveSuite, resolveLiveProvider, resolveTestProvider } from "./test-env";

const CLOUD_KEYS = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"];
const PINNED = [
  ...CLOUD_KEYS,
  "TEST_PROVIDER",
  "LMSTUDIO_BASE_URL",
  "CALICHE_LIVE",
  "CALICHE_RUN_LMSTUDIO_INTEGRATION",
];
/** Port 9 (discard) has no listener, so the lmstudio probe fails fast. */
const UNREACHABLE_LMSTUDIO = "http://127.0.0.1:9/v1";

describe("test provider resolution", () => {
  const saved = new Map<string, string | undefined>();

  beforeEach(() => {
    for (const key of PINNED) {
      saved.set(key, process.env[key]);
    }
    for (const key of CLOUD_KEYS) {
      process.env[key] = "sk-dummy";
    }
    delete process.env.TEST_PROVIDER;
    delete process.env.CALICHE_LIVE;
    delete process.env.CALICHE_RUN_LMSTUDIO_INTEGRATION;
  });

  afterEach(() => {
    for (const [key, value] of saved) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
    saved.clear();
  });

  test("cloud keys without TEST_PROVIDER resolve to lmstudio", () => {
    expect(resolveTestProvider()).toBe("lmstudio");
  });

  test("cloud keys without TEST_PROVIDER or lmstudio leave no live provider", async () => {
    process.env.LMSTUDIO_BASE_URL = UNREACHABLE_LMSTUDIO;
    expect(await resolveLiveProvider()).toBeUndefined();
  });

  test("TEST_PROVIDER is trimmed and lowercased", () => {
    process.env.TEST_PROVIDER = "  OpenRouter ";
    expect(resolveTestProvider()).toBe("openrouter");
  });

  test("an unknown TEST_PROVIDER throws and lists the valid values", async () => {
    process.env.TEST_PROVIDER = "opneai";
    const expected =
      'Unknown TEST_PROVIDER "opneai". Use one of: lmstudio, openai, anthropic, openrouter, ollama, gemini.';
    expect(() => resolveTestProvider()).toThrow(expected);
    await expect(resolveLiveProvider()).rejects.toThrow(expected);
  });

  test("liveSuite targets an explicit TEST_PROVIDER when live mode is on", async () => {
    process.env.CALICHE_LIVE = "1";
    process.env.TEST_PROVIDER = "ollama";
    expect((await liveSuite("Probe")).provider).toBe("ollama");
  });

  test("an explicit TEST_PROVIDER is used as given when live mode is on", async () => {
    process.env.CALICHE_LIVE = "1";
    process.env.TEST_PROVIDER = "openai";
    expect([resolveTestProvider(), await resolveLiveProvider()]).toEqual(["openai", "openai"]);
  });
});
