import { afterEach, describe, expect, it } from "bun:test";
import { AgentBuilder } from "../src";
import {
  agentBuilderFromEnv,
  configureBuilderFromEnv,
  resolveApiKeyFromEnv,
  resolveModelFromEnv,
  resolveProviderFromEnv,
} from "../src/runtime/env";

const KEYS = [
  "CALICHE_PROVIDER",
  "CALICHE_MODEL",
  "MODEL",
  "OPENROUTER_API_KEY",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "OPENAI_BASE_URL",
  "OPENAI_API_BASE",
  "OPENROUTER_MODEL",
  "OPENAI_MODEL",
  "LMSTUDIO_BASE_URL",
  "LM_STUDIO_BASE_URL",
] as const;

const original = new Map<string, string | undefined>();

function snapshotEnv(): void {
  for (const key of KEYS) {
    original.set(key, process.env[key]);
    delete process.env[key];
  }
}

function restoreEnv(): void {
  for (const key of KEYS) {
    const value = original.get(key);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
}

describe("runtime env helpers", () => {
  afterEach(() => {
    restoreEnv();
  });

  it("prefers openrouter when openrouter credentials are present", () => {
    snapshotEnv();
    process.env.OPENROUTER_API_KEY = "sk-or-test";
    expect(resolveProviderFromEnv()).toBe("openrouter");
    expect(resolveApiKeyFromEnv()).toBe("sk-or-test");
    expect(resolveModelFromEnv()).toBe("openai/gpt-4o-mini");
  });

  it("honors CALICHE_PROVIDER overrides", () => {
    snapshotEnv();
    process.env.CALICHE_PROVIDER = "anthropic";
    process.env.ANTHROPIC_API_KEY = "anth-key";
    expect(resolveProviderFromEnv()).toBe("anthropic");
    expect(resolveApiKeyFromEnv()).toBe("anth-key");
  });

  it("configures an AgentBuilder without adding AgentBuilder.fromEnv", () => {
    snapshotEnv();
    process.env.OPENAI_API_KEY = "sk-test";
    process.env.OPENAI_MODEL = "gpt-4o-mini";

    expect((AgentBuilder as unknown as { fromEnv?: unknown }).fromEnv).toBeUndefined();

    const builder = configureBuilderFromEnv(new AgentBuilder());
    expect(builder).toBeInstanceOf(AgentBuilder);

    const fromHelper = agentBuilderFromEnv({ temperature: 0 });
    expect(fromHelper).toBeInstanceOf(AgentBuilder);
  });

  it("defaults local-inference model from MODEL env", () => {
    snapshotEnv();
    process.env.CALICHE_PROVIDER = "local-inference";
    process.env.MODEL = "tiny-local";
    expect(resolveProviderFromEnv()).toBe("local-inference");
    expect(resolveModelFromEnv("local-inference")).toBe("tiny-local");
  });

  it("resolves lmstudio from either base-url env name", () => {
    snapshotEnv();
    process.env.LM_STUDIO_BASE_URL = "http://localhost:1234/v1";
    expect(resolveProviderFromEnv()).toBe("lmstudio");
  });
});
