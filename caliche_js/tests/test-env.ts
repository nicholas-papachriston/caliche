/**
 * Shared provider/model resolution for integration tests.
 * Only an explicit TEST_PROVIDER selects a provider. Otherwise tests use local lmstudio,
 * so a cloud API key in the environment never causes paid model calls.
 */

import { test } from "bun:test";
import { AgentBuilder } from "../src/index";

export const TEST_PROVIDERS = [
  "lmstudio",
  "openai",
  "anthropic",
  "openrouter",
  "ollama",
  "gemini",
] as const;

export type TestProvider = (typeof TEST_PROVIDERS)[number];

/** The provider when TEST_PROVIDER is unset: local, so no paid model calls. */
export const DEFAULT_TEST_PROVIDER: TestProvider = "lmstudio";

function isTestProvider(value: string): value is TestProvider {
  return (TEST_PROVIDERS as readonly string[]).includes(value);
}

/** TEST_PROVIDER, or undefined when unset. A value outside TEST_PROVIDERS throws. */
export function explicitTestProvider(): TestProvider | undefined {
  const explicit = process.env.TEST_PROVIDER?.trim().toLowerCase();
  if (!explicit) {
    return undefined;
  }
  if (!isTestProvider(explicit)) {
    throw new Error(
      `Unknown TEST_PROVIDER "${explicit}". Use one of: ${TEST_PROVIDERS.join(", ")}.`
    );
  }
  return explicit;
}

export function resolveTestProvider(): TestProvider {
  return explicitTestProvider() ?? DEFAULT_TEST_PROVIDER;
}

export function resolveTestModel(provider: TestProvider = resolveTestProvider()): string {
  if (process.env.TEST_MODEL) {
    return process.env.TEST_MODEL;
  }
  switch (provider) {
    case "openrouter":
      return process.env.OPENROUTER_MODEL || process.env.MODEL || "openai/gpt-4o-mini";
    case "openai":
      return process.env.OPENAI_MODEL || "gpt-4o-mini";
    case "anthropic":
      return "claude-3-haiku-20240307";
    case "ollama":
      return process.env.OLLAMA_MODEL || "llama3.2";
    case "gemini":
      return process.env.GEMINI_MODEL || "gemini-2.0-flash";
    case "lmstudio":
      return process.env.LMSTUDIO_MODEL || "local-model";
    default: {
      const _exhaustive: never = provider;
      return _exhaustive;
    }
  }
}

export function resolveTestApiKey(
  provider: TestProvider = resolveTestProvider()
): string | undefined {
  switch (provider) {
    case "openrouter":
      return process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY;
    case "openai":
      return process.env.OPENAI_API_KEY;
    case "anthropic":
      return process.env.ANTHROPIC_API_KEY;
    case "gemini":
      return process.env.GEMINI_API_KEY;
    case "lmstudio":
    case "ollama":
      return undefined;
    default: {
      const _exhaustive: never = provider;
      return _exhaustive;
    }
  }
}

const LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1";
const LIVE_PROBE_TIMEOUT_MS = 1000;
export const LIVE_SKIP_REASON =
  "set CALICHE_LIVE=1 to run live-model tests (and TEST_PROVIDER or a local LM Studio)";

async function isLmStudioReachable(): Promise<boolean> {
  const baseUrl = (process.env.LMSTUDIO_BASE_URL || LMSTUDIO_DEFAULT_BASE_URL).replace(/\/+$/, "");
  try {
    const response = await fetch(`${baseUrl}/models`, {
      signal: AbortSignal.timeout(LIVE_PROBE_TIMEOUT_MS),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Provider for tests that call a live model. Requires CALICHE_LIVE=1.
 * Then only an explicit TEST_PROVIDER or a reachable local LM Studio counts.
 * A cloud API key alone never enables paid model calls.
 */
export async function resolveLiveProvider(): Promise<TestProvider | undefined> {
  // Validate TEST_PROVIDER even when live mode is off, so a typo fails the suite.
  const explicit = explicitTestProvider();
  // Live tests run only when CALICHE_LIVE=1 or CALICHE_RUN_LMSTUDIO_INTEGRATION=true.
  if (process.env.CALICHE_LIVE !== "1" && process.env.CALICHE_RUN_LMSTUDIO_INTEGRATION !== "true") {
    return undefined;
  }
  if (explicit) {
    return explicit;
  }
  return (await isLmStudioReachable()) ? DEFAULT_TEST_PROVIDER : undefined;
}

/** Print one visible line when a suite skips its live-model tests. */
export function reportLiveSkip(suite: string, provider: TestProvider | undefined): void {
  if (provider === undefined) {
    console.log(`${suite}: skipped live-model tests: ${LIVE_SKIP_REASON}`);
  }
}

/** An AgentBuilder for `provider` and its test model. Local providers get no API key. */
export function testAgentBuilder(provider: TestProvider = resolveTestProvider()): AgentBuilder {
  const builder = new AgentBuilder().provider(provider).model(resolveTestModel(provider));
  const apiKey = resolveTestApiKey(provider);
  if (provider !== "lmstudio" && provider !== "ollama" && apiKey) {
    builder.apiKey(apiKey);
  }
  return builder;
}

export interface LiveSuite {
  /** `test`, skipped when no live provider is available. */
  readonly liveTest: ReturnType<typeof test.skipIf>;
  readonly provider: TestProvider;
  /** A fresh AgentBuilder for the suite's provider. */
  readonly agentBuilder: () => AgentBuilder;
}

/**
 * Setup for a suite with live-model tests. Without a live provider, `liveTest` skips and
 * the suite prints one line saying so; the builder then targets the default provider.
 */
export async function liveSuite(suite: string): Promise<LiveSuite> {
  const live = await resolveLiveProvider();
  reportLiveSkip(suite, live);
  const provider = live ?? DEFAULT_TEST_PROVIDER;
  return {
    liveTest: test.skipIf(live === undefined),
    provider,
    agentBuilder: () => testAgentBuilder(provider),
  };
}
