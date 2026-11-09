import type { Provider } from "../streaming/events";

/**
 * Environment-driven provider/model/credential helpers.
 *
 * These functions are pure TypeScript and do not load the native binding.
 */

export type EnvProvider = Provider;

export interface EnvRuntimeOptions {
  readonly provider?: EnvProvider;
  readonly model?: string;
  readonly apiKey?: string;
  readonly baseUrl?: string;
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly systemPrompt?: string;
  /** When true, prefer OpenRouter if OPENROUTER_API_KEY is set. Default true. */
  readonly preferOpenRouter?: boolean;
}

function firstNonEmpty(...values: Array<string | undefined>): string | undefined {
  for (const value of values) {
    const trimmed = value?.trim();
    if (trimmed) {
      return trimmed;
    }
  }
  return undefined;
}

function looksLikeOpenRouterKey(apiKey: string | undefined): boolean {
  return typeof apiKey === "string" && apiKey.startsWith("sk-or");
}

function openaiBaseUrl(): string | undefined {
  return firstNonEmpty(process.env.OPENAI_BASE_URL, process.env.OPENAI_API_BASE);
}

interface ModelFallback {
  readonly envVars: readonly string[];
  readonly fallback: string;
}

function defaultModelSpec(provider: EnvProvider): ModelFallback {
  switch (provider) {
    case "openrouter":
      return { envVars: ["OPENROUTER_MODEL"], fallback: "openai/gpt-4o-mini" };
    case "openai":
    case "azure-openai":
      return { envVars: ["OPENAI_MODEL"], fallback: "gpt-4o-mini" };
    case "anthropic":
      return { envVars: ["ANTHROPIC_MODEL"], fallback: "claude-3-haiku-20240307" };
    case "ollama":
      return { envVars: ["OLLAMA_MODEL"], fallback: "llama3.2" };
    case "google":
    case "gemini":
      return { envVars: ["GEMINI_MODEL"], fallback: "gemini-2.0-flash" };
    case "lmstudio":
      return { envVars: ["LMSTUDIO_MODEL"], fallback: "local-model" };
    case "bedrock":
    case "perplexity":
    case "cohere":
    case "cloudflare-ai-gateway":
    case "local-inference":
    case "external":
      return { envVars: ["MODEL"], fallback: "gpt-4o-mini" };
    default: {
      const _exhaustive: never = provider;
      return _exhaustive;
    }
  }
}

/**
 * Resolve the LLM provider from environment variables.
 *
 * Precedence:
 * 1. `CALICHE_PROVIDER`
 * 2. OpenRouter heuristics (`OPENROUTER_API_KEY`, `sk-or...`, openrouter base URL)
 * 3. `OPENAI_API_KEY` -> openai
 * 4. `ANTHROPIC_API_KEY` -> anthropic
 * 5. fallback `openai`
 */
export function resolveProviderFromEnv(options: EnvRuntimeOptions = {}): EnvProvider {
  if (options.provider) {
    return options.provider;
  }

  const forced = process.env.CALICHE_PROVIDER?.trim().toLowerCase();
  if (forced) {
    return forced as EnvProvider;
  }

  const preferOpenRouter = options.preferOpenRouter !== false;
  const base = (openaiBaseUrl() ?? "").toLowerCase();
  const openrouterKey = process.env.OPENROUTER_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;

  if (
    preferOpenRouter &&
    (openrouterKey || looksLikeOpenRouterKey(openaiKey) || base.includes("openrouter.ai"))
  ) {
    return "openrouter";
  }
  if (openaiKey) {
    return "openai";
  }
  if (process.env.ANTHROPIC_API_KEY) {
    return "anthropic";
  }
  if (process.env.LMSTUDIO_BASE_URL || process.env.LM_STUDIO_BASE_URL) {
    return "lmstudio";
  }
  return "openai";
}

/**
 * Resolve a model id for the given (or env-resolved) provider.
 */
export function resolveModelFromEnv(
  provider: EnvProvider = resolveProviderFromEnv(),
  options: EnvRuntimeOptions = {}
): string {
  if (options.model) {
    return options.model;
  }
  const explicit = firstNonEmpty(process.env.CALICHE_MODEL, process.env.MODEL);
  if (explicit) {
    return explicit;
  }

  const spec = defaultModelSpec(provider);
  const fromEnv = firstNonEmpty(...spec.envVars.map((key) => process.env[key]));
  return firstNonEmpty(fromEnv, spec.fallback) ?? spec.fallback;
}

/**
 * Resolve an API key for the given (or env-resolved) provider.
 */
export function resolveApiKeyFromEnv(
  provider: EnvProvider = resolveProviderFromEnv(),
  options: EnvRuntimeOptions = {}
): string | undefined {
  if (options.apiKey) {
    return options.apiKey;
  }

  switch (provider) {
    case "openrouter":
      return firstNonEmpty(process.env.OPENROUTER_API_KEY, process.env.OPENAI_API_KEY);
    case "openai":
    case "azure-openai":
    case "lmstudio":
    case "ollama":
    case "cloudflare-ai-gateway":
      return firstNonEmpty(process.env.OPENAI_API_KEY);
    case "anthropic":
      return firstNonEmpty(process.env.ANTHROPIC_API_KEY);
    case "google":
    case "gemini":
      return firstNonEmpty(process.env.GEMINI_API_KEY, process.env.GOOGLE_API_KEY);
    case "perplexity":
      return firstNonEmpty(process.env.PERPLEXITY_API_KEY);
    case "cohere":
      return firstNonEmpty(process.env.COHERE_API_KEY);
    case "bedrock":
    case "local-inference":
    case "external":
      return firstNonEmpty(process.env.OPENAI_API_KEY);
    default: {
      const _exhaustive: never = provider;
      return _exhaustive;
    }
  }
}

/**
 * Resolve a provider base URL when one is configured in the environment.
 */
export function resolveBaseUrlFromEnv(
  provider: EnvProvider = resolveProviderFromEnv(),
  options: EnvRuntimeOptions = {}
): string | undefined {
  if (options.baseUrl) {
    return options.baseUrl;
  }
  if (provider === "lmstudio") {
    return firstNonEmpty(
      process.env.LMSTUDIO_BASE_URL,
      process.env.LM_STUDIO_BASE_URL,
      "http://localhost:1234/v1"
    );
  }
  if (provider === "ollama") {
    return firstNonEmpty(process.env.OLLAMA_BASE_URL, "http://localhost:11434");
  }
  // OpenRouter uses the dedicated provider path; do not force OPENAI_BASE_URL onto it.
  if (provider === "openrouter") {
    return undefined;
  }
  return openaiBaseUrl();
}
