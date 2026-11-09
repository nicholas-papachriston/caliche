import { AgentBuilder } from "../agent/builder-base";
import {
  type EnvRuntimeOptions,
  resolveApiKeyFromEnv,
  resolveBaseUrlFromEnv,
  resolveModelFromEnv,
  resolveProviderFromEnv,
} from "./env-resolve";

/**
 * Environment-driven AgentBuilder helpers.
 *
 * These are standalone functions (not `AgentBuilder.fromEnv`) so builders stay
 * explicit and testable while still offering a one-line env bootstrap.
 */

export {
  type EnvProvider,
  type EnvRuntimeOptions,
  resolveApiKeyFromEnv,
  resolveBaseUrlFromEnv,
  resolveModelFromEnv,
  resolveProviderFromEnv,
} from "./env-resolve";

/**
 * Apply env-resolved provider/model/credentials onto an existing builder.
 */
export function configureBuilderFromEnv(
  builder: AgentBuilder,
  options: EnvRuntimeOptions = {}
): AgentBuilder {
  const provider = resolveProviderFromEnv(options);
  const model = resolveModelFromEnv(provider, options);
  const apiKey = resolveApiKeyFromEnv(provider, options);
  const baseUrl = resolveBaseUrlFromEnv(provider, options);

  builder.provider(provider).model(model);
  if (apiKey) {
    builder.apiKey(apiKey);
  }
  if (baseUrl) {
    builder.baseUrl(baseUrl);
  }
  if (typeof options.temperature === "number") {
    builder.temperature(options.temperature);
  }
  if (typeof options.maxTokens === "number") {
    builder.maxTokens(options.maxTokens);
  }
  if (options.systemPrompt) {
    builder.systemPrompt(options.systemPrompt);
  }
  return builder;
}

/**
 * Create a new AgentBuilder preconfigured from environment variables.
 */
export function agentBuilderFromEnv(options: EnvRuntimeOptions = {}): AgentBuilder {
  return configureBuilderFromEnv(new AgentBuilder(), options);
}
