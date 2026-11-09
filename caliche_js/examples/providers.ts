/**
 * Provider Configuration and Switching Example
 *
 * This example demonstrates:
 * - Programmatic configuration for all LLM providers
 * - Advanced provider options (timeouts, retries, headers)
 * - Custom retry strategies
 * - Comparing responses across different providers
 * - Implementing fallback logic on provider failure
 * - Provider-specific features and limitations
 *
 * Best practices shown:
 * - Graceful provider fallback
 * - Provider selection based on use case
 * - Error handling for provider failures
 * - Configuration without environment variables
 *
 * Run with:
 * ```bash
 * bun run examples/providers.ts
 * ```
 */

import { type Agent, AgentBuilder, type Provider } from "../src/index";

async function openAIProvider(): Promise<void> {
  console.log("=== Example 1: OpenAI Provider ===\n");

  try {
    // Create agent with OpenAI provider
    const agent: Agent = await new AgentBuilder().provider("openai").model("gpt-4o").build();

    console.log(`OpenAI provider: ${agent.provider}, model: ${agent.model}`);

    const response = await agent.chat("Say hello in one sentence.");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function anthropicProvider(): Promise<void> {
  console.log("=== Example 2: Anthropic Provider ===\n");

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("anthropic")
      .model("claude-sonnet-4-5-20250929")
      .build();

    console.log(`Anthropic provider: ${agent.provider}, model: ${agent.model}`);

    const response = await agent.chat("Say hello in one sentence.");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function googleProvider(): Promise<void> {
  console.log("=== Example 3: Google Provider ===\n");

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("google")
      .model("models/gemini-2.5-flash")
      .build();

    console.log(`Google provider: ${agent.provider}, model: ${agent.model}`);

    const response = await agent.chat("Say hello in one sentence.");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function openRouterProvider(): Promise<void> {
  console.log("=== Example 4: OpenRouter Provider ===\n");

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("openrouter")
      .model("anthropic/claude-haiku-4.5")
      .build();

    console.log(`OpenRouter provider: ${agent.provider}, model: ${agent.model}`);

    const response = await agent.chat("Say hello in one sentence.");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function perplexityProvider(): Promise<void> {
  console.log("=== Example 5: Perplexity Provider ===\n");

  try {
    const agent: Agent = await new AgentBuilder().provider("perplexity").model("sonar").build();

    console.log(`Perplexity provider: ${agent.provider}, model: ${agent.model}`);

    const response = await agent.chat("Say hello in one sentence.");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function lmStudioProvider(): Promise<void> {
  console.log("=== Example 6: LMStudio Provider ===\n");

  try {
    const agent: Agent = await new AgentBuilder().provider("lmstudio").model("local-model").build();

    console.log(`LMStudio provider: ${agent.provider}, model: ${agent.model}`);

    const response = await agent.chat("Say hello in one sentence.");
    console.log(`Response: ${response.response}\n`);
  } catch (error) {
    console.log(`(Skipped: ${error instanceof Error ? error.message : "LM Studio not running"})\n`);
  }
}

async function providerComparison(): Promise<void> {
  console.log("=== Example 7: Provider Comparison ===\n");

  const providers = [
    { name: "openai", model: "gpt-4o" },
    { name: "anthropic", model: "claude-sonnet-4-5-20250929" },
    { name: "google", model: "models/gemini-2.5-flash" },
  ];

  const query = "What is 2 + 2? Answer in one word.";

  for (const provider of providers) {
    try {
      console.log(`Testing ${provider.name}...`);
      const agent: Agent = await new AgentBuilder()
        .provider(provider.name as Provider)
        .model(provider.model)
        .build();

      const response = await agent.chat(query);
      console.log(`  ${provider.name}: ${response.response}`);
      if (response.usage.total_tokens) {
        console.log(`  Tokens: ${response.usage.total_tokens}`);
      }
      console.log();
    } catch (_error) {
      console.log(`  ${provider.name}: (not configured)\n`);
    }
  }
}

async function providerFallback(): Promise<void> {
  console.log("=== Example 8: Provider Fallback ===\n");

  const providers = ["openai", "anthropic", "bedrock", "google"];

  for (const providerName of providers) {
    try {
      console.log(`Trying ${providerName}...`);
      const agent: Agent = await new AgentBuilder()
        .provider(providerName as Provider)
        .model("default")
        .build();

      const response = await agent.chat("Hello!");
      console.log(`${providerName} succeeded: ${response.response}\n`);
      return; // Success, exit
    } catch (error) {
      console.log(
        `Failed: ${providerName} failed: ${error instanceof Error ? error.message : "Unknown error"}`
      );
    }
  }

  console.log("\nAll providers failed. Please configure at least one provider.\n");
}

async function main(): Promise<void> {
  console.log("=== Provider Configuration Examples ===\n");

  await openAIProvider();
  await anthropicProvider();
  await googleProvider();
  await openRouterProvider();
  await perplexityProvider();
  await lmStudioProvider();
  await providerComparison();
  await providerFallback();

  console.log("=== Examples Complete ===");
  console.log("\nTips:");
  console.log("- Use explicit provider configuration for production");
  console.log("- Implement fallback logic for high availability");
  console.log("- Compare providers to find the best fit for your use case");
  console.log("- Each provider has different capabilities and pricing");
}

main().catch(console.error);
