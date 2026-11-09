/**
 * Google Gemini Provider Example
 *
 * This example demonstrates:
 * - Using Google Gemini provider with AgentBuilder
 * - Model selection (gemini-2.5-flash, etc.)
 * - Streaming support with Google provider
 * - Comparison with other providers
 *
 * Best practices shown:
 * - Environment variable configuration
 * - Error handling
 * - Token usage tracking
 *
 * Run with:
 * ```bash
 * bun run examples/google.ts
 * ```
 *
 * Prerequisites:
 * - Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable
 */

import {
  type Agent,
  AgentBuilder,
  isMetadataEvent,
  isTextChunkEvent,
  type StreamIterator,
} from "../src/index";

async function basicGoogleUsage(): Promise<void> {
  console.log("=== Example 1: Basic Google Gemini Usage ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("google")
    .model("models/gemini-2.5-flash")
    .enableTool("calculator")
    .build();

  try {
    const response = await agent.chat("What is 15 multiplied by 23?");

    console.log("Query: What is 15 multiplied by 23?");
    console.log(`Response: ${response.response}`);
    if (response.usage.total_tokens) {
      console.log(`Tokens used: ${response.usage.total_tokens}`);
    }
    console.log();
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function googleStreaming(): Promise<void> {
  console.log("\n=== Example 2: Streaming with Google Provider ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("google")
    .model("models/gemini-2.5-flash")
    .build();

  try {
    console.log("Query: Write a haiku about programming");
    console.log("Response (streaming):\n");

    const stream: StreamIterator = agent.streamChat("Write a haiku about programming");

    for await (const event of stream) {
      if (isTextChunkEvent(event)) {
        process.stdout.write(event.content);
      } else if (isMetadataEvent(event)) {
        console.log("\n\nMetadata:", {
          usage: event.usage,
          finish_reason: event.finish_reason,
        });
      }
    }

    console.log("\n");
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function providerComparison(): Promise<void> {
  console.log("\n=== Example 3: Provider Comparison ===\n");

  const query = "Explain quantum computing in one sentence";

  try {
    console.log("Testing Google Gemini...");
    const googleAgent: Agent = await new AgentBuilder()
      .provider("google")
      .model("models/gemini-2.5-flash")
      .build();

    const response = await googleAgent.chat(query);
    console.log(`Google: ${response.response}\n`);
  } catch (_error) {
    console.log(`Google: (not configured)\n`);
  }
}

async function main(): Promise<void> {
  try {
    await basicGoogleUsage();
    await googleStreaming();
    await providerComparison();

    console.log("=== Examples Complete ===");
    console.log("\nTips:");
    console.log("- Google Gemini offers fast inference");
    console.log("- Streaming provides real-time responses");
    console.log("- Use GOOGLE_API_KEY or GEMINI_API_KEY environment variable");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main().catch(console.error);
