/**
 * AWS Bedrock Provider Example
 *
 * This example demonstrates AWS Bedrock streaming capabilities with the fully
 * type-safe streaming implementation.
 *
 * Best practices shown:
 * - Bedrock provider configuration with AWS credentials
 * - Real-time streaming with proper event handling
 * - Token usage tracking from Bedrock streaming events
 * - Error handling for AWS-specific errors
 * - Multiple Bedrock model examples
 * - Tool calls in streaming mode
 *
 * Prerequisites:
 * - AWS credentials configured (via environment variables or IAM role)
 * - AWS_REGION set (e.g., "us-east-1")
 * - BEDROCK_MODEL_ID set (optional, defaults to Claude Sonnet)
 *
 * Run with:
 * ```bash
 * export AWS_REGION=us-east-1
 * export AWS_ACCESS_KEY_ID=your_key
 * export AWS_SECRET_ACCESS_KEY=your_secret
 * bun run examples/bedrock.ts
 * ```
 */

import {
  type Agent,
  AgentBuilder,
  isMetadataEvent,
  isTextChunkEvent,
  type StreamIterator,
} from "../src/index";

async function basicBedrockStreaming(): Promise<void> {
  console.log("=== Example 1: Basic Bedrock Streaming ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    .build();

  console.log("Query: Explain quantum computing in one sentence");
  console.log("Response (streaming):\n");

  const stream: StreamIterator = agent.streamChat("Explain quantum computing in one sentence");

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
}

async function bedrockModels(): Promise<void> {
  console.log("\n=== Example 2: Different Bedrock Models ===\n");

  const models = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-opus-4-20250514-v1:0",
  ];

  for (const model of models) {
    try {
      console.log(`Testing model: ${model}`);
      const agent: Agent = await new AgentBuilder().provider("bedrock").model(model).build();

      const response = await agent.chat("Say hello in one sentence.");
      console.log(`Response: ${response.response}\n`);
    } catch (error) {
      console.log(`(Skipped: ${error instanceof Error ? error.message : "Model not available"})\n`);
    }
  }
}

async function bedrockStreamingWithTools(): Promise<void> {
  console.log("\n=== Example 3: Streaming with Tools ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    .enableTool("calculator")
    .build();

  console.log("Query: What is 15 * 23? Show your work.");
  console.log("Response (streaming):\n");

  const stream: StreamIterator = agent.streamChat("What is 15 * 23? Show your work.");

  for await (const event of stream) {
    switch (event.type) {
      case "text_chunk":
        process.stdout.write(event.content);
        break;

      case "tool_call_start":
        console.log(`\n[Tool Call] ${event.name}`);
        console.log(`Arguments: ${JSON.stringify(event.arguments)}\n`);
        break;

      case "tool_call_result":
        console.log(`\n[Tool Result] ${event.result}\n`);
        break;

      case "metadata":
        if (event.usage) {
          console.log(`\n\nToken usage: ${event.usage.total_tokens || "N/A"}`);
        }
        break;
    }
  }

  console.log("\n");
}

async function longFormStreaming(): Promise<void> {
  console.log("\n=== Example 4: Long-form Content Streaming ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    .build();

  console.log("Query: Write a short story about a robot.");
  console.log("Response (streaming):\n");

  const stream: StreamIterator = agent.streamChat("Write a short story about a robot.");

  for await (const event of stream) {
    if (isTextChunkEvent(event)) {
      process.stdout.write(event.content);
    }
  }

  console.log("\n");
}

async function main(): Promise<void> {
  try {
    await basicBedrockStreaming();
    await bedrockModels();
    await bedrockStreamingWithTools();
    await longFormStreaming();

    console.log("=== Examples Complete ===");
    console.log("\nTips:");
    console.log("- Use Bedrock for AWS-native integration");
    console.log("- Streaming provides better user experience");
    console.log("- Different models have different capabilities and costs");
  } catch (error) {
    console.error("Error:", error);
    if (error instanceof Error && error.message.includes("credentials")) {
      console.log("\nNote: Make sure AWS credentials are configured:");
      console.log("  - Set AWS_REGION environment variable");
      console.log("  - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY");
      console.log("  - Or use IAM role if running on EC2");
    }
    process.exit(1);
  }
}

main().catch(console.error);
