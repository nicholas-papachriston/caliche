/**
 * Streaming Responses Example
 *
 * This example demonstrates how to use the Agent's streaming API directly
 * for real-time token-by-token output.
 *
 * Best practices shown:
 * - Proper stream event handling
 * - Real-time token display with flushing
 * - Tool call detection in streams
 * - Graceful error handling
 * - Token usage tracking
 *
 * Run with:
 * ```bash
 * bun run examples/streaming.ts
 * */

import {
  type Agent,
  AgentBuilder,
  isMetadataEvent,
  isTextChunkEvent,
  type StreamIterator,
} from "../src/index";

async function basicStreaming(): Promise<void> {
  console.log("=== Example 1: Basic Streaming ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .build();

  console.log("Query: Write a haiku about programming");
  console.log("Response (streaming):");

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
}

async function streamingWithTools(): Promise<void> {
  console.log("\n=== Example 2: Streaming with Tool Calls ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .enableTool("calculator")
    .build();

  console.log("Query: What is 15 * 23? Show your work.");
  console.log("Response (streaming):");

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

async function main(): Promise<void> {
  try {
    await basicStreaming();
    await streamingWithTools();

    console.log("=== Examples Complete ===");
    console.log("\nTips:");
    console.log("- Use streaming for better user experience");
    console.log("- Handle tool calls in streams for interactive agents");
    console.log("- Monitor token usage for cost tracking");
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main().catch(console.error);
