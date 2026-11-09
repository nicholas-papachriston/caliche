/**
 * Durable Streams Example
 *
 * Shows how to configure an agent with a durable stream backend so that events
 * can be persisted for replay and resume (e.g. when using the API or a harness
 * with SSE and Last-Event-ID).
 *
 * Run with:
 *   bun run examples/durable_streams.ts
 *
 * For Redis backend, use DurableStreamBackend.redis("redis://localhost:6379/").
 */

import {
  AgentBuilder,
  DurableStreamBackend,
  isMetadataEvent,
  isTextChunkEvent,
  type StreamEvent,
} from "../src/index";

async function inMemoryBackend(): Promise<void> {
  console.log("=== In-memory durable stream (capacity 10_000) ===\n");

  const agent = await new AgentBuilder()
    .durableStreamBackend(DurableStreamBackend.memory(10_000))
    .provider("openai")
    .model("gpt-4o-mini")
    .build();

  console.log("Agent built with in-memory durable stream.\n");

  const response = await agent.chat("Say hello in one short sentence.");
  console.log(`Response: ${response.response}\n`);
  if (response.usage?.total_tokens) {
    console.log(`Tokens: ${response.usage.total_tokens}\n`);
  }
}

async function redisBackend(): Promise<void> {
  console.log("=== Redis durable stream ===\n");

  try {
    const agent = await new AgentBuilder()
      .durableStreamBackend(DurableStreamBackend.redis("redis://127.0.0.1:6379/"))
      .provider("openai")
      .model("gpt-4o-mini")
      .build();

    console.log("Agent built with Redis durable stream.\n");

    const response = await agent.chat("Say hello in one short sentence.");
    console.log(`Response: ${response.response}\n`);
    if (response.usage?.total_tokens) {
      console.log(`Tokens: ${response.usage.total_tokens}\n`);
    }
  } catch (e) {
    console.error("Redis backend failed (is Redis running?):", e, "\n");
  }
}

async function streamingWithDurableBackend(): Promise<void> {
  console.log("=== Streaming with durable stream ===\n");

  const agent = await new AgentBuilder()
    .durableStreamBackend(DurableStreamBackend.memory(5_000))
    .provider("openai")
    .model("gpt-4o-mini")
    .build();

  const stream = agent.streamChat("Say hello in one short sentence.");
  process.stdout.write("Response (streaming): ");
  for await (const event of stream) {
    const e = event as StreamEvent;
    if (isTextChunkEvent(e)) {
      process.stdout.write(e.content);
    }
    if (isMetadataEvent(e) && e.usage?.total_tokens) {
      console.log(`\nTokens: ${e.usage.total_tokens}`);
    }
  }
  console.log("\n");
}

async function main(): Promise<void> {
  await inMemoryBackend();
  await redisBackend();
  await streamingWithDurableBackend();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
