/**
 * Memory Strategies Example
 *
 * Demonstrates the three memory strategies available in Caliche:
 * - Buffer Memory: Fixed window of recent messages
 * - Summary Memory: Automatically summarizes older messages
 * - Entity Memory: Tracks and injects relevant entities
 *
 * bun run examples/memory.ts
 */

import { AgentBuilder } from "../src/index";

async function demonstrateBufferMemory() {
  console.log("=== Buffer Memory Example ===\n");

  // Buffer memory retains a fixed window of recent messages
  // Best for: Short conversations with predictable token usage
  // Using LM Studio (no API key required - runs locally at http://localhost:1234/v1)
  const agent = await new AgentBuilder()
    .provider("lmstudio")
    .model("local-model")
    .bufferMemory({
      max_tokens: 4096, // Maximum tokens to retain
      min_messages: 4, // Always keep at least 4 messages
      preserve_system: true, // Preserve system message
      summary_placeholder: true, // Add placeholder when truncating
    })
    .build();

  console.log("Buffer memory configured:");
  console.log("- Retains up to 4096 tokens");
  console.log("- Keeps minimum 4 messages");
  console.log("- Preserves system message");
  console.log("- Adds placeholder when older messages are truncated\n");

  // Example conversation
  const response1 = await agent.chat("My name is Alice and I'm a software engineer.");
  console.log("Response 1:", `${response1.response.substring(0, 100)}...\n`);

  const response2 = await agent.chat("What's my profession?");
  console.log("Response 2:", response2.response);
  console.log("(Agent remembers: Alice is a software engineer)\n");
}

async function demonstrateSummaryMemory() {
  console.log("=== Summary Memory Example ===\n");

  // Summary memory automatically summarizes older messages
  // Best for: Extended conversations where you need to retain context
  // Using LM Studio (no API key required - runs locally at http://localhost:1234/v1)
  const agent = await new AgentBuilder()
    .provider("lmstudio")
    .model("local-model")
    .summaryMemory({
      max_tokens: 8192, // Maximum token budget
      summarize_threshold_pct: 0.8, // Trigger summarization at 80%
      chunk_size: 12, // Messages per summarization batch
      max_summaries: 8, // Maximum summaries to retain
    })
    .build();

  console.log("Summary memory configured:");
  console.log("- Maximum 8192 tokens");
  console.log("- Summarizes when 80% of token budget is used");
  console.log("- Processes 12 messages per batch");
  console.log("- Retains up to 8 summaries\n");

  // Example: Long conversation
  const messages = [
    "I'm planning a trip to Japan.",
    "I want to visit Tokyo, Kyoto, and Osaka.",
    "I'm interested in temples, gardens, and food.",
    "I'll be traveling in spring.",
    "I have a budget of $3000.",
  ];

  for (const message of messages) {
    const response = await agent.chat(message);
    console.log(`Q: ${message}`);
    console.log(`A: ${response.response.substring(0, 100)}...\n`);
  }

  // Later query that requires context from earlier messages
  const finalResponse = await agent.chat("Summarize my travel plans.");
  console.log("Final response:", `${finalResponse.response.substring(0, 200)}...\n`);
}

async function demonstrateEntityMemory() {
  console.log("=== Entity Memory Example ===\n");

  // Entity memory tracks important entities (people, places, concepts)
  // Best for: Conversations where tracking specific entities is important
  // Using LM Studio (no API key required - runs locally at http://localhost:1234/v1)
  const agent = await new AgentBuilder()
    .provider("lmstudio")
    .model("local-model")
    .entityMemory({
      max_tokens: 6144, // Maximum token budget
      max_entities: 128, // Maximum entities to track
      extraction_interval: 3, // Extract entities every 3 messages
      similarity_threshold: 0.65, // Relevance threshold for injection
    })
    .build();

  console.log("Entity memory configured:");
  console.log("- Maximum 6144 tokens");
  console.log("- Tracks up to 128 entities");
  console.log("- Extracts entities every 3 messages");
  console.log("- Uses 0.65 similarity threshold\n");

  // Example conversation with entities
  const response1 = await agent.chat("My name is Bob. I work at Acme Corp as a data scientist.");
  console.log("Response 1:", `${response1.response.substring(0, 100)}...\n`);

  const response2 = await agent.chat("I live in San Francisco with my wife Sarah.");
  console.log("Response 2:", `${response2.response.substring(0, 100)}...\n`);

  const response3 = await agent.chat("We have two cats: Fluffy and Whiskers.");
  console.log("Response 3:", `${response3.response.substring(0, 100)}...\n`);

  // Query that references tracked entities
  const response4 = await agent.chat("Tell me about my family.");
  console.log("Response 4:", response4.response);
  console.log("(Agent tracks: Bob, Acme Corp, San Francisco, Sarah, Fluffy, Whiskers)\n");
}

async function compareMemoryStrategies() {
  console.log("=== Memory Strategy Comparison ===\n");

  console.log("Buffer Memory:");
  console.log("  Pro: Simple and predictable");
  console.log("  Pro: Low overhead");
  console.log("  Con: Limited context window");
  console.log("  Best for: Short conversations, simple use cases\n");

  console.log("Summary Memory:");
  console.log("  Pro: Retains context over many messages");
  console.log("  Pro: Automatically manages token budget");
  console.log("  Con: Requires summarization (costs tokens)");
  console.log("  Best for: Long conversations, extended context\n");

  console.log("Entity Memory:");
  console.log("  Pro: Tracks important entities");
  console.log("  Pro: Injects relevant context");
  console.log("  Con: More complex, requires entity extraction");
  console.log("  Best for: Entity-focused conversations\n");
}

async function main() {
  console.log("Using LM Studio (no API key required - runs locally at http://localhost:1234/v1)\n");

  try {
    await demonstrateBufferMemory();
    await demonstrateSummaryMemory();
    await demonstrateEntityMemory();
    await compareMemoryStrategies();
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

main().catch(console.error);
