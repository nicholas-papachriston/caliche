/**
 * Agent Creation and Configuration Example
 *
 * This example demonstrates:
 * - Creating agents with explicit configuration
 * - Runtime configuration with a single builder path
 * - Custom system prompts for specialized agents
 * - Explicit tool selection
 * - Multiple chat turns with context
 * - Shared state management
 * - Memory persistence
 *
 * Best practices shown:
 * - Environment variable validation
 * - Proper error handling
 * - Token usage tracking
 * - Consistent, fluent configuration patterns
 * - Type-safe agent creation
 *
 * Run with:
 * ```bash
 * bun run examples/agent.ts
 * ```
 *
 * Set environment variables and consume them through runtimeConfig:
 * ```bash
 * export OPENAI_API_KEY=...
 * bun run examples/agent.ts
 * */

import * as fs from "node:fs/promises";
import * as path from "node:path";
import {
  type Agent,
  AgentBuilder,
  type AgentRuntimeConfig,
  type ChatResponse,
  isMetadataEvent,
  isTextChunkEvent,
  type MemoryPersistence,
  type MemorySnapshot,
  SharedState,
  type StreamIterator,
  type TokenUsage,
} from "../src/index";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

// File-based memory persistence
class FileMemoryPersistence implements MemoryPersistence {
  private baseDir: string;

  constructor(baseDir: string = "./conversations") {
    this.baseDir = baseDir;
  }

  private getFilePath(conversationId: string): string {
    return path.join(this.baseDir, `${conversationId}.json`);
  }

  async save_snapshot(conversationId: string, snapshot: MemorySnapshot): Promise<void> {
    try {
      await fs.mkdir(this.baseDir, { recursive: true });
      const filePath = this.getFilePath(conversationId);
      await fs.writeFile(
        filePath,
        JSON.stringify(
          {
            messages: snapshot.messages,
            metadata: snapshot.metadata,
            saved_at: new Date().toISOString(),
          },
          null,
          2
        )
      );
      console.log(`[Persistence] Saved snapshot for conversation: ${conversationId}`);
    } catch (error) {
      console.error(`[Persistence] Failed to save snapshot: ${error}`);
      throw error;
    }
  }

  async load_snapshot(conversationId: string): Promise<MemorySnapshot | undefined> {
    try {
      const filePath = this.getFilePath(conversationId);
      const data = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(data);
      console.log(`[Persistence] Loaded snapshot for conversation: ${conversationId}`);
      return {
        messages: parsed.messages || [],
        metadata: parsed.metadata || {},
      };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        console.log(`[Persistence] No existing snapshot for conversation: ${conversationId}`);
        return undefined;
      }
      console.error(`[Persistence] Failed to load snapshot: ${error}`);
      throw error;
    }
  }

  async clear_snapshot(conversationId: string): Promise<void> {
    try {
      const filePath = this.getFilePath(conversationId);
      await fs.unlink(filePath);
      console.log(`[Persistence] Cleared snapshot for conversation: ${conversationId}`);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        console.error(`[Persistence] Failed to clear snapshot: ${error}`);
        throw error;
      }
    }
  }
}

async function explicitConfiguration(): Promise<void> {
  console.log("=== Example 1: Explicit Configuration ===\n");

  // Create and configure an agent with explicit settings
  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .systemPrompt("You are a helpful assistant.")
    .build();

  console.log(`Agent created with provider: ${agent.provider}, model: ${agent.model}`);

  // Send a message with typed response
  const response: ChatResponse = await agent.chat("What is 15 * 23?");

  console.log("\nQuery: What is 15 * 23?");
  console.log("Response:", response.response);
  console.log("Usage:", response.usage);

  // Type-safe access to usage
  const usage: TokenUsage = response.usage;
  if (usage.total_tokens) {
    console.log(`Total tokens used: ${usage.total_tokens}`);
  }
}

async function multipleChatTurns(): Promise<void> {
  console.log("\n=== Example 2: Multiple Chat Turns (Context) ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .build();

  // First turn
  const response1 = await agent.chat("Remember this number: 42. What is that number plus 10?");
  console.log("Query: Remember this number: 42. What is that number plus 10?");
  console.log("Response:", response1.response);

  // Second turn - agent remembers context
  const response2 = await agent.chat("What was the number I asked you to remember?");
  console.log("\nQuery: What was the number I asked you to remember?");
  console.log("Response:", response2.response);
}

async function runtimeConfiguration(): Promise<void> {
  console.log("\n=== Example 3: Runtime Configuration ===\n");
  const config: AgentRuntimeConfig = {
    provider: "bedrock",
    model: "anthropic.claude-3-sonnet-20240229-v1:0",
    credentials: {
      defaultApiKey: process.env.OPENAI_API_KEY,
    },
    timeouts: {
      streamEventTimeoutSecs: 240,
    },
  };
  const agent: Agent = await new AgentBuilder()
    .runtimeConfig(config)
    .enableTools(["calculator", "fs"])
    .systemPrompt("You are a helpful assistant.")
    .build();

  console.log(`Agent created from runtimeConfig: ${agent.provider}, ${agent.model}`);
  const response: ChatResponse = await agent.chat("Echo this: Hello from runtime-config agent!");
  console.log("Query: Echo this: Hello from runtime-config agent!");
  console.log("Response:", response.response);
}

async function customSystemPrompt(): Promise<void> {
  console.log("\n=== Example 5: Custom System Prompt ===\n");

  const codingAgent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .systemPrompt(
      `You are a Rust programming expert. 
When answering questions, focus on Rust best practices, 
memory safety, and performance considerations.`
    )
    .enableTool("fs")
    .enableTool("fs")
    .build();

  const response = await codingAgent.chat("What are the key principles of Rust memory safety?");
  console.log("Query: What are the key principles of Rust memory safety?");
  console.log("Response:", response.response);
}

async function allDefaultTools(): Promise<void> {
  console.log("\n=== Example 6: All Default Tools ===\n");

  try {
    const fullAgent: Agent = await new AgentBuilder()
      .provider("bedrock")
      .enableAllDefaultTools()
      .build();

    console.log(`Agent created with all default tools enabled`);
    console.log(`Provider: ${fullAgent.provider}, Model: ${fullAgent.model}`);
    console.log("(Note: listTools() method not available in JS bindings yet)");
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function streamingChat(): Promise<void> {
  console.log("\n=== Example 8: Streaming Chat ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .systemPrompt("You are a storyteller.")
    .build();

  const stream: StreamIterator = agent.streamChat("Tell me a very short story about TypeScript.");

  // Type-safe streaming with for-await-of
  for await (const event of stream) {
    // Type guards provide narrowing
    if (isTextChunkEvent(event)) {
      // event is now typed as TextChunkEvent
      process.stdout.write(event.content);
    } else if (isMetadataEvent(event)) {
      // event is now typed as MetadataEvent
      console.log("\n\nMetadata:", {
        usage: event.usage,
        finish_reason: event.finish_reason,
      });
    }
  }
}

async function toolUsage(): Promise<void> {
  console.log("\n=== Example 9: Tool Usage ===\n");

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .systemPrompt("You are a helpful assistant with access to tools.")
    .enableTools(["calculator"])
    .build();

  // Strongly typed args inferred from JSON schema (no casts needed).
  const localMathSchema = {
    type: "object",
    properties: {
      a: { type: "number" },
      b: { type: "number" },
    },
    required: ["a", "b"],
  } as const;

  agent.addTool("local_math", "Adds two numbers locally", localMathSchema, async (args) => {
    return { success: true, content: String(args.a + args.b) };
  });

  // Dynamic schemas default handler args to unknown, so narrow before use.
  const dynamicSchema = '{"type":"object","properties":{"input":{"type":"string"}}}';
  agent.addTool("echo_unknown", "Echoes unknown payloads safely", dynamicSchema, async (args) => {
    if (isRecord(args) && "input" in args) {
      const input = typeof args.input === "string" ? args.input : JSON.stringify(args.input);
      return { success: true, content: input ?? "" };
    }
    return { success: false, content: "", error: "Expected an object with an `input` field" };
  });

  const stream: StreamIterator = agent.streamChat("What is 2 + 2?");

  for await (const event of stream) {
    switch (event.type) {
      case "text_chunk":
        process.stdout.write(event.content);
        break;

      case "tool_call_start":
        console.log(`\nTool called: ${event.name}`);
        console.log(`Arguments: ${event.arguments}`);
        break;

      case "tool_call_result":
        console.log(`\nTool result: ${event.result}`);
        break;

      case "metadata":
        if (event.usage) {
          console.log("\nToken usage:", event.usage);
        }
        break;
    }
  }
}

async function memoryPersistence(): Promise<void> {
  console.log("\n=== Example 10: Memory Persistence ===\n");

  const persistence = new FileMemoryPersistence("./conversations");
  const conversationId = `chat-${Date.now()}`;

  const agent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .systemPrompt("You are a helpful assistant.")
    .bufferMemory({
      max_tokens: 4096,
      min_messages: 4,
      preserve_system: true,
      summary_placeholder: true,
    })
    .persistMemoryWith(persistence, conversationId, true)
    .build();

  console.log(`Conversation ID: ${conversationId}`);

  // First conversation
  const response1 = await agent.chat("Hello! My name is Alice.");
  console.log("Query: Hello! My name is Alice.");
  console.log("Response:", response1.response);

  const response2 = await agent.chat("What's my name?");
  console.log("\nQuery: What's my name?");
  console.log("Response:", response2.response);

  // Clear and restart
  await persistence.clear_snapshot(conversationId);
  console.log("\nCleared conversation, starting fresh...");

  const freshAgent: Agent = await new AgentBuilder()
    .provider("bedrock")
    .model("anthropic.claude-3-sonnet-20240229-v1:0")
    .systemPrompt("You are a helpful assistant.")
    .bufferMemory({
      max_tokens: 4096,
      min_messages: 4,
      preserve_system: true,
      summary_placeholder: true,
    })
    .persistMemoryWith(persistence, conversationId, true)
    .build();

  const response3 = await freshAgent.chat("Hello, I'm Bob.");
  console.log("\nQuery: Hello, I'm Bob.");
  console.log("Response:", response3.response);

  const response4 = await freshAgent.chat("What's my name?");
  console.log("\nQuery: What's my name?");
  console.log("Response:", response4.response);
}

async function sharedStateManagement(): Promise<void> {
  console.log("\n=== Example 11: Shared State Management ===\n");

  // Create a shared state that multiple agents can access
  const sharedState = new SharedState();

  // Initialize state with some data
  console.log("1. Setting initial state values...");
  await sharedState.set("/order_id", "ORD-12345");
  await sharedState.set("/customer/name", "Alice");
  await sharedState.set("/customer/email", "alice@example.com");

  console.log("   Set order_id: ORD-12345");
  console.log("   Set customer name: Alice");
  console.log("   Set customer email: alice@example.com\n");

  // Get individual values
  console.log("2. Retrieving values from state...");
  const orderId = await sharedState.get("/order_id");
  const customerName = await sharedState.get("/customer/name");

  console.log(`   Order ID: ${orderId}`);
  console.log(`   Customer Name: ${customerName}\n`);

  // Get all state
  console.log("3. Getting entire state...");
  const allState = await sharedState.getAll();
  console.log(`   Full state: ${JSON.stringify(allState, null, 2)}\n`);

  // Update state
  console.log("4. Updating order status...");
  await sharedState.set("/order/status", "processing");
  const status = await sharedState.get("/order/status");
  console.log(`   Order Status: ${status}\n`);

  // Demonstrate nested paths
  console.log("5. Setting nested values...");
  await sharedState.set("/order/items/0/name", "Widget");
  await sharedState.set("/order/items/0/quantity", 2);

  const itemName = await sharedState.get("/order/items/0/name");
  const itemQuantity = await sharedState.get("/order/items/0/quantity");

  console.log(`   Item Name: ${itemName}`);
  console.log(`   Item Quantity: ${itemQuantity}\n`);

  // Final state
  console.log("6. Final state:");
  const finalState = await sharedState.getAll();
  console.log(JSON.stringify(finalState, null, 2));
}

async function errorHandling(): Promise<void> {
  console.log("\n=== Example 12: Error Handling ===\n");

  try {
    const agent: Agent = await new AgentBuilder()
      .provider("bedrock")
      .model("anthropic.claude-3-sonnet-20240229-v1:0")
      .build();

    const response: ChatResponse = await agent.chat("Hello");
    console.log("Response:", response.response);
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    } else {
      console.error("Unknown error:", error);
    }
  }
}

// Main execution
async function main(): Promise<void> {
  try {
    // Configuration pattern examples
    await explicitConfiguration();
    await multipleChatTurns();
    await runtimeConfiguration();
    await customSystemPrompt();
    await allDefaultTools();

    // Feature examples
    await streamingChat();
    await toolUsage();
    await memoryPersistence();
    await sharedStateManagement();
    await errorHandling();

    console.log("\n=== Examples Complete ===");
    console.log("\nTips:");
    console.log("- Use one canonical path: new AgentBuilder().runtimeConfig(...).build()");
    console.log("- Add fluent overrides only where needed for local specialization");
    console.log("- Shared state enables agents to coordinate and share data");
    console.log("- Memory persistence allows conversations to survive restarts");
  } catch (error) {
    console.error("Fatal error:", error);
    process.exit(1);
  }
}

main().catch(console.error);
