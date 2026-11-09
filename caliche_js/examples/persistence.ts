/**
 * Example: Memory Persistence in TypeScript
 *
 * Demonstrates how to use memory persistence to save and restore conversation state.
 *
 * bun run examples/memory-persistence.ts
 */

import * as fs from "node:fs/promises";
import * as path from "node:path";
import { AgentBuilder, type MemoryPersistence, type MemorySnapshot } from "../src/index";

// File-based persistence implementation
const filePersistence: MemoryPersistence = {
  save_snapshot: async (conversationId: string, snapshot: MemorySnapshot): Promise<void> => {
    const dir = "./conversations";
    await fs.mkdir(dir, { recursive: true });
    const filePath = path.join(dir, `${conversationId}.json`);
    await fs.writeFile(
      filePath,
      JSON.stringify(
        {
          messages: snapshot.messages,
          metadata: snapshot.metadata,
        },
        null,
        2
      )
    );
  },

  load_snapshot: async (conversationId: string): Promise<MemorySnapshot | undefined> => {
    const filePath = path.join("./conversations", `${conversationId}.json`);
    try {
      const data = await fs.readFile(filePath, "utf-8");
      const parsed = JSON.parse(data);
      return {
        messages: parsed.messages,
        metadata: parsed.metadata || {},
      };
    } catch {
      return undefined;
    }
  },

  clear_snapshot: async (conversationId: string): Promise<void> => {
    const filePath = path.join("./conversations", `${conversationId}.json`);
    try {
      await fs.unlink(filePath);
    } catch {
      // File doesn't exist, ignore
    }
  },
};

async function main() {
  const conversationId = "user-123-session-1";

  // Create agent with persistence
  // Using LM Studio (no API key required - runs locally at http://localhost:1234/v1)
  const agent = await new AgentBuilder()
    .provider("lmstudio")
    .model("local-model")
    .persistMemoryWith(filePersistence, conversationId, true)
    .build();

  // First conversation
  console.log("=== First Session ===");
  const response1 = await agent.chat("My name is Alice and I'm a software engineer.");
  console.log(`Agent: ${response1.response}\n`);

  const response2 = await agent.chat("What's my profession?");
  console.log(`Agent: ${response2.response}\n`);

  // Later, create a new agent with the same conversation ID
  console.log("=== Second Session (Resumed) ===");
  const agent2 = await new AgentBuilder()
    .provider("lmstudio")
    .model("local-model")
    .persistMemoryWith(filePersistence, conversationId, true)
    .build();

  // Agent remembers previous conversation
  const response3 = await agent2.chat("What's my name and profession?");
  console.log(`Agent: ${response3.response}\n`);

  // Clear the conversation
  await filePersistence.clear_snapshot(conversationId);
  console.log("Conversation cleared.");
}

main().catch(console.error);
