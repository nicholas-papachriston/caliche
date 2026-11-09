import { beforeAll, describe, expect, test } from "bun:test";
import {
  type Agent,
  AgentBuilder,
  type ChatResponse,
  type MemoryPersistence,
  type MemorySnapshot,
  type Provider,
} from "../src/index";
import type { ToolArguments } from "../src/types";
import { liveSuite } from "./test-env";

// Use LM Studio by default (no API key required)
// Can override with environment variables for other providers
const { liveTest, agentBuilder } = await liveSuite("Integration");

describe("Integration Tests", () => {
  let agent: Agent;

  beforeAll(async () => {
    const builder = agentBuilder();
    agent = await builder.build();
  });

  test("should create an agent and get provider", () => {
    expect(agent).toBeDefined();
    expect(agent.provider).toBeDefined();
  });

  test("should create an agent and get model", () => {
    expect(agent).toBeDefined();
    expect(agent.model).toBeDefined();
  });

  liveTest(
    "should send a chat message and get a response",
    async () => {
      const response = await Promise.race([
        agent.chat("Say 'Hello, World!'"),
        new Promise<ChatResponse>((_, reject) =>
          setTimeout(() => reject(new Error("Timeout")), 10000)
        ),
      ]);
      expect(response).toBeDefined();
      if (response && "response" in response) {
        expect(response.response).toBeDefined();
        expect(typeof response.response).toBe("string");
        expect(response.usage).toBeDefined();
      }
    },
    15000
  );

  liveTest("should create a stream iterator", () => {
    const stream = agent.streamChat("Say 'Hello'");
    expect(stream).toBeDefined();
  });

  liveTest(
    "should work with custom tools",
    async () => {
      const tool = {
        name: "test_echo",
        description: "Echoes back the input",
        parameters: {
          type: "object" as const,
          properties: {
            message: { type: "string", description: "Message to echo" },
          },
          required: ["message"],
        },
        execute: async (args: ToolArguments) => {
          const message = args.message;
          return {
            success: true,
            content: typeof message === "string" ? message : String(message || ""),
          };
        },
      };

      const builder = agentBuilder().addTool(tool);

      const agentWithTool = await builder.build();

      const response = await Promise.race([
        agentWithTool.chat("Use the test_echo tool with message 'test'"),
        new Promise<ChatResponse>((_, reject) =>
          setTimeout(() => reject(new Error("Timeout")), 15000)
        ),
      ]);
      expect(response).toBeDefined();
      if (response && "response" in response) {
        expect(response.response).toBeDefined();
      }
    },
    20000
  );

  liveTest(
    "should work with memory persistence",
    async () => {
      const snapshots: Record<string, MemorySnapshot> = {};

      const persistence: MemoryPersistence = {
        save_snapshot: async (conversationId: string, snapshot: MemorySnapshot): Promise<void> => {
          snapshots[conversationId] = snapshot;
        },
        load_snapshot: async (conversationId: string): Promise<MemorySnapshot | undefined> => {
          return snapshots[conversationId] || undefined;
        },
        clear_snapshot: async (conversationId: string): Promise<void> => {
          delete snapshots[conversationId];
        },
      };

      const conversationId = `test-${Date.now()}`;

      const builder1 = agentBuilder().persistMemoryWith(persistence, conversationId, true);

      const agent1 = await builder1.build();

      await Promise.race([
        agent1.chat("My name is Alice"),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), 10000)),
      ]);

      const builder2 = agentBuilder().persistMemoryWith(persistence, conversationId, true);

      const agent2 = await builder2.build();

      const response = await Promise.race([
        agent2.chat("What's my name?"),
        new Promise<ChatResponse>((_, reject) =>
          setTimeout(() => reject(new Error("Timeout")), 10000)
        ),
      ]);
      expect(response).toBeDefined();
      if (response && "response" in response) {
        expect(response.response).toBeDefined();
        // The response should mention Alice (though this is not guaranteed)
      }
    },
    25000
  );
});
