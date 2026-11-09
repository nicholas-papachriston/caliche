import { beforeAll, describe, expect, test } from "bun:test";
import {
  AgentBuilder,
  type DoneEvent,
  isDoneEvent,
  isMetadataEvent,
  isTextChunkEvent,
  type MetadataEvent,
  type StreamEvent,
  type TextChunkEvent,
} from "../src/index";
import { liveSuite } from "./test-env";

const { liveTest, agentBuilder } = await liveSuite("Streaming");

describe("Streaming", () => {
  let agent: Awaited<ReturnType<typeof AgentBuilder.prototype.build>>;

  beforeAll(async () => {
    const builder = agentBuilder();

    agent = await builder.build();
  });

  liveTest("should create a StreamIterator instance", () => {
    const stream = agent.streamChat("Say 'Hello'");
    expect(stream).toBeDefined();
    expect(stream).toBeInstanceOf(Object);
  });

  liveTest("StreamIterator should be async iterable", () => {
    const stream = agent.streamChat("Say 'Hello'");
    expect(stream[Symbol.asyncIterator]).toBeDefined();
    expect(typeof stream[Symbol.asyncIterator]).toBe("function");
  });

  liveTest(
    "should iterate over stream events with for await...of",
    async () => {
      const stream = agent.streamChat("Say 'Hello'");
      const events: StreamEvent[] = [];

      for await (const event of stream) {
        events.push(event);
        if (event.type === "done") {
          break;
        }
      }

      expect(events.length).toBeGreaterThan(0);
      expect(events.some((e) => e.type === "text_chunk" || e.type === "done")).toBe(true);
    },
    30000
  );

  liveTest(
    "should receive text chunk events",
    async () => {
      const stream = agent.streamChat("Say 'Hello'");
      const textChunks: string[] = [];

      for await (const event of stream) {
        if (isTextChunkEvent(event)) {
          textChunks.push(event.content);
        }
        if (isDoneEvent(event)) {
          break;
        }
      }

      expect(textChunks.length).toBeGreaterThan(0);
    },
    30000
  );

  liveTest(
    "should receive metadata event with usage",
    async () => {
      const stream = agent.streamChat("Say 'Hello'");
      let metadataReceived = false;

      for await (const event of stream) {
        if (isMetadataEvent(event)) {
          metadataReceived = true;
          expect(event.usage).toBeDefined();
          break;
        }
        if (isDoneEvent(event)) {
          break;
        }
      }

      expect(metadataReceived).toBe(true);
    },
    30000
  );

  liveTest(
    "should receive done event",
    async () => {
      const stream = agent.streamChat("Say 'Hello'");
      let doneReceived = false;

      for await (const event of stream) {
        if (isDoneEvent(event)) {
          doneReceived = true;
          break;
        }
      }

      expect(doneReceived).toBe(true);
    },
    30000
  );

  liveTest(
    "should handle stream cancellation",
    async () => {
      const stream = agent.streamChat("Count from 1 to 100");
      let eventCount = 0;

      for await (const _event of stream) {
        eventCount++;
        if (eventCount >= 3) {
          break;
        }
      }

      expect(eventCount).toBeLessThanOrEqual(3);
    },
    30000
  );

  liveTest(
    "should accumulate text from multiple chunks",
    async () => {
      const stream = agent.streamChat("Say 'Hello World'");
      let fullText = "";

      for await (const event of stream) {
        if (isTextChunkEvent(event)) {
          fullText += event.content;
        }
        if (isDoneEvent(event)) {
          break;
        }
      }

      expect(fullText.length).toBeGreaterThan(0);
    },
    30000
  );

  test("isTextChunkEvent should identify text chunk events", () => {
    const event: StreamEvent = { type: "text_chunk", content: "Hello" } as TextChunkEvent;
    expect(isTextChunkEvent(event)).toBe(true);
  });

  test("isTextChunkEvent should reject non-text chunk events", () => {
    const event: StreamEvent = { type: "metadata", metadata: {} } as MetadataEvent;
    expect(isTextChunkEvent(event)).toBe(false);
  });

  test("isMetadataEvent should identify metadata events", () => {
    const event: StreamEvent = {
      type: "metadata",
      metadata: {},
      usage: { total_tokens: 100 },
    } as MetadataEvent;
    expect(isMetadataEvent(event)).toBe(true);
  });

  test("isMetadataEvent should reject non-metadata events", () => {
    const event: StreamEvent = { type: "text_chunk", content: "Hello" } as TextChunkEvent;
    expect(isMetadataEvent(event)).toBe(false);
  });

  test("isDoneEvent should identify done events", () => {
    const event: StreamEvent = { type: "done" } as DoneEvent;
    expect(isDoneEvent(event)).toBe(true);
  });

  test("isDoneEvent should reject non-done events", () => {
    const event: StreamEvent = { type: "text_chunk", content: "Hello" } as TextChunkEvent;
    expect(isDoneEvent(event)).toBe(false);
  });
});
