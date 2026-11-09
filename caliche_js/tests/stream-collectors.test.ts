import { describe, expect, it } from "bun:test";
import { collectStreamResult, collectText, collectWithUsage } from "../src/streaming/collectors";
import type { StreamEvent } from "../src/streaming/events";

async function* events(): AsyncGenerator<StreamEvent> {
  yield { type: "text_chunk", content: "Hello" };
  yield { type: "text_chunk", content: " world" };
  yield {
    type: "tool_call_start",
    id: "1",
    name: "lookup",
    arguments: { q: "x" },
  };
  yield {
    type: "metadata",
    finish_reason: "stop",
    usage: { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 },
    metadata: {},
  };
  yield { type: "done" };
}

describe("stream collectors", () => {
  it("collects text", async () => {
    expect(await collectText(events())).toBe("Hello world");
  });

  it("collects usage and finish reason", async () => {
    const result = await collectWithUsage(events());
    expect(result.text).toBe("Hello world");
    expect(result.finishReason).toBe("stop");
    expect(result.usage?.total_tokens).toBe(3);
  });

  it("collects a full stream result", async () => {
    const result = await collectStreamResult(events());
    expect(result.toolCalls).toHaveLength(1);
    expect(result.toolCalls[0]?.name).toBe("lookup");
    expect(result.errors).toEqual([]);
  });
});
