import { describe, expect, it } from "bun:test";
import {
  createInitialAgentStreamState,
  fromEventEnvelope,
  normalizeHarnessDisciplineMetrics,
  reduceAgentStreamEvent,
  type StreamEvent,
  type ToolCallResultEvent,
  toEventEnvelope,
} from "../src/streaming/events";
import { hydrateStreamEvent, normalizeRawStreamEvent } from "../src/streaming/stream-normalization";

describe("stream event normalization", () => {
  it("parses versioned envelopes into typed events", () => {
    const event = normalizeRawStreamEvent({
      version: "1",
      type: "text_chunk",
      payload: { type: "text_chunk", content: "hello" },
    });
    expect(event.type).toBe("text_chunk");
    if (event.type === "text_chunk") {
      expect(event.content).toBe("hello");
    }
  });

  it("unwraps nested event field before normalizing", () => {
    const event = normalizeRawStreamEvent({
      event: {
        type: "done",
      },
    });
    expect(event.type).toBe("done");
  });

  it("returns error for non-objects", () => {
    const event = normalizeRawStreamEvent(null);
    expect(event.type).toBe("error");
    if (event.type === "error") {
      expect(event.message).toContain("not an object");
    }
  });

  it("returns error when type is missing", () => {
    const event = normalizeRawStreamEvent({ content: "orphan" });
    expect(event.type).toBe("error");
    if (event.type === "error") {
      expect(event.message).toContain("missing a type field");
    }
  });

  it("round-trips event_id through envelopes", () => {
    const original: StreamEvent = {
      type: "text_chunk",
      content: "chunk",
      event_id: "evt-1",
    };
    const envelope = toEventEnvelope(original);
    const restored = fromEventEnvelope(envelope);
    expect(restored.event_id).toBe("evt-1");
    expect(restored.type).toBe("text_chunk");
  });

  it("accumulates text and tracks policy violations", () => {
    let state = createInitialAgentStreamState();
    state = reduceAgentStreamEvent(state, { type: "text_chunk", content: "Hi" });
    state = reduceAgentStreamEvent(state, {
      type: "policy_violation",
      policy: "require_tool:search",
      required_tool: "search",
      message: "Required tool missing",
      recoverable: true,
    });
    expect(state.text).toBe("Hi");
    expect(state.policyViolation?.required_tool).toBe("search");
    expect(state.events).toHaveLength(2);
  });

  it("hydrates tool_call_result payload from versioned envelopes", () => {
    const event = normalizeRawStreamEvent({
      version: "1",
      type: "tool_call_result",
      event_id: "evt-tool-1",
      payload: {
        type: "tool_call_result",
        id: "call-9",
        name: "lookup",
        result: {
          success: true,
          content: '{"answer":7}',
        },
      },
    });
    expect(event.type).toBe("tool_call_result");
    if (event.type === "tool_call_result") {
      expect(event.event_id).toBe("evt-tool-1");
      expect(event.id).toBe("call-9");
      expect(event.result.payload).toEqual({ answer: 7 });
    }
  });

  it("preserves payload event_id over envelope event_id", () => {
    const restored = fromEventEnvelope({
      version: "1",
      type: "text_chunk",
      event_id: "envelope-id",
      payload: {
        type: "text_chunk",
        content: "hi",
        event_id: "payload-id",
      },
    });
    expect(restored.type).toBe("text_chunk");
    expect(restored.event_id).toBe("payload-id");
  });

  it("normalizes media stream envelope types and keeps rust-shaped fields", () => {
    const audio = normalizeRawStreamEvent({
      version: "1",
      type: "audio_chunk",
      event_id: "audio-1",
      payload: {
        type: "audio_chunk",
        data: [1, 2, 3],
        format: "wav",
      },
    });
    expect(audio.type).toBe("audio_chunk");
    expect(audio.event_id).toBe("audio-1");
    expect((audio as { data?: number[]; format?: string }).data).toEqual([1, 2, 3]);
    expect((audio as { format?: string }).format).toBe("wav");

    const image = normalizeRawStreamEvent({
      version: "1",
      type: "image_generated",
      payload: {
        type: "image_generated",
        data: [9],
        media_type: "image/png",
        revised_prompt: "brighter",
      },
    });
    expect(image.type).toBe("image_generated");
    expect((image as { media_type?: string }).media_type).toBe("image/png");
  });

  it("preserves metadata performance metrics from native events", () => {
    const event = normalizeRawStreamEvent({
      type: "metadata",
      usage: { total_tokens: 12 },
      finish_reason: "stop",
      metrics: {
        total_duration_ms: 40,
        llm_duration_ms: 30,
        step_count: 1,
      },
      metadata: {},
    });

    expect(event.type).toBe("metadata");
    if (event.type === "metadata") {
      expect(event.metrics?.total_duration_ms).toBe(40);
      expect(event.metrics?.llm_duration_ms).toBe(30);
      expect(event.metrics?.step_count).toBe(1);
    }
  });

  it("normalizes harness discipline metrics with safe defaults", () => {
    const metrics = normalizeHarnessDisciplineMetrics({
      tokens_per_turn: 100.5,
      tool_calls: 3,
      tool_calls_per_turn: 1.5,
      compactions: 1,
      readonly_filter_hits: 2,
      estimated_cost_usd: 0.01,
      summary: "100.5 tokens/turn, 1.5 tool calls/turn",
    });

    expect(metrics.tool_calls).toBe(3);
    expect(metrics.compactions).toBe(1);
    expect(metrics.summary).toContain("tokens/turn");
    expect(normalizeHarnessDisciplineMetrics(null).tool_calls).toBe(0);
  });
});

describe("hydrateStreamEvent payload preservation", () => {
  it("preserves existing payload on tool_call_result", () => {
    const event: ToolCallResultEvent = {
      type: "tool_call_result",
      id: "call-1",
      name: "lookup",
      result: {
        success: true,
        content: '{"ignored":true}',
        payload: { kept: true },
      },
    };
    const hydrated = hydrateStreamEvent(event);
    if (hydrated.type === "tool_call_result") {
      expect(hydrated.result.payload).toEqual({ kept: true });
    }
  });

  it("rehydrates when payload is explicitly null", () => {
    const event = {
      type: "tool_call_result",
      id: "call-null",
      name: "lookup",
      result: {
        success: true,
        content: '{"answer":9}',
        payload: null,
      },
    } as unknown as ToolCallResultEvent;
    const hydrated = hydrateStreamEvent(event);
    if (hydrated.type === "tool_call_result") {
      expect(hydrated.result.payload).toEqual({ answer: 9 });
    }
  });

  it("parses JSON content into payload when payload is absent", () => {
    const event: ToolCallResultEvent = {
      type: "tool_call_result",
      id: "call-2",
      name: "lookup",
      result: {
        success: true,
        content: '{"answer":42}',
      },
    };
    const hydrated = hydrateStreamEvent(event);
    if (hydrated.type === "tool_call_result") {
      expect(hydrated.result.payload).toEqual({ answer: 42 });
    }
  });

  it("leaves non-json tool results unchanged", () => {
    const event: ToolCallResultEvent = {
      type: "tool_call_result",
      id: "call-3",
      name: "lookup",
      result: {
        success: true,
        content: "plain text",
      },
    };
    const hydrated = hydrateStreamEvent(event);
    if (hydrated.type === "tool_call_result") {
      expect(hydrated.result.payload).toBeUndefined();
      expect(hydrated.result.content).toBe("plain text");
    }
  });

  it("skips hydration when typedToolResults is false", () => {
    const event: ToolCallResultEvent = {
      type: "tool_call_result",
      id: "call-4",
      name: "lookup",
      result: {
        success: true,
        content: '{"answer":1}',
      },
    };
    const hydrated = hydrateStreamEvent(event, false);
    if (hydrated.type === "tool_call_result") {
      expect(hydrated.result.payload).toBeUndefined();
    }
  });
});
