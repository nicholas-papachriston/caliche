import type { JsonValue } from "../types";
import { fromEventEnvelope, type StreamEvent, type StreamEventEnvelope } from "./events";

/**
 * Normalize a raw stream event from the native binding into a typed {@link StreamEvent}.
 */
export function normalizeRawStreamEvent(raw: unknown, typedToolResults = true): StreamEvent {
  if (!raw || typeof raw !== "object") {
    return {
      type: "error",
      message: "Stream event from native binding was not an object",
      recoverable: false,
      event_id: null,
    };
  }
  const candidate = "event" in raw && raw.event && typeof raw.event === "object" ? raw.event : raw;
  if ("version" in candidate && "payload" in candidate && "type" in candidate) {
    const envelope = candidate as StreamEventEnvelope;
    return hydrateStreamEvent(fromEventEnvelope(envelope), typedToolResults);
  }
  if (!("type" in candidate)) {
    return {
      type: "error",
      message: "Stream event from native binding is missing a type field",
      recoverable: false,
      event_id: null,
    };
  }
  return hydrateStreamEvent(candidate as StreamEvent, typedToolResults);
}

/**
 * Parse JSON tool result content into a structured payload when absent.
 */
export function hydrateStreamEvent(event: StreamEvent, typedToolResults = true): StreamEvent {
  if (event.type === "tool_call_result" && typedToolResults) {
    if (event.result.payload !== undefined && event.result.payload !== null) {
      return event;
    }
    const content = event.result.content;
    let payload: JsonValue | undefined;
    try {
      payload = JSON.parse(content) as JsonValue;
    } catch {
      payload = undefined;
    }
    if (payload === undefined) {
      return event;
    }
    return {
      ...event,
      result: {
        ...event.result,
        payload,
      },
    };
  }
  return event;
}
