import type {
  ErrorEvent,
  MetadataEvent,
  StreamEvent,
  TextChunkEvent,
  TokenUsage,
  ToolCallStartEvent,
} from "./events";

export interface CollectedStreamResult {
  readonly text: string;
  readonly usage?: TokenUsage;
  readonly finishReason?: string;
  readonly toolCalls: ReadonlyArray<{
    readonly id: string;
    readonly name: string;
    readonly arguments: unknown;
  }>;
  readonly errors: readonly string[];
}

interface StreamAccumulator {
  readonly textParts: string[];
  readonly toolCalls: Array<{ id: string; name: string; arguments: unknown }>;
  readonly errors: string[];
  usage?: TokenUsage;
  finishReason?: string;
}

type PassiveStreamEvent = Exclude<
  StreamEvent,
  TextChunkEvent | ToolCallStartEvent | MetadataEvent | ErrorEvent
>;

function applyPassiveStreamEvent(event: PassiveStreamEvent): void {
  switch (event.type) {
    case "thinking_chunk":
    case "tool_call_result":
    case "audio_chunk":
    case "image_generated":
    case "attachment_processed":
    case "limit_reached":
    case "compaction":
    case "phase_transition":
    case "status":
    case "agent_start":
    case "agent_end":
    case "turn_start":
    case "turn_end":
    case "policy_violation":
    case "done":
      return;
    default: {
      const _exhaustive: never = event;
      void _exhaustive;
    }
  }
}

function applyStreamEvent(event: StreamEvent, acc: StreamAccumulator): void {
  switch (event.type) {
    case "text_chunk":
      acc.textParts.push(event.content);
      return;
    case "tool_call_start":
      acc.toolCalls.push({
        id: event.id,
        name: event.name,
        arguments: event.arguments,
      });
      return;
    case "metadata":
      if (event.usage) {
        acc.usage = event.usage;
      }
      if (typeof event.finish_reason === "string") {
        acc.finishReason = event.finish_reason;
      }
      return;
    case "error":
      acc.errors.push(event.message ?? "stream error");
      return;
    default:
      applyPassiveStreamEvent(event);
  }
}

/**
 * Collect concatenated text from a stream of Caliche events.
 */
export async function collectText(stream: AsyncIterable<StreamEvent>): Promise<string> {
  const result = await collectStreamResult(stream);
  return result.text;
}

/**
 * Collect text plus usage/finish metadata from a stream.
 */
export async function collectWithUsage(
  stream: AsyncIterable<StreamEvent>
): Promise<{ text: string; usage?: TokenUsage; finishReason?: string }> {
  const result = await collectStreamResult(stream);
  return {
    text: result.text,
    usage: result.usage,
    finishReason: result.finishReason,
  };
}

/**
 * Fully reduce a stream into text, tool calls, usage, and errors.
 */
export async function collectStreamResult(
  stream: AsyncIterable<StreamEvent>
): Promise<CollectedStreamResult> {
  const acc: StreamAccumulator = {
    textParts: [],
    toolCalls: [],
    errors: [],
  };

  for await (const event of stream) {
    applyStreamEvent(event, acc);
  }

  return {
    text: acc.textParts.join(""),
    usage: acc.usage,
    finishReason: acc.finishReason,
    toolCalls: acc.toolCalls,
    errors: acc.errors,
  };
}
