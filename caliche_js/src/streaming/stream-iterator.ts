import { type NativeStreamIterator, native, type StreamChatOptions } from "../native/module";
import {
  type PolicyViolationEvent,
  type StreamEvent,
  type StreamEventEnvelope,
  type ToolRequirementPolicy,
  toEventEnvelope,
} from "./events";
import { normalizeRawStreamEvent } from "./stream-normalization";

/**
 * Optional replay backend for durable stream resume.
 *
 * When a native DurableStreamReader is not exposed, callers (or Agent wiring)
 * can supply this interface so {@link StreamIterator.replayFrom} yields real events.
 */
export interface StreamReplaySource {
  replayFrom(
    eventId: string
  ):
    | AsyncIterable<StreamEvent>
    | Iterable<StreamEvent>
    | Promise<AsyncIterable<StreamEvent> | Iterable<StreamEvent>>;
}

type NativeReplayApi = {
  streamReplayFrom?: (
    iter: NativeStreamIterator,
    eventId: string
  ) =>
    | AsyncIterable<unknown>
    | Iterable<unknown>
    | Promise<AsyncIterable<unknown> | Iterable<unknown>>;
  DurableStreamReader?: {
    replayFrom?: (
      eventId: string,
      options?: { resumeFrom?: string }
    ) =>
      | AsyncIterable<unknown>
      | Iterable<unknown>
      | Promise<AsyncIterable<unknown> | Iterable<unknown>>;
  };
};

function isStreamStartId(eventId: string): boolean {
  return eventId === "" || eventId === "0";
}

async function* iterateMaybeAsync<T>(
  source: AsyncIterable<T> | Iterable<T>
): AsyncGenerator<T, void, unknown> {
  const maybeAsync = source as AsyncIterable<T>;
  if (typeof maybeAsync[Symbol.asyncIterator] === "function") {
    for await (const item of maybeAsync) {
      yield item;
    }
    return;
  }
  for (const item of source as Iterable<T>) {
    yield item;
  }
}

/**
 * Stream iterator for streaming chat responses
 *
 * Implements the async iterator protocol for convenient streaming.
 *
 * @example
 * ```typescript
 * const stream = agent.streamChat('Hello');
 * for await (const event of stream) {
 *   if (isTextChunkEvent(event)) {
 *     process.stdout.write(event.content);
 *   }
 * }
 * ```
 */
export class StreamIterator implements AsyncIterable<StreamEvent> {
  /** Stream event schema version exposed for native-binding parity checks. */
  readonly eventSchemaVersion: "1" = "1";

  private readonly _iter: NativeStreamIterator;
  private readonly parsePseudoToolProtocol: boolean;
  private toolCallCounter = 0;
  private readonly typedToolResults: boolean;
  private readonly message?: string;
  private readonly toolRequirementPolicies: readonly ToolRequirementPolicy[];
  private readonly usedTools = new Set<string>();
  private policyViolationEventEmitted = false;
  private pendingEvent: StreamEvent | null = null;
  private readonly resumeFrom?: string;
  private readonly replaySource?: StreamReplaySource;
  private readonly bufferForReplay: boolean;
  private readonly replayBuffer: StreamEvent[] = [];
  private replaySeq = 0;

  /**
   * @internal
   */
  constructor(
    nativeIter: NativeStreamIterator,
    parsePseudoToolProtocol: boolean = false,
    context?: {
      readonly options?: StreamChatOptions | null;
      readonly message?: string;
      readonly toolRequirementPolicies?: readonly ToolRequirementPolicy[];
      readonly replaySource?: StreamReplaySource;
      /** When true, buffer yielded events for later {@link replayFrom}. */
      readonly bufferForReplay?: boolean;
    }
  ) {
    this._iter = nativeIter;
    this.parsePseudoToolProtocol = parsePseudoToolProtocol;
    this.typedToolResults = context?.options?.typedToolResults ?? true;
    this.message = context?.message;
    this.toolRequirementPolicies = context?.toolRequirementPolicies ?? [];
    this.resumeFrom = context?.options?.resumeFrom;
    this.replaySource = context?.replaySource;
    this.bufferForReplay =
      context?.bufferForReplay ??
      (Boolean(context?.replaySource) || Boolean(context?.options?.resumeFrom));
  }

  private normalizeRawEvent(raw: unknown): StreamEvent {
    return normalizeRawStreamEvent(raw, this.typedToolResults);
  }

  private withReplayId(event: StreamEvent): StreamEvent {
    if (event.event_id) {
      return event;
    }
    this.replaySeq += 1;
    return { ...event, event_id: `evt-${this.replaySeq}` };
  }

  private bufferEvent(event: StreamEvent): StreamEvent {
    if (!this.bufferForReplay) {
      return event;
    }
    const withId = this.withReplayId(event);
    this.replayBuffer.push(withId);
    return withId;
  }

  private evaluatePolicyViolation(): PolicyViolationEvent | null {
    if (
      this.policyViolationEventEmitted ||
      !this.message ||
      this.toolRequirementPolicies.length === 0
    ) {
      return null;
    }
    for (const policy of this.toolRequirementPolicies) {
      if (!policy.predicate({ message: this.message })) {
        continue;
      }
      if (this.usedTools.has(policy.toolName)) {
        continue;
      }
      this.policyViolationEventEmitted = true;
      return {
        type: "policy_violation",
        policy: policy.name ?? `require_tool:${policy.toolName}`,
        required_tool: policy.toolName,
        message:
          policy.violationMessage ??
          `Required tool '${policy.toolName}' was not called before stream completion.`,
        recoverable: true,
        event_id: null,
      };
    }
    return null;
  }

  private parsePseudoToolEvent(content: string): StreamEvent | null {
    if (!this.parsePseudoToolProtocol) {
      return null;
    }
    const toolUseMatch = content.match(/^\s*Tool use:\s*(.+)$/i);
    if (toolUseMatch) {
      const raw = toolUseMatch[1]?.trim() ?? "";
      const name = raw.split(/\s+/)[0] ?? "tool";
      this.toolCallCounter += 1;
      return {
        type: "tool_call_start",
        id: `pseudo-tool-${this.toolCallCounter}`,
        name,
        arguments: { raw },
        event_id: null,
      };
    }
    const toolResultMatch = content.match(/^\s*Tool result:\s*(.+)$/i);
    if (toolResultMatch) {
      return {
        type: "tool_call_result",
        id: `pseudo-tool-${this.toolCallCounter || 1}`,
        name: "tool",
        result: {
          success: true,
          content: toolResultMatch[1]?.trim() ?? "",
        },
        event_id: null,
      };
    }
    return null;
  }

  /**
   * Get the next event from the stream
   *
   * @returns Promise resolving to the next event, or null if stream ended
   */
  async next(): Promise<StreamEvent | null> {
    if (this.pendingEvent) {
      const pending = this.pendingEvent;
      this.pendingEvent = null;
      return this.bufferEvent(pending);
    }
    const raw = await native.streamNext(this._iter);
    if (!raw) {
      return null;
    }
    const normalized = this.normalizeRawEvent(raw);
    if (normalized.type === "text_chunk") {
      const maybePseudo = this.parsePseudoToolEvent(normalized.content);
      if (maybePseudo) {
        return this.bufferEvent(maybePseudo);
      }
    }
    if (normalized.type === "tool_call_start") {
      this.usedTools.add(normalized.name);
    }
    if (normalized.type === "done") {
      const policyViolation = this.evaluatePolicyViolation();
      if (policyViolation) {
        this.pendingEvent = normalized;
        return this.bufferEvent(policyViolation);
      }
    }
    return this.bufferEvent(normalized);
  }

  async nextEnvelope(): Promise<StreamEventEnvelope | null> {
    const nextEvent = await this.next();
    if (!nextEvent) {
      return null;
    }
    return toEventEnvelope(nextEvent);
  }

  /**
   * Cancel the active stream.
   */
  cancel(): void {
    if (typeof native.streamCancel === "function") {
      native.streamCancel(this._iter);
    }
  }

  /**
   * Make the iterator async iterable
   */
  async *[Symbol.asyncIterator](): AsyncGenerator<StreamEvent, void, unknown> {
    while (true) {
      const event = await this.next();
      if (event === null) {
        break;
      }
      yield event;
    }
  }

  envelopes(): AsyncIterable<StreamEventEnvelope> {
    const self = this;
    return {
      async *[Symbol.asyncIterator](): AsyncGenerator<StreamEventEnvelope, void, unknown> {
        while (true) {
          const envelope = await self.nextEnvelope();
          if (envelope === null) {
            break;
          }
          yield envelope;
        }
      },
    };
  }

  /**
   * Replay stream events after `eventId`.
   *
   * Resolution order:
   * 1. Injected {@link StreamReplaySource} (durable backend callback)
   * 2. Native `streamReplayFrom` / `DurableStreamReader.replayFrom` when present
   * 3. In-memory buffer populated while consuming this iterator (when resume/durable config is set)
   */
  replayFrom(eventId: string): AsyncIterable<StreamEvent> {
    const self = this;
    return {
      async *[Symbol.asyncIterator](): AsyncGenerator<StreamEvent, void, unknown> {
        if (self.replaySource) {
          const source = await self.replaySource.replayFrom(eventId);
          yield* iterateMaybeAsync(source);
          return;
        }

        const nativeApi = native as unknown as NativeReplayApi;
        if (typeof nativeApi.streamReplayFrom === "function") {
          const source = await nativeApi.streamReplayFrom(self._iter, eventId);
          for await (const raw of iterateMaybeAsync(source)) {
            yield self.normalizeRawEvent(raw);
          }
          return;
        }
        if (typeof nativeApi.DurableStreamReader?.replayFrom === "function") {
          const source = await nativeApi.DurableStreamReader.replayFrom(eventId, {
            resumeFrom: self.resumeFrom,
          });
          for await (const raw of iterateMaybeAsync(source)) {
            yield self.normalizeRawEvent(raw);
          }
          return;
        }

        let past = isStreamStartId(eventId);
        for (const event of self.replayBuffer) {
          if (!past) {
            if (event.event_id === eventId) {
              past = true;
            }
            continue;
          }
          yield event;
        }
      },
    };
  }
}
