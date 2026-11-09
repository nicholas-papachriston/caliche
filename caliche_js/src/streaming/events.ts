import type {
  ToolResult as CanonicalToolResult,
  ChainOutput,
  JsonValue,
  MemorySnapshotMessages,
  Metadata,
  RegisteredTool,
  ToolArguments,
  ToolExecutionResult,
  ToolParameterSchema,
} from "../types";

/**
 * Token usage information
 */
export interface TokenUsage {
  readonly prompt_tokens?: number;
  readonly completion_tokens?: number;
  readonly total_tokens?: number;
  readonly last_prompt_tokens?: number;
}

/**
 * Tool call returned by a non-agentic LLM completion.
 */
export interface CompletionToolCall {
  readonly id: string;
  readonly name: string;
  readonly arguments: JsonValue;
  readonly thought_signature?: string;
}

/**
 * Completion response with content, usage, and finish reason
 */
export interface CompletionResponse {
  readonly content: string;
  readonly finish_reason?: string;
  readonly usage?: TokenUsage;
  readonly tool_calls?: readonly CompletionToolCall[];
}

/**
 * Performance metrics for a chat completion
 */
export interface PerformanceMetrics {
  readonly total_duration_ms?: number;
  readonly llm_duration_ms?: number;
  readonly tool_duration_ms?: number;
  readonly overhead_ms?: number;
  readonly step_count?: number;
  readonly time_to_first_token_ms?: number;
  readonly steps?: readonly {
    step: number;
    llm_duration_ms?: number;
    tool_duration_ms?: number;
    tools_executed: readonly string[];
  }[];
}

/**
 * Harness discipline metrics for turns/tasks.
 */
export interface HarnessDisciplineMetrics {
  readonly tokens_per_turn?: number;
  readonly tool_calls: number;
  readonly tool_calls_per_turn?: number;
  readonly compactions: number;
  readonly readonly_filter_hits: number;
  readonly estimated_cost_usd?: number;
  readonly summary?: string;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function numberOrZero(value: unknown): number {
  return optionalNumber(value) ?? 0;
}

/**
 * Normalize a Rust/API-shaped discipline metrics object into the SDK interface.
 */
export function normalizeHarnessDisciplineMetrics(value: unknown): HarnessDisciplineMetrics {
  const object = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const summary = object.summary;
  return {
    tokens_per_turn: optionalNumber(object.tokens_per_turn),
    tool_calls: numberOrZero(object.tool_calls),
    tool_calls_per_turn: optionalNumber(object.tool_calls_per_turn),
    compactions: numberOrZero(object.compactions),
    readonly_filter_hits: numberOrZero(object.readonly_filter_hits),
    estimated_cost_usd: optionalNumber(object.estimated_cost_usd),
    summary: typeof summary === "string" ? summary : undefined,
  };
}

/**
 * Chat response with message and token usage
 */
export interface ChatResponse {
  readonly response: string;
  readonly usage: TokenUsage;
  readonly metrics?: PerformanceMetrics;
}

/**
 * Tool information returned by listTools()
 */
export interface ToolInfo {
  readonly name: string;
  readonly description: string;
  readonly parameters: JsonValue;
}

/**
 * Simple message format for JavaScript bindings
 */
export interface MessageJs {
  readonly role: "user" | "assistant" | "system" | "tool";
  readonly content: string;
  readonly content_parts?: readonly ContentPartJs[];
}

export interface ContentPartJs {
  readonly type:
    | "text"
    | "image"
    | "image_url"
    | "audio"
    | "audio_url"
    | "video"
    | "video_url"
    | "document"
    | "document_url";
  readonly text?: string;
  readonly data?: string;
  readonly media_type?: string;
  readonly url?: string;
  readonly filename?: string;
}

export interface AttachmentInput {
  readonly type: "image" | "audio" | "video" | "document";
  readonly data: string; // base64
  readonly media_type: string;
  readonly filename?: string;
}

/**
 * User context for request tracking and multi-tenancy
 */
export interface UserContextJs {
  readonly user_id: string;
  readonly metadata: JsonValue;
}

/**
 * Tool call information
 */
export interface ToolCallJs {
  readonly id: string;
  readonly name: string;
  readonly arguments: JsonValue;
}

/**
 * Result of a single reasoning step
 */
export interface StepResultJs {
  readonly response: string;
  readonly tool_calls: readonly ToolCallJs[];
  readonly updated_messages: readonly MessageJs[];
  readonly usage: TokenUsage;
}

/**
 * Supported LLM providers
 */
export type Provider =
  | "openai"
  | "anthropic"
  | "bedrock"
  | "openrouter"
  | "perplexity"
  | "lmstudio"
  | "ollama"
  | "google"
  | "gemini"
  | "cohere"
  | "azure-openai"
  | "cloudflare-ai-gateway"
  | "local-inference"
  | "external";

/**
 * Stream event types
 */
export type StreamEventType =
  | "text_chunk"
  | "thinking_chunk"
  | "tool_call_start"
  | "tool_call_result"
  | "audio_chunk"
  | "image_generated"
  | "attachment_processed"
  | "metadata"
  | "limit_reached"
  | "compaction"
  | "phase_transition"
  | "status"
  | "agent_start"
  | "agent_end"
  | "turn_start"
  | "turn_end"
  | "policy_violation"
  | "error"
  | "done";

/**
 * Base stream event
 */
interface BaseStreamEvent {
  readonly type: StreamEventType;
  /** Present when durable streams are used; use for resumeFrom on reconnect */
  readonly event_id?: string | null;
}

/**
 * Text chunk event from streaming response
 */
export interface TextChunkEvent extends BaseStreamEvent {
  readonly type: "text_chunk";
  readonly content: string;
}

/**
 * Thinking chunk event from reasoning-capable models.
 */
export interface ThinkingChunkEvent extends BaseStreamEvent {
  readonly type: "thinking_chunk";
  readonly content: string;
  readonly action_id?: string;
}

/**
 * Tool call start event
 */
export interface ToolCallStartEvent extends BaseStreamEvent {
  readonly type: "tool_call_start";
  readonly id: string;
  readonly name: string;
  readonly arguments: ToolArguments;
  readonly thought_signature?: string;
  readonly action_id?: string;
}

/**
 * Tool call result event
 */
export interface ToolCallResultEvent extends BaseStreamEvent {
  readonly type: "tool_call_result";
  readonly id: string;
  readonly name: string;
  readonly result: ToolResult;
  readonly action_id?: string;
}

/**
 * Metadata event with usage and finish reason
 */
export interface MetadataEvent extends BaseStreamEvent {
  readonly type: "metadata";
  readonly usage?: TokenUsage;
  readonly finish_reason?: string;
  readonly metrics?: PerformanceMetrics;
  readonly metadata: Metadata;
}

export interface PolicyViolationEvent extends BaseStreamEvent {
  readonly type: "policy_violation";
  readonly policy: string;
  readonly required_tool: string;
  readonly message: string;
  readonly recoverable: boolean;
}

export interface ErrorEvent extends BaseStreamEvent {
  readonly type: "error";
  readonly message: string;
  readonly code?: string;
  readonly recoverable?: boolean;
}

/**
 * Done event indicating stream completion
 */
export interface DoneEvent extends BaseStreamEvent {
  readonly type: "done";
}

export interface LimitReachedEvent extends BaseStreamEvent {
  readonly type: "limit_reached";
  readonly max_iterations: number;
  readonly usage: TokenUsage;
}

export interface CompactionChange {
  readonly kind: "omitted" | "truncated" | "compacted";
  readonly source: string;
  readonly reason: string;
}

export interface CompactionReport {
  readonly changes: readonly CompactionChange[];
}

export interface CompactionEvent extends BaseStreamEvent {
  readonly type: "compaction";
  readonly tokens_before: number;
  readonly tokens_after: number;
  readonly messages_removed: number;
  readonly summary?: string;
  readonly report?: CompactionReport | null;
}

export interface PhaseTransitionEvent extends BaseStreamEvent {
  readonly type: "phase_transition";
  readonly phase_index: number;
  readonly phase_total: number;
  readonly tool_names: readonly string[];
}

/** Progress line while the model has not answered yet (e.g. a local model loading). */
export interface StatusEvent extends BaseStreamEvent {
  readonly type: "status";
  readonly message: string;
}

export interface AgentStartEvent extends BaseStreamEvent {
  readonly type: "agent_start";
}

export interface AgentEndEvent extends BaseStreamEvent {
  readonly type: "agent_end";
}

export interface TurnStartEvent extends BaseStreamEvent {
  readonly type: "turn_start";
}

export interface TurnEndEvent extends BaseStreamEvent {
  readonly type: "turn_end";
}

export interface AudioChunkEvent extends BaseStreamEvent {
  readonly type: "audio_chunk";
  readonly content: string;
  readonly media_type?: string;
}

export interface ImageGeneratedEvent extends BaseStreamEvent {
  readonly type: "image_generated";
  readonly content: string;
  readonly media_type?: string;
  readonly filename?: string;
}

export interface AttachmentProcessedEvent extends BaseStreamEvent {
  readonly type: "attachment_processed";
  readonly attachment_type: string;
  readonly filename?: string;
  readonly media_type?: string;
}

/**
 * Union type for all stream events
 */
export type StreamEvent =
  | TextChunkEvent
  | ThinkingChunkEvent
  | ToolCallStartEvent
  | ToolCallResultEvent
  | AudioChunkEvent
  | ImageGeneratedEvent
  | AttachmentProcessedEvent
  | MetadataEvent
  | LimitReachedEvent
  | CompactionEvent
  | PhaseTransitionEvent
  | StatusEvent
  | AgentStartEvent
  | AgentEndEvent
  | TurnStartEvent
  | TurnEndEvent
  | PolicyViolationEvent
  | ErrorEvent
  | DoneEvent;

export interface StreamEventEnvelope<T extends StreamEvent = StreamEvent> {
  readonly version: "1";
  readonly type: T["type"];
  readonly payload: T;
  /** Present when durable streams are used; use for resumeFrom on reconnect */
  readonly event_id?: string | null;
}

/** Result from tool execution. */
export type ToolResult = CanonicalToolResult;

/**
 * Tool interface for custom tools
 */
export type Tool<
  TArgs = ToolArguments,
  TResult extends ToolExecutionResult = ToolExecutionResult,
> = RegisteredTool<TArgs, TResult>;

/**
 * Tool schema for LLM function calling
 */
export interface ToolSchema {
  readonly name: string;
  readonly description: string;
  readonly parameters: ToolParameterSchema;
}

/**
 * Configuration for creating a tool from a Zod schema
 */
export interface ZodToolConfig {
  /** Tool name */
  name: string;
  /** Tool description */
  description: string;
}

type ZodLikeSchema<T> = {
  parse(data: unknown): T;
};

/**
 * Creates a Tool from a Zod schema and execute function.
 *
 * Converts the Zod schema to JSON Schema and validates inputs before execution.
 * Argument types are inferred from `schema.parse` / `z.infer<typeof schema>`.
 *
 * @example
 * ```typescript
 * import { z } from "zod";
 * import { createTool } from "@caliche/core";
 *
 * const weatherSchema = z.object({
 *   city: z.string().describe("The city name"),
 *   unit: z.enum(["celsius", "fahrenheit"]).default("celsius")
 * });
 *
 * const weatherTool = createTool({
 *   name: "get_weather",
 *   description: "Get the weather for a city",
 *   schema: weatherSchema,
 *   execute: async (args) => ({
 *     success: true,
 *     content: `Weather in ${args.city}: 20°${args.unit}`,
 *   }),
 * });
 * ```
 */
export function createTool<TArgs>(
  config: ZodToolConfig & {
    /** Zod schema for tool parameters */
    schema: ZodLikeSchema<TArgs>;
    /** Execute function with typed arguments */
    execute: (args: TArgs) => Promise<ToolExecutionResult> | ToolExecutionResult;
  }
): Tool<TArgs> {
  let zodToJsonSchema: (schema: unknown, options?: Record<string, unknown>) => unknown;

  try {
    // Ensure zod is available (even though we don't use it directly here)
    require("zod");
    // Try both default and named export patterns
    const zodToJsonSchemaModule = require("zod-to-json-schema") as {
      default?: (schema: unknown, options?: Record<string, unknown>) => unknown;
      zodToJsonSchema?: (schema: unknown, options?: Record<string, unknown>) => unknown;
    };
    const resolved =
      zodToJsonSchemaModule.default ||
      zodToJsonSchemaModule.zodToJsonSchema ||
      (zodToJsonSchemaModule as unknown as (
        schema: unknown,
        options?: Record<string, unknown>
      ) => unknown);
    if (typeof resolved !== "function") {
      throw new Error("zod-to-json-schema export is not a function");
    }
    zodToJsonSchema = resolved;
  } catch (_error) {
    throw new Error(
      "zod and zod-to-json-schema are required for createTool. " +
        "Install them with: npm install zod zod-to-json-schema"
    );
  }

  // Convert Zod schema to JSON schema
  // Use jsonSchema7 target to get standard JSON Schema format
  let jsonSchema: Record<string, unknown>;
  try {
    jsonSchema = zodToJsonSchema(config.schema, {
      target: "jsonSchema7",
      $refStrategy: "none",
    }) as Record<string, unknown>;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Failed to convert Zod schema to JSON schema: ${message}. ` +
        "This might be due to a compatibility issue between zod-to-json-schema and your Zod version."
    );
  }

  // Ensure we have a valid object structure
  // Handle cases where zod-to-json-schema might return incomplete schemas
  // (e.g., compatibility issues with Zod v4)
  let parameters: ToolParameterSchema;

  if (typeof jsonSchema === "object" && jsonSchema !== null) {
    const properties = jsonSchema.properties as Record<string, unknown> | undefined;
    // Check if it has properties (indicating it's an object schema)
    if (properties && Object.keys(properties).length > 0) {
      // Valid object schema with properties
      parameters = {
        type: "object",
        properties,
        required: Array.isArray(jsonSchema.required) ? (jsonSchema.required as string[]) : [],
        ...(jsonSchema.additionalProperties !== undefined && {
          additionalProperties: jsonSchema.additionalProperties as boolean,
        }),
        ...(typeof jsonSchema.description === "string" && { description: jsonSchema.description }),
      } as ToolParameterSchema;
    } else if (jsonSchema.type === "object") {
      // Object type but no properties - might be empty or incomplete
      parameters = {
        type: "object",
        properties: (properties || {}) as ToolParameterSchema["properties"],
        required: Array.isArray(jsonSchema.required) ? (jsonSchema.required as string[]) : [],
      };
    } else {
      // Fallback: try to extract schema from Zod's internal structure
      // This handles cases where zod-to-json-schema doesn't work properly
      // We'll rely on Zod's parse() for validation instead
      console.warn(
        `zod-to-json-schema returned incomplete schema for tool "${config.name}". ` +
          "Falling back to basic object schema. Validation will still work via Zod."
      );
      parameters = {
        type: "object",
        properties: {},
        required: [],
      };
    }
  } else {
    throw new Error(
      "Invalid JSON schema result from zod-to-json-schema. " +
        "Expected an object but got: " +
        typeof jsonSchema
    );
  }

  return {
    name: config.name,
    description: config.description,
    parameters,
    execute: async (args: TArgs): Promise<ToolExecutionResult> => {
      try {
        // Validate and parse arguments using Zod schema
        const validatedArgs = config.schema.parse(args);

        // Execute with validated arguments
        return await config.execute(validatedArgs);
      } catch (error: unknown) {
        // Check if it's a ZodError
        const err = error as {
          constructor?: { name?: string };
          name?: string;
          issues?: Array<{ path: (string | number)[]; message: string }>;
          errors?: Array<{ path: (string | number)[]; message: string }>;
          message?: string;
        };
        if (
          err &&
          (err.constructor?.name === "ZodError" || err.name === "ZodError" || err.issues)
        ) {
          const errors = err.issues || err.errors || [];
          if (errors.length > 0) {
            return {
              success: false,
              content: "",
              error: `Validation error: ${errors.map((e) => `${e.path?.join(".") || ""}: ${e.message || String(e)}`).join(", ")}`,
            };
          }
        }
        // If it's not a ZodError or we can't extract errors, return generic error
        return {
          success: false,
          content: "",
          error: err.message || `Tool execution failed: ${String(error)}`,
        };
      }
    },
  };
}

/**
 * Toolkit interface for bundling related tools
 */
export interface Toolkit {
  readonly name: string;
  get_tools(): Tool[];
}

/**
 * Memory snapshot for persistence
 */
export interface MemorySnapshot {
  readonly messages: MemorySnapshotMessages;
  readonly metadata: Metadata;
}

/**
 * Configuration for buffer-based conversation memory.
 *
 * Buffer memory retains a fixed window of recent messages, automatically
 * truncating older messages when the token limit is exceeded.
 */
export interface BufferMemoryConfig {
  /**
   * Maximum number of tokens to retain in memory (default: 4096)
   */
  readonly max_tokens?: number;

  /**
   * Minimum number of messages to keep regardless of token limits (default: 4)
   */
  readonly min_messages?: number;

  /**
   * Preserve the initial system message if encountered (default: true)
   */
  readonly preserve_system?: boolean;

  /**
   * Inject a placeholder summary when older messages are truncated (default: true)
   */
  readonly summary_placeholder?: boolean;
}

/**
 * Configuration for summarization-based memory.
 *
 * Summary memory automatically summarizes older messages when the token
 * threshold is reached, allowing for longer conversation context.
 */
export interface SummaryMemoryConfig {
  /**
   * Maximum token budget for the combined prompt (default: 8192)
   */
  readonly max_tokens?: number;

  /**
   * Percentage (0.0 - 1.0) at which summarization should be triggered (default: 0.8)
   */
  readonly summarize_threshold_pct?: number;

  /**
   * Number of messages to include in each summarization batch (default: 12)
   */
  readonly chunk_size?: number;

  /**
   * Maximum number of summaries to retain (default: 8)
   */
  readonly max_summaries?: number;
}

/**
 * Configuration for entity-tracking memory.
 *
 * Entity memory extracts and tracks important entities (people, places, concepts)
 * from conversations, injecting relevant entity context into prompts.
 */
export interface EntityMemoryConfig {
  /**
   * Maximum token budget including entity context (default: 6144)
   */
  readonly max_tokens?: number;

  /**
   * Maximum number of tracked entities (default: 128)
   */
  readonly max_entities?: number;

  /**
   * Extract entities every N messages (default: 3)
   */
  readonly extraction_interval?: number;

  /**
   * Minimum relevance threshold for injecting entities into context (default: 0.65)
   */
  readonly similarity_threshold?: number;
}

/**
 * Configuration for vector-based semantic memory
 */
export interface VectorMemoryConfig {
  /**
   * Maximum token budget for the combined context (buffer + retrieved memories) (default: 8192)
   */
  readonly max_tokens?: number;

  /**
   * Maximum number of conversation turns to store as embeddings (default: 1000)
   */
  readonly max_memories?: number;

  /**
   * Number of most relevant memories to retrieve (default: 5)
   */
  readonly top_k?: number;

  /**
   * Minimum similarity threshold (0.0-1.0) for retrieving memories (default: 0.7)
   */
  readonly similarity_threshold?: number;

  /**
   * Token budget reserved for recent buffer messages (default: 4096)
   */
  readonly buffer_tokens?: number;

  /**
   * Optional namespace for isolating memories in the vector backend
   */
  readonly namespace?: string;
}

/**
 * Memory persistence interface for custom persistence backends
 *
 * Memory persistence allows you to save and restore conversation state
 * across sessions, enabling stateful agents that remember previous interactions.
 */
export interface MemoryPersistence {
  /**
   * Save a memory snapshot for the given conversation
   *
   * @param conversationId - Unique identifier for the conversation
   * @param snapshot - MemorySnapshot containing messages and metadata
   * @returns Promise that resolves when save is complete
   */
  save_snapshot(conversationId: string, snapshot: MemorySnapshot): Promise<void>;

  /**
   * Load a memory snapshot for the given conversation
   *
   * @param conversationId - Unique identifier for the conversation
   * @returns Promise resolving to the snapshot, or undefined if not found
   */
  load_snapshot(conversationId: string): Promise<MemorySnapshot | undefined>;

  /**
   * Clear the memory snapshot for the given conversation
   *
   * @param conversationId - Unique identifier for the conversation
   * @returns Promise that resolves when clear is complete
   */
  clear_snapshot(conversationId: string): Promise<void>;
}

/**
 * Execution step in an agent flow
 */
export interface ExecutionStep {
  readonly agent_name: string;
  readonly input: string;
  readonly response?: string;
  readonly handoff?: {
    readonly target_agent: string;
    readonly reason: string;
    readonly instructions?: string;
    readonly persona?: string;
  };
  readonly usage: TokenUsage;
}

/**
 * Result of executing an agent flow
 */
export interface FlowExecutionResult {
  readonly final_response?: string;
  readonly execution_log: readonly ExecutionStep[];
  readonly total_token_usage: TokenUsage;
  readonly final_handoff?: {
    readonly target_agent: string;
    readonly reason: string;
    readonly instructions?: string;
    readonly persona?: string;
  };
}

export interface AgentRuntimeCredentials {
  readonly defaultApiKey?: string;
  readonly scopedApiKeys?: Readonly<Record<string, string>>;
  readonly aws?: {
    readonly accessKey: string;
    readonly secretKey: string;
    readonly region: string;
  };
  readonly baseUrl?: string;
}

export interface AgentRuntimeTimeouts {
  readonly streamEventTimeoutSecs?: number;
}

export interface AgentRuntimeConfig {
  readonly provider?: Provider;
  readonly model?: string;
  readonly credentials?: AgentRuntimeCredentials;
  readonly timeouts?: AgentRuntimeTimeouts;
}

export class AgentRuntimeConfigError extends Error {
  readonly issues: readonly string[];

  constructor(issues: readonly string[]) {
    super(`Invalid AgentRuntimeConfig: ${issues.join("; ")}`);
    this.name = "AgentRuntimeConfigError";
    this.issues = [...issues];
  }
}

export interface ToolRequirementPolicy {
  readonly name?: string;
  readonly toolName: string;
  readonly predicate: (input: { readonly message: string }) => boolean;
  readonly violationMessage?: string;
}

/**
 * Result from chain execution
 */
export interface ChainResult {
  readonly output: ChainOutput;
}

/**
 * Reasoning step in a reasoning process
 */
export interface ReasoningStep {
  readonly step_type: "thought" | "action" | "observation" | "plan" | "reflection" | "answer";
  readonly content: string;
  readonly metadata?: Metadata;
}

/**
 * Result from reasoning pattern execution
 */
export interface ReasoningResult {
  readonly final_answer: string;
  readonly steps: readonly ReasoningStep[];
  readonly success: boolean;
  readonly error?: string;
  readonly token_usage?: TokenUsage;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guard for text chunk events
 */
export function isTextChunkEvent(event: StreamEvent): event is TextChunkEvent {
  return event.type === "text_chunk";
}

/**
 * Type guard for tool call events
 */
export function isToolCallStartEvent(event: StreamEvent): event is ToolCallStartEvent {
  return event.type === "tool_call_start";
}

/**
 * Type guard for tool result events
 */
export function isToolCallResultEvent(event: StreamEvent): event is ToolCallResultEvent {
  return event.type === "tool_call_result";
}

/**
 * Type guard for metadata events
 */
export function isMetadataEvent(event: StreamEvent): event is MetadataEvent {
  return event.type === "metadata";
}

/**
 * Type guard for policy violation events
 */
export function isPolicyViolationEvent(event: StreamEvent): event is PolicyViolationEvent {
  return event.type === "policy_violation";
}

/**
 * Type guard for error events
 */
export function isErrorEvent(event: StreamEvent): event is ErrorEvent {
  return event.type === "error";
}

/**
 * Type guard for done events
 */
export function isDoneEvent(event: StreamEvent): event is DoneEvent {
  return event.type === "done";
}

export interface AgentStreamState {
  readonly text: string;
  readonly done: boolean;
  readonly events: readonly StreamEvent[];
  readonly usage?: TokenUsage;
  readonly finishReason?: string;
  readonly error?: string;
  readonly policyViolation?: PolicyViolationEvent;
}

export function createInitialAgentStreamState(): AgentStreamState {
  return {
    text: "",
    done: false,
    events: [],
  };
}

export function reduceAgentStreamEvent(
  state: AgentStreamState,
  event: StreamEvent
): AgentStreamState {
  const events = [...state.events, event];
  switch (event.type) {
    case "text_chunk":
    case "thinking_chunk":
      return {
        ...state,
        text: state.text + event.content,
        events,
      };
    case "metadata":
      return {
        ...state,
        usage: event.usage,
        finishReason: event.finish_reason,
        events,
      };
    case "policy_violation":
      return {
        ...state,
        policyViolation: event,
        events,
      };
    case "error":
      return {
        ...state,
        error: event.message,
        events,
      };
    case "done":
      return {
        ...state,
        done: true,
        events,
      };
    default:
      return {
        ...state,
        events,
      };
  }
}

export function toEventEnvelope(event: StreamEvent): StreamEventEnvelope {
  return {
    version: "1",
    type: event.type,
    payload: event,
    event_id: event.event_id ?? null,
  };
}

export function fromEventEnvelope(envelope: StreamEventEnvelope): StreamEvent {
  const payload = envelope.payload as StreamEvent;
  if (payload.event_id === undefined) {
    return { ...payload, event_id: envelope.event_id ?? null };
  }
  return payload;
}
