/**
 * Common type definitions for Caliche JavaScript bindings
 *
 * This file provides type-safe alternatives to `any` throughout the codebase.
 */

/**
 * JSON-compatible value types
 * Represents any value that can be serialized to/from JSON
 */
export type JsonValue = string | number | boolean | null | JsonObject | JsonArray;

/**
 * JSON object type
 */
export type JsonObject = {
  readonly [key: string]: JsonValue;
};

/**
 * JSON array type
 */
export type JsonArray = readonly JsonValue[];

/**
 * Record type for key-value pairs with JSON-compatible values
 * This is a more specific alternative to Record<string, any>
 */
export type JsonRecord = Record<string, JsonValue>;

/**
 * JSON Schema property definition
 */
export interface JsonSchemaProperty {
  readonly type?: string;
  readonly description?: string;
  readonly enum?: readonly JsonValue[];
  readonly default?: JsonValue;
  readonly format?: string;
  readonly items?: JsonSchemaProperty | JsonSchema;
  readonly properties?: Record<string, JsonSchemaProperty>;
  readonly required?: readonly string[];
  readonly oneOf?: readonly JsonSchema[];
  readonly anyOf?: readonly JsonSchema[];
  readonly allOf?: readonly JsonSchema[];
  readonly additionalProperties?: boolean | JsonSchemaProperty | JsonSchema;
  readonly minimum?: number;
  readonly maximum?: number;
  readonly minLength?: number;
  readonly maxLength?: number;
  readonly pattern?: string;
  readonly examples?: readonly JsonValue[];
}

/**
 * JSON Schema definition
 */
export interface JsonSchema {
  readonly type?: string;
  readonly description?: string;
  readonly properties?: Record<string, JsonSchemaProperty>;
  readonly required?: readonly string[];
  readonly additionalProperties?: boolean | JsonSchemaProperty | JsonSchema;
  readonly enum?: readonly JsonValue[];
  readonly oneOf?: readonly JsonSchema[];
  readonly anyOf?: readonly JsonSchema[];
  readonly allOf?: readonly JsonSchema[];
  readonly items?: JsonSchemaProperty | JsonSchema;
  readonly default?: JsonValue;
  readonly examples?: readonly JsonValue[];
}

/**
 * Tool parameter schema (JSON Schema for tools)
 */
export interface ToolParameterSchema {
  readonly type: "object";
  readonly properties: Record<string, JsonSchemaProperty>;
  readonly required?: readonly string[];
  readonly additionalProperties?: boolean | JsonSchemaProperty | JsonSchema;
}

/**
 * Tool handler args default.
 *
 * Defaulting to unknown forces explicit narrowing in handlers when
 * schema-based inference is not used.
 */
export type ToolHandlerArgs = unknown;

/**
 * Map JSON schema "type" values to TypeScript values.
 */
type JsonSchemaTypeName = "string" | "number" | "integer" | "boolean" | "null" | "object" | "array";

/**
 * Infer a TypeScript value type from a JSON schema property.
 */
type InferJsonSchemaPropertyType<T extends JsonSchemaProperty> = T["type"] extends "string"
  ? string
  : T["type"] extends "number" | "integer"
    ? number
    : T["type"] extends "boolean"
      ? boolean
      : T["type"] extends "null"
        ? null
        : T["type"] extends "array"
          ? T["items"] extends JsonSchemaProperty
            ? Array<InferJsonSchemaPropertyType<T["items"]>>
            : T["items"] extends JsonSchema
              ? Array<InferJsonSchemaType<T["items"]>>
              : JsonValue[]
          : T["type"] extends "object"
            ? T["properties"] extends Record<string, JsonSchemaProperty>
              ? {
                  [K in keyof T["properties"]]: InferJsonSchemaPropertyType<T["properties"][K]>;
                }
              : JsonRecord
            : JsonValue;

/**
 * Infer a TypeScript value type from a JSON schema object.
 */
type InferJsonSchemaType<T extends JsonSchema> = T["type"] extends JsonSchemaTypeName
  ? T["type"] extends "object"
    ? T["properties"] extends Record<string, JsonSchemaProperty>
      ? {
          [K in keyof T["properties"]]: InferJsonSchemaPropertyType<T["properties"][K]>;
        }
      : JsonRecord
    : T["type"] extends "array"
      ? T["items"] extends JsonSchemaProperty
        ? Array<InferJsonSchemaPropertyType<T["items"]>>
        : T["items"] extends JsonSchema
          ? Array<InferJsonSchemaType<T["items"]>>
          : JsonValue[]
      : T["type"] extends "string"
        ? string
        : T["type"] extends "number" | "integer"
          ? number
          : T["type"] extends "boolean"
            ? boolean
            : T["type"] extends "null"
              ? null
              : JsonValue
  : JsonValue;

/**
 * Infer tool arguments from a ToolParameterSchema.
 */
export type InferToolArguments<TSchema extends ToolParameterSchema> = {
  [K in keyof TSchema["properties"]]: InferJsonSchemaPropertyType<TSchema["properties"][K]>;
};

/**
 * Object-like tool arguments (`JsonRecord`).
 */
export type ToolArguments = JsonRecord;

/**
 * Tool result interface.
 */
export interface ToolResult {
  readonly success: boolean;
  readonly content: string;
  /** Parsed structured payload when content is valid JSON. */
  readonly payload?: JsonValue;
  readonly error?: string;
}

/**
 * Tool execution result - can be ToolResult, a record, or a string.
 */
export type ToolExecutionResult = ToolResult | JsonRecord | string;

/**
 * Typed tool handler contract used across builder and runtime registration.
 */
export type ToolHandler<
  TArgs = ToolHandlerArgs,
  TResult extends ToolExecutionResult = ToolExecutionResult,
> = (args: TArgs) => Promise<TResult> | TResult;

/**
 * Canonical runtime tool registration contract.
 */
export interface RegisteredTool<
  TArgs = ToolArguments,
  TResult extends ToolExecutionResult = ToolExecutionResult,
  TSchema extends ToolParameterSchema = ToolParameterSchema,
> {
  readonly name: string;
  readonly description: string;
  readonly parameters: TSchema;
  execute: ToolHandler<TArgs, TResult>;
}

/**
 * Message content type - can be string or structured content
 */
export type MessageContent = string | JsonObject;

/**
 * Message role type
 */
export type MessageRole = "user" | "assistant" | "system" | "tool";

/**
 * Message interface
 */
export interface Message {
  readonly role: MessageRole;
  readonly content: MessageContent;
  readonly tool_call_id?: string;
  readonly name?: string;
}

/**
 * Memory snapshot messages
 */
export type MemorySnapshotMessages = readonly Message[];

/**
 * LangChain embedding model interface
 */
export interface LangChainEmbeddingModel {
  readonly model?: string;
  readonly modelName?: string;
  embedDocuments?(texts: string[]): Promise<number[][]>;
  aembedDocuments?(texts: string[]): Promise<number[][]>;
}

/**
 * LangChain message content type
 */
export type LangChainMessageContent =
  | string
  | Array<{ type: string; text?: string; image_url?: JsonObject }>
  | JsonObject;

/**
 * LangChain tool call interface
 */
export interface LangChainToolCall {
  readonly id?: string;
  readonly name?: string;
  readonly args?: JsonRecord;
}

/**
 * LangChain response interface
 */
export interface LangChainResponse {
  readonly content?: LangChainMessageContent;
  readonly tool_calls?: readonly LangChainToolCall[];
  readonly response_metadata?: {
    readonly finish_reason?: string;
    readonly token_usage?: {
      readonly prompt_tokens?: number;
      readonly completion_tokens?: number;
      readonly total_tokens?: number;
    };
  };
}

/**
 * LangChain chunk interface
 */
export interface LangChainChunk {
  readonly content?: string;
  readonly tool_calls?: readonly LangChainToolCall[];
}

/**
 * LangChain document interface
 */
export interface LangChainDocument {
  readonly page_content?: string;
  readonly content?: string;
  readonly metadata?: JsonRecord;
}

/**
 * LangChain memory interface
 */
export interface LangChainMemory {
  readonly chatHistory?: {
    readonly messages?: readonly LangChainMessage[];
    addUserMessage?(content: string): void;
    addAIChatMessage?(content: string): void;
  };
  readonly chat_memory?: {
    readonly messages?: readonly LangChainMessage[];
    addUserMessage?(content: string): void;
    addAIMessage?(content: string): void;
  };
  readonly messages?: readonly LangChainMessage[];
}

/**
 * LangChain message interface
 */
export interface LangChainMessage {
  readonly type?: "human" | "user" | "ai" | "assistant" | "system" | "tool";
  readonly content?: string | LangChainMessageContent;
}

/**
 * LangChain retriever interface
 */
export interface LangChainRetriever {
  getRelevantDocuments?(query: string): Promise<readonly LangChainDocument[]>;
  agetRelevantDocuments?(query: string): Promise<readonly LangChainDocument[]>;
}

/**
 * LangChain chain interface
 */
export interface LangChainChain {
  invoke?(inputs: JsonRecord): Promise<JsonValue>;
  ainvoke?(inputs: JsonRecord): Promise<JsonValue>;
  run?(inputs: JsonRecord): Promise<JsonValue>;
  arun?(inputs: JsonRecord): Promise<JsonValue>;
}

/**
 * Provider name type - extends the base Provider type
 */
export type ProviderName =
  | "openai"
  | "anthropic"
  | "bedrock"
  | "openrouter"
  | "perplexity"
  | "lmstudio"
  | "google"
  | "gemini"
  | "cohere"
  | "azure-openai"
  | string; // Allow custom provider names

/**
 * Backend configuration for RAG service
 */
export type RagBackendConfig = JsonRecord;

/**
 * Workflow execution result
 */
export type WorkflowExecutionResult = JsonRecord;

/**
 * Evaluation context
 */
export type EvaluationContext = JsonRecord;

/**
 * Chain input/output
 */
export type ChainInput = JsonRecord;
export type ChainOutput = JsonRecord;

/**
 * Prompt template variables
 */
export type PromptVariables = JsonRecord;

/**
 * Context update type
 */
export type ContextUpdate = JsonRecord;

/**
 * Metadata type
 */
export type Metadata = JsonRecord;

/**
 * Error object with optional message
 */
export interface ErrorLike {
  readonly message?: string;
  readonly error?: {
    readonly message?: string;
  };
}
