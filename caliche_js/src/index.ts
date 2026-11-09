/**
 * Caliche TypeScript SDK
 *
 * Provides AI agent functionality with support for multiple LLM providers.
 */

// Type declaration for zod (peer dependency, loaded at runtime)
declare namespace z {
  interface ZodTypeAny {
    parse(data: unknown): unknown;
  }
  class ZodError extends Error {
    errors: Array<{ path: (string | number)[]; message: string }>;
  }
}

import { native } from "./native/module";

export { Agent } from "./agent/agent";
export { AgentBuilder } from "./agent/builder";
export {
  DurableStreamBackend,
  type DurableStreamBackendInstance,
  type HnswConfig,
  type StreamChatOptions,
  type TierConfigSpec,
  type VectorBackendSpec,
  type VectorBackendTieredOptions,
  type VectorBackendTierSpec,
} from "./native/module";
export {
  agentBuilderFromEnv,
  configureBuilderFromEnv,
  type EnvProvider,
  type EnvRuntimeOptions,
  resolveApiKeyFromEnv,
  resolveBaseUrlFromEnv,
  resolveModelFromEnv,
  resolveProviderFromEnv,
} from "./runtime/env";
export {
  type CollectedStreamResult,
  collectStreamResult,
  collectText,
  collectWithUsage,
} from "./streaming/collectors";
export {
  type AgentEndEvent,
  type AgentRuntimeConfig,
  AgentRuntimeConfigError,
  type AgentRuntimeCredentials,
  type AgentRuntimeTimeouts,
  type AgentStartEvent,
  type AgentStreamState,
  type AttachmentInput,
  type ChainResult,
  type ChatResponse,
  type CompactionChange,
  type CompactionEvent,
  type CompactionReport,
  type CompletionResponse,
  type CompletionToolCall,
  type ContentPartJs,
  createInitialAgentStreamState,
  createTool,
  type DoneEvent,
  type ErrorEvent,
  type FlowExecutionResult,
  fromEventEnvelope,
  type HarnessDisciplineMetrics,
  isDoneEvent,
  isErrorEvent,
  isMetadataEvent,
  isPolicyViolationEvent,
  isTextChunkEvent,
  isToolCallResultEvent,
  isToolCallStartEvent,
  type LimitReachedEvent,
  type MessageJs,
  type MetadataEvent,
  normalizeHarnessDisciplineMetrics,
  type PerformanceMetrics,
  type PhaseTransitionEvent,
  type StatusEvent,
  type PolicyViolationEvent,
  type Provider,
  type ReasoningResult,
  type ReasoningStep,
  reduceAgentStreamEvent,
  type StepResultJs,
  type StreamEvent,
  type StreamEventEnvelope,
  type StreamEventType,
  type TextChunkEvent,
  type ThinkingChunkEvent,
  type TokenUsage,
  type Tool,
  type ToolCallJs,
  type ToolCallResultEvent,
  type ToolCallStartEvent,
  type ToolInfo,
  type Toolkit,
  type ToolRequirementPolicy,
  type ToolResult,
  type ToolSchema,
  type TurnEndEvent,
  type TurnStartEvent,
  toEventEnvelope,
  type UserContextJs,
} from "./streaming/events";
export { StreamIterator, type StreamReplaySource } from "./streaming/stream-iterator";

// Cleanup Utilities

/**
 * Wait for all pending async tasks to complete
 *
 * This is useful for scripts that need to ensure all background tasks
 * are finished before exiting. Call this before process.exit() for
 * clean shutdown.
 *
 * @returns Promise that resolves when pending tasks have had time to complete
 *
 * @example
 * ```typescript
 * async function main() {
 *   // ... use agent ...
 *
 *   // Wait for any pending tasks to complete
 *   await waitForPendingTasks();
 *
 *   // Now safe to exit
 *   process.exit(0);
 * }
 * ```
 */
export async function waitForPendingTasks(): Promise<void> {
  await native.waitForPendingTasks();
}

// Chains and patterns
export {
  DebatePattern,
  HierarchicalPattern,
  LLMChain,
  MapReduceChain,
  PipelinePattern,
  PlanExecutePattern,
  ReActPattern,
  ReflexionPattern,
  SelfAskPattern,
  SequentialChain,
  SupervisorPattern,
  TreeOfThoughtsPattern,
} from "./chains";
// Evaluation
export type { EvaluationDataset, EvaluationResult, TestCase } from "./evaluation";
export {
  ContainsEvaluator,
  EvaluationDatasetLoader,
  ExactMatchEvaluator,
  LLM_EVALUATOR_NEEDS_PROVIDER,
  LLMEvaluator,
} from "./evaluation";
// LLM client
export { AgentFlow, FlowBuilder, LlmClient, LlmClientBuilder, SharedState } from "./llm";
// RAG
export type {
  EnhancedQueryOptions,
  EnhancedQueryResult,
  GraphSearchResult,
  RagChunkHit,
  RagIndexResult,
  RagQueryResult,
  RagStats,
} from "./rag";
export {
  BuiltinTool,
  RAG_CUSTOM_BACKEND_UNSUPPORTED,
  RAG_SERVICE_REQUIRES_FEATURE,
  RAG_TOOL_REQUIRES_FEATURE,
  RagService,
  RagTool,
} from "./rag";
// Workflow orchestration
export type {
  AgentCapability,
  AgentStat,
  CoordinatorStats,
  DelegationResult,
  ExecutionResult,
  LoadBalancingStrategy,
  MergeStrategy,
  MessageBusStats,
  PendingInterrupt,
  WorkflowExecutionResult,
  WorkflowState,
} from "./workflow";
export {
  AgentCoordinator,
  AgentTemplate,
  MessageBus,
  PromptTemplate,
  PromptTemplateStore,
  StorageConfig,
  TemplateRegistry,
  Workflow,
  WorkflowBuilder,
  WorkflowEngine,
} from "./workflow";

// ============================================================================
// Adapters
// ============================================================================

export {
  CalicheMcpBridge,
  type CalicheMcpServerBinding,
  type McpPromptDescriptor,
  type McpResourceDescriptor,
} from "./adapters/ai-sdk-mcp";
export {
  CalicheLanguageModel,
  type CalicheMcpSnapshot,
  type CalicheModelOptions,
  caliche,
} from "./adapters/ai-sdk-provider";
export {
  LangChainAdapter,
  LangChainChainAdapter,
  LangChainMemoryAdapter,
  LangChainRetrieverAdapter,
} from "./adapters/langchain";
export { LangChainEmbeddingProvider } from "./adapters/langchain-embeddings";
export { LangChainLLMProvider } from "./adapters/langchain-llm";

// ============================================================================
// Hooks, Sessions, and In-Process MCP
// ============================================================================

export { createCalculatorTool, createCodeExecutionTool, createHttpTool } from "./builtin-tools";
export {
  type HookContext,
  HookMatcher,
  type HookResult,
  type LifecycleHook,
  PermissionMode,
} from "./hooks";
export {
  JsonlSessionEntry,
  JsonlSessionHeader,
  JsonlSessionStore,
  JsonlSessionTree,
  newEntryId,
  parseJsonlSession,
} from "./jsonl-session";
export { InProcessMcpServer, InProcessMcpServerBuilder, type InProcessMcpTool } from "./mcp";
export { createApplyPatchTool, createFsTool, MutationStyle } from "./mutation-style";
export { AgentSession } from "./session";
export {
  defaultInspectionManager,
  InspectionAction,
  InspectionHookAdapter,
  PathSecurityInspector,
  PermissionModeInspector,
  ToolInspectionManager,
} from "./tool-inspection";
export { ToolRegistry } from "./tool-registry";
export type {
  InferToolArguments,
  RegisteredTool,
  ToolArguments,
  ToolExecutionResult,
  ToolHandler,
  ToolHandlerArgs,
  ToolParameterSchema,
} from "./types";

/** Host/catalog fit helpers for in-process local inference (native local-inference feature). */
export const probeLocalInferenceHost = native.probeLocalInferenceHost?.bind(native);
export const fitLocalInferenceCatalog = native.fitLocalInferenceCatalog?.bind(native);
export const fitLocalInferenceModel = native.fitLocalInferenceModel?.bind(native);
export const safeLocalInferenceModels = native.safeLocalInferenceModels?.bind(native);

// ============================================================================
// MCP (Model Context Protocol)
// ============================================================================

export type { McpTool } from "./mcp";
export {
  LATEST_PROTOCOL_VERSION,
  McpServer,
  McpServerBuilder,
  SUPPORTED_PROTOCOL_VERSIONS,
} from "./mcp";
export { decodeSseChunk, decodeSseStream, encodeSseEvent } from "./transport/sse";
