import type {
  ChainResult,
  ChatResponse,
  CompletionResponse,
  FlowExecutionResult,
  MemoryPersistence,
  MessageJs,
  ReasoningResult,
  StepResultJs,
  StreamEvent,
  Tool,
  ToolInfo,
  Toolkit,
  ToolSchema,
  UserContextJs,
} from "../streaming/events";
import type {
  ChainInput,
  EvaluationContext,
  JsonRecord,
  JsonValue,
  PromptVariables,
  ToolParameterSchema,
} from "../types";
import type {
  CoordinatorStats,
  DelegationResult,
  EnhancedQueryResult,
  EvaluationDataset,
  EvaluationResult,
  ExecutionResult,
  GraphSearchResult,
  MessageBusStats,
  PendingInterrupt,
  RagIndexResult,
  RagQueryResult,
  RagStats,
  WorkflowExecutionResult,
} from "../types/native-results";

// Native module type definitions
/** Instance of a durable stream backend (from DurableStreamBackend.memory() or .redis()). */
export type DurableStreamBackendInstance = {};

interface NativeAgentBuilder {
  setProvider(provider: string): void;
  setModel(model: string): void;
  setSystemPrompt(prompt: string): void;
  setIncludeContextInSystem(value: boolean): void;
  setDurableStreamBackend?(backend: DurableStreamBackendInstance): void;
  setDurableStreamSqlite?(dbPath: string): void;
  setDurableStreamPostgres?(connectionString: string): void;
  setDurableStreamS3?(bucket: string, prefix: string, region?: string): void;
  jsonMode(): void;
  jsonSchema(name: string, schema: string): void;
  enableTool(tool: string): void;
  bufferMemory(
    maxTokens: number,
    minMessages: number,
    preserveSystem: boolean,
    summaryPlaceholder: boolean
  ): void;
  bufferMemoryConfig?(config: {
    max_tokens?: number;
    min_messages?: number;
    preserve_system?: boolean;
    summary_placeholder?: boolean;
  }): void;
  summaryMemory(
    maxTokens: number,
    summarizeThresholdPct: number,
    chunkSize: number,
    maxSummaries: number
  ): void;
  entityMemory(
    maxTokens: number,
    maxEntities: number,
    extractionInterval: number,
    similarityThreshold: number
  ): void;
  enableTools(tools: readonly string[]): void;
  enableAllDefaultTools(): void;
  enableExtendedDefaultTools?(): void;
  mutationStyle?(style: unknown): void;
  enableToolInspection?(workspaceRoot?: string | null): void;
  addInspectionHook?(matcher: unknown, hook: unknown): void;
  addTool(tool: Tool): void;
  registerToolkit(toolkit: Toolkit): void;
  setTemperature(temp: number): void;
  setMaxTokens(tokens: number): void;
  setTopP(topP: number): void;
  setApiKey(key: string): void;
  setApiKeyFor?(target: string, key: string): void;
  setAwsCredentials(accessKey: string, secretKey: string, region: string): void;
  setBaseUrl(url: string): void;
  setMaxIterations(maxIterations: number): void;
  setToolChoice?(choice: string): void;
  setAutoDiscoverMcp?(enabled: boolean): void;
  setAutoDiscoverSkills?(enabled: boolean): void;
  setExternalProvider(
    name: string,
    modelName: string,
    completeFn: (request: JsonRecord) => Promise<string>,
    streamFn: (request: JsonRecord) => AsyncGenerator<string, void, unknown>,
    supportsNativeTools: boolean,
    supportsStreaming: boolean
  ): void;
  setExternalEmbeddingProvider(
    name: string,
    embedFn: (request: JsonRecord) => Promise<string>
  ): void;
  setVectorBackendTiered(
    tiersJson: string,
    writeStrategy?: string,
    backgroundSync?: boolean,
    syncIntervalSecs?: number,
    searchStrategy?: string
  ): void;
  setVectorBackendMemory(ttlSecs?: number, maxEntries?: number): void;
  setVectorBackendRedis(url: string, prefix: string): void;
  setVectorBackendPostgres(
    connectionString: string,
    tableName: string,
    vectorDimension?: number,
    maxConnections?: number,
    hnswConfig?: HnswConfig
  ): void;
  setVectorBackendQdrant(
    url: string,
    vectorDimension: number,
    collectionPrefix?: string,
    apiKey?: string
  ): void;
  setVectorBackendQdrantGrpc(
    grpcUrl: string,
    vectorDimension: number,
    collectionPrefix?: string,
    apiKey?: string,
    batchSize?: number
  ): void;
  setVectorBackendPinecone(
    baseUrl: string,
    apiKey: string,
    dimension: number,
    timeoutSecs?: number
  ): void;
  setVectorBackendChroma(
    baseUrl: string,
    collectionPrefix?: string,
    expectedDimension?: number,
    timeoutSecs?: number
  ): void;
  setVectorBackendS3(
    bucket: string,
    keyPrefix: string,
    enableCompression?: boolean,
    batchSize?: number,
    cacheIndices?: boolean
  ): void;
  setVectorBackendDynamodb?(
    tableName: string,
    indexName: string,
    region: string,
    vectorDimension: number,
    ensureTable?: boolean
  ): void;
  setVectorBackendColdforge(
    bucket: string,
    keyPrefix: string,
    region: string,
    vectorDimension: number,
    cacheSizeMb?: number,
    numCentroids?: number
  ): void;
  vectorBackendFromEnv(): void;
  withMetrics(enabled: boolean): void;
  persistMemoryWith(
    saveFn: (err: Error | null, payload: JsonValue) => Promise<string> | string,
    loadFn: (err: Error | null, payload: JsonValue) => Promise<string> | string,
    clearFn: (err: Error | null, payload: JsonValue) => Promise<string> | string,
    conversationId: string,
    autosave: boolean
  ): void;
  setAzureOpenAIConfig?(
    resourceName: string,
    deploymentId: string,
    apiKey: string,
    apiVersion: string
  ): void;
  vectorMemory(
    maxTokens: number,
    maxMemories: number,
    topK: number,
    similarityThreshold: number,
    bufferTokens: number
  ): void;
  sharedRagTool?(ragTool: NativeRagTool): void;
  setPermissionMode(mode: NativePermissionMode): void;
  setAllowedTools(tools: readonly string[]): void;
  addHook(
    matcher: NativeHookMatcher,
    preToolUseFn: (ctxValue: unknown) => string | Promise<string>,
    postToolUseFn?: (combinedValue: unknown) => string | Promise<string>
  ): void;
  addMcpServer(name: string, server: NativeInProcessMcpServer): void;
  build(): Promise<NativeAgent>;
}

type NativePermissionMode = unknown;
type NativeHookMatcher = unknown;
type NativeInProcessMcpServer = unknown;

/** One tier in a tiered vector backend: backend spec + tier config */
export interface VectorBackendTierSpec {
  backend: VectorBackendSpec;
  tierConfig: TierConfigSpec;
}

/** Backend spec for one tier (discriminated by type) */
export type VectorBackendSpec =
  | { type: "memory"; ttlSecs?: number; maxEntries?: number }
  | { type: "redis"; url: string; prefix: string }
  | {
      type: "postgres";
      connectionString: string;
      tableName?: string;
      vectorDimension?: number;
      maxConnections?: number;
    }
  | {
      type: "qdrant";
      url: string;
      vectorDimension: number;
      collectionPrefix?: string;
      apiKey?: string;
      batchSize?: number;
      timeoutSecs?: number;
    }
  | {
      type: "qdrant_grpc";
      grpcUrl: string;
      vectorDimension: number;
      collectionPrefix?: string;
      apiKey?: string;
      batchSize?: number;
    }
  | { type: "pinecone"; baseUrl: string; apiKey: string; dimension: number; namespace?: string }
  | { type: "chroma"; url: string; collectionName?: string; dimension: number }
  | { type: "s3"; bucket: string; prefix: string; dimension: number; region?: string }
  | {
      type: "coldforge";
      bucket: string;
      prefix: string;
      dimension: number;
      region?: string;
      compression?: boolean;
    }
  | {
      type: "dynamodb";
      tableName: string;
      indexName?: string;
      dimension: number;
      region?: string;
      endpointUrl?: string;
      ensureTable?: boolean;
    };

/** Tier config for capacity and promotion */
export interface TierConfigSpec {
  name: string;
  maxVectors?: number;
  maxBytes?: number;
  promotionThreshold?: number;
}

/** Options for tiered vector backend */
export interface VectorBackendTieredOptions {
  writeStrategy?: "write_through" | "write_to_source" | "write_to_source_and_hot";
  /** Default: search all tiers. Use `hot_then_source_fallback` to skip intermediate cache tiers on the query path. */
  searchStrategy?: "search_all_tiers" | "hot_then_source_fallback";
  backgroundSync?: boolean;
  syncIntervalSecs?: number;
}
/** Options for streamChat (e.g. resume from durable stream) */
export interface StreamChatOptions {
  /** Resume from this event ID when using a durable stream backend */
  resumeFrom?: string;
  /** Preferred stream event schema version. Defaults to "1". */
  eventSchemaVersion?: "v0" | "1";
  /** Attempt to parse tool result content into structured payloads. Defaults to true. */
  typedToolResults?: boolean;
}

interface NativeAgent {
  cloneForRequest(): Promise<NativeAgent>;
  chat(message: string): Promise<ChatResponse>;
  streamChat(message: string): NativeStreamIterator;
  streamChatWithOptions?(message: string, options?: StreamChatOptions | null): NativeStreamIterator;
  streamChatWithHistory?(message: string, history: MessageJs[]): NativeStreamIterator;
  streamChatWithHistoryAndOptions?(
    message: string,
    history: MessageJs[],
    options?: StreamChatOptions | null
  ): NativeStreamIterator;
  getProvider(): string;
  getModel(): string;
  listTools(): ToolInfo[];
  toolRegistry?(): Promise<unknown>;
  chatWithHistory(message: string, history: MessageJs[]): Promise<ChatResponse>;
  addTool(
    name: string,
    description: string,
    parameters: JsonValue,
    executeFn: (err: Error | null, arg: JsonValue) => void
  ): void;
  setUserContext(userId: string | null, metadata: JsonValue | null): void;
  getUserContext(): UserContextJs | null;
  getMessages(): Promise<MessageJs[]>;
  step(messages: MessageJs[]): Promise<StepResultJs>;
}
interface NativeStreamIterator {
  [key: string]: unknown;
}
interface NativeSharedState {
  get(path: string): Promise<string | null>;
  set(path: string, valueJson: string): Promise<void>;
  getAll(): Promise<string>;
}
interface NativeFlowBuilder {
  addNode(name: string, agent: NativeAgent): void;
  setEntryPoint(name: string): void;
  build(): NativeAgentFlow;
}
type NativeAgentFlow = {};
interface NativeLlmClientBuilder {
  setProvider(provider: string): void;
  setModel(model: string): void;
  setApiKey(key: string): void;
  setApiKeyFor(target: string, key: string): void;
  setAwsCredentials(
    accessKey: string,
    secretKey: string,
    region: string,
    sessionToken?: string
  ): void;
  setBaseUrl(url: string): void;
  setTemperature(temp: number): void;
  setMaxTokens(tokens: number): void;
  setTimeout(seconds: number): void;
  setMaxRetries(retries: number): void;
  setTools(tools: readonly ToolSchema[]): void;
  setToolChoice(choice: string): void;
  build(): Promise<NativeLlmClient>;
}
type NativeLlmClient = {};
/** NAPI `JsWorkflowBuilder` instance */
interface NativeWorkflowBuilder {
  addAgent(name: string, promptTemplate: string): void;
  addAgentWithCapability(name: string, capability: string, promptTemplate: string): void;
  addCondition(name: string, predicate: string): void;
  addParallel(branches: Array<{ name: string; prompt: string }>): void;
  fanOut(branches: Array<{ name: string; prompt: string }>): void;
  collectMerge(mergeNodeName: string, strategy: string): void;
  addParallelWithMergeStrategy(
    branches: Array<{ name: string; prompt: string }>,
    strategy: string
  ): void;
  addMerge(name: string, strategy: string): void;
  addLoop(name: string, maxIterations: number, condition: string, bodyPrompt: string): void;
  addHumanInterrupt(name: string, message: string, requiredFields?: string[]): void;
  connectNodes(from: string, to: string, condition?: string): void;
  setEntryPoint(name: string): void;
  build(): NativeWorkflow;
}

/** NAPI `JsWorkflow` instance */
interface NativeWorkflow {
  id(): string;
  name(): string;
  description(): string;
  toJson(): string;
  execute(initialState?: unknown): Promise<{ result: unknown; state: unknown }>;
}
/** NAPI `JsWorkflowEngine` instance */
interface NativeWorkflowEngine {
  registerWorkflow(workflow: NativeWorkflow): Promise<void>;
}
interface NativeAgentCoordinator {
  registerAgent(
    name: string,
    role: string,
    capabilities: string[],
    provider: string,
    model: string
  ): Promise<string>;
  unregisterAgent(agentId: string): Promise<void>;
  delegateTask(message: string, capability?: string): Promise<DelegationResult>;
  executeOnAgent(agentId: string, message: string): Promise<ExecutionResult>;
  sendMessage(fromAgentId: string, toAgentId: string, content: string): Promise<void>;
  broadcast(fromAgentId: string, content: string): Promise<void>;
  getStats(): Promise<CoordinatorStats>;
  listAgentIds(): string[];
  agentCount(): number;
  getMessageBus(): NativeMessageBus;
}
interface NativeMessageBus {
  getStats(): Promise<MessageBusStats>;
  queueSize(): Promise<number>;
}
interface NativeDebatePattern {
  execute(): Promise<any>;
  getId(): string;
  getName(): string;
  getDescription(): string;
}
interface NativeHierarchicalPattern {
  addLayer(
    name: string,
    agentCount: number,
    promptTemplate: string,
    capability?: string
  ): NativeHierarchicalPattern;
  execute(): Promise<any>;
  getId(): string;
  getName(): string;
  getDescription(): string;
}
interface NativePipelinePattern {
  addStage(
    name: string,
    prompt: string,
    capability?: string,
    transform?: string
  ): NativePipelinePattern;
  execute(): Promise<any>;
  getId(): string;
  getName(): string;
  getDescription(): string;
}
interface NativeSupervisorPattern {
  execute(): Promise<any>;
  getId(): string;
  getName(): string;
  getDescription(): string;
}
type NativeExactMatchEvaluator = {
  evaluate(
    output: string,
    expected: string,
    context?: EvaluationContext
  ): Promise<EvaluationResult>;
};
type NativeContainsEvaluator = {
  evaluate(
    output: string,
    expected: string,
    context?: EvaluationContext
  ): Promise<EvaluationResult>;
};
type NativeLLMEvaluator = {
  evaluate(
    output: string,
    expected: string,
    context?: EvaluationContext
  ): Promise<EvaluationResult>;
};
type NativeRagService = {
  indexDocument(content: string, docId: string): Promise<RagIndexResult>;
  query(query: string, topK: number, minSimilarity?: number): Promise<RagQueryResult>;
  enhancedQuery(
    query: string,
    topK: number,
    minSimilarity?: number,
    searchMode?: "dense" | "hybrid" | "full_text" | "hybrid_full_text",
    denseWeight?: number,
    sparseWeight?: number,
    rerank?: boolean,
    metadataFilter?: unknown,
    maxContextTokens?: number
  ): Promise<EnhancedQueryResult>;
  graphSearch(query: string, topK: number, maxHops?: number): Promise<GraphSearchResult>;
  stats(): Promise<RagStats>;
  clear(): Promise<void>;
};
type NativeRagTool = {
  service(): NativeRagService;
};

/**
 * HNSW (Hierarchical Navigable Small World) index configuration
 */
export interface HnswConfig {
  /** Number of connections per layer (default: 16) */
  m: number;
  /** Search width during construction (default: 200) */
  ef_construction: number;
  /** Search width during query (default: 50) */
  ef_search: number;
  /** Maximum connections per node (default: 16) */
  max_nb_connection: number;
  /** Candidate multiplier before exact reranking (default: 4) */
  rerank_overfetch?: number;
}

interface NativeModule {
  // Class constructors
  AgentBuilder: {
    new (): NativeAgentBuilder;
  };
  LlmClientBuilder: {
    new (): NativeLlmClientBuilder;
  };
  SharedState: {
    new (): NativeSharedState;
  };
  FlowBuilder: {
    new (): NativeFlowBuilder;
  };
  DebatePattern: {
    new: (
      name: string,
      topic: string,
      debaterCount: number,
      maxRounds: number,
      moderatorPrompt?: string,
      description?: string
    ) => NativeDebatePattern;
  };
  HierarchicalPattern: {
    new: (name: string, description?: string) => NativeHierarchicalPattern;
  };
  PipelinePattern: {
    new: (name: string, description?: string) => NativePipelinePattern;
  };
  SupervisorPattern: {
    new: (
      name: string,
      supervisorPrompt: string,
      workerCount: number,
      workerCapability?: string,
      description?: string
    ) => NativeSupervisorPattern;
  };
  RagService: {
    new: (provider: string, model: string) => Promise<NativeRagService>;
  };
  RagTool: {
    new: (provider: string, model: string) => Promise<NativeRagTool>;
  };
  AgentCoordinator: {
    new (): NativeAgentCoordinator;
    withStrategy(strategy: string): NativeAgentCoordinator;
  };
  DurableStreamBackend: {
    memory(capacity: number): DurableStreamBackendInstance;
    redis(redisUrl: string): DurableStreamBackendInstance;
  };
  LlmChain: {
    new (agent: NativeAgent, promptTemplate: string): NativeLLMChain;
  };
  SequentialChain: {
    new (): NativeSequentialChain;
  };
  MapReduceChain: {
    new (
      mapAgent: NativeAgent,
      mapPrompt: string,
      reduceAgent: NativeAgent,
      reducePrompt: string
    ): NativeMapReduceChain;
  };
  /**
   * NAPI exports `static new(...)`. Use property typings (not method/`new()` construct
   * signatures) so oxfmt cannot rewrite them into construct signatures.
   */
  ReActPattern: {
    new: (provider: string, model: string, maxIterations: number) => Promise<NativeReActPattern>;
  };
  PlanExecutePattern: {
    new: (
      provider: string,
      model: string,
      maxIterations: number
    ) => Promise<NativePlanExecutePattern>;
  };
  ReflexionPattern: {
    new: (
      provider: string,
      model: string,
      maxIterations: number
    ) => Promise<NativeReflexionPattern>;
  };
  SelfAskPattern: {
    new: (provider: string, model: string, maxIterations?: number) => Promise<NativeSelfAskPattern>;
  };
  TreeOfThoughtsPattern: {
    new: (
      provider: string,
      model: string,
      maxIterations: number,
      breadth: number
    ) => Promise<NativeTreeOfThoughtsPattern>;
  };
  // Agent builder functions
  createAgentBuilder(): NativeAgentBuilder;
  builderSetProvider(builder: NativeAgentBuilder, provider: string): void;
  builderSetModel(builder: NativeAgentBuilder, model: string): void;
  builderSetSystemPrompt(builder: NativeAgentBuilder, prompt: string): void;
  builderEnableTool(builder: NativeAgentBuilder, tool: string): void;
  builderEnableTools(builder: NativeAgentBuilder, tools: readonly string[]): void;
  builderEnableAllDefaultTools(builder: NativeAgentBuilder): void;
  builderAddTool(
    builder: NativeAgentBuilder,
    name: string,
    description: string,
    parameters: JsonValue,
    executeFn: (...callbackArgs: unknown[]) => Promise<string> | string
  ): void;
  builderRegisterToolkit(builder: NativeAgentBuilder, toolkit: Toolkit): void;
  builderSetTemperature(builder: NativeAgentBuilder, temp: number): void;
  builderSetMaxTokens(builder: NativeAgentBuilder, tokens: number): void;
  builderSetTopP(builder: NativeAgentBuilder, topP: number): void;
  builderExternalProvider(
    builder: NativeAgentBuilder,
    name: string,
    modelName: string,
    completeFn: (request: JsonRecord) => Promise<string>,
    streamFn: (request: JsonRecord) => AsyncGenerator<string, void, unknown>,
    supportsNativeTools: boolean,
    supportsStreaming: boolean
  ): void;
  builderExternalEmbeddingProvider(
    builder: NativeAgentBuilder,
    name: string,
    embedFn: (request: JsonRecord) => Promise<string>
  ): void;
  builderSetApiKey(builder: NativeAgentBuilder, key: string): void;
  builderSetAwsCredentials(
    builder: NativeAgentBuilder,
    accessKey: string,
    secretKey: string,
    region: string
  ): void;
  builderSetBaseUrl(builder: NativeAgentBuilder, url: string): void;
  builderSetMaxIterations(builder: NativeAgentBuilder, maxIterations: number): void;
  builderSetAzureOpenAIConfig?(
    builder: NativeAgentBuilder,
    resourceName: string,
    deploymentId: string,
    apiKey?: string,
    apiVersion?: string
  ): void;
  setAzureOpenAIConfig?(
    builder: NativeAgentBuilder,
    resourceName: string,
    deploymentId: string,
    apiKey?: string,
    apiVersion?: string
  ): void;
  bufferMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    minMessages: number,
    preserveSystem: boolean,
    summaryPlaceholder: boolean
  ): void;
  summaryMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    summarizeThresholdPct: number,
    chunkSize: number,
    maxSummaries: number
  ): void;
  entityMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    maxEntities: number,
    extractionInterval: number,
    similarityThreshold: number
  ): void;
  vectorMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    maxMemories: number,
    topK: number,
    similarityThreshold: number,
    bufferTokens: number,
    namespace?: string
  ): void;
  builderBufferMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    minMessages: number,
    preserveSystem: boolean,
    summaryPlaceholder: boolean
  ): void;
  builderSummaryMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    summarizeThresholdPct: number,
    chunkSize: number,
    maxSummaries: number
  ): void;
  builderEntityMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    maxEntities: number,
    extractionInterval: number,
    similarityThreshold: number
  ): void;
  builderVectorMemory(
    builder: NativeAgentBuilder,
    maxTokens: number,
    maxMemories: number,
    topK: number,
    similarityThreshold: number,
    bufferTokens: number,
    namespace?: string
  ): void;
  builderPersistMemoryWith(
    builder: NativeAgentBuilder,
    persistence: MemoryPersistence,
    conversationId: string,
    autosave: boolean
  ): void;
  builderSharedRagTool?(builder: NativeAgentBuilder, ragTool: NativeRagTool): void;
  // Vector Backends
  builderSetVectorBackendMemory(
    builder: NativeAgentBuilder,
    ttlSecs?: number,
    maxEntries?: number
  ): void;
  builderSetVectorBackendRedis(builder: NativeAgentBuilder, url: string, prefix: string): void;
  builderSetVectorBackendPostgres(
    builder: NativeAgentBuilder,
    connectionString: string,
    tableName: string,
    vectorDimension?: number,
    maxConnections?: number,
    hnswConfig?: HnswConfig
  ): void;
  builderWithMetrics(builder: NativeAgentBuilder, enabled: boolean): void;
  builderSetVectorBackendQdrant(
    builder: NativeAgentBuilder,
    url: string,
    vectorDimension: number,
    collectionPrefix?: string,
    apiKey?: string
  ): void;
  builderSetVectorBackendQdrantGrpc(
    builder: NativeAgentBuilder,
    grpcUrl: string,
    vectorDimension: number,
    collectionPrefix?: string,
    apiKey?: string,
    batchSize?: number
  ): void;
  builderSetVectorBackendPinecone(
    builder: NativeAgentBuilder,
    baseUrl: string,
    apiKey: string,
    dimension: number,
    timeoutSecs?: number
  ): void;
  builderSetVectorBackendChroma(
    builder: NativeAgentBuilder,
    baseUrl: string,
    collectionPrefix?: string,
    expectedDimension?: number,
    timeoutSecs?: number
  ): void;
  builderSetVectorBackendS3(
    builder: NativeAgentBuilder,
    bucket: string,
    keyPrefix: string,
    enableCompression?: boolean,
    batchSize?: number,
    cacheIndices?: boolean
  ): void;
  builderSetVectorBackendColdforge(
    builder: NativeAgentBuilder,
    bucket: string,
    keyPrefix: string,
    region: string,
    vectorDimension: number,
    cacheSizeMb?: number,
    numCentroids?: number
  ): void;
  builderSetVectorBackendDynamodb?(
    builder: NativeAgentBuilder,
    tableName: string,
    indexName: string,
    region: string,
    vectorDimension: number,
    ensureTable?: boolean
  ): void;
  builderVectorBackendFromEnv(builder: NativeAgentBuilder): void;
  builderBuild(builder: NativeAgentBuilder): Promise<NativeAgent>;
  agentChat(agent: NativeAgent, message: string): Promise<ChatResponse>;
  agentStreamChat(agent: NativeAgent, message: string): NativeStreamIterator;
  agentGetProvider(agent: NativeAgent): string;
  agentGetModel(agent: NativeAgent): string;
  streamNext(iter: NativeStreamIterator): Promise<StreamEvent | null | undefined>;
  streamCancel?(iter: NativeStreamIterator): void;
  agentFlowRun(flow: NativeAgentFlow, initialInput: string): Promise<FlowExecutionResult>;
  agentFlowToMermaid(flow: NativeAgentFlow): string;
  agentFlowEntryPoint(flow: NativeAgentFlow): string;
  agentFlowNodeNames(flow: NativeAgentFlow): string[];
  waitForPendingTasks(): Promise<void>;
  // LLM Client
  llmClientComplete(
    client: NativeLlmClient,
    messages: Array<{ role: string; content: string }>
  ): Promise<CompletionResponse>;
  llmClientStream(
    client: NativeLlmClient,
    messages: Array<{ role: string; content: string }>
  ): NativeStreamIterator;
  llmClientCompleteText(client: NativeLlmClient, prompt: string): Promise<string>;
  llmClientStreamText(client: NativeLlmClient, prompt: string): NativeStreamIterator;
  llmClientGetProviderName(client: NativeLlmClient): string;
  llmClientGetModel(client: NativeLlmClient): string;
  // Chains
  createLLMChain(agent: NativeAgent, promptTemplate: string): NativeLLMChain;
  createSequentialChain(): NativeSequentialChain;
  sequentialChainAddChain(
    chain: NativeSequentialChain,
    name: string,
    llmChain: NativeLLMChain
  ): void;
  createMapReduceChain(
    mapAgent: NativeAgent,
    mapPrompt: string,
    reduceAgent: NativeAgent,
    reducePrompt: string
  ): NativeMapReduceChain;
  // Chain wrapper functions
  LlmChainFromAgent(
    agent: NativeAgent,
    promptTemplate: string
  ): NativeLLMChain | Promise<NativeLLMChain>;
  MapReduceChainFromAgents(
    mapAgent: NativeAgent,
    mapPrompt: string,
    reduceAgent: NativeAgent,
    reducePrompt: string
  ): NativeMapReduceChain | Promise<NativeMapReduceChain>;
  // Prompts
  createPromptTemplate(template: string): NativePromptTemplate;
  promptTemplateFormat(template: NativePromptTemplate, variables: PromptVariables): string;
  promptTemplateVariables(template: NativePromptTemplate): string[];
  createSummarizeTemplate(): NativePromptTemplate;
  createAnalyzeTemplate(): NativePromptTemplate;
  createTranslateTemplate(): NativePromptTemplate;
  // Storage
  createStorageConfigFilesystem(basePath: string): NativeStorageConfig;
  createStorageConfigAwsS3?(
    bucket: string,
    region?: string,
    accessKeyId?: string,
    secretAccessKey?: string,
    sessionToken?: string
  ): NativeStorageConfig;
  createStorageConfigCloudflareR2?(
    bucket: string,
    accountId: string,
    accessKeyId: string,
    secretAccessKey: string,
    endpoint?: string
  ): NativeStorageConfig;
  storageConfigWithCache(config: NativeStorageConfig, enabled: boolean): NativeStorageConfig;
  storageConfigWithCacheTtl(config: NativeStorageConfig, ttlSeconds: number): NativeStorageConfig;
  storageConfigWithCacheMaxSize(
    config: NativeStorageConfig,
    maxSizeMb: number
  ): NativeStorageConfig;
  // Templates
  createTemplateRegistry(): NativeTemplateRegistry;
  createTemplateRegistryWithStorage(config: NativeStorageConfig): Promise<NativeTemplateRegistry>;
  templateRegistryAddTemplateDir(registry: NativeTemplateRegistry, dir: string): void;
  templateRegistryLoadTemplates(registry: NativeTemplateRegistry): Promise<number>;
  templateRegistryGet(
    registry: NativeTemplateRegistry,
    id: string
  ): Promise<NativeAgentTemplate | null>;
  templateRegistryList(registry: NativeTemplateRegistry): Promise<NativeAgentTemplate[]>;
  // Workflow Orchestration
  createWorkflowBuilder(name: string): NativeWorkflowBuilder;
  workflowEngineStartExecution(engine: NativeWorkflowEngine, workflowId: string): Promise<string>;
  workflowEngineGetExecutionResult(
    engine: NativeWorkflowEngine,
    executionId: string
  ): Promise<WorkflowExecutionResult>;
  workflowEngineGetExecutionState(
    engine: NativeWorkflowEngine,
    executionId: string
  ): Promise<string>;
  workflowEngineGetPendingInterrupts(engine: NativeWorkflowEngine): Promise<PendingInterrupt[]>;
  workflowEngineResumeExecution(
    engine: NativeWorkflowEngine,
    interruptId: string,
    feedback: JsonRecord
  ): Promise<void>;
  workflowEnginePauseExecution(engine: NativeWorkflowEngine, executionId: string): Promise<void>;
  workflowEngineCancelExecution(engine: NativeWorkflowEngine, executionId: string): Promise<void>;
  createWorkflowEngine(): NativeWorkflowEngine;
  // Evaluation
  createExactMatchEvaluator(caseSensitive?: boolean): NativeExactMatchEvaluator;
  createContainsEvaluator(caseSensitive?: boolean): NativeContainsEvaluator;
  createLLMEvaluator(criteria: string, provider?: string, model?: string): NativeLLMEvaluator;
  loadDatasetFromJsonl(path: string): Promise<EvaluationDataset>;
  loadDatasetFromJson(path: string): Promise<EvaluationDataset>;
  // Prompt Template Store
  createPromptTemplateStore?(config: NativeStorageConfig): Promise<NativePromptTemplateStore>;
  promptTemplateStoreSave?(
    store: NativePromptTemplateStore,
    name: string,
    template: NativePromptTemplate
  ): Promise<void>;
  promptTemplateStoreLoad?(
    store: NativePromptTemplateStore,
    name: string
  ): Promise<NativePromptTemplate>;
  promptTemplateStoreList?(store: NativePromptTemplateStore): Promise<string[]>;
  // Built-in Tools
  BuiltinTool?: {
    new (): NativeBuiltinTool;
    calculator(): NativeBuiltinTool;
    fs(mutationStyle?: unknown): NativeBuiltinTool;
    applyPatch(): NativeBuiltinTool;
    http(): NativeBuiltinTool;
    codeExecution(): NativeBuiltinTool;
  };
  AgentSession: {
    new (id: string, history?: readonly MessageJs[] | null, sharedState?: unknown): unknown;
    withHistory?(id: string, history: readonly MessageJs[]): unknown;
    with_history?(id: string, history: readonly MessageJs[]): unknown;
  };
  McpServerBuilder?: { new (): unknown };
  InProcessMcpServerBuilder?: { new (name: string): unknown };
  inProcessMcpServerBuilderVersion?(builder: unknown, version: string): void;
  JsPermissionMode: { new (mode: string): unknown };
  JsPermissionModeDefault?(): unknown;
  JsPermissionModeAcceptEdits?(): unknown;
  JsPermissionModeAcceptAll?(): unknown;
  JsPermissionModeDenyAll?(): unknown;
  JsPermissionModeInteractive?(): unknown;
  JsHookMatcher: { new (matcherType: string, toolName?: string | null): unknown };
  JsHookMatcherTool?(toolName: string): unknown;
  JsHookMatcherAll?(): unknown;
  MutationStyle: {
    new (style: string): unknown;
    editWrite(): unknown;
    applyPatch(): unknown;
  };
  ToolRegistry: { new (): unknown };
  InspectionAction: {
    allow(): unknown;
    deny(reason: string): unknown;
    requireApproval(reason: string): unknown;
  };
  PathSecurityInspector: { new (workspaceRoot: string): unknown };
  PermissionModeInspector: { new (mode: unknown): unknown };
  ToolInspectionManager: { new (): unknown };
  InspectionHookAdapter: { new (manager: unknown): unknown };
  defaultInspectionManager(permissionMode: unknown, workspaceRoot?: string | null): unknown;
  JsonlSessionHeader: { new (id: string, cwd?: string | null): unknown };
  JsonlSessionEntry: {
    message(id: string, message: unknown, parentId?: string | null): unknown;
    branchSummary(id: string, fromId: string, summary: string, parentId?: string | null): unknown;
    compactionSummary(
      id: string,
      summary: string,
      parentId?: string | null,
      tokensBefore?: number | null
    ): unknown;
    modelChange(id: string, provider: string, modelId: string, parentId?: string | null): unknown;
  };
  JsonlSessionTree: { new (header: unknown): unknown };
  JsonlSessionStore: { new (path: string): unknown };
  parseJsonlSession(text: string): unknown;
  newEntryId(): string;
  createCalculatorTool?(): NativeBuiltinTool;
  createFsTool?(mutationStyle?: unknown): NativeBuiltinTool;
  createHttpTool?(): NativeBuiltinTool;
  createCodeExecutionTool?(): NativeBuiltinTool;
  /** Host RAM/VRAM probe for local inference (requires local-inference feature). */
  probeLocalInferenceHost?(): {
    totalRamBytes: number;
    availableRamBytes: number;
    availableVramBytes?: number | null;
    availableDiskBytes?: number | null;
    recommendedDevice: string;
    notes: string[];
  };
  fitLocalInferenceCatalog?(): Array<{
    modelId: string;
    displayName: string;
    kind: string;
    status: string;
    estimatedWeightBytes: number;
    estimatedRuntimeBytes: number;
    recommendedNCtx: number;
    maxSafeNCtx: number;
    usableRamBytes: number;
    reasons: string[];
  }>;
  fitLocalInferenceModel?(id: string): {
    modelId: string;
    displayName: string;
    kind: string;
    status: string;
    estimatedWeightBytes: number;
    estimatedRuntimeBytes: number;
    recommendedNCtx: number;
    maxSafeNCtx: number;
    usableRamBytes: number;
    reasons: string[];
  };
  safeLocalInferenceModels?(): Array<{
    modelId: string;
    displayName: string;
    kind: string;
    status: string;
    estimatedWeightBytes: number;
    estimatedRuntimeBytes: number;
    recommendedNCtx: number;
    maxSafeNCtx: number;
    usableRamBytes: number;
    reasons: string[];
  }>;
  builtinToolName?(tool: NativeBuiltinTool): string;
  builtinToolDescription?(tool: NativeBuiltinTool): string;
  builtinToolParameters?(tool: NativeBuiltinTool): ToolParameterSchema;
}

type NativeLLMChain = {
  run(input: ChainInput): Promise<ChainResult>;
};
type NativeSequentialChain = {
  addChainInstance(name: string, chain: NativeLLMChain): void;
  run(input: ChainInput): Promise<ChainResult>;
};
type NativeMapReduceChain = {
  run(items: Array<ChainInput>): Promise<ChainResult>;
};
type NativeReActPattern = {
  execute(query: string): Promise<ReasoningResult>;
};
type NativePlanExecutePattern = {
  execute(query: string): Promise<ReasoningResult>;
};
type NativeReflexionPattern = {
  execute(query: string): Promise<ReasoningResult>;
};
type NativeSelfAskPattern = {
  execute(query: string): Promise<ReasoningResult>;
};
type NativeTreeOfThoughtsPattern = {
  execute(query: string): Promise<ReasoningResult>;
};
type NativePromptTemplate = {};
type NativeStorageConfig = {};
type NativeTemplateRegistry = {};
type NativeAgentTemplate = {
  id(): string;
  name(): string;
  description(): string;
  version(): string;
};
type NativePromptTemplateStore = {};
type NativeBuiltinTool = {};

// Import native module with proper typing
export const native = require("@caliche/runtime-node") as NativeModule;

/** Durable stream backend factory (memory or Redis). Use with AgentBuilder.durableStreamBackend(). */
export const DurableStreamBackend = native.DurableStreamBackend;

export type {
  NativeAgent,
  NativeAgentBuilder,
  NativeAgentCoordinator,
  NativeBuiltinTool,
  NativeAgentFlow,
  NativeAgentTemplate,
  NativeContainsEvaluator,
  NativeDebatePattern,
  NativeExactMatchEvaluator,
  NativeFlowBuilder,
  NativeHierarchicalPattern,
  NativeLLMChain,
  NativeLLMEvaluator,
  NativeLlmClient,
  NativeLlmClientBuilder,
  NativeMapReduceChain,
  NativeMessageBus,
  NativeModule,
  NativePipelinePattern,
  NativePlanExecutePattern,
  NativePromptTemplate,
  NativePromptTemplateStore,
  NativeRagService,
  NativeRagTool,
  NativeReActPattern,
  NativeReflexionPattern,
  NativeSelfAskPattern,
  NativeSequentialChain,
  NativeSharedState,
  NativeStorageConfig,
  NativeStreamIterator,
  NativeSupervisorPattern,
  NativeTemplateRegistry,
  NativeTreeOfThoughtsPattern,
  NativeWorkflow,
  NativeWorkflowBuilder,
  NativeWorkflowEngine,
};
