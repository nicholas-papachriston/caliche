import type { Agent } from "../agent/agent";
import {
  type NativeBuiltinTool,
  type NativeRagService,
  type NativeRagTool,
  native,
} from "../native/module";
import type { Tool } from "../streaming/events";
import type { RagBackendConfig, ToolArguments } from "../types";
import type {
  EnhancedQueryOptions,
  EnhancedQueryResult,
  GraphSearchResult,
  RagChunkHit,
  RagIndexResult,
  RagQueryResult,
  RagStats,
} from "../types/native-results";

export type {
  EnhancedQueryOptions,
  EnhancedQueryResult,
  GraphSearchResult,
  RagChunkHit,
  RagIndexResult,
  RagQueryResult,
  RagStats,
};

function textFromRecord(record: Record<string, unknown>, fallback: string): string {
  if (typeof record.text === "string") {
    return record.text;
  }
  if (typeof record.content === "string") {
    return record.content;
  }
  return fallback;
}

function hitFromUnknown(item: unknown, objectFallback: string): RagChunkHit | undefined {
  if (typeof item === "string") {
    return { text: item };
  }
  if (!item || typeof item !== "object") {
    return undefined;
  }
  const record = item as Record<string, unknown>;
  return {
    id: typeof record.id === "string" ? record.id : undefined,
    text: textFromRecord(record, objectFallback),
    score: typeof record.score === "number" ? record.score : undefined,
    metadata:
      record.metadata && typeof record.metadata === "object"
        ? (record.metadata as RagChunkHit["metadata"])
        : undefined,
  };
}

function collectHits(items: unknown[], stringifyObjects: boolean): RagChunkHit[] {
  const hits: RagChunkHit[] = [];
  for (const item of items) {
    const fallback = stringifyObjects ? JSON.stringify(item) : "";
    const hit = hitFromUnknown(item, fallback);
    if (hit) {
      hits.push(hit);
    }
  }
  return hits;
}

function normalizeQueryResult(raw: unknown): RagQueryResult {
  const value = (raw ?? {}) as {
    results?: unknown;
    hits?: unknown;
    top_similarity?: number;
    topSimilarity?: number;
    usage?: RagQueryResult["usage"];
  };

  const hits = Array.isArray(value.hits)
    ? collectHits(value.hits, false)
    : Array.isArray(value.results)
      ? collectHits(value.results, true)
      : [];

  return {
    results: hits.map((hit) => hit.text),
    hits,
    top_similarity: value.top_similarity ?? value.topSimilarity,
    usage: value.usage,
  };
}

function embeddingModelFor(agent: Agent): string {
  if (process.env.CALICHE_EMBEDDING_MODEL) {
    return process.env.CALICHE_EMBEDDING_MODEL;
  }
  if (process.env.TEST_EMBEDDING_MODEL) {
    return process.env.TEST_EMBEDDING_MODEL;
  }
  const provider = String(agent.provider);
  if (provider === "openrouter") {
    return "openai/text-embedding-3-small";
  }
  if (provider === "openai") {
    return "text-embedding-3-small";
  }
  return agent.model;
}

/** Thrown by RagService constructors when the native binding lacks `tools-rag`. */
export const RAG_SERVICE_REQUIRES_FEATURE = "RAG Service requires 'tools-rag' feature";
/** Thrown by RagTool.new when the native binding lacks `tools-rag`. */
export const RAG_TOOL_REQUIRES_FEATURE = "RAG Tool requires 'tools-rag' feature";
/** Thrown by RagService.withBackend when given a non-empty backend config. */
export const RAG_CUSTOM_BACKEND_UNSUPPORTED =
  "RagService.withBackend does not accept custom backend configs in the JS bindings yet. " +
  "Use RagService.new(agent) for the default in-memory backend, or configure vector " +
  "backends on AgentBuilder (setVectorBackend* / vectorBackendFromEnv) and share RagTool.service().";

/**
 * RAG Service for document indexing and retrieval
 *
 * Provides programmatic access to RAG operations including document indexing,
 * semantic search, hybrid search, reranking, and graph-enhanced search.
 *
 * @example
 * ```typescript
 * const agent = await new AgentBuilder()
 *   .provider('openai')
 *   .model('gpt-4')
 *   .apiKey('sk-...')
 *   .build();
 *
 * const rag = await RagService.new(agent);
 *
 * // Index documents
 * await rag.indexDocument('Python is a programming language...', 'python_intro');
 *
 * // Query
 * const results = await rag.query('What is Python?', 5, 0.3);
 * console.log(results.results);
 * ```
 */
export class RagService {
  private readonly _service: NativeRagService;

  private constructor(service: NativeRagService) {
    this._service = service;
  }

  /**
   * True when the native binding was built with `tools-rag`.
   */
  static isAvailable(): boolean {
    return typeof native.RagService?.new === "function";
  }

  /**
   * Create a new RAG service with default in-memory backend
   *
   * @param agent - Agent with embedding provider configured
   * @returns Promise resolving to RagService instance
   */
  static async new(agent: Agent): Promise<RagService> {
    if (!RagService.isAvailable()) {
      throw new Error(RAG_SERVICE_REQUIRES_FEATURE);
    }
    const service = await native.RagService.new(String(agent.provider), embeddingModelFor(agent));
    return new RagService(service);
  }

  /**
   * Create a RAG service with a custom vector backend.
   *
   * JS bindings currently expose the default in-memory RagService constructor
   * only. Pass no config (or an empty object) to use that default. For custom
   * vector backends, configure the agent builder (`setVectorBackend*` /
   * `vectorBackendFromEnv`) and share the resulting RagTool/service instead.
   */
  static async withBackend(agent: Agent, backendConfig?: RagBackendConfig): Promise<RagService> {
    if (backendConfig && Object.keys(backendConfig).length > 0) {
      throw new Error(RAG_CUSTOM_BACKEND_UNSUPPORTED);
    }
    return RagService.new(agent);
  }

  /**
   * Create a RagService from a native service instance.
   * This is used internally to share backends between RagTool and RagService.
   *
   * @param service - Native RagService instance
   * @returns RagService instance
   * @internal
   */
  static fromNative(service: NativeRagService): RagService {
    return new RagService(service);
  }

  /**
   * Index a document by splitting it into chunks and generating embeddings
   *
   * @param content - Document content to index
   * @param docId - Unique identifier for the document
   * @returns Promise resolving to index result with chunk count and usage
   */
  async indexDocument(content: string, docId: string): Promise<RagIndexResult> {
    const raw = (await this._service.indexDocument(content, docId)) as RagIndexResult & {
      chunkCount?: number;
    };
    return {
      chunk_count: raw.chunk_count ?? raw.chunkCount ?? 0,
      usage: raw.usage,
    };
  }

  /**
   * Index multiple documents sequentially.
   */
  async indexDocuments(
    documents: ReadonlyArray<{ readonly content: string; readonly docId: string }>
  ): Promise<readonly RagIndexResult[]> {
    const results: RagIndexResult[] = [];
    for (const document of documents) {
      results.push(await this.indexDocument(document.content, document.docId));
    }
    return results;
  }

  /**
   * Query for relevant chunks using semantic search
   *
   * @param query - Query string
   * @param topK - Number of results to return
   * @param minSimilarity - Minimum similarity threshold (default: 0.3)
   * @returns Promise resolving to query results
   */
  async query(query: string, topK: number, minSimilarity?: number): Promise<RagQueryResult> {
    const raw = await this._service.query(query, topK, minSimilarity);
    return normalizeQueryResult(raw);
  }

  /**
   * Enhanced query with hybrid search, reranking, and filtering
   *
   * @param options - Query options
   * @returns Promise resolving to enhanced query results
   */
  async enhancedQuery(options: EnhancedQueryOptions): Promise<EnhancedQueryResult> {
    return await this._service.enhancedQuery(
      options.query,
      options.top_k,
      options.min_similarity,
      options.search_mode,
      options.dense_weight,
      options.sparse_weight,
      options.rerank,
      options.metadata_filter,
      options.max_context_tokens
    );
  }

  /**
   * Perform graph-enhanced search (requires graph backend)
   *
   * @param query - Query string
   * @param topK - Number of results to return
   * @param maxHops - Maximum graph traversal hops (default: 2)
   * @returns Promise resolving to graph search results
   */
  async graphSearch(query: string, topK: number, maxHops?: number): Promise<GraphSearchResult> {
    return await this._service.graphSearch(query, topK, maxHops);
  }

  /**
   * Get statistics about indexed documents
   *
   * @returns Promise resolving to statistics
   */
  async stats(): Promise<RagStats> {
    const raw = (await this._service.stats()) as RagStats & {
      embeddingCount?: number;
      textCount?: number;
    };
    return {
      embedding_count: raw.embedding_count ?? raw.embeddingCount ?? 0,
      text_count: raw.text_count ?? raw.textCount ?? 0,
    };
  }

  /**
   * Clear all indexed documents
   *
   * @returns Promise that resolves when clear is complete
   */
  async clear(): Promise<void> {
    return await this._service.clear();
  }
}

/**
 * RAG Tool for use with agents.
 *
 * This tool wraps RagService and provides a Tool interface for use in agents.
 * Use this to create a shared RAG tool that can be used by multiple agents,
 * ensuring they all share the same vector store backend.
 *
 * @example
 * ```typescript
 * import { AgentBuilder, ProviderType, RagTool } from 'caliche';
 *
 * const agent = await new AgentBuilder()
 *   .provider(ProviderType.OpenAI)
 *   .model('gpt-4o')
 *   .apiKey('sk-...')
 *   .build();
 *
 * const ragTool = await RagTool.new(agent);
 *
 * // Get the RagService to share the backend
 * const ragService = ragTool.service();
 *
 * // Use the shared tool in another agent
 * const agent2 = await new AgentBuilder()
 *   .provider(ProviderType.OpenAI)
 *   .model('gpt-4o')
 *   .apiKey('sk-...')
 *   .sharedRagTool(ragTool)
 *   .build();
 * ```
 */
export class RagTool {
  readonly _tool: NativeRagTool;

  private constructor(tool: NativeRagTool) {
    this._tool = tool;
  }

  /**
   * Check if RAG tool feature is available
   *
   * @returns true if RAG tool feature is available, false otherwise
   */
  static isAvailable(): boolean {
    return typeof native.RagTool?.new === "function";
  }

  /**
   * Create a new RAG tool with default in-memory backend
   *
   * @param agent - Agent with embedding provider configured
   * @returns Promise resolving to RagTool instance
   */
  static async new(agent: Agent): Promise<RagTool> {
    if (!RagTool.isAvailable()) {
      throw new Error(RAG_TOOL_REQUIRES_FEATURE);
    }
    const tool = await native.RagTool.new(String(agent.provider), embeddingModelFor(agent));
    return new RagTool(tool);
  }

  /**
   * Get the underlying RAG service.
   *
   * This allows sharing the same backend between RagTool and RagService instances.
   *
   * @returns A RagService instance that shares the same backend as this RagTool
   */
  service(): RagService {
    return RagService.fromNative(this._tool.service());
  }
}

/**
 * Built-in Tool wrapper
 *
 * Provides access to Caliche's built-in tools like Calculator, Echo, ReadFile, etc.
 *
 * @example
 * ```typescript
 * const calculator = BuiltinTool.calculator();
 * const fs = BuiltinTool.fs();
 * ```
 */
export class BuiltinTool {
  readonly _tool: NativeBuiltinTool & {
    name(): string;
    description(): string;
    parameters(): unknown;
    execute(arguments_: Record<string, unknown>): {
      success: boolean;
      content: string;
      error?: string;
      typed_payload?: unknown;
    };
  };

  private constructor(tool: BuiltinTool["_tool"]) {
    this._tool = tool;
  }

  /** @internal native handle for registry registration */
  get inner(): BuiltinTool["_tool"] {
    return this._tool;
  }

  /**
   * Create a CalculatorTool instance
   */
  static calculator(): BuiltinTool {
    if (!native.BuiltinTool?.calculator) {
      throw new Error("Built-in tools not available");
    }
    return new BuiltinTool(native.BuiltinTool.calculator() as BuiltinTool["_tool"]);
  }

  /**
   * Create an FsTool instance.
   *
   * Pass `MutationStyle.applyPatch()` to disable write/edit on `fs`.
   */
  static fs(mutationStyle?: { inner: unknown } | null): BuiltinTool {
    if (!native.BuiltinTool?.fs) {
      throw new Error("Built-in tools not available");
    }
    return new BuiltinTool(
      native.BuiltinTool.fs(mutationStyle?.inner ?? null) as BuiltinTool["_tool"]
    );
  }

  /**
   * Create an ApplyPatchTool instance (Codex-style freeform patch).
   */
  static applyPatch(): BuiltinTool {
    if (!native.BuiltinTool?.applyPatch) {
      throw new Error("Built-in tools not available");
    }
    return new BuiltinTool(native.BuiltinTool.applyPatch() as BuiltinTool["_tool"]);
  }

  /**
   * Create an HttpTool instance (unified HTTP tool supporting request, fetch_html, and api_call operations)
   */
  static http(): BuiltinTool {
    if (!native.BuiltinTool?.http) {
      throw new Error("Built-in tools not available");
    }
    return new BuiltinTool(native.BuiltinTool.http() as BuiltinTool["_tool"]);
  }

  /**
   * Create a CodeExecutionTool instance
   */
  static codeExecution(): BuiltinTool {
    if (!native.BuiltinTool?.codeExecution) {
      throw new Error("Built-in tools not available");
    }
    return new BuiltinTool(native.BuiltinTool.codeExecution() as BuiltinTool["_tool"]);
  }

  /**
   * Get the tool name
   */
  name(): string {
    return this._tool.name();
  }

  /**
   * Get the tool description
   */
  description(): string {
    return this._tool.description();
  }

  /**
   * Get the tool parameters schema
   */
  parameters(): any {
    return this._tool.parameters();
  }

  /**
   * Execute the tool directly (shared Tokio runtime).
   */
  execute(arguments_: Record<string, unknown>): {
    success: boolean;
    content: string;
    error?: string;
    typed_payload?: unknown;
  } {
    return this._tool.execute(arguments_);
  }

  /**
   * Convert to Tool interface for use with AgentBuilder
   */
  toTool(): Tool {
    return {
      name: this.name(),
      description: this.description(),
      parameters: this.parameters(),
      execute: async (args: ToolArguments) => {
        return this.execute(args as Record<string, unknown>);
      },
    };
  }
}
