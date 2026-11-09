// Effect 4 is ESM-only. This package compiles as CJS, so the import resolution mode is required to see its types.
import type { Effect as EffectType } from "effect/Effect" with { "resolution-mode": "import" };
import type { HookContext, HookMatcher, LifecycleHook, PermissionMode } from "../hooks";
import type { InProcessMcpServer } from "../mcp";
import type { MutationStyle } from "../mutation-style";
import { napiToolPayload } from "../napi-callback";
import {
  type DurableStreamBackendInstance,
  type NativeAgentBuilder,
  native,
} from "../native/module";
import {
  type AgentRuntimeConfig,
  AgentRuntimeConfigError,
  type AgentRuntimeCredentials,
  type CompletionResponse,
  type MessageJs,
  type Provider,
  type Tool,
  type Toolkit,
  type ToolRequirementPolicy,
} from "../streaming/events";
import type { InspectionHookAdapter } from "../tool-inspection";
import type { JsonRecord, JsonValue, MemorySnapshotMessages, ToolArguments } from "../types";
import { Agent } from "./agent";
import { agentBuilderMemoryMethods } from "./builder-memory";
import { agentBuilderVectorBackendMethods } from "./builder-vector-backend";

type EffectModule = typeof import("effect/Effect", { with: { "resolution-mode": "import" } });

interface MemorySnapshot {
  readonly messages: MemorySnapshotMessages;
}

interface MemoryPersistence {
  save_snapshot(conversationId: string, snapshot: MemorySnapshot): Promise<void>;
  load_snapshot(conversationId: string): Promise<MemorySnapshot | undefined>;
  clear_snapshot(conversationId: string): Promise<void>;
}

interface ExternalProviderConfig {
  readonly name: string;
  readonly modelName: string;
  readonly completeFn: (request: JsonRecord) => Promise<string>;
  readonly streamFn: (request: JsonRecord) => AsyncGenerator<string, void, unknown>;
  readonly supportsNativeTools: boolean;
  readonly supportsStreaming: boolean;
}

interface ExternalEmbeddingProviderConfig {
  readonly name: string;
  readonly embedFn: (request: JsonRecord) => Promise<string>;
}

interface MemoryPersistenceConfig {
  readonly persistence: MemoryPersistence;
  readonly conversationId: string;
  readonly autosave: boolean;
}

function applyRuntimeTimeouts(config: AgentRuntimeConfig, issues: string[]): void {
  const timeout = config.timeouts?.streamEventTimeoutSecs;
  if (timeout === undefined) {
    return;
  }
  if (timeout <= 0) {
    issues.push("timeouts.streamEventTimeoutSecs must be > 0");
    return;
  }
  process.env.CALICHE_STREAM_EVENT_TIMEOUT_SECS = String(timeout);
}

function providerEnvApiKey(provider: Provider | undefined): string | undefined {
  if (provider === "openrouter") {
    return process.env.OPENROUTER_API_KEY;
  }
  if (provider === "perplexity") {
    return process.env.PERPLEXITY_API_KEY;
  }
  if (provider === "anthropic") {
    return process.env.ANTHROPIC_API_KEY;
  }
  return process.env.OPENAI_API_KEY;
}

function applyRuntimeCredentials(
  builder: AgentBuilder,
  config: AgentRuntimeConfig,
  currentProvider: Provider | null,
  issues: string[]
): void {
  const runtimeCreds: AgentRuntimeCredentials | undefined = config.credentials;
  const scoped = runtimeCreds?.scopedApiKeys ?? {};
  for (const [target, key] of Object.entries(scoped)) {
    if (!key || key.trim().length === 0) {
      issues.push(`credentials.scopedApiKeys.${target} must be non-empty`);
      continue;
    }
    builder.apiKeyFor(target, key);
  }

  const provider = config.provider ?? currentProvider ?? undefined;
  const resolvedDefaultKey = runtimeCreds?.defaultApiKey ?? providerEnvApiKey(provider);
  if (resolvedDefaultKey && resolvedDefaultKey.trim().length > 0) {
    builder.apiKey(resolvedDefaultKey);
  }

  if (runtimeCreds?.aws) {
    builder.awsCredentials(
      runtimeCreds.aws.accessKey,
      runtimeCreds.aws.secretKey,
      runtimeCreds.aws.region
    );
  }

  if (runtimeCreds?.baseUrl) {
    builder.baseUrl(runtimeCreds.baseUrl);
  }
}

// ============================================================================
// AgentBuilder
// ============================================================================

/**
 * Builder for configuring and creating agents
 *
 * @example
 * ```typescript
 * const agent = await new AgentBuilder()
 *   .provider("openai")
 *   .model("gpt-4")
 *   .enableTool("calculator")
 *   .build();
 * ```
 */
export class AgentBuilder {
  private readonly _builder: NativeAgentBuilder;
  private _provider: Provider | null = null;
  private _seedMessages: MessageJs[] = [];
  private _pseudoToolProtocolEnabled = false;
  private readonly _scopedApiKeys: Record<string, string> = {};
  private readonly _toolRequirementPolicies: ToolRequirementPolicy[] = [];
  private readonly _customTools: Tool[] = [];
  private _externalProvider: ExternalProviderConfig | null = null;
  private _externalEmbeddingProvider: ExternalEmbeddingProviderConfig | null = null;
  private _memoryPersistence: MemoryPersistenceConfig | null = null;

  /**
   * Create a new agent builder
   */
  constructor() {
    this._builder = new native.AgentBuilder();
  }

  /**
   * Set the LLM provider
   *
   * @param provider - Provider name (e.g., "openai", "anthropic", "bedrock")
   * @returns This builder for method chaining
   */
  provider(provider: Provider): this {
    this._provider = provider;
    this._builder.setProvider(provider);
    return this;
  }

  /**
   * Set the model name
   *
   * @param model - Model identifier (e.g., "gpt-4", "claude-3-opus")
   * @returns This builder for method chaining
   */
  model(model: string): this {
    this._builder.setModel(model);
    return this;
  }

  /**
   * Set the system prompt
   *
   * @param prompt - System prompt text
   * @returns This builder for method chaining
   */
  systemPrompt(prompt: string): this {
    this._builder.setSystemPrompt(prompt);
    return this;
  }

  /**
   * Control whether runtime context is injected into the system prompt.
   * Benchmarks and thin wrappers typically disable this to match raw SDK shapes.
   */
  includeContextInSystem(value: boolean): this {
    this._builder.setIncludeContextInSystem(value);
    return this;
  }

  /**
   * Seed structured conversation history for this agent instance.
   */
  messages(history: readonly MessageJs[]): this {
    this._seedMessages = [...history];
    return this;
  }

  /**
   * Enable parsing of plain-text pseudo tool protocol into structured events.
   */
  parsePseudoToolProtocol(enabled: boolean = true): this {
    this._pseudoToolProtocolEnabled = enabled;
    return this;
  }

  /**
   * Enable JSON object mode - response will be valid JSON
   *
   * @returns This builder for method chaining
   *
   * @example
   * ```typescript
   * const agent = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4o")
   *   .jsonMode()
   *   .build();
   * ```
   */
  jsonMode(): this {
    this._builder.jsonMode();
    return this;
  }

  /**
   * Enable JSON schema mode with validation
   *
   * @param name - Schema name for the LLM
   * @param schema - JSON Schema object
   * @returns This builder for method chaining
   *
   * @example
   * ```typescript
   * const schema = {
   *   type: "object",
   *   properties: {
   *     name: { type: "string" },
   *     age: { type: "integer" }
   *   },
   *   required: ["name", "age"]
   * };
   *
   * const agent = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4o")
   *   .jsonSchema("Person", schema)
   *   .build();
   * ```
   */
  jsonSchema(name: string, schema: Record<string, unknown>): this {
    this._builder.jsonSchema(name, JSON.stringify(schema));
    return this;
  }

  /**
   * Enable a single tool by name
   *
   * @param tool - Tool name (e.g., "calculator", "web_search")
   * @returns This builder for method chaining
   */
  enableTool(tool: string): this {
    this._builder.enableTool(tool);
    return this;
  }

  /**
   * Enable multiple tools by name
   *
   * @param tools - Array of tool names
   * @returns This builder for method chaining
   */
  enableTools(tools: readonly string[]): this {
    this._builder.enableTools(tools);
    return this;
  }

  /**
   * Enable all default tools
   *
   * @returns This builder for method chaining
   */
  enableAllDefaultTools(): this {
    this._builder.enableAllDefaultTools();
    return this;
  }

  /**
   * Enable the broader default tool set (grep/git/etc. beyond coding defaults).
   */
  enableExtendedDefaultTools(): this {
    if (typeof this._builder.enableExtendedDefaultTools === "function") {
      this._builder.enableExtendedDefaultTools();
    }
    return this;
  }

  /**
   * Select file mutation style (`EditWrite` XOR `ApplyPatch`).
   */
  mutationStyle(style: MutationStyle): this {
    if (typeof this._builder.mutationStyle === "function") {
      this._builder.mutationStyle(style.inner);
    }
    return this;
  }

  /**
   * Enable Goose-style tool inspection (path security + permission mode).
   */
  enableToolInspection(workspaceRoot?: string | null): this {
    if (typeof this._builder.enableToolInspection === "function") {
      this._builder.enableToolInspection(workspaceRoot ?? null);
    }
    return this;
  }

  /**
   * Add a native inspection hook adapter.
   */
  addInspectionHook(matcher: HookMatcher, hook: InspectionHookAdapter): this {
    if (typeof this._builder.addInspectionHook === "function") {
      this._builder.addInspectionHook(matcher.inner, hook.inner);
    }
    return this;
  }

  /**
   * Add a custom tool
   *
   * @param tool - Tool object implementing the Tool interface
   * @returns This builder for method chaining
   */
  addTool(tool: Tool): this {
    // Store custom tools in JavaScript and pass them during build
    this._customTools.push(tool);
    return this;
  }

  /**
   * Register a toolkit (collection of related tools)
   *
   * @param toolkit - Toolkit object implementing the Toolkit interface
   * @returns This builder for method chaining
   */
  registerToolkit(toolkit: Toolkit): this {
    this._builder.registerToolkit(toolkit);
    return this;
  }

  /**
   * Set temperature for LLM responses
   *
   * @param temp - Temperature value (0.0 to 2.0)
   * @returns This builder for method chaining
   */
  temperature(temp: number): this {
    this._builder.setTemperature(temp);
    return this;
  }

  /**
   * Set maximum tokens for responses
   *
   * @param tokens - Maximum token count
   * @returns This builder for method chaining
   */
  maxTokens(tokens: number): this {
    this._builder.setMaxTokens(tokens);
    return this;
  }

  /**
   * Set top-p (nucleus sampling) parameter
   *
   * @param topP - Top-p value (0.0 to 1.0)
   * @returns This builder for method chaining
   */
  topP(topP: number): this {
    this._builder.setTopP(topP);
    return this;
  }

  /**
   * Set a LangChain provider using a LangChainLLMProvider adapter
   *
   * @param provider - LangChainLLMProvider instance wrapping a LangChain chat model
   * @returns This builder for method chaining
   *
   * @example
   * ```typescript
   * import { ChatOpenAI } from "@langchain/openai";
   * import { LangChainLLMProvider } from "@caliche/adapters-langchain";
   *
   * const llm = new ChatOpenAI({ model: "gpt-4o" });
   * const provider = new LangChainLLMProvider(llm);
   *
   * const agent = await new AgentBuilder()
   *   .langchainProvider(provider)
   *   .enableTool("calculator")
   *   .build();
   * ```
   */
  /**
   * Set a LangChain provider using a LangChainLLMProvider adapter
   *
   * @param provider - LangChainLLMProvider instance wrapping a LangChain chat model
   * @returns This builder for method chaining
   *
   * @example
   * ```typescript
   * import { ChatOpenAI } from "@langchain/openai";
   * import { LangChainLLMProvider } from "@caliche/adapters-langchain";
   *
   * const llm = new ChatOpenAI({ model: "gpt-4o" });
   * const provider = new LangChainLLMProvider(llm);
   *
   * const agent = await new AgentBuilder()
   *   .langchainProvider(provider)
   *   .enableTool("calculator")
   *   .build();
   * ```
   */
  langchainProvider(provider: {
    complete(request: JsonRecord | any): Promise<CompletionResponse | any>;
    stream(request: JsonRecord | any): AsyncGenerator<any, void, unknown>;
    modelName?: string;
    supportsNativeTools?: boolean;
    supportsStreaming?: boolean;
  }): this {
    // Create wrapper functions that can be called from Rust
    // These will serialize/deserialize JSON for communication
    const completeFn = async (request: JsonRecord) => {
      const requestObj = typeof request === "string" ? JSON.parse(request) : request;
      const response = await provider.complete(requestObj);
      // MUST return JSON string for Rust side sync bridge
      return JSON.stringify(response);
    };

    const streamFn = async function* (request: JsonRecord): AsyncGenerator<string, void, unknown> {
      const requestObj = typeof request === "string" ? JSON.parse(request) : request;
      const events = [];
      for await (const event of provider.stream(requestObj)) {
        events.push(event);
      }
      yield JSON.stringify(events);
    };

    const name = provider.modelName || "langchain-llm";
    const modelName = provider.modelName || "langchain-llm";
    const supportsNativeTools = provider.supportsNativeTools !== false;
    const supportsStreaming = provider.supportsStreaming !== false;

    // Bridged into the native builder in `build()`.
    this._externalProvider = {
      name,
      modelName,
      completeFn,
      streamFn,
      supportsNativeTools,
      supportsStreaming,
    };
    return this;
  }

  /**
   * Set an AI SDK-compatible provider using `caliche(...)` or CalicheLanguageModel.
   *
   * @param provider - Provider instance exposing complete/stream methods
   * @returns This builder for method chaining
   */
  aiSdkProvider(provider: {
    complete(request: JsonRecord | any): Promise<CompletionResponse | any>;
    stream(request: JsonRecord | any): AsyncGenerator<any, void, unknown>;
    modelId?: string;
    supportsNativeTools?: boolean;
    supportsStreaming?: boolean;
  }): this {
    const completeFn = async (request: JsonRecord) => {
      const requestObj = typeof request === "string" ? JSON.parse(request) : request;
      const response = await provider.complete(requestObj);
      return JSON.stringify(response);
    };

    const streamFn = async function* (request: JsonRecord): AsyncGenerator<string, void, unknown> {
      const requestObj = typeof request === "string" ? JSON.parse(request) : request;
      const events = [];
      for await (const event of provider.stream(requestObj)) {
        events.push(event);
      }
      yield JSON.stringify(events);
    };

    const name = provider.modelId || "caliche-ai-sdk";
    const modelName = provider.modelId || "caliche-ai-sdk";
    const supportsNativeTools = provider.supportsNativeTools !== false;
    const supportsStreaming = provider.supportsStreaming !== false;

    this._externalProvider = {
      name,
      modelName,
      completeFn,
      streamFn,
      supportsNativeTools,
      supportsStreaming,
    };
    return this;
  }

  /**
   * Set a LangChain embedding provider using a LangChainEmbeddingProvider adapter
   *
   * @param provider - LangChainEmbeddingProvider instance wrapping a LangChain embeddings model
   * @returns This builder for method chaining
   *
   * @example
   * ```typescript
   * import { OpenAIEmbeddings } from "@langchain/openai";
   * import { LangChainEmbeddingProvider } from "@caliche/adapters-langchain";
   *
   * const embeddings = new OpenAIEmbeddings();
   * const provider = new LangChainEmbeddingProvider(embeddings);
   *
   * const agent = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4o")
   *   .langchainEmbeddings(provider)
   *   .vectorMemory(...)
   *   .build();
   * ```
   */
  langchainEmbeddings(provider: {
    embed(request: JsonRecord | { texts: string[] }): Promise<JsonRecord | any>;
    modelName?: string;
  }): this {
    const embedFn = async (request: JsonRecord) => {
      const requestObj = typeof request === "string" ? JSON.parse(request) : request;
      const response = await provider.embed(requestObj);
      // MUST return JSON string for Rust side sync bridge
      return typeof response === "string" ? response : JSON.stringify(response ?? {});
    };

    const name = provider.modelName || "langchain-embeddings";

    this._externalEmbeddingProvider = {
      name,
      embedFn,
    };
    return this;
  }

  /**
   * Set API key for authentication
   *
   * @param key - API key string
   * @returns This builder for method chaining
   */
  apiKey(key: string): this {
    this._scopedApiKeys.default = key;
    this._builder.setApiKey(key);
    return this;
  }

  /**
   * Set a scoped API key for a provider/tool target.
   */
  apiKeyFor(target: string, key: string): this {
    const normalizedTarget = target.trim().toLowerCase();
    this._scopedApiKeys[normalizedTarget] = key;
    if (typeof this._builder.setApiKeyFor === "function") {
      this._builder.setApiKeyFor(normalizedTarget, key);
    }
    return this;
  }

  /**
   * Set AWS credentials for Bedrock provider
   *
   * @param accessKey - AWS access key
   * @param secretKey - AWS secret key
   * @param region - AWS region
   * @returns This builder for method chaining
   */
  awsCredentials(accessKey: string, secretKey: string, region: string): this {
    this._builder.setAwsCredentials(accessKey, secretKey, region);
    return this;
  }

  /**
   * Set base URL for LMStudio provider
   *
   * @param url - Base URL (e.g., "http://localhost:1234/v1")
   * @returns This builder for method chaining
   */
  baseUrl(url: string): this {
    this._builder.setBaseUrl(url);
    return this;
  }

  runtimeConfig(config: AgentRuntimeConfig): this {
    const issues: string[] = [];
    if (config.provider) {
      this.provider(config.provider);
    }
    if (config.model) {
      this.model(config.model);
    }
    applyRuntimeTimeouts(config, issues);
    applyRuntimeCredentials(this, config, this._provider, issues);
    if (issues.length > 0) {
      throw new AgentRuntimeConfigError(issues);
    }
    return this;
  }

  /**
   * Set the maximum number of tool call iterations (default: 50)
   *
   * By default, agents have a limit of 50 tool call iterations. Use this method
   * to set a different limit, or use Number.MAX_SAFE_INTEGER for unlimited iterations.
   * Each iteration consists of one LLM call and its tool executions.
   *
   * @param maxIterations - Maximum number of iterations (use Number.MAX_SAFE_INTEGER for unlimited)
   * @returns This builder for method chaining
   *
   * @example
   * ```typescript
   * // Default limit of 50 iterations
   * const agent1 = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .build();
   *
   * // Limited to 20 iterations
   * const agent2 = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .maxIterations(20) // Limit to 20 iterations
   *   .build();
   *
   * // Unlimited iterations
   * const agent3 = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .maxIterations(Number.MAX_SAFE_INTEGER) // Unlimited iterations
   *   .build();
   * ```
   */
  maxIterations(maxIterations: number): this {
    this._builder.setMaxIterations(maxIterations);
    return this;
  }

  /**
   * Control how the LLM uses tools for agent chat turns.
   *
   * - `"auto"`: model decides (default)
   * - `"none"`: disable tool usage
   * - `"required"`: force at least one tool call
   * - specific tool name: force that tool
   */
  toolChoice(choice: string): this {
    if (typeof this._builder.setToolChoice === "function") {
      this._builder.setToolChoice(choice);
    }
    return this;
  }

  /**
   * Control whether `build()` auto-attaches MCP servers discovered from host
   * config files (for example `~/.cursor/mcp.json`). Defaults to disabled.
   */
  autoDiscoverMcp(enabled: boolean): this {
    if (typeof this._builder.setAutoDiscoverMcp === "function") {
      this._builder.setAutoDiscoverMcp(enabled);
    }
    return this;
  }

  /**
   * Control whether `build()` indexes on-disk skills (`SKILL.md`) for progressive
   * disclosure in the system prompt. Defaults to disabled so SDK agents stay
   * token-lean unless the caller opts in.
   */
  autoDiscoverSkills(enabled: boolean): this {
    if (typeof this._builder.setAutoDiscoverSkills === "function") {
      this._builder.setAutoDiscoverSkills(enabled);
    }
    return this;
  }

  requireToolWhen(
    predicate: ToolRequirementPolicy["predicate"],
    toolName: string,
    options?: {
      readonly name?: string;
      readonly violationMessage?: string;
    }
  ): this {
    this._toolRequirementPolicies.push({
      name: options?.name,
      toolName,
      predicate,
      violationMessage: options?.violationMessage,
    });
    return this;
  }

  /**
   * Set the permission mode for tool execution
   *
   * @param mode - The permission mode to use
   * @returns The builder instance for method chaining
   *
   * @example
   * ```typescript
   * // Auto-accept file edits
   * const agent = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .permissionMode(PermissionMode.acceptEdits())
   *   .build();
   *
   * // Deny all tool calls
   * const agent2 = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .permissionMode(PermissionMode.denyAll())
   *   .build();
   * ```
   */
  permissionMode(mode: PermissionMode): this {
    // Pass the inner native object
    this._builder.setPermissionMode(mode.inner);
    return this;
  }

  /**
   * Set the list of allowed tools (if not set, all tools are allowed)
   *
   * @param tools - Array of tool names that are allowed
   * @returns The builder instance for method chaining
   *
   * @example
   * ```typescript
   * const agent = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .enableAllDefaultTools()
   *   .allowedTools(["calculator", "fs"])
   *   .build();
   * ```
   */
  allowedTools(tools: readonly string[]): this {
    this._builder.setAllowedTools([...tools]);
    return this;
  }

  /**
   * Add a lifecycle hook with a matcher
   *
   * @param matcher - The hook matcher (tool name or "all")
   * @param hook - The lifecycle hook implementation
   * @returns The builder instance for method chaining
   *
   * @example
   * ```typescript
   * const securityHook: LifecycleHook = {
   *   onPreToolUse(ctx) {
   *     if (ctx.tool_name === "Bash" && ctx.tool_input.command?.includes("rm -rf")) {
   *       return { permission_decision: { decision: "deny", reason: "Dangerous command" } };
   *     }
   *     return { permission_decision: { decision: "allow" } };
   *   }
   * };
   *
   * const agent = await new AgentBuilder()
   *   .provider("openai")
   *   .model("gpt-4")
   *   .hook(HookMatcher.tool("Bash"), securityHook)
   *   .build();
   * ```
   */
  hook(matcher: HookMatcher, hook: LifecycleHook): this {
    // Pass the inner native object
    const nativeMatcher = matcher.inner;

    // Create wrapper functions that handle both sync and async hooks
    // The Rust side passes a JSON Value, so we need to parse it as HookContext
    // NAPI-RS will automatically convert these JS functions to ThreadsafeFunction
    const preFn = (ctxValue: any): string | Promise<string> => {
      // Parse the JSON Value as HookContext
      const ctx: HookContext = typeof ctxValue === "string" ? JSON.parse(ctxValue) : ctxValue;

      // Call the hook method
      const result = hook.onPreToolUse(ctx);

      // If result is a Promise, await it and stringify
      if (result instanceof Promise) {
        return result.then((r) => JSON.stringify(r));
      }

      // Sync result - stringify directly
      return JSON.stringify(result);
    };

    let postFn: ((combinedValue: any) => string | Promise<string>) | undefined;
    if (hook.onPostToolUse) {
      postFn = (combinedValue: any): string | Promise<string> => {
        // Parse the combined JSON Value (contains context and result)
        const parsed =
          typeof combinedValue === "string" ? JSON.parse(combinedValue) : combinedValue;
        const ctx: HookContext = parsed.context;
        const result = parsed.result;

        // Call the hook method
        const hookResult = hook.onPostToolUse?.(ctx, result);

        // If result is a Promise, await it and stringify
        if (hookResult instanceof Promise) {
          return hookResult.then((r) => JSON.stringify(r));
        }

        // Sync result - stringify directly
        return JSON.stringify(hookResult);
      };
    }

    // Call the native addHook method - NAPI-RS will auto-convert JS functions to ThreadsafeFunction
    // The wrapper functions handle Promise unwrapping, so the Rust side always receives a string
    this._builder.addHook(nativeMatcher, preFn, postFn);

    return this;
  }

  mcpServer(name: string, server: InProcessMcpServer): this {
    this._builder.addMcpServer(name, server.native);
    return this;
  }

  durableStreamBackend(backend: DurableStreamBackendInstance): this {
    if (typeof this._builder.setDurableStreamBackend === "function") {
      this._builder.setDurableStreamBackend(backend);
    }
    return this;
  }

  durableStreamSqlite(dbPath: string): this {
    if (typeof this._builder.setDurableStreamSqlite === "function") {
      this._builder.setDurableStreamSqlite(dbPath);
    }
    return this;
  }

  durableStreamPostgres(connectionString: string): this {
    if (typeof this._builder.setDurableStreamPostgres === "function") {
      this._builder.setDurableStreamPostgres(connectionString);
    }
    return this;
  }

  durableStreamS3(bucket: string, prefix: string, region?: string): this {
    if (typeof this._builder.setDurableStreamS3 === "function") {
      this._builder.setDurableStreamS3(bucket, prefix, region);
    }
    return this;
  }

  azureOpenAIConfig(
    resourceName: string,
    deploymentId: string,
    apiKey?: string,
    apiVersion?: string
  ): this {
    if (typeof this._builder.setAzureOpenAIConfig === "function") {
      this._builder.setAzureOpenAIConfig(
        resourceName,
        deploymentId,
        apiKey ?? "",
        apiVersion ?? ""
      );
    } else if (native.setAzureOpenAIConfig) {
      native.setAzureOpenAIConfig(this._builder, resourceName, deploymentId, apiKey, apiVersion);
    }
    return this;
  }

  async build(): Promise<Agent> {
    const externalProvider = this._externalProvider;
    if (externalProvider) {
      this._builder.setExternalProvider(
        externalProvider.name,
        externalProvider.modelName,
        externalProvider.completeFn,
        externalProvider.streamFn,
        externalProvider.supportsNativeTools,
        externalProvider.supportsStreaming
      );
    }

    const externalEmbeddingProvider = this._externalEmbeddingProvider;
    if (externalEmbeddingProvider) {
      this._builder.setExternalEmbeddingProvider(
        externalEmbeddingProvider.name,
        externalEmbeddingProvider.embedFn
      );
    }

    for (const tool of this._customTools) {
      // NAPI ThreadsafeFunction uses error-first callbacks: (err, payload) => ...
      const executeFn = async (...callbackArgs: unknown[]): Promise<string> => {
        try {
          const args = napiToolPayload(callbackArgs) as ToolArguments;
          const result = await tool.execute(args);
          if (typeof result === "object" && result !== null) {
            if ("success" in result && "content" in result) {
              return JSON.stringify(result);
            }
            return JSON.stringify({ success: true, content: JSON.stringify(result) });
          }
          return JSON.stringify({ success: true, content: String(result) });
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : String(error);
          return JSON.stringify({ success: false, content: "", error: message });
        }
      };

      native.builderAddTool(
        this._builder,
        tool.name,
        tool.description,
        tool.parameters as unknown as JsonValue,
        executeFn
      );
    }

    const memoryPersistence = this._memoryPersistence;
    if (memoryPersistence) {
      const { persistence, conversationId, autosave } = memoryPersistence;
      // NAPI ThreadsafeFunction uses an error-first callback shape: (err, arg) => ...
      const saveFn = async (err: Error | null, payload: JsonValue): Promise<string> => {
        if (err) {
          throw err;
        }
        const [convId, snapshot] = payload as [string, MemorySnapshot];
        await persistence.save_snapshot(convId, snapshot);
        return "";
      };
      const loadFn = async (err: Error | null, payload: JsonValue): Promise<string> => {
        if (err) {
          throw err;
        }
        const convId = (Array.isArray(payload) ? payload[0] : payload) as string;
        const snapshot = await persistence.load_snapshot(convId);
        return snapshot === undefined ? "null" : JSON.stringify(snapshot);
      };
      const clearFn = async (err: Error | null, payload: JsonValue): Promise<string> => {
        if (err) {
          throw err;
        }
        const convId = (Array.isArray(payload) ? payload[0] : payload) as string;
        await persistence.clear_snapshot(convId);
        return "";
      };
      this._builder.persistMemoryWith(saveFn, loadFn, clearFn, conversationId, autosave);
    }

    const providerApiKey = this._provider ? this._scopedApiKeys[this._provider] : undefined;
    if (providerApiKey) {
      this._builder.setApiKey(providerApiKey);
    }

    const nativeAgent = await this._builder.build();
    return new Agent(
      nativeAgent,
      this._seedMessages,
      this._pseudoToolProtocolEnabled,
      this._toolRequirementPolicies
    );
  }

  buildEffect(): EffectType<Agent, Error, never> {
    let effectModule: EffectModule;
    try {
      effectModule = require("effect/Effect") as EffectModule;
    } catch (error) {
      const message =
        "buildEffect() requires the 'effect' package. Install it with `bun add effect` or `npm install effect`.";
      throw new Error(
        error instanceof Error ? `${message} Original error: ${error.message}` : message
      );
    }

    return effectModule.tryPromise({
      try: () => this.build(),
      catch: (error) => (error instanceof Error ? error : new Error(String(error))),
    });
  }
}

Object.assign(AgentBuilder.prototype, agentBuilderMemoryMethods, agentBuilderVectorBackendMethods);
