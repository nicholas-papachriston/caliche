import type {
  JSONValue,
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider";
import { AgentBuilder } from "../agent/builder";
import type { PermissionMode } from "../hooks";
import { LlmClientBuilder } from "../llm/client";
import type {
  CompletionResponse,
  CompletionToolCall,
  MessageJs,
  Provider,
  StreamEvent,
  TokenUsage,
  Tool,
  ToolSchema,
} from "../streaming/events";
import type { JsonRecord, ToolExecutionResult } from "../types";
import {
  CalicheMcpBridge,
  type CalicheMcpServerBinding,
  type McpPromptDescriptor,
  type McpResourceDescriptor,
} from "./ai-sdk-mcp";

export interface CalicheModelOptions {
  readonly provider?: Provider | string;
  readonly model?: string;
  readonly apiKey?: string;
  readonly baseUrl?: string;
  readonly systemPrompt?: string;
  readonly temperature?: number;
  readonly maxTokens?: number;
  readonly maxIterations?: number;
  readonly permissionMode?: PermissionMode;
  /**
   * When `"agent"`, run through Caliche's agent loop (tools/MCP execute in-process).
   * Default `"model"` uses LlmClient and lets the AI SDK own client-side tool execution.
   */
  readonly mode?: "model" | "agent";
  readonly tools?: readonly Tool[];
  readonly mcpServers?: readonly CalicheMcpServerBinding[];
  readonly onNeedsApproval?: (ctx: {
    readonly toolName: string;
    readonly args: Record<string, unknown>;
  }) => boolean | Promise<boolean>;
}

export interface CalicheMcpSnapshot {
  readonly resources: readonly McpResourceDescriptor[];
  readonly prompts: readonly McpPromptDescriptor[];
  readonly tools: ReadonlyArray<{
    readonly name: string;
    readonly description: string;
    readonly parameters: unknown;
  }>;
}

type UnifiedFinishReason = LanguageModelV3FinishReason["unified"];

function parseModelId(modelId: string): { provider: Provider; model: string } {
  const slash = modelId.indexOf("/");
  if (slash <= 0) {
    return { provider: "openai", model: modelId };
  }
  return {
    provider: modelId.slice(0, slash) as Provider,
    model: modelId.slice(slash + 1),
  };
}

function emptyUsage(): LanguageModelV3Usage {
  return {
    inputTokens: {
      total: undefined,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: undefined,
      text: undefined,
      reasoning: undefined,
    },
  };
}

function toV3Usage(usage?: TokenUsage | null): LanguageModelV3Usage {
  if (!usage) {
    return emptyUsage();
  }
  return {
    inputTokens: {
      total: usage.prompt_tokens,
      noCache: undefined,
      cacheRead: undefined,
      cacheWrite: undefined,
    },
    outputTokens: {
      total: usage.completion_tokens,
      text: usage.completion_tokens,
      reasoning: undefined,
    },
    raw: {
      prompt_tokens: usage.prompt_tokens ?? null,
      completion_tokens: usage.completion_tokens ?? null,
      total_tokens: usage.total_tokens ?? null,
    },
  };
}

function toFinishReason(
  raw: string | undefined,
  hasToolCalls: boolean
): LanguageModelV3FinishReason {
  const normalized = (raw ?? (hasToolCalls ? "tool-calls" : "stop")).toLowerCase();
  let unified: UnifiedFinishReason = "other";
  if (normalized.includes("length") || normalized.includes("max_token")) {
    unified = "length";
  } else if (normalized.includes("content") && normalized.includes("filter")) {
    unified = "content-filter";
  } else if (normalized.includes("tool") || hasToolCalls) {
    unified = "tool-calls";
  } else if (normalized.includes("error")) {
    unified = "error";
  } else if (
    normalized.includes("stop") ||
    normalized === "end_turn" ||
    normalized === "completed"
  ) {
    unified = "stop";
  }
  return { unified, raw };
}

function extractTextParts(
  content: LanguageModelV3CallOptions["prompt"][number]["content"]
): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object" || !("type" in part)) {
      continue;
    }
    if (part.type === "text" && "text" in part && typeof part.text === "string") {
      parts.push(part.text);
      continue;
    }
    if (part.type === "tool-call") {
      parts.push(
        `[tool-call ${part.toolName} ${typeof part.input === "string" ? part.input : JSON.stringify(part.input)}]`
      );
      continue;
    }
    if (part.type === "tool-result") {
      parts.push(`[tool-result ${part.toolName} ${JSON.stringify(part.output)}]`);
    }
  }
  return parts.join("\n");
}

function promptToMessages(
  prompt: LanguageModelV3CallOptions["prompt"],
  systemPrompt?: string
): MessageJs[] {
  const messages: MessageJs[] = [];
  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }
  for (const message of prompt) {
    const role = message.role;
    if (role === "system") {
      messages.push({ role: "system", content: message.content });
      continue;
    }
    messages.push({
      role: role === "assistant" ? "assistant" : role === "tool" ? "tool" : "user",
      content: extractTextParts(message.content),
    });
  }
  return messages.length > 0 ? messages : [{ role: "user", content: "Hello" }];
}

function callToolsToSchemas(options: LanguageModelV3CallOptions): ToolSchema[] {
  const schemas: ToolSchema[] = [];
  for (const tool of options.tools ?? []) {
    if (tool.type !== "function") {
      continue;
    }
    schemas.push({
      name: tool.name,
      description: tool.description ?? `Tool: ${tool.name}`,
      parameters: (tool.inputSchema ?? {
        type: "object",
        properties: {},
      }) as ToolSchema["parameters"],
    });
  }
  return schemas;
}

/** Marker returned by call-option tool stubs so the AI SDK can own execution. */
export interface CalicheClientToolMarker {
  readonly __calicheClientTool: true;
  readonly name: string;
  readonly args: Record<string, unknown>;
}

function clientToolMarker(name: string, args: unknown): CalicheClientToolMarker {
  return {
    __calicheClientTool: true,
    name,
    args:
      args && typeof args === "object" && !Array.isArray(args)
        ? (args as Record<string, unknown>)
        : { value: args },
  };
}

function clientToolMarkerResult(name: string, args: unknown): ToolExecutionResult {
  // Marker is consumed by the AI SDK adapter; surface it as a JsonRecord tool result.
  return clientToolMarker(name, args) as unknown as JsonRecord;
}

function filterCallOptionSchemas(
  schemas: readonly ToolSchema[],
  reservedNames: ReadonlySet<string>
): ToolSchema[] {
  return schemas.filter((schema) => !reservedNames.has(schema.name));
}

function mapToolChoice(options: LanguageModelV3CallOptions): string | undefined {
  const choice = options.toolChoice;
  if (!choice) {
    return undefined;
  }
  switch (choice.type) {
    case "auto":
      return "auto";
    case "none":
      return "none";
    case "required":
      return "required";
    case "tool":
      return choice.toolName;
    default: {
      const _exhaustive: never = choice;
      return _exhaustive;
    }
  }
}

function completionToolCalls(response: CompletionResponse): readonly CompletionToolCall[] {
  return response.tool_calls ?? [];
}

function contentFromCompletion(response: CompletionResponse): LanguageModelV3Content[] {
  const content: LanguageModelV3Content[] = [];
  if (response.content) {
    content.push({ type: "text", text: response.content });
  }
  for (const call of completionToolCalls(response)) {
    content.push({
      type: "tool-call",
      toolCallId: call.id || `call_${call.name}`,
      toolName: call.name,
      input:
        typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {}),
    });
  }
  return content;
}

function stringifyArgs(args: unknown): string {
  if (typeof args === "string") {
    return args;
  }
  try {
    return JSON.stringify(args ?? {});
  } catch {
    return "{}";
  }
}

function toJsonValue(value: unknown): NonNullable<JSONValue> {
  if (value === null || value === undefined) {
    return {};
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  try {
    return JSON.parse(JSON.stringify(value)) as NonNullable<JSONValue>;
  } catch {
    return String(value);
  }
}

interface FinalStreamMetadataState {
  sawMetadata: boolean;
  usage?: TokenUsage;
  finishReason?: string;
}

function updateFinalStreamMetadata(
  state: FinalStreamMetadataState,
  event: { usage?: TokenUsage; finish_reason?: string }
): void {
  state.sawMetadata = true;
  state.usage = event.usage ?? state.usage;
  state.finishReason = event.finish_reason ?? state.finishReason;
}

/**
 * Caliche-backed AI SDK Language Model (V3).
 *
 * Default mode uses {@link LlmClient} so `generateText` / `streamText` own the
 * tool loop. Set `mode: "agent"` to run Caliche's agent loop instead.
 */
export class CalicheLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly provider = "caliche";
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private readonly options: CalicheModelOptions;

  constructor(modelId: string, options: CalicheModelOptions = {}) {
    this.modelId = modelId;
    this.options = options;
  }

  private resolvedProviderModel(): { provider: Provider; model: string } {
    const parsed = parseModelId(this.modelId);
    return {
      provider: (this.options.provider as Provider | undefined) ?? parsed.provider,
      model: this.options.model ?? parsed.model,
    };
  }

  private async createLlmClient(options: LanguageModelV3CallOptions) {
    const { provider, model } = this.resolvedProviderModel();
    const builder = new LlmClientBuilder().provider(provider).model(model);

    if (this.options.apiKey) {
      builder.apiKey(this.options.apiKey);
    }
    if (this.options.baseUrl) {
      builder.baseUrl(this.options.baseUrl);
    }
    if (typeof this.options.temperature === "number") {
      builder.temperature(this.options.temperature);
    }
    if (typeof options.temperature === "number") {
      builder.temperature(options.temperature);
    }
    if (typeof this.options.maxTokens === "number") {
      builder.maxTokens(this.options.maxTokens);
    }
    if (typeof options.maxOutputTokens === "number") {
      builder.maxTokens(options.maxOutputTokens);
    }

    const tools = callToolsToSchemas(options);
    if (tools.length > 0) {
      builder.tools(tools);
    }
    const toolChoice = mapToolChoice(options);
    if (toolChoice) {
      builder.toolChoice(toolChoice);
    }

    return builder.build();
  }

  private async createAgent(options: LanguageModelV3CallOptions) {
    const { provider, model } = this.resolvedProviderModel();
    const builder = new AgentBuilder().provider(provider).model(model);

    if (this.options.systemPrompt) {
      builder.systemPrompt(this.options.systemPrompt);
    }
    if (this.options.apiKey) {
      builder.apiKey(this.options.apiKey);
    }
    if (this.options.baseUrl) {
      builder.baseUrl(this.options.baseUrl);
    }
    if (typeof this.options.temperature === "number") {
      builder.temperature(this.options.temperature);
    }
    if (typeof options.temperature === "number") {
      builder.temperature(options.temperature);
    }
    if (typeof this.options.maxTokens === "number") {
      builder.maxTokens(this.options.maxTokens);
    }
    if (typeof options.maxOutputTokens === "number") {
      builder.maxTokens(options.maxOutputTokens);
    }
    if (typeof this.options.maxIterations === "number") {
      builder.maxIterations(this.options.maxIterations);
    }
    if (this.options.permissionMode) {
      builder.permissionMode(this.options.permissionMode);
    }

    for (const mcp of this.options.mcpServers ?? []) {
      builder.mcpServer(mcp.name, mcp.server);
    }

    const reservedToolNames = new Set<string>();
    for (const tool of this.options.tools ?? []) {
      reservedToolNames.add(tool.name);
      builder.addTool(this.wrapApproval(tool));
    }

    const mcpBridge = new CalicheMcpBridge(this.options.mcpServers ?? []);
    const mcpTools = mcpBridge.hasServers() ? await mcpBridge.asAgentTools() : [];
    for (const mcpTool of mcpTools) {
      reservedToolNames.add(mcpTool.name);
    }

    // Call-option tools are schema-only in V3; expose them as client-tool stubs so the
    // agent can still call them, while AI SDK remains the preferred executor in model mode.
    // Skip names already registered via builder tools or MCP.
    for (const schema of filterCallOptionSchemas(callToolsToSchemas(options), reservedToolNames)) {
      builder.addTool({
        name: schema.name,
        description: schema.description,
        parameters: schema.parameters,
        execute: async (args) => {
          const approved = await this.approve(schema.name, args as Record<string, unknown>);
          if (!approved) {
            return {
              success: false,
              content: "",
              error: `Tool "${schema.name}" requires explicit approval`,
            };
          }
          return clientToolMarkerResult(schema.name, args ?? {});
        },
      });
    }

    for (const mcpTool of mcpTools) {
      builder.addTool(this.wrapApproval(mcpTool));
    }

    return builder.build();
  }

  private wrapApproval(tool: Tool): Tool {
    return {
      ...tool,
      execute: async (args) => {
        const approved = await this.approve(tool.name, args as Record<string, unknown>);
        if (!approved) {
          return {
            success: false,
            content: "",
            error: `Tool "${tool.name}" requires explicit approval`,
          };
        }
        return tool.execute(args);
      },
    };
  }

  private async approve(toolName: string, args: Record<string, unknown>): Promise<boolean> {
    if (typeof this.options.onNeedsApproval !== "function") {
      return true;
    }
    return Boolean(await this.options.onNeedsApproval({ toolName, args }));
  }

  private warningsFor(options: LanguageModelV3CallOptions): SharedV3Warning[] {
    const warnings: SharedV3Warning[] = [];
    if (options.topK !== undefined) {
      warnings.push({ type: "unsupported", feature: "topK" });
    }
    if (options.presencePenalty !== undefined) {
      warnings.push({ type: "unsupported", feature: "presencePenalty" });
    }
    if (options.frequencyPenalty !== undefined) {
      warnings.push({ type: "unsupported", feature: "frequencyPenalty" });
    }
    if (options.seed !== undefined) {
      warnings.push({ type: "unsupported", feature: "seed" });
    }
    return warnings;
  }

  async doGenerate(options: LanguageModelV3CallOptions) {
    const warnings = this.warningsFor(options);
    const messages = promptToMessages(options.prompt, this.options.systemPrompt);
    const mode = this.options.mode ?? "model";

    if (mode === "agent") {
      const agent = await this.createAgent(options);
      const last = messages[messages.length - 1] ?? { role: "user", content: "Hello" };
      const history = messages.slice(0, -1);
      const response = await agent.chatWithHistory(last.content, history);
      const completion: CompletionResponse = {
        content: response.response ?? "",
        usage: response.usage,
        finish_reason: "stop",
      };
      const content = contentFromCompletion(completion);
      return {
        content,
        finishReason: toFinishReason(completion.finish_reason, false),
        usage: toV3Usage(completion.usage),
        warnings,
        response: {
          modelId: this.modelId,
        },
      };
    }

    const client = await this.createLlmClient(options);
    const response = (await client.complete(messages)) as CompletionResponse;
    const toolCalls = completionToolCalls(response);
    const content = contentFromCompletion(response);
    return {
      content,
      finishReason: toFinishReason(response.finish_reason, toolCalls.length > 0),
      usage: toV3Usage(response.usage),
      warnings,
      response: {
        modelId: this.modelId,
      },
    };
  }

  async doStream(options: LanguageModelV3CallOptions) {
    const warnings = this.warningsFor(options);
    const messages = promptToMessages(options.prompt, this.options.systemPrompt);
    const mode = this.options.mode ?? "model";
    const self = this;

    let streamSource: AsyncIterable<StreamEvent>;
    if (mode === "agent") {
      const agent = await this.createAgent(options);
      const serialized = messages
        .map((message) => `${message.role}: ${message.content}`)
        .join("\n");
      streamSource = agent.streamChat(serialized);
    } else {
      const client = await this.createLlmClient(options);
      streamSource = client.stream(messages);
    }

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start(controller) {
        const textId = `text-${Date.now()}`;
        controller.enqueue({ type: "stream-start", warnings });
        controller.enqueue({ type: "text-start", id: textId });

        (async () => {
          let hasToolCalls = false;
          const finalMetadata: FinalStreamMetadataState = { sawMetadata: false };

          for await (const event of streamSource) {
            if (options.abortSignal?.aborted) {
              controller.error(new Error("Aborted"));
              return;
            }

            switch (event.type) {
              case "text_chunk":
                controller.enqueue({
                  type: "text-delta",
                  id: textId,
                  delta: event.content,
                });
                break;
              case "tool_call_start": {
                hasToolCalls = true;
                controller.enqueue({
                  type: "tool-call",
                  toolCallId: event.id,
                  toolName: event.name,
                  input: stringifyArgs(event.arguments),
                });
                break;
              }
              case "tool_call_result":
                controller.enqueue({
                  type: "tool-result",
                  toolCallId: event.id,
                  toolName: event.name,
                  result: toJsonValue(event.result),
                  isError: event.result?.success === false,
                });
                break;
              case "metadata": {
                updateFinalStreamMetadata(finalMetadata, event);
                break;
              }
              case "error":
                controller.enqueue({ type: "error", error: event.message });
                break;
              default:
                break;
            }
          }

          controller.enqueue({ type: "text-end", id: textId });
          if (finalMetadata.sawMetadata) {
            controller.enqueue({
              type: "response-metadata",
              id: `resp-${Date.now()}`,
              modelId: self.modelId,
            });
          }
          controller.enqueue({
            type: "finish",
            finishReason: toFinishReason(finalMetadata.finishReason ?? "stop", hasToolCalls),
            usage: finalMetadata.usage ? toV3Usage(finalMetadata.usage) : emptyUsage(),
          });
          controller.close();
        })().catch((error: unknown) => {
          controller.error(error);
        });
      },
    });

    return { stream };
  }

  async getMcpSnapshot(): Promise<CalicheMcpSnapshot> {
    const bridge = new CalicheMcpBridge(this.options.mcpServers ?? []);
    return {
      resources: bridge.listResources(),
      prompts: bridge.listPrompts(),
      tools: await bridge.listTools(),
    };
  }
}

/**
 * Create a Caliche-backed AI SDK language model.
 *
 * @example
 * ```typescript
 * import { generateText } from "ai";
 * import { caliche } from "@caliche/adapters-ai-sdk";
 *
 * const result = await generateText({
 *   model: caliche("openai/gpt-4o-mini", { apiKey: process.env.OPENAI_API_KEY }),
 *   prompt: "Hello",
 * });
 * ```
 */
export function caliche(
  modelId: string,
  options: CalicheModelOptions = {}
): CalicheLanguageModel {
  return new CalicheLanguageModel(modelId, options);
}

/** @internal test helper */
export function __testOnly_parseModelId(modelId: string): { provider: Provider; model: string } {
  return parseModelId(modelId);
}

/** @internal test helper */
export function __testOnly_toV3Usage(usage?: TokenUsage | null): LanguageModelV3Usage {
  return toV3Usage(usage);
}

/** @internal test helper */
export function __testOnly_toFinishReason(
  raw: string | undefined,
  hasToolCalls: boolean
): LanguageModelV3FinishReason {
  return toFinishReason(raw, hasToolCalls);
}

/** @internal test helper */
export function __testOnly_collectFinalStreamMetadata(
  events: ReadonlyArray<{ usage?: TokenUsage; finish_reason?: string }>
): FinalStreamMetadataState {
  const state: FinalStreamMetadataState = { sawMetadata: false };
  for (const event of events) {
    updateFinalStreamMetadata(state, event);
  }
  return state;
}

/** @internal test helper */
export function __testOnly_clientToolMarker(
  name: string,
  args: unknown
): CalicheClientToolMarker {
  return clientToolMarker(name, args);
}

/** @internal test helper */
export function __testOnly_filterCallOptionSchemas(
  schemas: readonly ToolSchema[],
  reservedNames: ReadonlySet<string>
): ToolSchema[] {
  return filterCallOptionSchemas(schemas, reservedNames);
}
