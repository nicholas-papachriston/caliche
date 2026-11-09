/**
 * LangChain LLM Provider Adapter for Caliche
 *
 * This adapter allows you to use LangChain chat models (ChatOpenAI, ChatAnthropic, etc.)
 * as providers for Caliche agents.
 *
 * @example
 * ```typescript
 * import { ChatOpenAI } from "@langchain/openai";
 * import { LangChainLLMProvider } from "@caliche/adapters-langchain";
 * import { AgentBuilder } from "@caliche/core";
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

import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type { BaseMessage } from "@langchain/core/messages";
import type { StreamEvent } from "../index";
import type { JsonRecord, JsonValue, LangChainMessageContent, LangChainToolCall } from "../types";

export interface CompletionRequest {
  messages: string | Array<{ role: string; content: LangChainMessageContent }>;
  max_tokens?: number;
  temperature?: number;
  tools?: string | JsonRecord;
  tool_choice?: string | JsonValue;
  response_format?: JsonRecord;
}

export interface CompletionResponse {
  content: string;
  finish_reason?: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  tool_calls?: readonly LangChainToolCall[];
}

/**
 * Adapter that wraps a LangChain chat model to make it compatible with Caliche.
 */
export class LangChainLLMProvider {
  private llm: BaseChatModel;
  public readonly supportsNativeTools: boolean;
  public readonly supportsStreaming: boolean;

  constructor(
    llm: BaseChatModel,
    supportsNativeTools: boolean = true,
    supportsStreaming: boolean = true
  ) {
    this.llm = llm;
    this.supportsNativeTools = supportsNativeTools;
    this.supportsStreaming = supportsStreaming;
  }

  /**
   * Perform a completion request using the LangChain LLM.
   */
  async complete(request: CompletionRequest | JsonRecord): Promise<CompletionResponse> {
    const requestObj = this.normalizeRequest(request);
    const messagesData = this.extractMessages(requestObj);

    // Convert Caliche messages to LangChain messages
    const langchainMessages = this.convertMessagesToLangChain(messagesData);

    // Handle tools if present
    type BoundLlm = BaseChatModel & {
      bindTools?: (tools: unknown[]) => BaseChatModel | any;
      temperature?: number;
      maxTokens?: number;
    };
    let boundLlm: BoundLlm = this.llm as BoundLlm;
    if (requestObj.tools) {
      const toolsData =
        typeof requestObj.tools === "string" ? JSON.parse(requestObj.tools) : requestObj.tools;

      const langchainTools = this.convertToolsToLangChain(toolsData);
      if (langchainTools && langchainTools.length > 0 && typeof boundLlm.bindTools === "function") {
        // Type assertion needed because LangChain's bindTools has stricter types
        boundLlm = boundLlm.bindTools(langchainTools as any) as BoundLlm;
      }
    }

    // Set parameters
    if (requestObj.temperature !== undefined) {
      boundLlm.temperature = requestObj.temperature;
    }
    if (requestObj.max_tokens !== undefined) {
      boundLlm.maxTokens = requestObj.max_tokens;
    }

    // Call LangChain LLM
    try {
      const response = await boundLlm.invoke(langchainMessages);

      // Convert LangChain response to Caliche format
      // LangChain returns AIMessageChunk which has different structure
      return this.convertLangChainResponse(response as any);
    } catch (error) {
      // Extract more detailed error information
      let errorMessage = "Unknown error";
      if (error instanceof Error) {
        errorMessage = error.message;
        // Include stack trace in development
        if (process.env.NODE_ENV === "development") {
          errorMessage += `\n${error.stack}`;
        }
      } else if (typeof error === "object" && error !== null) {
        // Try to extract error details from LangChain error objects
        const errorObj = error as { message?: string; error?: { message?: string } };
        if (errorObj.message) {
          errorMessage = errorObj.message;
        } else if (errorObj.error?.message) {
          errorMessage = errorObj.error.message;
        } else {
          errorMessage = JSON.stringify(error);
        }
      } else {
        errorMessage = String(error);
      }
      throw new Error(`LangChain LLM invocation failed: ${errorMessage}`);
    }
  }

  /**
   * Stream completion tokens from the LangChain LLM.
   */
  async *stream(
    request: CompletionRequest | JsonRecord
  ): AsyncGenerator<StreamEvent, void, unknown> {
    const requestObj = this.normalizeRequest(request);
    const messagesData = this.extractMessages(requestObj);

    const langchainMessages = this.convertMessagesToLangChain(messagesData);

    // Handle tools if present
    type BoundLlm = BaseChatModel & {
      bindTools?: (tools: unknown[]) => BaseChatModel | any;
      temperature?: number;
      maxTokens?: number;
    };
    let boundLlm: BoundLlm = this.llm as BoundLlm;
    if (requestObj.tools) {
      const toolsData =
        typeof requestObj.tools === "string" ? JSON.parse(requestObj.tools) : requestObj.tools;

      const langchainTools = this.convertToolsToLangChain(toolsData);
      if (langchainTools && langchainTools.length > 0 && typeof boundLlm.bindTools === "function") {
        // Type assertion needed because LangChain's bindTools has stricter types
        boundLlm = boundLlm.bindTools(langchainTools as any) as BoundLlm;
      }
    }

    // Set parameters
    if (requestObj.temperature !== undefined) {
      boundLlm.temperature = requestObj.temperature;
    }
    if (requestObj.max_tokens !== undefined) {
      boundLlm.maxTokens = requestObj.max_tokens;
    }

    // Stream from LangChain
    try {
      for await (const chunk of await boundLlm.stream(langchainMessages)) {
        // Convert LangChain chunk to Caliche StreamEvent
        const event = this.convertLangChainChunk(chunk as any);
        if (event) {
          yield event;
        }
      }
    } catch (error) {
      // StreamEvent doesn't support error type, so we throw instead
      throw new Error(
        `LangChain streaming failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }

    // Yield done event
    yield { type: "done" };
  }

  private convertMessagesToLangChain(
    messagesData: Array<{ role: string; content: LangChainMessageContent }>
  ): BaseMessage[] {
    // Dynamic import to avoid requiring langchain-core at module level
    const {
      HumanMessage,
      AIMessage,
      SystemMessage,
      ToolMessage,
    } = require("@langchain/core/messages");

    const langchainMessages: BaseMessage[] = [];
    for (const msgData of messagesData) {
      const role = String(msgData.role || "user").toLowerCase();
      let content = msgData.content;

      // Handle structured content
      if (typeof content === "object" && content !== null) {
        if (Array.isArray(content)) {
          // Handle array of content blocks
          const textParts = content
            .map((block: any) => block.text || block.content || "")
            .filter((text: string) => text);
          content = textParts.join(" ") || "";
        } else {
          content = (content as any).text || (content as any).content || "";
        }
      } else {
        content = String(content || "");
      }

      if (role === "user" || role === "human") {
        langchainMessages.push(new HumanMessage(content));
      } else if (role === "assistant" || role === "ai") {
        langchainMessages.push(new AIMessage(content));
      } else if (role === "system") {
        langchainMessages.push(new SystemMessage(content));
      } else if (role === "tool") {
        const toolCallId = (msgData as any).tool_call_id || "";
        langchainMessages.push(new ToolMessage(content, toolCallId));
      } else {
        // Be permissive with unknown roles to avoid empty-message provider errors.
        langchainMessages.push(new HumanMessage(content));
      }
    }

    if (langchainMessages.length === 0) {
      throw new Error("LangChainLLMProvider: completion request contained no messages");
    }

    return langchainMessages;
  }

  private normalizeRequest(request: CompletionRequest | JsonRecord): CompletionRequest {
    let raw: any = {};
    if (typeof request === "string") {
      try {
        raw = JSON.parse(request);
      } catch {
        raw = { prompt: request };
      }
    } else if (request && typeof request === "object") {
      raw = request;
    }

    // Rust bridge payloads may be nested under common envelope keys.
    const req = [raw?.request, raw?.payload, raw?.body, raw].find(
      (value) => value && typeof value === "object"
    );

    return (req || {}) as CompletionRequest;
  }

  private extractMessages(
    requestObj: CompletionRequest
  ): Array<{ role: string; content: LangChainMessageContent }> {
    const rawMessages = (requestObj as any).messages ?? (requestObj as any).input;

    if (typeof rawMessages === "string") {
      try {
        const parsed = JSON.parse(rawMessages);
        if (Array.isArray(parsed)) {
          return parsed;
        }
        if (parsed && typeof parsed === "object") {
          return [parsed as { role: string; content: LangChainMessageContent }];
        }
      } catch {
        // If it's plain text rather than JSON, treat it as a user message.
        return [{ role: "user", content: rawMessages }];
      }
      return [];
    }

    if (Array.isArray(rawMessages) && rawMessages.length > 0) {
      return rawMessages as Array<{ role: string; content: LangChainMessageContent }>;
    }

    if (rawMessages && typeof rawMessages === "object") {
      return [rawMessages as { role: string; content: LangChainMessageContent }];
    }

    const prompt = (requestObj as any).prompt;
    if (typeof prompt === "string" && prompt.length > 0) {
      return [{ role: "user", content: prompt }];
    }

    for (const key of ["text", "query", "message", "content"]) {
      const value = (requestObj as any)[key];
      if (typeof value === "string" && value.length > 0) {
        return [{ role: "user", content: value }];
      }
    }

    return [];
  }

  private convertToolsToLangChain(tools: JsonRecord[]): unknown[] | null {
    if (!tools || tools.length === 0) {
      return null;
    }

    try {
      // Dynamic import to avoid requiring @langchain/core at module level
      const { DynamicStructuredTool } = require("@langchain/core/tools");

      const langchainTools: unknown[] = [];

      for (const tool of tools) {
        // Extract tool information from Caliche format
        const name = tool.name as string;
        const description = (tool.description as string) || "";
        const parameters = (tool.parameters as JsonRecord) || {};

        if (!name) {
          console.warn("Skipping tool without name:", tool);
          continue;
        }

        // Ensure parameters is a valid JSON Schema object
        const schema = {
          type: "object",
          ...parameters,
        } as JsonRecord;

        // Create a LangChain tool with the JSON Schema
        // Note: We provide a no-op function since tool execution is handled by Caliche
        // The schema is what matters for LLM tool binding
        const langchainTool = new DynamicStructuredTool({
          name,
          description,
          schema: schema as any, // LangChain accepts JSON Schema directly
          func: async (_args: JsonRecord) => {
            // This function should never be called in practice since Caliche handles execution
            // But LangChain requires a function, so we provide a stub
            return JSON.stringify({
              success: false,
              content: "Tool execution is handled by Caliche, not LangChain",
              error: "This is a schema-only tool definition",
            });
          },
        });

        langchainTools.push(langchainTool);
      }

      return langchainTools.length > 0 ? langchainTools : null;
    } catch (error) {
      // If @langchain/core is not available, log a warning and return null
      console.warn(
        "Failed to convert tools to LangChain format. Make sure @langchain/core is installed:",
        error instanceof Error ? error.message : String(error)
      );
      return null;
    }
  }

  private convertLangChainResponse(response: {
    content?: LangChainMessageContent;
    tool_calls?: readonly LangChainToolCall[];
    response_metadata?: {
      finish_reason?: string;
      token_usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
      };
    };
  }): CompletionResponse {
    let content = "";
    let toolCalls: LangChainToolCall[] | undefined;

    // Extract content
    if (response.content) {
      content = String(response.content);
    }

    // Extract tool calls if present
    if (response.tool_calls && response.tool_calls.length > 0) {
      toolCalls = response.tool_calls.map((tc) => ({
        id: tc.id || "",
        name: tc.name || "",
        args: tc.args || {},
      }));
    }

    // Build response
    const result: CompletionResponse = {
      content: content,
      finish_reason: response.response_metadata?.finish_reason,
      tool_calls: toolCalls,
      usage: response.response_metadata?.token_usage
        ? {
            prompt_tokens: response.response_metadata.token_usage.prompt_tokens,
            completion_tokens: response.response_metadata.token_usage.completion_tokens,
            total_tokens: response.response_metadata.token_usage.total_tokens,
          }
        : undefined,
    };

    return result;
  }

  private convertLangChainChunk(chunk: {
    content?: string;
    tool_calls?: readonly LangChainToolCall[];
  }): StreamEvent | null {
    if (chunk.content) {
      return {
        type: "text_chunk",
        content: String(chunk.content),
      };
    }
    return null;
  }
}
