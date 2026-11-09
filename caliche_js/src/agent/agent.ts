import { napiToolPayload } from "../napi-callback";
import type { NativeAgent, StreamChatOptions } from "../native/module";
import type {
  AttachmentInput,
  ChatResponse,
  ContentPartJs,
  MessageJs,
  Provider,
  StepResultJs,
  ToolInfo,
  ToolRequirementPolicy,
  UserContextJs,
} from "../streaming/events";
import { StreamIterator } from "../streaming/stream-iterator";
import { ToolRegistry } from "../tool-registry";
import type {
  InferToolArguments,
  JsonValue,
  ToolExecutionResult,
  ToolHandler,
  ToolHandlerArgs,
  ToolParameterSchema,
} from "../types";

/**
 * Agent for chat interactions with LLMs
 *
 * @example
 * ```typescript
 * const response = await agent.chat('Hello!');
 * console.log(response.response);
 * ```
 */
export class Agent {
  readonly agent: NativeAgent;
  private readonly seedHistory: MessageJs[];
  private readonly parsePseudoToolProtocol: boolean;
  private readonly toolRequirementPolicies: readonly ToolRequirementPolicy[];

  /**
   * @internal
   * Create an agent (use AgentBuilder instead)
   */
  constructor(
    nativeAgent: NativeAgent,
    seedHistory: readonly MessageJs[] = [],
    parsePseudoToolProtocol: boolean = false,
    toolRequirementPolicies: readonly ToolRequirementPolicy[] = []
  ) {
    this.agent = nativeAgent;
    this.seedHistory = [...seedHistory];
    this.parsePseudoToolProtocol = parsePseudoToolProtocol;
    this.toolRequirementPolicies = [...toolRequirementPolicies];
  }

  /**
   * Create an isolated request-scoped agent for concurrent work.
   *
   * The clone shares immutable provider and tool configuration while receiving
   * independent session and conversation state. Prefer this over rebuilding an
   * AgentBuilder per request or calling chat concurrently on one Agent.
   */
  async cloneForRequest(): Promise<Agent> {
    const nativeAgent = await this.agent.cloneForRequest();
    return new Agent(
      nativeAgent,
      this.seedHistory,
      this.parsePseudoToolProtocol,
      this.toolRequirementPolicies
    );
  }

  /**
   * Send a chat message and get a response
   *
   * @param message - The message to send to the agent
   * @returns Promise resolving to the response and token usage
   * @throws If the chat request fails
   *
   * @example
   * ```typescript
   * const response = await agent.chat('What is the capital of France?');
   * console.log(response.response); // "Paris is the capital of France."
   * console.log(response.usage.total_tokens); // e.g., 42
   * ```
   */
  async chat(message: string): Promise<ChatResponse> {
    if (this.seedHistory.length > 0) {
      return await this.agent.chatWithHistory(message, this.seedHistory);
    }
    return await this.agent.chat(message);
  }

  /**
   * Send a chat message and return only the response text.
   */
  async chatText(message: string): Promise<string> {
    const response = await this.chat(message);
    return response.response ?? "";
  }

  /**
   * Start a streaming chat for real-time responses
   *
   * @param message - The message to send to the agent
   * @returns Stream iterator for receiving events
   *
   * @example
   * ```typescript
   * const stream = agent.streamChat('Tell me a story');
   * for await (const event of stream) {
   *   if (isTextChunkEvent(event)) {
   *     process.stdout.write(event.content);
   *   }
   * }
   * ```
   */
  streamChat(message: string, options?: StreamChatOptions | null): StreamIterator {
    const agent = this.agent;
    const nativeIter =
      this.seedHistory.length > 0
        ? options && typeof agent.streamChatWithHistoryAndOptions === "function"
          ? agent.streamChatWithHistoryAndOptions(message, this.seedHistory, options)
          : typeof agent.streamChatWithHistory === "function"
            ? agent.streamChatWithHistory(message, this.seedHistory)
            : agent.streamChat(message)
        : options && typeof agent.streamChatWithOptions === "function"
          ? agent.streamChatWithOptions(message, options)
          : agent.streamChat(message);
    return new StreamIterator(nativeIter, this.parsePseudoToolProtocol, {
      options,
      message,
      toolRequirementPolicies: this.toolRequirementPolicies,
    });
  }

  /**
   * Get the agent's configured provider
   *
   * @example
   * ```typescript
   * console.log(agent.provider); // "openai"
   * ```
   */
  get provider(): Provider {
    return this.agent.getProvider() as Provider;
  }

  /**
   * Get the agent's configured model
   *
   * @example
   * ```typescript
   * console.log(agent.model); // "gpt-4"
   * ```
   */
  get model(): string {
    return this.agent.getModel();
  }

  /**
   * List all available tools for this agent
   *
   * @returns Array of tool information including name, description, and parameters
   *
   * @example
   * ```typescript
   * const tools = agent.listTools();
   * for (const tool of tools) {
   *   console.log(`${tool.name}: ${tool.description}`);
   * }
   * ```
   */
  listTools(): ToolInfo[] {
    return this.agent.listTools();
  }

  /**
   * Access the agent's tool registry (active/deferred controls).
   */
  async toolRegistry(): Promise<ToolRegistry> {
    if (typeof this.agent.toolRegistry !== "function") {
      throw new Error("toolRegistry is not available on this native binding");
    }
    const registry = await this.agent.toolRegistry();
    return ToolRegistry.fromNative(registry);
  }

  /**
   * Send a chat message with explicit conversation history
   *
   * This method allows you to provide the full conversation history manually,
   * which is useful for custom memory management or when working with reasoning patterns.
   *
   * @param message - The message to send to the agent
   * @param history - Array of previous messages in the conversation
   * @returns Promise resolving to the response and token usage
   *
   * @example
   * ```typescript
   * const history = [
   *   { role: "user", content: "Hello" },
   *   { role: "assistant", content: "Hi there!" }
   * ];
   * const response = await agent.chatWithHistory("What did I say?", history);
   * ```
   */
  async chatWithHistory(message: string, history: MessageJs[]): Promise<ChatResponse> {
    return await this.agent.chatWithHistory(message, history);
  }

  async chatWithAttachments(
    message: string,
    attachments: readonly AttachmentInput[]
  ): Promise<ChatResponse> {
    const contentParts: ContentPartJs[] = [
      { type: "text", text: message },
      ...attachments.map((attachment) => ({
        type: attachment.type,
        data: attachment.data,
        media_type: attachment.media_type,
        filename: attachment.filename,
      })),
    ];
    const history: MessageJs[] = [
      ...this.seedHistory,
      { role: "user", content: message, content_parts: contentParts },
    ];
    return await this.agent.chatWithHistory(
      "Please process the previously attached multimodal content.",
      history
    );
  }

  streamWithAttachments(
    message: string,
    attachments: readonly AttachmentInput[],
    options?: StreamChatOptions | null
  ): StreamIterator {
    const contentParts: ContentPartJs[] = [
      { type: "text", text: message },
      ...attachments.map((attachment) => ({
        type: attachment.type,
        data: attachment.data,
        media_type: attachment.media_type,
        filename: attachment.filename,
      })),
    ];
    const history: MessageJs[] = [
      ...this.seedHistory,
      { role: "user", content: message, content_parts: contentParts },
    ];
    const agent = this.agent;
    const prompt = "Please process the previously attached multimodal content.";
    const nativeIter =
      options && typeof agent.streamChatWithHistoryAndOptions === "function"
        ? agent.streamChatWithHistoryAndOptions(prompt, history, options)
        : typeof agent.streamChatWithHistory === "function"
          ? agent.streamChatWithHistory(prompt, history)
          : agent.streamChat(prompt);
    return new StreamIterator(nativeIter, this.parsePseudoToolProtocol, {
      options,
      message: prompt,
      toolRequirementPolicies: this.toolRequirementPolicies,
    });
  }

  /**
   * Add a custom tool to the agent dynamically
   *
   * This allows you to add tools after the agent has been built, which is useful
   * for dynamic tool registration or plugin systems.
   *
   * @param name - The name of the tool
   * @param description - Description of what the tool does
   * @param parameters - JSON schema for the tool parameters
   * @param executeFn - Async function that executes the tool
   *
   * @example
   * ```typescript
   * agent.addTool(
   *   "my_tool",
   *   "Does something useful",
   *   {
   *     type: "object",
   *     properties: {
   *       input: { type: "string", description: "Input parameter" }
   *     }
   *   },
   *   async (args) => {
   *     return { success: true, content: `Processed: ${args.input}` };
   *   }
   * );
   * ```
   */
  addTool<
    TParameters extends JsonValue = JsonValue,
    TResult extends ToolExecutionResult = ToolExecutionResult,
  >(
    name: string,
    description: string,
    parameters: TParameters,
    executeFn: ToolHandler<
      TParameters extends ToolParameterSchema ? InferToolArguments<TParameters> : ToolHandlerArgs,
      TResult
    >
  ): void {
    // The Rust side uses call_js_sync_bridge which expects the function to return a string.
    // NAPI ThreadsafeFunction invokes JS as error-first: (err, payload) => returnValue.
    const isAsync =
      executeFn.constructor.name === "AsyncFunction" || executeFn.toString().includes("async");

    if (isAsync) {
      const asyncFn = executeFn as unknown as ToolHandler<JsonValue, ToolExecutionResult>;
      const wrappedFn = async (...callbackArgs: unknown[]): Promise<string> => {
        try {
          const args = napiToolPayload(callbackArgs) as JsonValue;
          const result = await asyncFn(args);
          return JSON.stringify(result);
        } catch (error) {
          return JSON.stringify({
            success: false,
            content: "",
            error: error instanceof Error ? error.message : String(error),
          });
        }
      };
      // Pass as any since ThreadsafeFunction typing is complex
      (this.agent as any).addTool(name, description, parameters, wrappedFn);
    } else {
      const syncFn = executeFn as unknown as ToolHandler<JsonValue, ToolExecutionResult>;
      const wrappedFn = (...callbackArgs: unknown[]): string => {
        try {
          const args = napiToolPayload(callbackArgs) as JsonValue;
          const result = syncFn(args);
          if (result instanceof Promise) {
            throw new Error(
              "Tool handler returned a Promise in sync path. Declare the handler as async so it is bridged correctly."
            );
          }
          return JSON.stringify(result);
        } catch (error) {
          return JSON.stringify({
            success: false,
            content: "",
            error: error instanceof Error ? error.message : String(error),
          });
        }
      };
      (this.agent as any).addTool(name, description, parameters, wrappedFn);
    }
  }

  /**
   * Set user context for request tracking and multi-tenancy
   *
   * User context is propagated to tools and can be used for filtering,
   * auditing, and tenant isolation.
   *
   * @param userId - The user ID (null to clear context)
   * @param metadata - Optional metadata object with custom claims
   *
   * @example
   * ```typescript
   * agent.setUserContext("user123", { tenant_id: "tenant1", role: "admin" });
   * ```
   */
  setUserContext(userId: string | null, metadata?: JsonValue): void {
    this.agent.setUserContext(userId, metadata || null);
  }

  /**
   * Get the current user context
   *
   * @returns User context object or null if not set
   *
   * @example
   * ```typescript
   * const context = agent.getUserContext();
   * if (context) {
   *   console.log(`User: ${context.user_id}`);
   * }
   * ```
   */
  getUserContext(): UserContextJs | null {
    return this.agent.getUserContext();
  }

  /**
   * Get conversation messages from memory
   *
   * Retrieves all messages stored in the agent's memory, if memory is enabled.
   * Returns an empty array if memory is not configured.
   *
   * @returns Promise resolving to array of messages
   *
   * @example
   * ```typescript
   * const messages = await agent.getMessages();
   * for (const msg of messages) {
   *   console.log(`${msg.role}: ${msg.content}`);
   * }
   * ```
   */
  async getMessages(): Promise<MessageJs[]> {
    return await this.agent.getMessages();
  }

  /**
   * Execute a single reasoning step
   *
   * This method executes one LLM call and its immediate tool executions,
   * returning the result without updating the agent's internal memory.
   * Useful for advanced reasoning patterns like Tree of Thoughts where
   * you need fine-grained control over the execution loop.
   *
   * @param messages - Full conversation history up to this point
   * @returns Promise resolving to step result with response, tool calls, and updated history
   *
   * @example
   * ```typescript
   * const messages = [
   *   { role: "user", content: "What is 2+2?" }
   * ];
   * const stepResult = await agent.step(messages);
   * console.log(stepResult.response); // "4"
   * console.log(stepResult.tool_calls); // Any tool calls made
   * ```
   */
  async step(messages: MessageJs[]): Promise<StepResultJs> {
    return await this.agent.step(messages);
  }
}
