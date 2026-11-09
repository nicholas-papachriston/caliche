import { type NativeLlmClient, type NativeLlmClientBuilder, native } from "../native/module";
import type { CompletionResponse, Provider, ToolSchema } from "../streaming/events";
import { StreamIterator } from "../streaming/stream-iterator";

/**
 * Builder for constructing non-agentic LLM clients
 *
 * Provides a lightweight client for direct LLM interactions without
 * tools, memory, or orchestration overhead.
 *
 * @example
 * ```typescript
 * const client = await new LlmClientBuilder()
 *   .provider("openai")
 *   .model("gpt-4o")
 *   .apiKey("sk-...")
 *   .build();
 * ```
 */
export class LlmClientBuilder {
  private readonly _builder: NativeLlmClientBuilder;

  /**
   * Create a new LLM client builder
   */
  constructor() {
    this._builder = new native.LlmClientBuilder();
  }

  /**
   * Set the LLM provider
   *
   * @param provider - Provider name (e.g., "openai", "anthropic", "bedrock")
   * @returns This builder for method chaining
   */
  provider(provider: Provider): this {
    this._builder.setProvider(provider);
    return this;
  }

  /**
   * Set the model name
   *
   * @param model - Model identifier (e.g., "gpt-4o", "claude-3-opus")
   * @returns This builder for method chaining
   */
  model(model: string): this {
    this._builder.setModel(model);
    return this;
  }

  /**
   * Set the API key for authentication
   *
   * @param key - API key string
   * @returns This builder for method chaining
   */
  apiKey(key: string): this {
    this._builder.setApiKey(key);
    return this;
  }

  /**
   * Set an API key scoped to a provider target.
   *
   * @param target - Provider target (e.g., "openai", "anthropic", "bedrock")
   * @param key - API key string
   * @returns This builder for method chaining
   */
  apiKeyFor(target: string, key: string): this {
    this._builder.setApiKeyFor(target, key);
    return this;
  }

  /**
   * Set AWS credentials for Bedrock provider
   *
   * @param accessKey - AWS access key ID
   * @param secretKey - AWS secret access key
   * @param region - AWS region
   * @param sessionToken - Optional session token
   * @returns This builder for method chaining
   */
  awsCredentials(
    accessKey: string,
    secretKey: string,
    region: string,
    sessionToken?: string
  ): this {
    this._builder.setAwsCredentials(accessKey, secretKey, region, sessionToken);
    return this;
  }

  /**
   * Set base URL (for LMStudio or custom endpoints)
   *
   * @param url - Base URL string
   * @returns This builder for method chaining
   */
  baseUrl(url: string): this {
    this._builder.setBaseUrl(url);
    return this;
  }

  /**
   * Set default temperature for completions
   *
   * @param temp - Temperature value (0.0 to 2.0)
   * @returns This builder for method chaining
   */
  temperature(temp: number): this {
    this._builder.setTemperature(temp);
    return this;
  }

  /**
   * Set default max tokens for completions
   *
   * @param tokens - Maximum token count
   * @returns This builder for method chaining
   */
  maxTokens(tokens: number): this {
    this._builder.setMaxTokens(tokens);
    return this;
  }

  /**
   * Set request timeout in seconds
   *
   * @param seconds - Timeout in seconds
   * @returns This builder for method chaining
   */
  timeout(seconds: number): this {
    this._builder.setTimeout(seconds);
    return this;
  }

  /**
   * Set maximum retry attempts
   *
   * @param retries - Maximum number of retries
   * @returns This builder for method chaining
   */
  maxRetries(retries: number): this {
    this._builder.setMaxRetries(retries);
    return this;
  }

  /**
   * Set default tools for completions
   *
   * Tools are optional and can be used for function calling.
   * If not set, no tools will be sent to the LLM.
   *
   * @param tools - Array of tool schemas
   * @returns This builder for method chaining
   */
  tools(tools: readonly ToolSchema[]): this {
    this._builder.setTools(tools);
    return this;
  }

  /**
   * Set default tool choice for completions
   *
   * Controls how the LLM uses tools:
   * - "auto": LLM decides whether to use tools (default)
   * - "none": Disable tool usage
   * - "required": Force tool usage
   * - Specific tool name: Force use of a specific tool
   *
   * @param choice - Tool choice string
   * @returns This builder for method chaining
   */
  toolChoice(choice: string): this {
    this._builder.setToolChoice(choice);
    return this;
  }

  /**
   * Build the LLM client
   *
   * @returns Promise resolving to the configured LlmClient
   */
  async build(): Promise<LlmClient> {
    const nativeClient = await this._builder.build();
    return new LlmClient(nativeClient);
  }
}

/**
 * Non-agentic LLM client for simple completions
 *
 * Provides direct access to LLM providers without tools, memory,
 * or orchestration overhead.
 *
 * @example
 * ```typescript
 * const client = await LlmClient.builder()
 *   .provider("openai")
 *   .model("gpt-4o")
 *   .apiKey("sk-...")
 *   .build();
 *
 * // Non-streaming
 * const response = await client.completeText("What is 2+2?");
 * console.log(response); // "4"
 *
 * // Streaming
 * const stream = client.streamText("Tell me a story");
 * for await (const event of stream) {
 *   if (event.type === "text_chunk") {
 *     process.stdout.write(event.content);
 *   }
 * }
 * ```
 */
export class LlmClient {
  readonly client: NativeLlmClient;

  /**
   * Create a new builder for constructing an LlmClient
   */
  static builder(): LlmClientBuilder {
    return new LlmClientBuilder();
  }

  /**
   * @internal
   * Create an LLM client (use LlmClientBuilder instead)
   */
  constructor(nativeClient: NativeLlmClient) {
    this.client = nativeClient;
  }

  /**
   * Complete a conversation with messages (non-streaming)
   *
   * @param messages - Array of message objects with role and content
   * @returns Promise resolving to the completion response
   */
  async complete(messages: Array<{ role: string; content: string }>): Promise<CompletionResponse> {
    return await native.llmClientComplete(this.client, messages);
  }

  /**
   * Stream a conversation with messages
   *
   * @param messages - Array of message objects with role and content
   * @returns Stream iterator for receiving events
   */
  stream(messages: Array<{ role: string; content: string }>): StreamIterator {
    const nativeIter = native.llmClientStream(this.client, messages);
    return new StreamIterator(nativeIter);
  }

  /**
   * Complete a single text prompt (convenience method)
   *
   * @param prompt - Text prompt string
   * @returns Promise resolving to the completion text
   */
  async completeText(prompt: string): Promise<string> {
    return await native.llmClientCompleteText(this.client, prompt);
  }

  /**
   * Stream a single text prompt (convenience method)
   *
   * @param prompt - Text prompt string
   * @returns Stream iterator for receiving events
   */
  streamText(prompt: string): StreamIterator {
    const nativeIter = native.llmClientStreamText(this.client, prompt);
    return new StreamIterator(nativeIter);
  }

  /**
   * Get the provider name
   */
  get providerName(): string {
    return native.llmClientGetProviderName(this.client);
  }

  /**
   * Get the model name
   */
  get model(): string {
    return native.llmClientGetModel(this.client);
  }
}
