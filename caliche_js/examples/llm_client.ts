/**
 * Non-Agentic LLM Client Example
 *
 * This example demonstrates:
 * - Using LlmClient for simple, single-shot LLM completions
 * - Non-streaming completions with messages
 * - Streaming completions
 * - Convenience methods for text-only prompts
 * - Builder pattern configuration
 *
 * Best practices shown:
 * - Direct LLM access without agent overhead
 * - Simple prompt/response workflows
 * - Streaming for real-time responses
 *
 * Run with:
 * ```bash
 * bun run examples/llm_client.ts
 * ```
 */

import { isTextChunkEvent, LlmClientBuilder, type ToolSchema } from "../src/index";

async function example1BasicCompletion(): Promise<void> {
  console.log("=== Example 1: Basic Non-Streaming Completion ===\n");

  try {
    const client = await new LlmClientBuilder()
      .provider("openai")
      .model("gpt-4o")
      .temperature(0.7)
      .maxTokens(100)
      .build();

    const response = await client.completeText("What is 2+2? Answer in one sentence.");
    console.log(`Response: ${response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function example2CompletionWithMessages(): Promise<void> {
  console.log("=== Example 2: Completion with Message History ===\n");

  try {
    const client = await new LlmClientBuilder().provider("openai").model("gpt-4o").build();

    const messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is the capital of France?" },
    ];

    const response = await client.complete(messages);
    console.log(`Response: ${response.content}`);
    if (response.usage) {
      console.log(`Tokens used: ${JSON.stringify(response.usage)}\n`);
    }
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function example3StreamingCompletion(): Promise<void> {
  console.log("=== Example 3: Streaming Completion ===\n");

  try {
    const client = await new LlmClientBuilder().provider("openai").model("gpt-4o").build();

    console.log("Streaming response: ");
    const stream = client.streamText("Tell me a short story about a robot in exactly 3 sentences.");

    for await (const event of stream) {
      if (isTextChunkEvent(event)) {
        process.stdout.write(event.content);
      } else if (event.type === "done") {
        console.log("\nStream completed\n");
        break;
      }
    }
  } catch (error) {
    console.log(
      `\n(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function example4StreamingWithMessages(): Promise<void> {
  console.log("=== Example 4: Streaming with Message History ===\n");

  try {
    const client = await new LlmClientBuilder().provider("openai").model("gpt-4o").build();

    const messages = [
      { role: "system", content: "You are a creative writing assistant." },
      { role: "user", content: "Write a haiku about programming." },
    ];

    console.log("Streaming response: ");
    const stream = client.stream(messages);

    for await (const event of stream) {
      if (isTextChunkEvent(event)) {
        process.stdout.write(event.content);
      } else if (event.type === "done") {
        console.log("\nStream completed\n");
        break;
      }
    }
  } catch (error) {
    console.log(
      `\n(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function example6AnthropicProvider(): Promise<void> {
  console.log("=== Example 6: Using Anthropic Provider ===\n");

  try {
    const client = await new LlmClientBuilder()
      .provider("anthropic")
      .model("claude-haiku-4-5-20251001")
      .build();

    console.log("Anthropic client created");
    const response = await client.completeText("Say hello in one sentence.");
    console.log(`Response: ${response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function example7CustomConfig(): Promise<void> {
  console.log("=== Example 7: Custom Configuration ===\n");

  try {
    const client = await new LlmClientBuilder()
      .provider("openai")
      .model("gpt-4o")
      .timeout(30) // 30 seconds
      .maxRetries(3)
      .temperature(0.5)
      .build();

    console.log("Client created with custom timeout and retries");
    const response = await client.completeText("What is the meaning of life? Answer briefly.");
    console.log(`Response: ${response}\n`);
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function example8WithTools(): Promise<void> {
  console.log("=== Example 8: Using Tools (Function Calling) ===\n");

  try {
    // Define a tool schema
    const weatherTool: ToolSchema = {
      name: "get_weather",
      description: "Get the current weather in a location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
          },
          unit: {
            type: "string",
            enum: ["celsius", "fahrenheit"],
            description: "The unit of temperature",
          },
        },
        required: ["location"],
      },
    };

    const client = await new LlmClientBuilder()
      .provider("anthropic")
      .model("claude-haiku-4-5-20251001")
      .tools([weatherTool])
      .toolChoice("auto") // Let the LLM decide when to use tools
      .build();

    console.log("Client created with tools");
    const messages = [{ role: "user", content: "What's the weather like in San Francisco?" }];

    const response = await client.complete(messages);
    console.log(`Response: ${response.content}`);

    // Note: Tool calls may be present in the response but are not exposed
    // in the JavaScript bindings. In a real application, you would check for
    // tool calls, execute them, and send the result back to the LLM
    // in a follow-up message.
    console.log("(Note: Tool calls are returned but not executed in this example)\n");
  } catch (error) {
    console.log(
      `(Skipped: ${error instanceof Error ? error.message : "Provider not configured"})\n`
    );
  }
}

async function main(): Promise<void> {
  console.log("=== Non-Agentic LLM Client Examples ===\n");

  await example1BasicCompletion();
  await example2CompletionWithMessages();
  await example3StreamingCompletion();
  await example4StreamingWithMessages();
  await example6AnthropicProvider();
  await example7CustomConfig();
  await example8WithTools();

  console.log("=== All Examples Complete ===");
}

main().catch(console.error);
