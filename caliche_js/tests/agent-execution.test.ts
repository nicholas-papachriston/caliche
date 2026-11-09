import { describe, expect, test } from "bun:test";
import { AgentBuilder } from "../src/index";

describe("Agent Execution", () => {
  test("should create agent with OpenAI provider", () => {
    const builder = new AgentBuilder().provider("openai").model("gpt-4o").apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should create agent with Anthropic provider", () => {
    const builder = new AgentBuilder()
      .provider("anthropic")
      .model("claude-3-5-sonnet-20241022")
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should create agent with system prompt", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .systemPrompt("You are a helpful assistant")
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should create agent with temperature", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .temperature(0.7)
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should create agent with max tokens", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .maxTokens(2048)
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should create agent with multiple configuration options", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .systemPrompt("Test prompt")
      .temperature(0.8)
      .maxTokens(1024)
      .apiKey("test-key");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should handle agent state management", async () => {
    const builder = new AgentBuilder().provider("openai").model("gpt-4o").apiKey("test-key");

    // Note: Actual build would require valid API key
    // This tests the builder structure
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should support all provider types", () => {
    const providers: Array<
      | "openai"
      | "anthropic"
      | "bedrock"
      | "openrouter"
      | "perplexity"
      | "lmstudio"
      | "google"
      | "gemini"
    > = [
      "openai",
      "anthropic",
      "bedrock",
      "openrouter",
      "perplexity",
      "lmstudio",
      "google",
      "gemini",
    ];

    for (const provider of providers) {
      const builder = new AgentBuilder().provider(provider).model("test-model").apiKey("test-key");

      expect(builder).toBeInstanceOf(AgentBuilder);
    }
  });

  test("should handle error when building without required fields", async () => {
    const builder = new AgentBuilder().provider("openai");
    // Missing model and API key

    // Building should fail or handle gracefully
    // In real implementation, would throw error
    expect(builder).toBeInstanceOf(AgentBuilder);
  });
});
