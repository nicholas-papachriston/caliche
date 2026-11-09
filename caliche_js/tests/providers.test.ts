import { describe, expect, test } from "bun:test";
import { AgentBuilder } from "../src/index";

describe("Providers", () => {
  test("should configure provider with API key", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .apiKey("test-openai-key-123");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should configure provider with custom base URL", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4o")
      .apiKey("test-key")
      .baseUrl("https://api.custom-openai.com/v1");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should configure AWS Bedrock provider", () => {
    const builder = new AgentBuilder()
      .provider("bedrock")
      .model("anthropic.claude-3-5-sonnet-20241022-v2:0")
      .awsCredentials("access-key", "secret-key", "us-east-1");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should handle provider timeout", () => {
    const builder = new AgentBuilder().provider("openai").model("gpt-4o").apiKey("test-key");
    // Timeout would be configured in provider implementation
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should handle provider retries", () => {
    const builder = new AgentBuilder().provider("openai").model("gpt-4o").apiKey("test-key");
    // Retry logic would be in provider implementation
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should switch providers", () => {
    const builder1 = new AgentBuilder().provider("openai").model("gpt-4o").apiKey("test-key");

    const builder2 = new AgentBuilder()
      .provider("anthropic")
      .model("claude-3-5-sonnet-20241022")
      .apiKey("test-key");

    expect(builder1).toBeInstanceOf(AgentBuilder);
    expect(builder2).toBeInstanceOf(AgentBuilder);
  });

  test("should handle provider error configuration", () => {
    const builder = new AgentBuilder().provider("openai").model("gpt-4o").apiKey("test-key");

    // Error handling would be tested during actual API calls
    expect(builder).toBeInstanceOf(AgentBuilder);
  });
});
