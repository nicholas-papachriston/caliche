import { describe, expect, test } from "bun:test";
import { LlmClient, LlmClientBuilder } from "../src/index";

// Live tests run only when CALICHE_LIVE=1 or CALICHE_RUN_LMSTUDIO_INTEGRATION=true.
const liveLlmTest = test.skipIf(
  process.env.CALICHE_LIVE !== "1" && process.env.CALICHE_RUN_LMSTUDIO_INTEGRATION !== "true",
);

describe("LlmClientBuilder", () => {
  test("should create a LlmClientBuilder instance", () => {
    const builder = new LlmClientBuilder();
    expect(builder).toBeInstanceOf(LlmClientBuilder);
  });

  test("should not expose fromEnv factory method", () => {
    expect((LlmClientBuilder as unknown as { fromEnv?: unknown }).fromEnv).toBeUndefined();
  });

  test("should chain provider method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.provider("openai");
    expect(result).toBe(builder);
  });

  test("should chain model method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.model("gpt-4o");
    expect(result).toBe(builder);
  });

  test("should chain apiKey method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.apiKey("test-openai-key");
    expect(result).toBe(builder);
  });

  test("should chain awsCredentials method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.awsCredentials("key", "secret", "us-east-1");
    expect(result).toBe(builder);
  });

  test("should chain awsCredentials with session token", () => {
    const builder = new LlmClientBuilder();
    const result = builder.awsCredentials("key", "secret", "us-east-1", "session-token");
    expect(result).toBe(builder);
  });

  test("should chain baseUrl method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.baseUrl("http://localhost:1234/v1");
    expect(result).toBe(builder);
  });

  test("should chain temperature method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.temperature(0.7);
    expect(result).toBe(builder);
  });

  test("should chain maxTokens method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.maxTokens(2048);
    expect(result).toBe(builder);
  });

  test("should chain timeout method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.timeout(30);
    expect(result).toBe(builder);
  });

  test("should chain maxRetries method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.maxRetries(3);
    expect(result).toBe(builder);
  });

  test("should chain tools method", () => {
    const builder = new LlmClientBuilder();
    const tools = [
      {
        name: "test_tool",
        description: "Test tool",
        parameters: { type: "object" as const, properties: {} },
      },
    ];
    const result = builder.tools(tools);
    expect(result).toBe(builder);
  });

  test("should chain toolChoice method", () => {
    const builder = new LlmClientBuilder();
    const result = builder.toolChoice("auto");
    expect(result).toBe(builder);
  });

  test("should support method chaining", () => {
    const builder = new LlmClientBuilder()
      .provider("openai")
      .model("gpt-4o")
      .apiKey("test-key")
      .temperature(0.7)
      .maxTokens(2048)
      .timeout(30)
      .maxRetries(3);

    expect(builder).toBeInstanceOf(LlmClientBuilder);
  });

  test("should have build method", () => {
    const builder = new LlmClientBuilder();
    expect(typeof builder.build).toBe("function");
  });
});

describe("LlmClient", () => {
  test("should have builder static method", () => {
    expect(LlmClient.builder).toBeDefined();
    expect(typeof LlmClient.builder).toBe("function");
  });

  test("should create builder via static method", () => {
    const builder = LlmClient.builder();
    expect(builder).toBeInstanceOf(LlmClientBuilder);
  });

  liveLlmTest("should build LlmClient", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(client).toBeInstanceOf(LlmClient);
  });

  liveLlmTest("should have complete method", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(typeof client.complete).toBe("function");
  });

  liveLlmTest("should have stream method", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(typeof client.stream).toBe("function");
  });

  liveLlmTest("should have completeText method", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(typeof client.completeText).toBe("function");
  });

  liveLlmTest("should have streamText method", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(typeof client.streamText).toBe("function");
  });

  liveLlmTest("should have providerName getter", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(typeof client.providerName).toBe("string");
  });

  liveLlmTest("should have model getter", async () => {
    const builder = new LlmClientBuilder().provider("lmstudio").model("local-model");
    const client = await builder.build();
    expect(typeof client.model).toBe("string");
  });
});
