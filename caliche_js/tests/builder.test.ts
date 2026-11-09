import { describe, expect, test } from "bun:test";
import { AgentBuilder, MutationStyle } from "../src/index";

describe("AgentBuilder", () => {
  test("should create a builder instance", () => {
    const builder = new AgentBuilder();
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should chain provider method", () => {
    const builder = new AgentBuilder();
    const result = builder.provider("openai");
    expect(result).toBe(builder);
  });

  test("should chain model method", () => {
    const builder = new AgentBuilder();
    const result = builder.model("gpt-4");
    expect(result).toBe(builder);
  });

  test("should chain systemPrompt method", () => {
    const builder = new AgentBuilder();
    const result = builder.systemPrompt("You are a helpful assistant.");
    expect(result).toBe(builder);
  });

  test("should chain enableTool method", () => {
    const builder = new AgentBuilder();
    const result = builder.enableTool("calculator");
    expect(result).toBe(builder);
  });

  test("should chain enableTools method", () => {
    const builder = new AgentBuilder();
    const result = builder.enableTools(["calculator", "web_search"]);
    expect(result).toBe(builder);
  });

  test("should chain enableAllDefaultTools method", () => {
    const builder = new AgentBuilder();
    const result = builder.enableAllDefaultTools();
    expect(result).toBe(builder);
  });

  test("should chain mutationStyle, enableToolInspection, and enableExtendedDefaultTools", () => {
    const builder = new AgentBuilder()
      .mutationStyle(MutationStyle.applyPatch())
      .enableToolInspection("/tmp/ws")
      .enableExtendedDefaultTools();
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should chain temperature method", () => {
    const builder = new AgentBuilder();
    const result = builder.temperature(0.7);
    expect(result).toBe(builder);
  });

  test("should chain maxTokens method", () => {
    const builder = new AgentBuilder();
    const result = builder.maxTokens(2048);
    expect(result).toBe(builder);
  });

  test("should chain topP method", () => {
    const builder = new AgentBuilder();
    const result = builder.topP(0.9);
    expect(result).toBe(builder);
  });

  test("should chain apiKey method", () => {
    const builder = new AgentBuilder();
    const result = builder.apiKey("test-openai-key");
    expect(result).toBe(builder);
  });

  test("should chain awsCredentials method", () => {
    const builder = new AgentBuilder();
    const result = builder.awsCredentials("key", "secret", "us-east-1");
    expect(result).toBe(builder);
  });

  test("should chain baseUrl method", () => {
    const builder = new AgentBuilder();
    const result = builder.baseUrl("http://localhost:1234/v1");
    expect(result).toBe(builder);
  });

  test("should chain bufferMemory method", () => {
    const builder = new AgentBuilder();
    const result = builder.bufferMemory({
      max_tokens: 4096,
      min_messages: 4,
      preserve_system: true,
      summary_placeholder: true,
    });
    expect(result).toBe(builder);
  });

  test("should chain bufferMemory with defaults", () => {
    const builder = new AgentBuilder();
    const result = builder.bufferMemory({});
    expect(result).toBe(builder);
  });

  test("should chain summaryMemory method", () => {
    const builder = new AgentBuilder();
    const result = builder.summaryMemory({
      max_tokens: 8192,
      summarize_threshold_pct: 0.8,
      chunk_size: 12,
      max_summaries: 8,
    });
    expect(result).toBe(builder);
  });

  test("should chain summaryMemory with defaults", () => {
    const builder = new AgentBuilder();
    const result = builder.summaryMemory({});
    expect(result).toBe(builder);
  });

  test("should chain entityMemory method", () => {
    const builder = new AgentBuilder();
    const result = builder.entityMemory({
      max_tokens: 6144,
      max_entities: 128,
      extraction_interval: 3,
      similarity_threshold: 0.65,
    });
    expect(result).toBe(builder);
  });

  test("should chain entityMemory with defaults", () => {
    const builder = new AgentBuilder();
    const result = builder.entityMemory({});
    expect(result).toBe(builder);
  });

  test("should chain addTool method", () => {
    const builder = new AgentBuilder();
    const tool = {
      name: "test_tool",
      description: "Test tool",
      parameters: { type: "object" as const, properties: {} },
      execute: async () => ({ success: true, content: "test" }),
    };
    const result = builder.addTool(tool);
    expect(result).toBe(builder);
  });

  test("should chain persistMemoryWith method", () => {
    const builder = new AgentBuilder();
    const persistence = {
      save_snapshot: async () => {},
      load_snapshot: async () => undefined,
      clear_snapshot: async () => {},
    };
    const result = builder.persistMemoryWith(persistence, "conv-1", true);
    expect(result).toBe(builder);
  });

  test("should support method chaining", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4")
      .systemPrompt("Test")
      .temperature(0.7)
      .maxTokens(2048)
      .enableTool("calculator")
      .bufferMemory({ max_tokens: 4096 });

    expect(builder).toBeInstanceOf(AgentBuilder);
  });
});
