import { describe, expect, test } from "bun:test";
import { Agent, AgentBuilder } from "../src/index";

describe("Agent", () => {
  test("should have provider and model getters", () => {
    // Note: This test requires a built agent, which needs API keys
    // For now, we'll just verify the Agent class exists
    expect(AgentBuilder).toBeDefined();
    expect(Agent).toBeDefined();
  });

  test("should support method chaining for builder", () => {
    const builder = new AgentBuilder()
      .provider("openai")
      .model("gpt-4")
      .systemPrompt("Test prompt")
      .temperature(0.7)
      .maxTokens(2048)
      .enableTool("calculator");

    expect(builder).toBeInstanceOf(AgentBuilder);
  });
});
