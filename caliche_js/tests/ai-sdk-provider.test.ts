import { describe, expect, it } from "bun:test";
import {
  __testOnly_clientToolMarker,
  __testOnly_collectFinalStreamMetadata,
  __testOnly_filterCallOptionSchemas,
  __testOnly_parseModelId,
  __testOnly_toFinishReason,
  __testOnly_toV3Usage,
  CalicheLanguageModel,
  caliche,
} from "../src/adapters/ai-sdk-provider";

describe("caliche LanguageModelV3", () => {
  it("creates a v3 language model", () => {
    const model = caliche("openai/gpt-4o-mini", { apiKey: "test" });
    expect(model).toBeInstanceOf(CalicheLanguageModel);
    expect(model.specificationVersion).toBe("v3");
    expect(model.provider).toBe("caliche");
    expect(model.modelId).toBe("openai/gpt-4o-mini");
  });

  it("parses provider/model ids", () => {
    expect(__testOnly_parseModelId("openrouter/openai/gpt-4o-mini")).toEqual({
      provider: "openrouter",
      model: "openai/gpt-4o-mini",
    });
    expect(__testOnly_parseModelId("gpt-4o-mini")).toEqual({
      provider: "openai",
      model: "gpt-4o-mini",
    });
  });

  it("parses anthropic and azure-openai model ids", () => {
    expect(__testOnly_parseModelId("anthropic/claude-sonnet-4-20250514")).toEqual({
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
    });
    expect(__testOnly_parseModelId("azure-openai/my-deployment")).toEqual({
      provider: "azure-openai",
      model: "my-deployment",
    });
  });

  it("maps null usage to empty v3 shape", () => {
    expect(__testOnly_toV3Usage(null)).toEqual({
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
    });
    expect(__testOnly_toV3Usage(undefined)).toEqual(__testOnly_toV3Usage(null));
  });

  it("maps usage and finish reasons to v3 shapes", () => {
    expect(
      __testOnly_toV3Usage({ prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 })
    ).toEqual({
      inputTokens: {
        total: 3,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: 5,
        text: 5,
        reasoning: undefined,
      },
      raw: {
        prompt_tokens: 3,
        completion_tokens: 5,
        total_tokens: 8,
      },
    });

    expect(__testOnly_toFinishReason("stop", false)).toEqual({ unified: "stop", raw: "stop" });
    expect(__testOnly_toFinishReason(undefined, true).unified).toBe("tool-calls");
    expect(__testOnly_toFinishReason("length", false).unified).toBe("length");
  });

  it("coalesces split stream metadata into one final state", () => {
    expect(
      __testOnly_collectFinalStreamMetadata([
        {
          usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
        },
        { finish_reason: "stop" },
      ])
    ).toEqual({
      sawMetadata: true,
      usage: { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 },
      finishReason: "stop",
    });
  });

  it("returns an empty MCP snapshot by default", async () => {
    const model = caliche("openai/gpt-4o-mini");
    const snapshot = await model.getMcpSnapshot();
    expect(snapshot.tools).toEqual([]);
    expect(snapshot.resources).toEqual([]);
    expect(snapshot.prompts).toEqual([]);
  });

  it("returns structured client-tool markers from call-option stubs", () => {
    expect(__testOnly_clientToolMarker("search", { q: "weather" })).toEqual({
      __calicheClientTool: true,
      name: "search",
      args: { q: "weather" },
    });
  });

  it("deduplicates call-option schemas against reserved tool names", () => {
    const schemas = [
      {
        name: "search",
        description: "Search",
        parameters: { type: "object" as const, properties: {} },
      },
      {
        name: "calc",
        description: "Calc",
        parameters: { type: "object" as const, properties: {} },
      },
      {
        name: "mcp__tool",
        description: "MCP",
        parameters: { type: "object" as const, properties: {} },
      },
    ];
    const filtered = __testOnly_filterCallOptionSchemas(schemas, new Set(["search", "mcp__tool"]));
    expect(filtered.map((schema) => schema.name)).toEqual(["calc"]);
  });
});
