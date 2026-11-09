import { describe, expect, test } from "bun:test";
import { z as z4 } from "zod";
import { z as z3 } from "zod/v3";
import { LangChainAdapter } from "../src/adapters/langchain";

const invoke = async () => "ok";

describe("LangChainAdapter with a Zod schema", () => {
  test("converts a Zod 4 schema with its properties", () => {
    const adapter = new LangChainAdapter({
      name: "lookup",
      description: "Look up a city",
      schema: z4.object({ city: z4.string(), days: z4.number().optional() }),
      invoke,
    });

    expect(adapter.parameters).toEqual({
      $schema: "http://json-schema.org/draft-07/schema#",
      type: "object",
      properties: { city: { type: "string" }, days: { type: "number" } },
      required: ["city"],
      additionalProperties: false,
    });
  });

  test("converts a Zod 3 schema through zod-to-json-schema", () => {
    const adapter = new LangChainAdapter({
      name: "lookup",
      description: "Look up a city",
      schema: z3.object({ city: z3.string(), days: z3.number().optional() }),
      invoke,
    });

    expect(adapter.parameters).toEqual({
      $schema: "http://json-schema.org/draft-07/schema#",
      type: "object",
      properties: { city: { type: "string" }, days: { type: "number" } },
      required: ["city"],
      additionalProperties: false,
    });
  });
});
