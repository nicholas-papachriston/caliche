import { type Agent, AgentBuilder } from "@caliche/core";
import type { Effect } from "effect/Effect";
import { expectError, expectType } from "tsd";

const builder = new AgentBuilder()
  .provider("openai")
  .model("gpt-4o")
  .apiKeyFor("openai", "test-openai-key");
expectType<Promise<Agent>>(builder.build());
expectType<Effect<Agent, Error, never>>(builder.buildEffect());

async function checkBuildInference(): Promise<void> {
  const agent = await new AgentBuilder().provider("openai").model("gpt-4o").build();
  expectType<Agent>(agent);
}
void checkBuildInference;

declare const agent: Agent;

const typedSchema = {
  type: "object",
  properties: {
    query: { type: "string" },
    limit: { type: "integer" },
  },
  required: ["query", "limit"],
} as const;

agent.addTool("typed_search", "Search with inferred args", typedSchema, async (args) => {
  expectType<string>(args.query);
  expectType<number>(args.limit);
  return { success: true, content: `${args.query}:${args.limit}` };
});

const dynamicSchema = '{"type":"object","properties":{"input":{"type":"string"}}}';
agent.addTool("dynamic_tool", "Unknown args by default", dynamicSchema, async (args) => {
  expectType<unknown>(args);
  return "ok";
});

expectError(
  agent.addTool("invalid_tool", "Should fail typed inference", typedSchema, async (args) => {
    return { success: true, content: args.missingField };
  })
);
