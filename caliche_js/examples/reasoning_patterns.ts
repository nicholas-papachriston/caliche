/**
 * Caliche Reasoning Patterns Examples
 *
 * Demonstrates advanced reasoning patterns including ReAct, Plan-Execute, and Reflexion.
 *
 * bun run examples/reasoning.ts
 */

import {
  AgentBuilder,
  PlanExecutePattern,
  ReActPattern,
  type ReasoningStep,
  ReflexionPattern,
} from "../src/index";

async function exampleReAct() {
  console.log("=== ReAct (Reasoning + Acting) Pattern ===\n");

  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .enableTool("calculator")
    .build();

  const react = new ReActPattern(agent, 5); // Max 5 iterations

  const result = await react.execute("What is 15 * 23? Show your work.");

  console.log("Final Answer:", result.final_answer);
  console.log("\nReasoning Steps:");
  result.steps.forEach((step: ReasoningStep, i: number) => {
    console.log(`\nStep ${i + 1} [${step.step_type}]:`);
    console.log(step.content);
    if (step.metadata) {
      console.log("Metadata:", step.metadata);
    }
  });

  if (result.token_usage) {
    console.log("\nToken Usage:", result.token_usage);
  }
  console.log();
}

async function examplePlanExecute() {
  console.log("=== Plan-Execute Pattern ===\n");

  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .enableTool("calculator")
    .build();

  const planner = new PlanExecutePattern(agent, 10); // Max 10 steps

  const result = await planner.execute(
    "Create a step-by-step plan for learning Rust programming, then execute the first step."
  );

  console.log("Final Answer:", result.final_answer);
  console.log("\nExecution Steps:");
  result.steps.forEach((step: ReasoningStep, i: number) => {
    console.log(`\nStep ${i + 1} [${step.step_type}]:`);
    console.log(step.content);
  });
  console.log();
}

async function exampleReflexion() {
  console.log("=== Reflexion Pattern ===\n");

  const agent = await new AgentBuilder()
    .provider("openai")
    .model("gpt-4o-mini")
    .apiKey(process.env.OPENAI_API_KEY || "")
    .build();

  const reflexion = new ReflexionPattern(agent, 3); // Max 3 iterations

  const result = await reflexion.execute(
    "Write a haiku about artificial intelligence, then critique and improve it."
  );

  console.log("Final Answer:", result.final_answer);
  console.log("\nReflection Steps:");
  result.steps.forEach((step: ReasoningStep, i: number) => {
    console.log(`\nStep ${i + 1} [${step.step_type}]:`);
    console.log(step.content);
  });
  console.log();
}

async function main() {
  try {
    await exampleReAct();
    await examplePlanExecute();
    await exampleReflexion();
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}
