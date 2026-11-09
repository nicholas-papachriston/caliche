/**
 * Evaluation Framework Example
 *
 * Demonstrates how to evaluate agent outputs using different evaluators
 * and load evaluation datasets.
 *
 * bun run examples/evaluation.ts
 */

import { AgentBuilder, ContainsEvaluator, ExactMatchEvaluator, LLMEvaluator } from "../src/index";

async function main() {
  console.log("Evaluation Framework Example\n");

  // Create an agent to evaluate
  const agent = await new AgentBuilder().provider("openai").model("gpt-4").build();

  // Test cases
  const testCases = [
    {
      input: "What is 2+2?",
      expected: "4",
    },
    {
      input: "What is the capital of France?",
      expected: "Paris",
    },
    {
      input: "Write a hello world program in Python",
      expected: "print",
    },
  ];

  // 1. Exact Match Evaluator
  console.log("1. Exact Match Evaluator");
  const exactEvaluator = ExactMatchEvaluator.create(true);

  for (const testCase of testCases.slice(0, 2)) {
    const response = await agent.chat(testCase.input);
    const result = await exactEvaluator.evaluate(response.response, testCase.expected);
    console.log(`  Input: ${testCase.input}`);
    console.log(`  Expected: ${testCase.expected}`);
    console.log(`  Output: ${response.response.substring(0, 50)}...`);
    console.log(`  Score: ${result.score}, Passed: ${result.passed}\n`);
  }

  // 2. Contains Evaluator
  console.log("2. Contains Evaluator");
  const containsEvaluator = ContainsEvaluator.create(false);

  const codingTestCase = testCases[2];
  if (codingTestCase) {
    const codingResponse = await agent.chat(codingTestCase.input);
    const containsResult = await containsEvaluator.evaluate(
      codingResponse.response,
      codingTestCase.expected
    );
    console.log(`  Input: ${codingTestCase.input}`);
    console.log(`  Expected to contain: ${codingTestCase.expected}`);
    console.log(`  Score: ${containsResult.score}, Passed: ${containsResult.passed}\n`);
  }

  // 3. LLM Evaluator
  console.log("3. LLM Evaluator");
  const llmEvaluator = LLMEvaluator.create(
    "Evaluate if the response is helpful, accurate, and well-formatted.",
    "openai",
    "gpt-4"
  );

  const llmResponse = await agent.chat("Explain quantum computing simply");
  const llmResult = await llmEvaluator.evaluate(
    llmResponse.response,
    "A clear, simple explanation of quantum computing",
    {
      context: "Educational content for beginners",
    }
  );
  console.log(`  Input: Explain quantum computing simply`);
  console.log(`  Score: ${llmResult.score}, Passed: ${llmResult.passed}`);
  if (llmResult.details?.reasoning) {
    console.log(`  Reasoning: ${llmResult.details.reasoning}\n`);
  }

  // 4. Load dataset from file (example)
  console.log("4. Dataset Loading");
  try {
    // This would load from a JSONL file if it exists
    // const dataset = await EvaluationDatasetLoader.fromJsonl("./test_dataset.jsonl");
    // console.log(`  Loaded dataset: ${dataset.name}`);
    // console.log(`  Test cases: ${dataset.test_cases.length}`);
    console.log("  (Dataset loading requires a valid JSONL file)");
  } catch (_error) {
    console.log("  Dataset file not found (this is expected in this example)");
  }

  console.log("\nEvaluation complete!");
}

main().catch(console.error);
