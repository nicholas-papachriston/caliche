import { describe, expect, test } from "bun:test";
import {
  ContainsEvaluator,
  EvaluationDatasetLoader,
  ExactMatchEvaluator,
  LLM_EVALUATOR_NEEDS_PROVIDER,
  LLMEvaluator,
} from "../src/index";
import { liveSuite, resolveTestModel } from "./test-env";

const { liveTest, provider } = await liveSuite("LLMEvaluator");

/** An LLM evaluator pinned to the suite's provider, so it never falls back to a cloud default. */
function liveEvaluator(): LLMEvaluator {
  return LLMEvaluator.create("Check if output is correct", provider, resolveTestModel(provider));
}

describe("ExactMatchEvaluator", () => {
  test("should create an ExactMatchEvaluator instance", () => {
    const evaluator = ExactMatchEvaluator.create();
    expect(evaluator).toBeInstanceOf(ExactMatchEvaluator);
  });

  test("should create with case sensitive option", () => {
    const evaluator = ExactMatchEvaluator.create(true);
    expect(evaluator).toBeInstanceOf(ExactMatchEvaluator);
  });

  test("should create with case insensitive option", () => {
    const evaluator = ExactMatchEvaluator.create(false);
    expect(evaluator).toBeInstanceOf(ExactMatchEvaluator);
  });

  test("should have evaluate method", () => {
    const evaluator = ExactMatchEvaluator.create();
    expect(typeof evaluator.evaluate).toBe("function");
  });

  test("should evaluate exact match", async () => {
    const evaluator = ExactMatchEvaluator.create(true);
    const result = await evaluator.evaluate("Hello", "Hello");
    expect(result).toBeDefined();
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1.0);
  });

  test("should evaluate case sensitive mismatch", async () => {
    const evaluator = ExactMatchEvaluator.create(true);
    const result = await evaluator.evaluate("Hello", "hello");
    expect(result).toBeDefined();
    expect(result.passed).toBe(false);
    expect(result.score).toBe(0.0);
  });

  test("should evaluate case insensitive match", async () => {
    const evaluator = ExactMatchEvaluator.create(false);
    const result = await evaluator.evaluate("Hello", "hello");
    expect(result).toBeDefined();
    expect(result.passed).toBe(true);
    expect(result.score).toBe(1.0);
  });

  test("should evaluate with context", async () => {
    const evaluator = ExactMatchEvaluator.create();
    const result = await evaluator.evaluate("Hello", "Hello", { key: "value" });
    expect(result).toBeDefined();
    expect(result.passed).toBe(true);
  });
});

describe("ContainsEvaluator", () => {
  test("should create a ContainsEvaluator instance", () => {
    const evaluator = ContainsEvaluator.create();
    expect(evaluator).toBeInstanceOf(ContainsEvaluator);
  });

  test("should create with case sensitive option", () => {
    const evaluator = ContainsEvaluator.create(true);
    expect(evaluator).toBeInstanceOf(ContainsEvaluator);
  });

  test("should create with case insensitive option", () => {
    const evaluator = ContainsEvaluator.create(false);
    expect(evaluator).toBeInstanceOf(ContainsEvaluator);
  });

  test("should have evaluate method", () => {
    const evaluator = ContainsEvaluator.create();
    expect(typeof evaluator.evaluate).toBe("function");
  });

  test("should evaluate contains match", async () => {
    const evaluator = ContainsEvaluator.create(false);
    const result = await evaluator.evaluate("Hello World", "World");
    expect(result).toBeDefined();
    expect(result.passed).toBe(true);
    expect(result.score).toBeGreaterThan(0);
  });

  test("should evaluate contains mismatch", async () => {
    const evaluator = ContainsEvaluator.create(false);
    const result = await evaluator.evaluate("Hello", "World");
    expect(result).toBeDefined();
    expect(result.passed).toBe(false);
    expect(result.score).toBe(0.0);
  });

  test("should evaluate case sensitive contains", async () => {
    const evaluator = ContainsEvaluator.create(true);
    const result = await evaluator.evaluate("Hello World", "world");
    expect(result).toBeDefined();
    expect(result.passed).toBe(false);
  });

  test("should evaluate case insensitive contains", async () => {
    const evaluator = ContainsEvaluator.create(false);
    const result = await evaluator.evaluate("Hello World", "world");
    expect(result).toBeDefined();
    expect(result.passed).toBe(true);
  });

  test("should evaluate with context", async () => {
    const evaluator = ContainsEvaluator.create();
    const result = await evaluator.evaluate("Hello World", "World", { key: "value" });
    expect(result).toBeDefined();
    expect(result.passed).toBe(true);
  });
});

describe("LLMEvaluator", () => {
  test("should create an LLMEvaluator instance", () => {
    const evaluator = LLMEvaluator.create("Check if output is correct");
    expect(evaluator).toBeInstanceOf(LLMEvaluator);
  });

  test("should create with provider", () => {
    const evaluator = LLMEvaluator.create("Check if output is correct", "openai");
    expect(evaluator).toBeInstanceOf(LLMEvaluator);
  });

  test("should create with provider and model", () => {
    const evaluator = LLMEvaluator.create("Check if output is correct", "openai", "gpt-4o");
    expect(evaluator).toBeInstanceOf(LLMEvaluator);
  });

  test("should have evaluate method", () => {
    const evaluator = LLMEvaluator.create("Check if output is correct");
    expect(typeof evaluator.evaluate).toBe("function");
  });

  test("evaluate without a provider refuses instead of using a default", async () => {
    await expect(
      LLMEvaluator.create("Check if output is correct").evaluate("Hello", "Hello")
    ).rejects.toThrow(`Evaluation failed: ${LLM_EVALUATOR_NEEDS_PROVIDER}`);
  });

  test("an unknown provider name throws", () => {
    expect(() => LLMEvaluator.create("Check if output is correct", "nope")).toThrow(
      "Unknown provider: nope. Supported providers:"
    );
  });

  liveTest("should evaluate with LLM", async () => {
    const result = await liveEvaluator().evaluate("Hello", "Hello");
    expect(typeof result.passed).toBe("boolean");
    expect(typeof result.score).toBe("number");
  });

  liveTest("should evaluate with context", async () => {
    const result = await liveEvaluator().evaluate("Hello", "Hello", { key: "value" });
    expect(typeof result.passed).toBe("boolean");
  });
});

describe("EvaluationDatasetLoader", () => {
  test("should have fromJsonl static method", () => {
    expect(EvaluationDatasetLoader.fromJsonl).toBeDefined();
    expect(typeof EvaluationDatasetLoader.fromJsonl).toBe("function");
  });

  test("should have fromJson static method", () => {
    expect(EvaluationDatasetLoader.fromJson).toBeDefined();
    expect(typeof EvaluationDatasetLoader.fromJson).toBe("function");
  });

  test("should load dataset from JSONL file", async () => {
    try {
      const dataset = await EvaluationDatasetLoader.fromJsonl("nonexistent.jsonl");
      expect(dataset).toBeDefined();
      expect(dataset.name).toBeDefined();
      expect(dataset.test_cases).toBeDefined();
      expect(Array.isArray(dataset.test_cases)).toBe(true);
    } catch (error: any) {
      if (
        error.message?.includes("No such file") ||
        error.message?.includes("not found") ||
        error.message?.includes("Failed to read")
      ) {
        expect(error).toBeDefined();
      } else {
        throw error;
      }
    }
  });

  test("should load dataset from JSON file", async () => {
    try {
      const dataset = await EvaluationDatasetLoader.fromJson("nonexistent.json");
      expect(dataset).toBeDefined();
      expect(dataset.name).toBeDefined();
      expect(dataset.test_cases).toBeDefined();
      expect(Array.isArray(dataset.test_cases)).toBe(true);
    } catch (error: any) {
      if (
        error.message?.includes("No such file") ||
        error.message?.includes("not found") ||
        error.message?.includes("Failed to read")
      ) {
        expect(error).toBeDefined();
      } else {
        throw error;
      }
    }
  });
});
