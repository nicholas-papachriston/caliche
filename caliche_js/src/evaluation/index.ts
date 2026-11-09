import {
  type NativeContainsEvaluator,
  type NativeExactMatchEvaluator,
  type NativeLLMEvaluator,
  native,
} from "../native/module";
import type { ContextUpdate, Metadata } from "../types";

/**
 * Evaluation result
 */
export interface EvaluationResult {
  readonly score: number;
  readonly passed: boolean;
  readonly details: Record<string, any>;
  readonly error?: string;
}

/**
 * Test case in an evaluation dataset
 */
export interface TestCase {
  readonly input: string;
  readonly expected_output: string;
  readonly context?: ContextUpdate;
  readonly metadata?: Metadata;
}

/**
 * Evaluation dataset
 */
export interface EvaluationDataset {
  readonly name: string;
  readonly description?: string;
  readonly test_cases: readonly TestCase[];
}

/**
 * Exact match evaluator
 */
export class ExactMatchEvaluator {
  private readonly _evaluator: NativeExactMatchEvaluator;

  private constructor(evaluator: NativeExactMatchEvaluator) {
    this._evaluator = evaluator;
  }

  /**
   * Create an exact match evaluator
   */
  static create(caseSensitive: boolean = true): ExactMatchEvaluator {
    return new ExactMatchEvaluator(native.createExactMatchEvaluator(caseSensitive));
  }

  /**
   * Evaluate an output against expected output
   */
  async evaluate(
    output: string,
    expected: string,
    context?: ContextUpdate
  ): Promise<EvaluationResult> {
    return await this._evaluator.evaluate(output, expected, context);
  }
}

/**
 * Contains evaluator
 */
export class ContainsEvaluator {
  private readonly _evaluator: NativeContainsEvaluator;

  private constructor(evaluator: NativeContainsEvaluator) {
    this._evaluator = evaluator;
  }

  /**
   * Create a contains evaluator
   */
  static create(caseSensitive: boolean = false): ContainsEvaluator {
    return new ContainsEvaluator(native.createContainsEvaluator(caseSensitive));
  }

  /**
   * Evaluate an output against expected output
   */
  async evaluate(
    output: string,
    expected: string,
    context?: ContextUpdate
  ): Promise<EvaluationResult> {
    return await this._evaluator.evaluate(output, expected, context);
  }
}

/** Thrown by LLMEvaluator.evaluate when the evaluator was created without a provider. */
export const LLM_EVALUATOR_NEEDS_PROVIDER =
  "LLMEvaluator needs a provider and does not pick one from the environment";

/**
 * LLM-based evaluator
 */
export class LLMEvaluator {
  private readonly _evaluator: NativeLLMEvaluator;

  private constructor(evaluator: NativeLLMEvaluator) {
    this._evaluator = evaluator;
  }

  /**
   * Create an LLM-based evaluator.
   *
   * `provider` is optional here but required by `evaluate`, which throws
   * {@link LLM_EVALUATOR_NEEDS_PROVIDER} instead of picking one from the environment.
   */
  static create(criteria: string, provider?: string, model?: string): LLMEvaluator {
    return new LLMEvaluator(native.createLLMEvaluator(criteria, provider, model));
  }

  /**
   * Evaluate an output against expected output
   */
  async evaluate(
    output: string,
    expected: string,
    context?: ContextUpdate
  ): Promise<EvaluationResult> {
    return await this._evaluator.evaluate(output, expected, context);
  }
}

/**
 * Evaluation dataset loader
 */
export class EvaluationDatasetLoader {
  /**
   * Load dataset from JSONL file
   */
  static async fromJsonl(path: string): Promise<EvaluationDataset> {
    return await native.loadDatasetFromJsonl(path);
  }

  /**
   * Load dataset from JSON file
   */
  static async fromJson(path: string): Promise<EvaluationDataset> {
    return await native.loadDatasetFromJson(path);
  }
}
