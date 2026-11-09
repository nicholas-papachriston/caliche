import type { Agent } from "../agent/agent";
import {
  type NativePlanExecutePattern,
  type NativeReActPattern,
  type NativeReflexionPattern,
  type NativeSelfAskPattern,
  type NativeTreeOfThoughtsPattern,
  native,
} from "../native/module";
import type { ReasoningResult } from "../streaming/events";

export class ReActPattern {
  private readonly _pattern: Promise<NativeReActPattern>;

  constructor(agent: Agent, maxIterations: number) {
    this._pattern = native.ReActPattern.new(agent.provider, agent.model, maxIterations);
  }

  /**
   * Execute the ReAct pattern
   *
   * @param query - The query to process
   * @returns Promise resolving to the reasoning result
   */
  async execute(query: string): Promise<ReasoningResult> {
    return await (await this._pattern).execute(query);
  }
}

/**
 * Plan-and-Execute pattern
 *
 * Separates planning from execution for complex workflows.
 *
 * @example
 * ```typescript
 * const planExecute = new PlanExecutePattern(agent, 10);
 * const result = await planExecute.execute("Research and write a report");
 * ```
 */
export class PlanExecutePattern {
  private readonly _pattern: Promise<NativePlanExecutePattern>;

  constructor(agent: Agent, maxIterations: number) {
    this._pattern = native.PlanExecutePattern.new(agent.provider, agent.model, maxIterations);
  }

  /**
   * Execute the Plan-Execute pattern
   *
   * @param query - The query to process
   * @returns Promise resolving to the reasoning result
   */
  async execute(query: string): Promise<ReasoningResult> {
    return await (await this._pattern).execute(query);
  }
}

/**
 * Reflexion (Self-Critique) pattern
 *
 * Uses self-reflection and error correction.
 *
 * @example
 * ```typescript
 * const reflexion = new ReflexionPattern(agent, 3);
 * const result = await reflexion.execute("Solve this problem");
 * ```
 */
export class ReflexionPattern {
  private readonly _pattern: Promise<NativeReflexionPattern>;

  constructor(agent: Agent, maxIterations: number) {
    this._pattern = native.ReflexionPattern.new(agent.provider, agent.model, maxIterations);
  }

  /**
   * Execute the Reflexion pattern
   *
   * @param query - The query to process
   * @returns Promise resolving to the reasoning result
   */
  async execute(query: string): Promise<ReasoningResult> {
    return await (await this._pattern).execute(query);
  }
}

/**
 * Self-Ask pattern
 *
 * Recursively decomposes complex questions into simpler sub-questions.
 *
 * @example
 * ```typescript
 * const selfAsk = new SelfAskPattern(agent, 5);
 * const result = await selfAsk.execute("What is the capital of France?");
 * ```
 */
export class SelfAskPattern {
  private readonly _pattern: Promise<NativeSelfAskPattern>;

  constructor(agent: Agent, maxDepth: number) {
    this._pattern = native.SelfAskPattern.new(agent.provider, agent.model, maxDepth);
  }

  /**
   * Execute the Self-Ask pattern
   *
   * @param query - The query to process
   * @returns Promise resolving to the reasoning result
   */
  async execute(query: string): Promise<ReasoningResult> {
    return await (await this._pattern).execute(query);
  }
}

/**
 * Tree of Thoughts pattern
 *
 * Explores multiple reasoning paths in parallel using BFS/DFS search.
 *
 * @example
 * ```typescript
 * const tot = new TreeOfThoughtsPattern(agent, 3, 3);
 * const result = await tot.execute("Solve this complex problem");
 * ```
 */
export class TreeOfThoughtsPattern {
  private readonly _pattern: Promise<NativeTreeOfThoughtsPattern>;

  constructor(agent: Agent, maxDepth: number, branchingFactor: number = 3) {
    this._pattern = native.TreeOfThoughtsPattern.new(
      agent.provider,
      agent.model,
      maxDepth,
      branchingFactor
    );
  }

  /**
   * Execute the Tree of Thoughts pattern
   *
   * @param query - The query to process
   * @returns Promise resolving to the reasoning result
   */
  async execute(query: string): Promise<ReasoningResult> {
    return await (await this._pattern).execute(query);
  }
}
