import type { Agent } from "../agent/agent";
import {
  type NativeLLMChain,
  type NativeMapReduceChain,
  type NativeSequentialChain,
  native,
} from "../native/module";
import type { ChainResult } from "../streaming/events";
import type { ChainInput } from "../types";

/**
 * LLM Chain for combining prompt templates with LLM calls
 *
 * @example
 * ```typescript
 * const chain = new LLMChain(agent, "Summarize: {text}");
 * const result = await chain.run({ text: "Long document..." });
 * console.log(result.output);
 * ```
 */
export class LLMChain {
  private _chain: NativeLLMChain | undefined;
  private readonly _ready: Promise<NativeLLMChain>;

  constructor(agent: Agent, promptTemplate: string) {
    this._ready = Promise.resolve(
      native.LlmChainFromAgent(agent.agent, promptTemplate) as
        | NativeLLMChain
        | Promise<NativeLLMChain>
    ).then((chain) => {
      this._chain = chain;
      return chain;
    });
  }

  /**
   * Wait until the native chain handle is ready.
   */
  async ready(): Promise<this> {
    await this._ready;
    return this;
  }

  /**
   * Get the internal native chain (for internal use)
   */
  get internalChain(): NativeLLMChain {
    if (!this._chain) {
      throw new Error("LLMChain is not ready yet; await chain.ready() or chain.run() first");
    }
    return this._chain;
  }

  /**
   * Run the chain with input variables
   *
   * @param input - Input variables as an object
   * @returns Promise resolving to the chain result
   */
  async run(input: ChainInput): Promise<ChainResult> {
    const chain = await this._ready;
    return await chain.run(input);
  }
}

/**
 * Sequential Chain for executing multiple chains in sequence
 *
 * @example
 * ```typescript
 * const seqChain = new SequentialChain();
 * seqChain.addChain("analyze", analyzeChain);
 * seqChain.addChain("summarize", summarizeChain);
 * const result = await seqChain.run({ input: "data" });
 * ```
 */
export class SequentialChain {
  private readonly _chain: NativeSequentialChain;
  private readonly _pending: Promise<void>[] = [];

  constructor() {
    this._chain = native.createSequentialChain();
  }

  /**
   * Add a chain to the sequence
   *
   * @param name - Name for this chain step
   * @param chain - The LLMChain to add
   */
  addChain(name: string, chain: LLMChain): void {
    this._pending.push(
      chain.ready().then(() => {
        this._chain.addChainInstance(name, chain.internalChain);
      })
    );
  }

  /**
   * Run the sequential chain
   *
   * @param input - Initial input variables
   * @returns Promise resolving to the final chain result
   */
  async run(input: ChainInput): Promise<ChainResult> {
    await Promise.all(this._pending);
    this._pending.length = 0;
    return await this._chain.run(input);
  }
}

/**
 * Map-Reduce Chain for parallel processing
 *
 * @example
 * ```typescript
 * const mapReduce = new MapReduceChain(
 *   mapAgent, "Summarize: {input}",
 *   reduceAgent, "Combine: {output}"
 * );
 * const result = await mapReduce.run([
 *   { input: "doc1" },
 *   { input: "doc2" }
 * ]);
 * ```
 */
export class MapReduceChain {
  private readonly _ready: Promise<NativeMapReduceChain>;

  constructor(mapAgent: Agent, mapPrompt: string, reduceAgent: Agent, reducePrompt: string) {
    this._ready = Promise.resolve(
      native.MapReduceChainFromAgents(
        mapAgent.agent,
        mapPrompt,
        reduceAgent.agent,
        reducePrompt
      ) as NativeMapReduceChain | Promise<NativeMapReduceChain>
    );
  }

  /**
   * Wait until the native chain handle is ready.
   */
  async ready(): Promise<this> {
    await this._ready;
    return this;
  }

  /**
   * Run the map-reduce chain
   *
   * @param items - Array of input items to process
   * @returns Promise resolving to the reduced result
   */
  async run(items: Array<ChainInput>): Promise<ChainResult> {
    const chain = await this._ready;
    return await chain.run(items);
  }
}
