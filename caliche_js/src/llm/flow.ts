import type { Agent } from "../agent/agent";
import {
  type NativeAgentFlow,
  type NativeFlowBuilder,
  type NativeSharedState,
  native,
} from "../native/module";
import type { FlowExecutionResult } from "../streaming/events";
import type { JsonRecord, JsonValue } from "../types";

/**
 * Shared state container for agents
 *
 * Allows multiple agents to share structured application state.
 *
 * @example
 * ```typescript
 * const state = new SharedState();
 * await state.set("/order_id", "ORD-123");
 * const orderId = await state.get("/order_id");
 * ```
 */
export class SharedState {
  private readonly _state: NativeSharedState;

  /**
   * Create a new shared state
   */
  constructor() {
    this._state = new native.SharedState();
  }

  /**
   * Get a value from the state using a JSON pointer path
   *
   * @param path - JSON pointer path (e.g., "/order_id" or "/customer/name")
   * @returns Promise resolving to the value, or null if not found
   */
  async get(path: string): Promise<any> {
    const jsonString = await this._state.get(path);
    if (jsonString == null) {
      return null;
    }
    return JSON.parse(jsonString);
  }

  /**
   * Set a value at a JSON pointer path
   *
   * @param path - JSON pointer path (e.g., "/order_id" or "/customer/name")
   * @param value - The value to set (must be JSON-serializable)
   */
  async set(path: string, value: JsonValue): Promise<void> {
    const jsonString = JSON.stringify(value);
    await this._state.set(path, jsonString);
  }

  /**
   * Get the entire state as an object
   *
   * @returns Promise resolving to the entire state object
   */
  async getAll(): Promise<JsonRecord> {
    const jsonString = await this._state.getAll();
    return JSON.parse(jsonString);
  }
}

/**
 * Builder for constructing agent flows
 *
 * @example
 * ```typescript
 * const flow = new FlowBuilder()
 *   .addNode("researcher", researchAgent)
 *   .addNode("writer", writerAgent)
 *   .setEntryPoint("researcher")
 *   .build();
 * ```
 */
export class FlowBuilder {
  private readonly _builder: NativeFlowBuilder;

  /**
   * Create a new flow builder
   */
  constructor() {
    this._builder = new native.FlowBuilder();
  }

  /**
   * Add an agent node to the flow
   *
   * @param name - Unique identifier for this agent node
   * @param agent - The agent instance
   * @returns This builder for method chaining
   */
  addNode(name: string, agent: Agent): this {
    this._builder.addNode(name, agent.agent);
    return this;
  }

  /**
   * Set the entry point for the flow
   *
   * @param name - Name of the entry point node
   * @returns This builder for method chaining
   */
  setEntryPoint(name: string): this {
    this._builder.setEntryPoint(name);
    return this;
  }

  /**
   * Build the agent flow
   *
   * @returns The configured AgentFlow
   */
  build(): AgentFlow {
    const nativeFlow = this._builder.build();
    return new AgentFlow(nativeFlow);
  }
}

/**
 * Agent flow for executing multi-agent workflows
 *
 * @example
 * ```typescript
 * const result = await flow.run("Research and write a summary");
 * console.log(result.final_response);
 * ```
 */
export class AgentFlow {
  private readonly _flow: NativeAgentFlow;

  /**
   * @internal
   */
  constructor(nativeFlow: NativeAgentFlow) {
    this._flow = nativeFlow;
  }

  /**
   * Run the flow with an initial input message
   *
   * @param initialInput - The initial message to start the flow
   * @returns Promise resolving to the flow execution result
   */
  async run(initialInput: string): Promise<FlowExecutionResult> {
    return await native.agentFlowRun(this._flow, initialInput);
  }

  /**
   * Generate a Mermaid diagram representation of the flow
   *
   * @returns Mermaid flowchart syntax string
   */
  toMermaid(): string {
    return native.agentFlowToMermaid(this._flow);
  }

  /**
   * Get the entry point node name
   *
   * @returns Entry point node name
   */
  get entryPoint(): string {
    return native.agentFlowEntryPoint(this._flow);
  }

  /**
   * Get all node names in the flow
   *
   * @returns Array of node names
   */
  get nodeNames(): string[] {
    return native.agentFlowNodeNames(this._flow);
  }
}
