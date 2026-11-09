import {
  type NativeWorkflow,
  type NativeWorkflowBuilder,
  type NativeWorkflowEngine,
  native,
} from "../native/module";
import type { ChainOutput, JsonRecord } from "../types";

export type MergeStrategy = "concatenate" | "vote" | "first_complete" | "all_complete" | string;

/**
 * Workflow execution state
 */
export type WorkflowState = "pending" | "running" | "paused" | "completed" | "failed" | "cancelled";

/**
 * Pending interrupt information
 */
export interface PendingInterrupt {
  readonly interrupt_id: string;
  readonly execution_id: string;
  readonly node_id: string;
  readonly node_name: string;
  readonly message: string;
  readonly required_fields?: readonly string[];
}

/**
 * Workflow execution result
 */
export interface WorkflowExecutionResult {
  readonly state: WorkflowState;
  readonly output: ChainOutput;
  readonly error?: string;
}

/**
 * Workflow builder for creating DAG-based workflows
 */
export class WorkflowBuilder {
  private readonly _builder: NativeWorkflowBuilder;

  private constructor(builder: NativeWorkflowBuilder) {
    this._builder = builder;
  }

  /**
   * Create a new workflow builder
   */
  static create(name: string = "workflow"): WorkflowBuilder {
    return new WorkflowBuilder(native.createWorkflowBuilder(name));
  }

  /**
   * Add an agent node to the workflow
   */
  addAgent(name: string, promptTemplate: string): this {
    this._builder.addAgent(name, promptTemplate);
    return this;
  }

  /**
   * Add an agent node with a specific capability
   */
  addAgentWithCapability(name: string, capability: string, promptTemplate: string): this {
    this._builder.addAgentWithCapability(name, capability, promptTemplate);
    return this;
  }

  /**
   * Add a condition node for branching
   */
  addCondition(name: string, predicate: string): this {
    this._builder.addCondition(name, predicate);
    return this;
  }

  /**
   * Add a parallel execution node (merge strategy `all_complete`)
   */
  addParallel(branches: Array<{ name: string; prompt: string }>): this {
    this._builder.addParallel(branches);
    return this;
  }

  /**
   * Fan out to parallel branches without merging; call `collectMerge` or `addMerge` next.
   */
  fanOut(branches: Array<{ name: string; prompt: string }>): this {
    this._builder.fanOut(branches);
    return this;
  }

  /**
   * Complete a pending fan-out with a merge node and strategy.
   */
  collectMerge(mergeNodeName: string, strategy: MergeStrategy): this {
    this._builder.collectMerge(mergeNodeName, strategy);
    return this;
  }

  /**
   * Parallel branches with an explicit merge strategy (single call).
   */
  addParallelWithMergeStrategy(
    branches: Array<{ name: string; prompt: string }>,
    strategy: MergeStrategy
  ): this {
    this._builder.addParallelWithMergeStrategy(branches, strategy);
    return this;
  }

  /**
   * Add a merge node for a pending fan-out (requires prior `fanOut`).
   */
  addMerge(name: string, strategy: MergeStrategy): this {
    this._builder.addMerge(name, strategy);
    return this;
  }

  /**
   * Add a loop node for iterative execution
   */
  addLoop(name: string, maxIterations: number, condition: string, bodyPrompt: string): this {
    this._builder.addLoop(name, maxIterations, condition, bodyPrompt);
    return this;
  }

  /**
   * Add a human-in-the-loop interrupt point
   */
  addHumanInterrupt(name: string, message: string, requiredFields?: string[]): this {
    this._builder.addHumanInterrupt(name, message, requiredFields);
    return this;
  }

  /**
   * Connect two nodes in the workflow
   */
  connectNodes(from: string, to: string, condition?: string): this {
    this._builder.connectNodes(from, to, condition);
    return this;
  }

  /**
   * Set the entry point of the workflow
   */
  setEntryPoint(name: string): this {
    this._builder.setEntryPoint(name);
    return this;
  }

  /**
   * Build the workflow
   */
  async build(): Promise<Workflow> {
    const workflow = this._builder.build();
    return new Workflow(workflow);
  }
}

/**
 * Workflow instance
 */
export class Workflow {
  private readonly _workflow: NativeWorkflow;

  constructor(workflow: NativeWorkflow) {
    this._workflow = workflow;
  }

  /**
   * Get workflow ID (for internal use)
   */
  get internalWorkflow(): NativeWorkflow {
    return this._workflow;
  }

  /**
   * Get the workflow ID
   */
  getId(): string {
    return this._workflow.id();
  }

  /**
   * Serialize workflow definition to JSON (node graph inspection / tests).
   */
  toJson(): string {
    return this._workflow.toJson();
  }
}

/**
 * Workflow engine for executing workflows
 */
export class WorkflowEngine {
  private readonly _engine: NativeWorkflowEngine;

  constructor(engine: NativeWorkflowEngine) {
    this._engine = engine;
  }

  /**
   * Create a new workflow engine
   */
  static create(): WorkflowEngine {
    return new WorkflowEngine(native.createWorkflowEngine());
  }

  /**
   * Register a workflow with the engine
   */
  async registerWorkflow(workflow: Workflow): Promise<void> {
    await this._engine.registerWorkflow(workflow.internalWorkflow);
  }

  /**
   * Start workflow execution
   * @param workflowId - The ID of the workflow to execute
   * @returns Execution ID
   */
  async startExecution(workflowId: string): Promise<string> {
    return await native.workflowEngineStartExecution(this._engine, workflowId);
  }

  /**
   * Get execution result
   * @param executionId - The execution ID
   */
  async getExecutionResult(executionId: string): Promise<WorkflowExecutionResult> {
    return await native.workflowEngineGetExecutionResult(this._engine, executionId);
  }

  /**
   * Get execution state
   * @param executionId - The execution ID
   */
  async getExecutionState(executionId: string): Promise<WorkflowState> {
    const state = await native.workflowEngineGetExecutionState(this._engine, executionId);
    return state as WorkflowState;
  }

  /**
   * Get pending interrupts requiring human input
   */
  async getPendingInterrupts(): Promise<PendingInterrupt[]> {
    return await native.workflowEngineGetPendingInterrupts(this._engine);
  }

  /**
   * Resume execution with human input
   * @param interruptId - The interrupt ID
   * @param feedback - Human feedback data
   */
  async resumeExecution(interruptId: string, feedback: JsonRecord): Promise<void> {
    await native.workflowEngineResumeExecution(this._engine, interruptId, feedback);
  }

  /**
   * Pause workflow execution
   * @param executionId - The execution ID
   */
  async pauseExecution(executionId: string): Promise<void> {
    await native.workflowEnginePauseExecution(this._engine, executionId);
  }

  /**
   * Cancel workflow execution
   * @param executionId - The execution ID
   */
  async cancelExecution(executionId: string): Promise<void> {
    await native.workflowEngineCancelExecution(this._engine, executionId);
  }
}
