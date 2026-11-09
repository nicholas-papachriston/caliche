import {
  type NativeDebatePattern,
  type NativeHierarchicalPattern,
  type NativePipelinePattern,
  type NativeSupervisorPattern,
  native,
} from "../native/module";

export class DebatePattern {
  private readonly _pattern: NativeDebatePattern;

  constructor(
    name: string,
    topic: string,
    debaterCount: number,
    maxRounds: number,
    moderatorPrompt?: string,
    description?: string
  ) {
    this._pattern = native.DebatePattern.new(
      name,
      topic,
      debaterCount,
      maxRounds,
      moderatorPrompt,
      description
    );
  }

  /**
   * Execute the debate pattern
   */
  async execute(): Promise<any> {
    return await this._pattern.execute();
  }

  /**
   * Get pattern ID
   */
  getId(): string {
    return this._pattern.getId();
  }

  /**
   * Get pattern name
   */
  getName(): string {
    return this._pattern.getName();
  }

  /**
   * Get pattern description
   */
  getDescription(): string {
    return this._pattern.getDescription();
  }
}

/**
 * Hierarchical pattern: multi-level agent teams with layers of coordination
 *
 * @example
 * ```typescript
 * const hierarchical = new HierarchicalPattern("hierarchical1");
 * hierarchical.addLayer("layer1", 3, "Process data");
 * const result = await hierarchical.execute(engine);
 * ```
 */
export class HierarchicalPattern {
  private _pattern: NativeHierarchicalPattern;

  constructor(name: string, description?: string) {
    this._pattern = native.HierarchicalPattern.new(name, description);
  }

  /**
   * Add a layer to the hierarchical pattern
   * Returns a new pattern instance (since the Rust API consumes self)
   */
  addLayer(
    name: string,
    agentCount: number,
    promptTemplate: string,
    capability?: string
  ): HierarchicalPattern {
    const newPattern = this._pattern.addLayer(name, agentCount, promptTemplate, capability);
    const result = Object.create(HierarchicalPattern.prototype) as HierarchicalPattern;
    (result as unknown as { _pattern: NativeHierarchicalPattern })._pattern = newPattern;
    return result;
  }

  /**
   * Execute the hierarchical pattern
   */
  async execute(): Promise<any> {
    return await this._pattern.execute();
  }

  /**
   * Get pattern ID
   */
  getId(): string {
    return this._pattern.getId();
  }

  /**
   * Get pattern name
   */
  getName(): string {
    return this._pattern.getName();
  }

  /**
   * Get pattern description
   */
  getDescription(): string {
    return this._pattern.getDescription();
  }
}

/**
 * Pipeline pattern: sequential data transformation through multiple stages
 *
 * @example
 * ```typescript
 * const pipeline = new PipelinePattern("pipeline1");
 * pipeline.addStage("stage1", "Process input");
 * pipeline.addStage("stage2", "Transform data");
 * const result = await pipeline.execute(engine);
 * ```
 */
export class PipelinePattern {
  private _pattern: NativePipelinePattern;

  constructor(name: string, description?: string) {
    this._pattern = native.PipelinePattern.new(name, description);
  }

  /**
   * Add a stage to the pipeline pattern
   * Returns a new pattern instance (since the Rust API consumes self)
   */
  addStage(name: string, prompt: string, capability?: string, transform?: string): PipelinePattern {
    const newPattern = this._pattern.addStage(name, prompt, capability, transform);
    const result = Object.create(PipelinePattern.prototype) as PipelinePattern;
    (result as unknown as { _pattern: NativePipelinePattern })._pattern = newPattern;
    return result;
  }

  /**
   * Execute the pipeline pattern
   */
  async execute(): Promise<any> {
    return await this._pattern.execute();
  }

  /**
   * Get pattern ID
   */
  getId(): string {
    return this._pattern.getId();
  }

  /**
   * Get pattern name
   */
  getName(): string {
    return this._pattern.getName();
  }

  /**
   * Get pattern description
   */
  getDescription(): string {
    return this._pattern.getDescription();
  }
}

/**
 * Supervisor pattern: one coordinator agent assigns tasks to multiple worker agents
 *
 * @example
 * ```typescript
 * const supervisor = new SupervisorPattern("supervisor1", "Coordinate tasks", 5);
 * const result = await supervisor.execute(engine);
 * ```
 */
export class SupervisorPattern {
  private readonly _pattern: NativeSupervisorPattern;

  constructor(
    name: string,
    supervisorPrompt: string,
    workerCount: number,
    workerCapability?: string,
    description?: string
  ) {
    this._pattern = native.SupervisorPattern.new(
      name,
      supervisorPrompt,
      workerCount,
      workerCapability,
      description
    );
  }

  /**
   * Execute the supervisor pattern
   */
  async execute(): Promise<any> {
    return await this._pattern.execute();
  }

  /**
   * Get pattern ID
   */
  getId(): string {
    return this._pattern.getId();
  }

  /**
   * Get pattern name
   */
  getName(): string {
    return this._pattern.getName();
  }

  /**
   * Get pattern description
   */
  getDescription(): string {
    return this._pattern.getDescription();
  }
}
