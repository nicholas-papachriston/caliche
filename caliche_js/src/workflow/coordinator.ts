import type { Agent } from "../agent/agent";
import { type NativeAgentCoordinator, type NativeMessageBus, native } from "../native/module";
import type { TokenUsage } from "../streaming/events";

export type LoadBalancingStrategy = "round_robin" | "least_connections" | "weighted";

/**
 * Agent capability type
 */
export type AgentCapability =
  | "research"
  | "writing"
  | "coding"
  | "data_analysis"
  | "planning"
  | "coordination"
  | string;

/**
 * Delegation result
 */
export interface DelegationResult {
  readonly agent_id: string;
  readonly response: string;
  readonly usage: TokenUsage;
}

/**
 * Execution result
 */
export interface ExecutionResult {
  readonly response: string;
  readonly usage: TokenUsage;
}

/**
 * Agent statistics
 */
export interface AgentStat {
  readonly agent_id: string;
  readonly name: string;
  readonly current_load: number;
  readonly max_concurrent_tasks: number;
}

/**
 * Coordinator statistics
 */
export interface CoordinatorStats {
  readonly total_agents: number;
  readonly total_load: number;
  readonly total_capacity: number;
  readonly agent_stats: readonly AgentStat[];
}

/**
 * Message bus statistics
 */
export interface MessageBusStats {
  readonly queue_size: number;
  readonly max_queue_size: number;
  readonly low_priority: number;
  readonly normal_priority: number;
  readonly high_priority: number;
  readonly critical_priority: number;
  readonly subscriber_count: number;
}

/**
 * Agent coordinator for managing multiple agents
 */
export class AgentCoordinator {
  private readonly _coordinator: NativeAgentCoordinator;

  constructor() {
    this._coordinator = new native.AgentCoordinator();
  }

  /**
   * Create a new agent coordinator
   */
  static create(): AgentCoordinator {
    return new AgentCoordinator();
  }

  /**
   * Create a coordinator with a specific load balancing strategy
   */
  static withStrategy(strategy: LoadBalancingStrategy): AgentCoordinator {
    const coordinator = Object.create(AgentCoordinator.prototype) as AgentCoordinator;
    (coordinator as unknown as { _coordinator: NativeAgentCoordinator })._coordinator =
      native.AgentCoordinator.withStrategy(strategy);
    return coordinator;
  }

  /**
   * Register an agent with the coordinator
   */
  async registerAgent(
    name: string,
    role: string,
    capabilities: AgentCapability[],
    agent: Agent
  ): Promise<string> {
    return await this._coordinator.registerAgent(
      name,
      role,
      capabilities,
      agent.provider,
      agent.model
    );
  }

  /**
   * Unregister an agent
   */
  async unregisterAgent(agentId: string): Promise<void> {
    await this._coordinator.unregisterAgent(agentId);
  }

  /**
   * Delegate a task to the best available agent
   */
  async delegateTask(message: string, capability?: AgentCapability): Promise<DelegationResult> {
    return await this._coordinator.delegateTask(message, capability ?? undefined);
  }

  /**
   * Execute a task on a specific agent
   */
  async executeOnAgent(agentId: string, message: string): Promise<ExecutionResult> {
    return await this._coordinator.executeOnAgent(agentId, message);
  }

  /**
   * Send a message from one agent to another
   */
  async sendMessage(fromAgentId: string, toAgentId: string, content: string): Promise<void> {
    await this._coordinator.sendMessage(fromAgentId, toAgentId, content);
  }

  /**
   * Broadcast a message to all agents
   */
  async broadcast(fromAgentId: string, content: string): Promise<void> {
    await this._coordinator.broadcast(fromAgentId, content);
  }

  /**
   * Get coordinator statistics
   */
  async getStats(): Promise<CoordinatorStats> {
    return await this._coordinator.getStats();
  }

  /**
   * List all registered agent IDs
   */
  listAgentIds(): string[] {
    return this._coordinator.listAgentIds();
  }

  /**
   * Get the number of registered agents
   */
  agentCount(): number {
    return this._coordinator.agentCount();
  }

  /**
   * Get the message bus for inter-agent communication
   */
  getMessageBus(): MessageBus {
    const bus = this._coordinator.getMessageBus();
    return new MessageBus(bus);
  }
}

/**
 * Message bus for inter-agent communication
 */
export class MessageBus {
  private readonly _bus: NativeMessageBus;

  constructor(bus: NativeMessageBus) {
    this._bus = bus;
  }

  /**
   * Get message bus statistics
   */
  async getStats(): Promise<MessageBusStats> {
    return await this._bus.getStats();
  }

  /**
   * Get current queue size
   */
  async queueSize(): Promise<number> {
    return await this._bus.queueSize();
  }
}
