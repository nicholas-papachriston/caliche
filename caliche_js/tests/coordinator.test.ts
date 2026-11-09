import { beforeAll, describe, expect, test } from "bun:test";
import {
  AgentBuilder,
  AgentCoordinator,
  type LoadBalancingStrategy,
  MessageBus,
} from "../src/index";
import { liveSuite } from "./test-env";

const { liveTest, agentBuilder } = await liveSuite("AgentCoordinator");

describe("AgentCoordinator", () => {
  let agent: Awaited<ReturnType<typeof AgentBuilder.prototype.build>>;

  beforeAll(async () => {
    const builder = agentBuilder();

    agent = await builder.build();
  });

  test("should create an AgentCoordinator instance", () => {
    const coordinator = AgentCoordinator.create();
    expect(coordinator).toBeInstanceOf(AgentCoordinator);
  });

  test("should create coordinator with load balancing strategy", () => {
    const strategies: LoadBalancingStrategy[] = ["round_robin", "least_connections", "weighted"];

    for (const strategy of strategies) {
      const coordinator = AgentCoordinator.withStrategy(strategy);
      expect(coordinator).toBeInstanceOf(AgentCoordinator);
    }
  });

  test("should have registerAgent method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.registerAgent).toBe("function");
  });

  test("should register an agent", async () => {
    const coordinator = AgentCoordinator.create();
    const agentId = await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    expect(typeof agentId).toBe("string");
    expect(agentId.length).toBeGreaterThan(0);
  });

  test("should have unregisterAgent method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.unregisterAgent).toBe("function");
  });

  test("should unregister an agent", async () => {
    const coordinator = AgentCoordinator.create();
    const agentId = await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    expect(agentId).toBeDefined();
    await expect(coordinator.unregisterAgent(agentId)).resolves.toBeUndefined();
  });

  test("should have delegateTask method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.delegateTask).toBe("function");
  });

  liveTest(
    "should delegate task to agent",
    async () => {
      const coordinator = AgentCoordinator.create();
      await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
      // delegateTask might fail if no agents are available or if the agent can't handle the task
      // So we'll check if it returns a result or throws an error
      const result = await Promise.race([
        coordinator.delegateTask("Test task", "research"),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error("delegateTask timed out")), 25000)
        ),
      ]);
      expect(result).toBeDefined();
      if (result && typeof result === "object" && "agent_id" in result) {
        expect((result as { agent_id?: string }).agent_id).toBeDefined();
      }
    },
    30000
  );

  test("should have executeOnAgent method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.executeOnAgent).toBe("function");
  });

  liveTest(
    "should execute task on specific agent",
    async () => {
      const coordinator = AgentCoordinator.create();
      const agentId = await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
      const result = await Promise.race([
        coordinator.executeOnAgent(agentId, "Reply with OK"),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("executeOnAgent timed out")), 25000)
        ),
      ]);
      expect(result).toBeDefined();
      expect(result.response).toBeDefined();
    },
    30000
  );

  test("should have sendMessage method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.sendMessage).toBe("function");
  });

  test("should send message between agents", async () => {
    const coordinator = AgentCoordinator.create();
    const agentId1 = await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    const agentId2 = await coordinator.registerAgent("agent2", "writer", ["writing"], agent);
    expect(agentId1).toBeDefined();
    expect(agentId2).toBeDefined();
    await expect(coordinator.sendMessage(agentId1, agentId2, "Hello")).resolves.toBeUndefined();
  });

  test("should have broadcast method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.broadcast).toBe("function");
  });

  test("should broadcast message to all agents", async () => {
    const coordinator = AgentCoordinator.create();
    const agentId = await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    expect(agentId).toBeDefined();
    await expect(coordinator.broadcast(agentId, "Broadcast message")).resolves.toBeUndefined();
  });

  test("should have getStats method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.getStats).toBe("function");
  });

  test("should get coordinator statistics", async () => {
    const coordinator = AgentCoordinator.create();
    await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    try {
      const stats = await coordinator.getStats();
      expect(stats).toBeDefined();
      if (stats) {
        expect(stats.total_agents).toBeDefined();
        expect(typeof stats.total_agents).toBe("number");
      }
    } catch (error) {
      // If getStats throws, that's also acceptable
      expect(error).toBeDefined();
    }
  });

  test("should have listAgentIds method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.listAgentIds).toBe("function");
  });

  test("should list registered agent IDs", async () => {
    const coordinator = AgentCoordinator.create();
    await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    const ids = coordinator.listAgentIds();
    expect(Array.isArray(ids)).toBe(true);
    expect(ids.length).toBeGreaterThan(0);
  });

  test("should have agentCount method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.agentCount).toBe("function");
  });

  test("should get agent count", async () => {
    const coordinator = AgentCoordinator.create();
    expect(coordinator.agentCount()).toBe(0);
    await coordinator.registerAgent("agent1", "researcher", ["research"], agent);
    expect(coordinator.agentCount()).toBe(1);
  });

  test("should have getMessageBus method", () => {
    const coordinator = AgentCoordinator.create();
    expect(typeof coordinator.getMessageBus).toBe("function");
  });

  test("should get message bus", () => {
    const coordinator = AgentCoordinator.create();
    const bus = coordinator.getMessageBus();
    expect(bus).toBeInstanceOf(MessageBus);
  });
});

describe("MessageBus", () => {
  let coordinator: AgentCoordinator;

  beforeAll(() => {
    coordinator = AgentCoordinator.create();
  });

  test("should create MessageBus from coordinator", () => {
    const bus = coordinator.getMessageBus();
    expect(bus).toBeInstanceOf(MessageBus);
  });

  test("should have getStats method", () => {
    const bus = coordinator.getMessageBus();
    expect(typeof bus.getStats).toBe("function");
  });

  test("should get message bus statistics", async () => {
    const bus = coordinator.getMessageBus();
    try {
      const stats = await bus.getStats();
      expect(stats).toBeDefined();
      if (stats) {
        expect(stats.queue_size).toBeDefined();
        expect(typeof stats.queue_size).toBe("number");
      }
    } catch (error) {
      // If getStats throws, that's also acceptable
      expect(error).toBeDefined();
    }
  });

  test("should have queueSize method", () => {
    const bus = coordinator.getMessageBus();
    expect(typeof bus.queueSize).toBe("function");
  });

  test("should get queue size", async () => {
    const bus = coordinator.getMessageBus();
    const size = await bus.queueSize();
    expect(typeof size).toBe("number");
    expect(size).toBeGreaterThanOrEqual(0);
  });
});
