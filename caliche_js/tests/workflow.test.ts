import { describe, expect, test } from "bun:test";
import { Workflow, WorkflowBuilder, WorkflowEngine, type WorkflowState } from "../src/index";

describe("Workflow", () => {
  describe("WorkflowBuilder", () => {
    test("should create a WorkflowBuilder instance", () => {
      const builder = WorkflowBuilder.create("test-workflow");
      expect(builder).toBeInstanceOf(WorkflowBuilder);
    });

    test("should create builder with custom name", () => {
      const builder = WorkflowBuilder.create("my-custom-workflow");
      expect(builder).toBeInstanceOf(WorkflowBuilder);
    });

    test("should have addAgent method", () => {
      const builder = WorkflowBuilder.create("test");
      expect(typeof builder.addAgent).toBe("function");
    });

    test("should add agent node", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addAgent("agent1", "Process: {input}");
      expect(result).toBe(builder);
    });

    test("should add agent with capability", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addAgentWithCapability("agent1", "research", "Research: {input}");
      expect(result).toBe(builder);
    });

    test("should add condition node", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addCondition("check", "{value} > 10");
      expect(result).toBe(builder);
    });

    test("should add parallel node", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addParallel([
        { name: "branch1", prompt: "Process 1: {input}" },
        { name: "branch2", prompt: "Process 2: {input}" },
      ]);
      expect(result).toBe(builder);
    });

    test("should add merge node after fan-out", async () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder
        .fanOut([
          { name: "b1", prompt: "p1" },
          { name: "b2", prompt: "p2" },
        ])
        .addMerge("merge1", "concatenate");
      expect(result).toBe(builder);
      const wf = await builder.build();
      const json = JSON.parse(wf.toJson()) as { nodes: Array<{ node_type: unknown }> };
      const mergeNodes = json.nodes.filter((n) => {
        const t = n.node_type as Record<string, unknown>;
        return t && typeof t === "object" && "Merge" in t;
      });
      expect(mergeNodes.length).toBeGreaterThanOrEqual(1);
      const merge = mergeNodes[0].node_type as { Merge: { strategy: string } };
      expect(merge.Merge.strategy).toBe("Concatenate");
    });

    test("addParallelWithMergeStrategy should set merge strategy", async () => {
      const builder = WorkflowBuilder.create("test");
      builder.addParallelWithMergeStrategy(
        [
          { name: "b1", prompt: "p1" },
          { name: "b2", prompt: "p2" },
        ],
        "vote"
      );
      const wf = await builder.build();
      const json = JSON.parse(wf.toJson()) as { nodes: Array<{ node_type: unknown }> };
      const mergeNode = json.nodes.find((n) => {
        const t = n.node_type as Record<string, unknown>;
        return t && typeof t === "object" && "Merge" in t;
      });
      expect(mergeNode).toBeDefined();
      const merge = mergeNode!.node_type as { Merge: { strategy: string } };
      expect(merge.Merge.strategy).toBe("Vote");
    });

    test("should add loop node", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addLoop("loop1", 5, "{count} < 10", "Iterate: {count}");
      expect(result).toBe(builder);
    });

    test("should add human interrupt", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addHumanInterrupt("interrupt1", "Please provide input");
      expect(result).toBe(builder);
    });

    test("should add human interrupt with required fields", () => {
      const builder = WorkflowBuilder.create("test");
      const result = builder.addHumanInterrupt("interrupt1", "Please provide input", [
        "name",
        "email",
      ]);
      expect(result).toBe(builder);
    });

    test("should connect nodes", () => {
      const builder = WorkflowBuilder.create("test");
      builder.addAgent("agent1", "Process: {input}");
      builder.addAgent("agent2", "Process: {input}");
      const result = builder.connectNodes("agent1", "agent2");
      expect(result).toBe(builder);
    });

    test("should connect nodes with condition", () => {
      const builder = WorkflowBuilder.create("test");
      builder.addAgent("agent1", "Process: {input}");
      builder.addAgent("agent2", "Process: {input}");
      const result = builder.connectNodes("agent1", "agent2", "{value} > 0");
      expect(result).toBe(builder);
    });

    test("should set entry point", () => {
      const builder = WorkflowBuilder.create("test");
      builder.addAgent("agent1", "Process: {input}");
      const result = builder.setEntryPoint("agent1");
      expect(result).toBe(builder);
    });

    test("should build workflow", async () => {
      const builder = WorkflowBuilder.create("test");
      builder.addAgent("agent1", "Process: {input}");
      builder.setEntryPoint("agent1");

      const workflow = await builder.build();
      expect(workflow).toBeInstanceOf(Workflow);
    });

    test("should support method chaining", () => {
      const builder = WorkflowBuilder.create("test")
        .addAgent("agent1", "Process: {input}")
        .addAgent("agent2", "Process: {input}")
        .connectNodes("agent1", "agent2")
        .setEntryPoint("agent1");

      expect(builder).toBeInstanceOf(WorkflowBuilder);
    });
  });

  describe("Workflow", () => {
    test("should get workflow ID", async () => {
      const builder = WorkflowBuilder.create("test-workflow");
      builder.addAgent("agent1", "Process: {input}");
      builder.setEntryPoint("agent1");

      const workflow = await builder.build();
      const id = workflow.getId();
      expect(typeof id).toBe("string");
      expect(id.length).toBeGreaterThan(0);
    });
  });

  describe("WorkflowEngine", () => {
    test("should create a WorkflowEngine instance", () => {
      const engine = WorkflowEngine.create();
      expect(engine).toBeInstanceOf(WorkflowEngine);
    });

    test("should have registerWorkflow method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.registerWorkflow).toBe("function");
    });

    test("should have startExecution method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.startExecution).toBe("function");
    });

    test("should have getExecutionResult method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.getExecutionResult).toBe("function");
    });

    test("should have getExecutionState method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.getExecutionState).toBe("function");
    });

    test("should have getPendingInterrupts method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.getPendingInterrupts).toBe("function");
    });

    test("should have resumeExecution method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.resumeExecution).toBe("function");
    });

    test("should have pauseExecution method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.pauseExecution).toBe("function");
    });

    test("should have cancelExecution method", () => {
      const engine = WorkflowEngine.create();
      expect(typeof engine.cancelExecution).toBe("function");
    });

    test("should register a workflow", async () => {
      const engine = WorkflowEngine.create();
      const builder = WorkflowBuilder.create("test");
      builder.addAgent("agent1", "Process: {input}");
      builder.setEntryPoint("agent1");
      const workflow = await builder.build();

      await engine.registerWorkflow(workflow);
    });
  });

  describe("Workflow State Transitions", () => {
    test("should have valid workflow state types", () => {
      const validStates: WorkflowState[] = [
        "pending",
        "running",
        "paused",
        "completed",
        "failed",
        "cancelled",
      ];

      for (const state of validStates) {
        expect(typeof state).toBe("string");
      }
    });
  });
});
