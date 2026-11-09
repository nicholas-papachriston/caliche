import { beforeAll, describe, expect, test } from "bun:test";
import {
  AgentBuilder,
  DebatePattern,
  HierarchicalPattern,
  PipelinePattern,
  PlanExecutePattern,
  ReActPattern,
  type ReasoningResult,
  ReflexionPattern,
  SelfAskPattern,
  SupervisorPattern,
  TreeOfThoughtsPattern,
  WorkflowEngine,
} from "../src/index";
import { liveSuite } from "./test-env";

const { liveTest, agentBuilder } = await liveSuite("Patterns");

describe("Reasoning Patterns", () => {
  let agent: Awaited<ReturnType<typeof AgentBuilder.prototype.build>>;

  beforeAll(async () => {
    const builder = agentBuilder();

    agent = await builder.build();
  });

  describe("ReActPattern", () => {
    test("should create a ReActPattern instance", () => {
      const pattern = new ReActPattern(agent, 5);
      expect(pattern).toBeInstanceOf(ReActPattern);
    });

    test("should have execute method", () => {
      const pattern = new ReActPattern(agent, 5);
      expect(typeof pattern.execute).toBe("function");
    });

    liveTest(
      "should execute ReAct pattern",
      async () => {
        const pattern = new ReActPattern(agent, 3);
        const result = await Promise.race([
          pattern.execute("What is 2+2?"),
          new Promise<ReasoningResult>((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 10000)
          ),
        ]);
        expect(result).toBeDefined();
        if (result && typeof result === "object" && "final_answer" in result) {
          const reasoningResult = result as ReasoningResult;
          // Check for final_answer (required) or other possible response fields
          expect(
            reasoningResult.final_answer !== undefined ||
              (reasoningResult as any).answer !== undefined ||
              (reasoningResult as any).response !== undefined
          ).toBe(true);
        }
      },
      15000
    );
  });

  describe("PlanExecutePattern", () => {
    test("should create a PlanExecutePattern instance", () => {
      const pattern = new PlanExecutePattern(agent, 10);
      expect(pattern).toBeInstanceOf(PlanExecutePattern);
    });

    test("should have execute method", () => {
      const pattern = new PlanExecutePattern(agent, 10);
      expect(typeof pattern.execute).toBe("function");
    });

    liveTest(
      "should execute Plan-Execute pattern",
      async () => {
        const pattern = new PlanExecutePattern(agent, 5);
        const result = await Promise.race([
          pattern.execute("Plan a simple task"),
          new Promise<ReasoningResult>((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 10000)
          ),
        ]);
        expect(result).toBeDefined();
        if (result && typeof result === "object" && "final_answer" in result) {
          const reasoningResult = result as ReasoningResult;
          // Check for final_answer (required) or other possible response fields
          expect(
            reasoningResult.final_answer !== undefined ||
              (reasoningResult as any).answer !== undefined ||
              (reasoningResult as any).response !== undefined
          ).toBe(true);
        }
      },
      15000
    );
  });

  describe("ReflexionPattern", () => {
    test("should create a ReflexionPattern instance", () => {
      const pattern = new ReflexionPattern(agent, 3);
      expect(pattern).toBeInstanceOf(ReflexionPattern);
    });

    test("should have execute method", () => {
      const pattern = new ReflexionPattern(agent, 3);
      expect(typeof pattern.execute).toBe("function");
    });

    liveTest(
      "should execute Reflexion pattern",
      async () => {
        const pattern = new ReflexionPattern(agent, 2);
        const result = await Promise.race([
          pattern.execute("Solve a simple problem"),
          new Promise<ReasoningResult>((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 10000)
          ),
        ]);
        expect(result).toBeDefined();
        if (result && typeof result === "object" && "final_answer" in result) {
          const reasoningResult = result as ReasoningResult;
          // Check for final_answer (required) or other possible response fields
          expect(
            reasoningResult.final_answer !== undefined ||
              (reasoningResult as any).answer !== undefined ||
              (reasoningResult as any).response !== undefined
          ).toBe(true);
        }
      },
      15000
    );
  });

  describe("SelfAskPattern", () => {
    test("should create a SelfAskPattern instance", () => {
      const pattern = new SelfAskPattern(agent, 5);
      expect(pattern).toBeInstanceOf(SelfAskPattern);
    });

    test("should have execute method", () => {
      const pattern = new SelfAskPattern(agent, 5);
      expect(typeof pattern.execute).toBe("function");
    });

    liveTest(
      "should execute Self-Ask pattern",
      async () => {
        const pattern = new SelfAskPattern(agent, 3);
        const result = await Promise.race([
          pattern.execute("What is the capital of France?"),
          new Promise<ReasoningResult>((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 10000)
          ),
        ]);
        expect(result).toBeDefined();
        if (result && typeof result === "object" && "final_answer" in result) {
          const reasoningResult = result as ReasoningResult;
          // Check for final_answer (required) or other possible response fields
          expect(
            reasoningResult.final_answer !== undefined ||
              (reasoningResult as any).answer !== undefined ||
              (reasoningResult as any).response !== undefined
          ).toBe(true);
        }
      },
      15000
    );
  });

  describe("TreeOfThoughtsPattern", () => {
    test("should create a TreeOfThoughtsPattern instance", () => {
      const pattern = new TreeOfThoughtsPattern(agent, 3, 3);
      expect(pattern).toBeInstanceOf(TreeOfThoughtsPattern);
    });

    test("should create with default branching factor", () => {
      const pattern = new TreeOfThoughtsPattern(agent, 3);
      expect(pattern).toBeInstanceOf(TreeOfThoughtsPattern);
    });

    test("should have execute method", () => {
      const pattern = new TreeOfThoughtsPattern(agent, 3, 3);
      expect(typeof pattern.execute).toBe("function");
    });

    liveTest(
      "should execute Tree of Thoughts pattern",
      async () => {
        const pattern = new TreeOfThoughtsPattern(agent, 2, 2);
        const result = await Promise.race([
          pattern.execute("Solve a simple problem"),
          new Promise<ReasoningResult>((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 10000)
          ),
        ]);
        expect(result).toBeDefined();
        if (result && typeof result === "object" && "final_answer" in result) {
          const reasoningResult = result as ReasoningResult;
          // Check for final_answer (required) or other possible response fields
          expect(
            reasoningResult.final_answer !== undefined ||
              (reasoningResult as any).answer !== undefined ||
              (reasoningResult as any).response !== undefined
          ).toBe(true);
        }
      },
      15000
    );
  });
});

describe("Multi-Agent Patterns", () => {
  beforeAll(() => {
    WorkflowEngine.create();
  });

  describe("DebatePattern", () => {
    test("should create a DebatePattern instance", () => {
      const pattern = new DebatePattern("debate1", "Test topic", 2, 3);
      expect(pattern).toBeInstanceOf(DebatePattern);
    });

    test("should create with moderator prompt", () => {
      const pattern = new DebatePattern("debate2", "Test topic", 2, 3, "Moderate the debate");
      expect(pattern).toBeInstanceOf(DebatePattern);
    });

    test("should create with description", () => {
      const pattern = new DebatePattern(
        "debate3",
        "Test topic",
        2,
        3,
        undefined,
        "Test description"
      );
      expect(pattern).toBeInstanceOf(DebatePattern);
    });

    test("should have execute method", () => {
      const pattern = new DebatePattern("debate1", "Test topic", 2, 3);
      expect(typeof pattern.execute).toBe("function");
    });

    test("should have getId method", () => {
      const pattern = new DebatePattern("debate1", "Test topic", 2, 3);
      expect(typeof pattern.getId).toBe("function");
      const id = pattern.getId();
      expect(typeof id).toBe("string");
    });

    test("should have getName method", () => {
      const pattern = new DebatePattern("debate1", "Test topic", 2, 3);
      expect(typeof pattern.getName).toBe("function");
      const name = pattern.getName();
      expect(name).toBe("debate1");
    });

    test("should have getDescription method", () => {
      const pattern = new DebatePattern("debate1", "Test topic", 2, 3);
      expect(typeof pattern.getDescription).toBe("function");
      const description = pattern.getDescription();
      expect(typeof description).toBe("string");
    });
  });

  describe("HierarchicalPattern", () => {
    test("should create a HierarchicalPattern instance", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      expect(pattern).toBeInstanceOf(HierarchicalPattern);
    });

    test("should create with description", () => {
      const pattern = new HierarchicalPattern("hierarchical2", "Test description");
      expect(pattern).toBeInstanceOf(HierarchicalPattern);
    });

    test("should have addLayer method", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      expect(typeof pattern.addLayer).toBe("function");
    });

    test("should add layer and return new instance", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      const newPattern = pattern.addLayer("layer1", 3, "Process data");
      expect(newPattern).toBeInstanceOf(HierarchicalPattern);
      expect(newPattern).not.toBe(pattern);
    });

    test("should add layer with capability", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      const newPattern = pattern.addLayer("layer1", 3, "Process data", "research");
      expect(newPattern).toBeInstanceOf(HierarchicalPattern);
    });

    test("should have execute method", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      expect(typeof pattern.execute).toBe("function");
    });

    test("should have getId method", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      expect(typeof pattern.getId).toBe("function");
      const id = pattern.getId();
      expect(typeof id).toBe("string");
    });

    test("should have getName method", () => {
      const pattern = new HierarchicalPattern("hierarchical1");
      expect(typeof pattern.getName).toBe("function");
      const name = pattern.getName();
      expect(name).toBe("hierarchical1");
    });

    test("should have getDescription method", () => {
      const pattern = new HierarchicalPattern("hierarchical1", "Test description");
      expect(typeof pattern.getDescription).toBe("function");
      const description = pattern.getDescription();
      expect(typeof description).toBe("string");
    });
  });

  describe("PipelinePattern", () => {
    test("should create a PipelinePattern instance", () => {
      const pattern = new PipelinePattern("pipeline1");
      expect(pattern).toBeInstanceOf(PipelinePattern);
    });

    test("should create with description", () => {
      const pattern = new PipelinePattern("pipeline2", "Test description");
      expect(pattern).toBeInstanceOf(PipelinePattern);
    });

    test("should have addStage method", () => {
      const pattern = new PipelinePattern("pipeline1");
      expect(typeof pattern.addStage).toBe("function");
    });

    test("should add stage and return new instance", () => {
      const pattern = new PipelinePattern("pipeline1");
      const newPattern = pattern.addStage("stage1", "Process input");
      expect(newPattern).toBeInstanceOf(PipelinePattern);
      expect(newPattern).not.toBe(pattern);
    });

    test("should add stage with capability", () => {
      const pattern = new PipelinePattern("pipeline1");
      const newPattern = pattern.addStage("stage1", "Process input", "research");
      expect(newPattern).toBeInstanceOf(PipelinePattern);
    });

    test("should add stage with transform", () => {
      const pattern = new PipelinePattern("pipeline1");
      const newPattern = pattern.addStage(
        "stage1",
        "Process input",
        "research",
        "{input} -> {output}"
      );
      expect(newPattern).toBeInstanceOf(PipelinePattern);
    });

    test("should have execute method", () => {
      const pattern = new PipelinePattern("pipeline1");
      expect(typeof pattern.execute).toBe("function");
    });

    test("should have getId method", () => {
      const pattern = new PipelinePattern("pipeline1");
      expect(typeof pattern.getId).toBe("function");
      const id = pattern.getId();
      expect(typeof id).toBe("string");
    });

    test("should have getName method", () => {
      const pattern = new PipelinePattern("pipeline1");
      expect(typeof pattern.getName).toBe("function");
      const name = pattern.getName();
      expect(name).toBe("pipeline1");
    });

    test("should have getDescription method", () => {
      const pattern = new PipelinePattern("pipeline1", "Test description");
      expect(typeof pattern.getDescription).toBe("function");
      const description = pattern.getDescription();
      expect(typeof description).toBe("string");
    });
  });

  describe("SupervisorPattern", () => {
    test("should create a SupervisorPattern instance", () => {
      const pattern = new SupervisorPattern("supervisor1", "Coordinate tasks", 5);
      expect(pattern).toBeInstanceOf(SupervisorPattern);
    });

    test("should create with worker capability", () => {
      const pattern = new SupervisorPattern("supervisor2", "Coordinate tasks", 5, "research");
      expect(pattern).toBeInstanceOf(SupervisorPattern);
    });

    test("should create with description", () => {
      const pattern = new SupervisorPattern(
        "supervisor3",
        "Coordinate tasks",
        5,
        undefined,
        "Test description"
      );
      expect(pattern).toBeInstanceOf(SupervisorPattern);
    });

    test("should have execute method", () => {
      const pattern = new SupervisorPattern("supervisor1", "Coordinate tasks", 5);
      expect(typeof pattern.execute).toBe("function");
    });

    test("should have getId method", () => {
      const pattern = new SupervisorPattern("supervisor1", "Coordinate tasks", 5);
      expect(typeof pattern.getId).toBe("function");
      const id = pattern.getId();
      expect(typeof id).toBe("string");
    });

    test("should have getName method", () => {
      const pattern = new SupervisorPattern("supervisor1", "Coordinate tasks", 5);
      expect(typeof pattern.getName).toBe("function");
      const name = pattern.getName();
      expect(name).toBe("supervisor1");
    });

    test("should have getDescription method", () => {
      const pattern = new SupervisorPattern(
        "supervisor1",
        "Coordinate tasks",
        5,
        undefined,
        "Test description"
      );
      expect(typeof pattern.getDescription).toBe("function");
      const description = pattern.getDescription();
      expect(typeof description).toBe("string");
    });
  });
});
