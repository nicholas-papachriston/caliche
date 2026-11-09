import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  Agent,
  AgentBuilder,
  AgentCoordinator,
  AgentFlow,
  AgentTemplate,
  BuiltinTool,
  ContainsEvaluator,
  DebatePattern,
  EvaluationDatasetLoader,
  ExactMatchEvaluator,
  FlowBuilder,
  HierarchicalPattern,
  isDoneEvent,
  isMetadataEvent,
  isTextChunkEvent,
  isToolCallResultEvent,
  isToolCallStartEvent,
  LATEST_PROTOCOL_VERSION,
  LangChainAdapter,
  LangChainChainAdapter,
  LangChainEmbeddingProvider,
  LangChainLLMProvider,
  LangChainMemoryAdapter,
  LangChainRetrieverAdapter,
  LLMChain,
  LLMEvaluator,
  LlmClient,
  LlmClientBuilder,
  MapReduceChain,
  McpServer,
  McpServerBuilder,
  MessageBus,
  PipelinePattern,
  PlanExecutePattern,
  PromptTemplate,
  PromptTemplateStore,
  RagService,
  RagTool,
  ReActPattern,
  ReflexionPattern,
  SelfAskPattern,
  SequentialChain,
  SharedState,
  StorageConfig,
  StreamIterator,
  SUPPORTED_PROTOCOL_VERSIONS,
  SupervisorPattern,
  TemplateRegistry,
  TreeOfThoughtsPattern,
  Workflow,
  WorkflowBuilder,
  WorkflowEngine,
  waitForPendingTasks,
} from "../src/index";

describe("Native Bindings", () => {
  test("should export AgentBuilder class", () => {
    expect(AgentBuilder).toBeDefined();
    expect(typeof AgentBuilder).toBe("function");
  });

  test("should export Agent class", () => {
    expect(Agent).toBeDefined();
    expect(typeof Agent).toBe("function");
  });

  test("should export LlmClientBuilder class", () => {
    expect(LlmClientBuilder).toBeDefined();
    expect(typeof LlmClientBuilder).toBe("function");
  });

  test("should export LlmClient class", () => {
    expect(LlmClient).toBeDefined();
    expect(typeof LlmClient).toBe("function");
  });

  test("should export SharedState class", () => {
    expect(SharedState).toBeDefined();
    expect(typeof SharedState).toBe("function");
  });

  test("should export FlowBuilder class", () => {
    expect(FlowBuilder).toBeDefined();
    expect(typeof FlowBuilder).toBe("function");
  });

  test("should export AgentFlow class", () => {
    expect(AgentFlow).toBeDefined();
    expect(typeof AgentFlow).toBe("function");
  });

  test("should export LLMChain class", () => {
    expect(LLMChain).toBeDefined();
    expect(typeof LLMChain).toBe("function");
  });

  test("should export SequentialChain class", () => {
    expect(SequentialChain).toBeDefined();
    expect(typeof SequentialChain).toBe("function");
  });

  test("should export MapReduceChain class", () => {
    expect(MapReduceChain).toBeDefined();
    expect(typeof MapReduceChain).toBe("function");
  });

  test("should export StreamIterator class", () => {
    expect(StreamIterator).toBeDefined();
    expect(typeof StreamIterator).toBe("function");
  });

  test("StreamIterator defaults to schema version 1", () => {
    const iterator = new StreamIterator({} as never);
    expect((iterator as unknown as { eventSchemaVersion: string }).eventSchemaVersion).toBe("1");
  });

  test("should export waitForPendingTasks function", () => {
    expect(waitForPendingTasks).toBeDefined();
    expect(typeof waitForPendingTasks).toBe("function");
  });

  test("should export stream event type guards", () => {
    expect(isTextChunkEvent).toBeDefined();
    expect(typeof isTextChunkEvent).toBe("function");
    expect(isToolCallStartEvent).toBeDefined();
    expect(typeof isToolCallStartEvent).toBe("function");
    expect(isToolCallResultEvent).toBeDefined();
    expect(typeof isToolCallResultEvent).toBe("function");
    expect(isMetadataEvent).toBeDefined();
    expect(typeof isMetadataEvent).toBe("function");
    expect(isDoneEvent).toBeDefined();
    expect(typeof isDoneEvent).toBe("function");
  });

  test("should export reasoning pattern classes", () => {
    expect(ReActPattern).toBeDefined();
    expect(typeof ReActPattern).toBe("function");
    expect(PlanExecutePattern).toBeDefined();
    expect(typeof PlanExecutePattern).toBe("function");
    expect(ReflexionPattern).toBeDefined();
    expect(typeof ReflexionPattern).toBe("function");
    expect(SelfAskPattern).toBeDefined();
    expect(typeof SelfAskPattern).toBe("function");
    expect(TreeOfThoughtsPattern).toBeDefined();
    expect(typeof TreeOfThoughtsPattern).toBe("function");
  });

  test("should export prompt template classes", () => {
    expect(PromptTemplate).toBeDefined();
    expect(typeof PromptTemplate).toBe("function");
    expect(PromptTemplateStore).toBeDefined();
    expect(typeof PromptTemplateStore).toBe("function");
    expect(TemplateRegistry).toBeDefined();
    expect(typeof TemplateRegistry).toBe("function");
    expect(AgentTemplate).toBeDefined();
    expect(typeof AgentTemplate).toBe("function");
  });

  test("should export StorageConfig class", () => {
    expect(StorageConfig).toBeDefined();
    expect(typeof StorageConfig).toBe("function");
  });

  test("should export workflow classes", () => {
    expect(WorkflowBuilder).toBeDefined();
    expect(typeof WorkflowBuilder).toBe("function");
    expect(Workflow).toBeDefined();
    expect(typeof Workflow).toBe("function");
    expect(WorkflowEngine).toBeDefined();
    expect(typeof WorkflowEngine).toBe("function");
  });

  test("should export coordinator classes", () => {
    expect(AgentCoordinator).toBeDefined();
    expect(typeof AgentCoordinator).toBe("function");
    expect(MessageBus).toBeDefined();
    expect(typeof MessageBus).toBe("function");
  });

  test("should export evaluator classes", () => {
    expect(ExactMatchEvaluator).toBeDefined();
    expect(typeof ExactMatchEvaluator).toBe("function");
    expect(ContainsEvaluator).toBeDefined();
    expect(typeof ContainsEvaluator).toBe("function");
    expect(LLMEvaluator).toBeDefined();
    expect(typeof LLMEvaluator).toBe("function");
    expect(EvaluationDatasetLoader).toBeDefined();
    expect(typeof EvaluationDatasetLoader).toBe("function");
  });

  test("should export RAG classes", () => {
    expect(RagService).toBeDefined();
    expect(typeof RagService).toBe("function");
    expect(RagTool).toBeDefined();
    expect(typeof RagTool).toBe("function");
  });

  test("should export BuiltinTool class", () => {
    expect(BuiltinTool).toBeDefined();
    expect(typeof BuiltinTool).toBe("function");
  });

  test("should export multi-agent pattern classes", () => {
    expect(DebatePattern).toBeDefined();
    expect(typeof DebatePattern).toBe("function");
    expect(HierarchicalPattern).toBeDefined();
    expect(typeof HierarchicalPattern).toBe("function");
    expect(PipelinePattern).toBeDefined();
    expect(typeof PipelinePattern).toBe("function");
    expect(SupervisorPattern).toBeDefined();
    expect(typeof SupervisorPattern).toBe("function");
  });

  test("should export LangChain adapters", () => {
    expect(LangChainAdapter).toBeDefined();
    expect(typeof LangChainAdapter).toBe("function");
    expect(LangChainChainAdapter).toBeDefined();
    expect(typeof LangChainChainAdapter).toBe("function");
    expect(LangChainMemoryAdapter).toBeDefined();
    expect(typeof LangChainMemoryAdapter).toBe("function");
    expect(LangChainRetrieverAdapter).toBeDefined();
    expect(typeof LangChainRetrieverAdapter).toBe("function");
    expect(LangChainEmbeddingProvider).toBeDefined();
    expect(typeof LangChainEmbeddingProvider).toBe("function");
    expect(LangChainLLMProvider).toBeDefined();
    expect(typeof LangChainLLMProvider).toBe("function");
  });

  test("should export MCP classes and constants", () => {
    expect(McpServer).toBeDefined();
    expect(typeof McpServer).toBe("function");
    expect(McpServerBuilder).toBeDefined();
    expect(typeof McpServerBuilder).toBe("function");
    expect(LATEST_PROTOCOL_VERSION).toBeDefined();
    expect(typeof LATEST_PROTOCOL_VERSION).toBe("string");
    expect(SUPPORTED_PROTOCOL_VERSIONS).toBeDefined();
    expect(Array.isArray(SUPPORTED_PROTOCOL_VERSIONS)).toBe(true);
  });

  test("AgentBuilder should not expose fromEnv factory method", () => {
    expect((AgentBuilder as unknown as { fromEnv?: unknown }).fromEnv).toBeUndefined();
  });

  test("LlmClientBuilder should not expose fromEnv factory method", () => {
    expect((LlmClientBuilder as unknown as { fromEnv?: unknown }).fromEnv).toBeUndefined();
  });

  test("should be able to instantiate AgentBuilder", () => {
    const builder = new AgentBuilder();
    expect(builder).toBeInstanceOf(AgentBuilder);
  });

  test("should be able to instantiate LlmClientBuilder", () => {
    const builder = new LlmClientBuilder();
    expect(builder).toBeInstanceOf(LlmClientBuilder);
  });

  test("should be able to instantiate SharedState", () => {
    const state = new SharedState();
    expect(state).toBeInstanceOf(SharedState);
  });

  test("should be able to instantiate FlowBuilder", () => {
    const builder = new FlowBuilder();
    expect(builder).toBeInstanceOf(FlowBuilder);
  });

  test("should be able to create WorkflowBuilder and WorkflowEngine", () => {
    const builder = WorkflowBuilder.create("test-workflow");
    const engine = WorkflowEngine.create();
    expect(builder).toBeInstanceOf(WorkflowBuilder);
    expect(engine).toBeInstanceOf(WorkflowEngine);
  });

  test("should be able to instantiate AgentCoordinator", () => {
    const coordinator = new AgentCoordinator();
    expect(coordinator).toBeInstanceOf(AgentCoordinator);
  });

  test("should validate native method existence on AgentBuilder", () => {
    const builder = new AgentBuilder();
    expect(typeof builder.provider).toBe("function");
    expect(typeof builder.model).toBe("function");
    expect(typeof builder.systemPrompt).toBe("function");
    expect(typeof builder.build).toBe("function");
  });

  test("should validate native method existence on StreamIterator", () => {
    expect(StreamIterator.prototype[Symbol.asyncIterator]).toBeDefined();
  });

  test("facade no longer references removed reasoning wrapper exports", () => {
    const source = readFileSync(join(import.meta.dir, "..", "src", "index.ts"), "utf8");
    expect(source.includes("createReActPattern")).toBe(false);
    expect(source.includes("reactPatternExecute")).toBe(false);
    expect(source.includes("createPlanExecutePattern")).toBe(false);
    expect(source.includes("planExecutePatternExecute")).toBe(false);
    expect(source.includes("createReflexionPattern")).toBe(false);
    expect(source.includes("reflexionPatternExecute")).toBe(false);
    expect(source.includes("createSelfAskPattern")).toBe(false);
    expect(source.includes("selfAskPatternExecute")).toBe(false);
    expect(source.includes("createTreeOfThoughtsPattern")).toBe(false);
    expect(source.includes("treeOfThoughtsPatternExecute")).toBe(false);
  });

  test("type declarations no longer expose removed pattern or reasoning wrappers", () => {
    const rootDeclarations = readFileSync(join(import.meta.dir, "..", "index.d.ts"), "utf8");
    const runtimeDeclarations = readFileSync(
      join(import.meta.dir, "..", "packages", "runtime-node", "native-loader.d.ts"),
      "utf8"
    );

    for (const declarations of [rootDeclarations, runtimeDeclarations]) {
      expect(declarations.includes("createDebatePattern")).toBe(false);
      expect(declarations.includes("debatePatternGetId")).toBe(false);
      expect(declarations.includes("hierarchicalPatternAddLayer")).toBe(false);
      expect(declarations.includes("pipelinePatternAddStage")).toBe(false);
      expect(declarations.includes("supervisorPatternGetId")).toBe(false);
      expect(declarations.includes("createPlanExecutePattern")).toBe(false);
      expect(declarations.includes("createReActPattern")).toBe(false);
      expect(declarations.includes("createReflexionPattern")).toBe(false);
      expect(declarations.includes("createSelfAskPattern")).toBe(false);
      expect(declarations.includes("createTreeOfThoughtsPattern")).toBe(false);
      expect(declarations.includes("planExecutePatternExecute")).toBe(false);
      expect(declarations.includes("reactPatternExecute")).toBe(false);
      expect(declarations.includes("reflexionPatternExecute")).toBe(false);
      expect(declarations.includes("selfAskPatternExecute")).toBe(false);
      expect(declarations.includes("treeOfThoughtsPatternExecute")).toBe(false);
      expect(
        declarations.includes(
          "static new(provider: string, model: string, maxIterations?: number | undefined | null): Promise<SelfAskPattern>"
        )
      ).toBe(true);
    }
  });
});
