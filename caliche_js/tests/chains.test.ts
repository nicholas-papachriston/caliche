import { beforeAll, describe, expect, test } from "bun:test";
import { AgentBuilder, LLMChain, MapReduceChain, SequentialChain } from "../src/index";
import { liveSuite } from "./test-env";

const { liveTest, agentBuilder } = await liveSuite("Chains");

describe("Chains", () => {
  let agent: Awaited<ReturnType<typeof AgentBuilder.prototype.build>>;

  beforeAll(async () => {
    const builder = agentBuilder();

    agent = await builder.build();
  });

  describe("LLMChain", () => {
    test("should create an LLMChain instance", () => {
      const chain = new LLMChain(agent, "Say hello to {name}");
      expect(chain).toBeInstanceOf(LLMChain);
    });

    test("should have a run method", () => {
      const chain = new LLMChain(agent, "Say hello to {name}");
      expect(typeof chain.run).toBe("function");
    });

    liveTest("should run chain with input variables", async () => {
      const chain = new LLMChain(agent, "Say hello to {name}");
      const result = await chain.run({ name: "Alice" });
      expect(result).toBeDefined();
      expect(typeof result).toBe("object");
    });

    liveTest("should handle prompt template with multiple variables", async () => {
      const chain = new LLMChain(agent, "Hello {name}, you are {age} years old");
      const result = await chain.run({ name: "Bob", age: "25" });
      expect(result).toBeDefined();
    });

    liveTest("should handle empty prompt template", async () => {
      const chain = new LLMChain(agent, "Hello");
      const result = await chain.run({});
      expect(result).toBeDefined();
    });
  });

  describe("SequentialChain", () => {
    test("should create a SequentialChain instance", () => {
      const chain = new SequentialChain();
      expect(chain).toBeInstanceOf(SequentialChain);
    });

    test("should have addChain method", () => {
      const chain = new SequentialChain();
      expect(typeof chain.addChain).toBe("function");
    });

    test("should have a run method", () => {
      const chain = new SequentialChain();
      expect(typeof chain.run).toBe("function");
    });

    test("should add chains to sequence", () => {
      const seqChain = new SequentialChain();
      const llmChain1 = new LLMChain(agent, "First: {input}");
      const llmChain2 = new LLMChain(agent, "Second: {input}");

      expect(() => {
        seqChain.addChain("first", llmChain1);
        seqChain.addChain("second", llmChain2);
      }).not.toThrow();
    });

    liveTest(
      "should run sequential chain",
      async () => {
        const seqChain = new SequentialChain();
        const llmChain = new LLMChain(agent, "Process: {input}");
        seqChain.addChain("process", llmChain);

        const result = await seqChain.run({ input: "test" });
        expect(result).toBeDefined();
      },
      30000
    );

    liveTest(
      "should chain multiple steps",
      async () => {
        const seqChain = new SequentialChain();
        const step1 = new LLMChain(agent, "Step 1: {input}");
        // Sequential steps receive prior chain output under `input`.
        const step2 = new LLMChain(agent, "Step 2: {input}");
        seqChain.addChain("step1", step1);
        seqChain.addChain("step2", step2);

        const result = await seqChain.run({ input: "data" });
        expect(result).toBeDefined();
      },
      45000
    );
  });

  describe("MapReduceChain", () => {
    test("should create a MapReduceChain instance", () => {
      const mapAgent = agent;
      const reduceAgent = agent;
      const chain = new MapReduceChain(mapAgent, "Map: {input}", reduceAgent, "Reduce: {output}");
      expect(chain).toBeInstanceOf(MapReduceChain);
    });

    test("should have a run method", () => {
      const mapAgent = agent;
      const reduceAgent = agent;
      const chain = new MapReduceChain(mapAgent, "Map: {input}", reduceAgent, "Reduce: {output}");
      expect(typeof chain.run).toBe("function");
    });

    liveTest(
      "should run map-reduce chain with array input",
      async () => {
        const mapAgent = agent;
        const reduceAgent = agent;
        const chain = new MapReduceChain(
          mapAgent,
          "Summarize: {input}",
          reduceAgent,
          "Combine: {output}"
        );

        const result = await chain.run([{ input: "First document" }, { input: "Second document" }]);
        expect(result).toBeDefined();
      },
      60000
    );

    liveTest(
      "should handle single item array",
      async () => {
        const mapAgent = agent;
        const reduceAgent = agent;
        const chain = new MapReduceChain(mapAgent, "Map: {input}", reduceAgent, "Reduce: {output}");

        const result = await chain.run([{ input: "single item" }]);
        expect(result).toBeDefined();
      },
      45000
    );

    liveTest("should handle empty array", async () => {
      const mapAgent = agent;
      const reduceAgent = agent;
      const chain = new MapReduceChain(mapAgent, "Map: {input}", reduceAgent, "Reduce: {output}");

      const result = await chain.run([]);
      expect(result).toBeDefined();
    });
  });
});
