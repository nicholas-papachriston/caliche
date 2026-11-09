import { beforeAll, describe, expect, test } from "bun:test";
import {
  RAG_CUSTOM_BACKEND_UNSUPPORTED,
  RAG_SERVICE_REQUIRES_FEATURE,
  RAG_TOOL_REQUIRES_FEATURE,
  RagService,
  RagTool,
} from "../src/index";
import { liveSuite } from "./test-env";

/** `tools-rag` is not a default feature, so most bindings lack the RAG classes. */
const RAG_AVAILABLE = RagService.isAvailable();
const { liveTest, agentBuilder } = await liveSuite("RAG");
const ragTest = test.skipIf(!RAG_AVAILABLE);
// Indexing and querying call the embedding model, so they also need a live provider.
const liveRagTest = RAG_AVAILABLE ? liveTest : test.skipIf(true);
if (!RAG_AVAILABLE) {
  console.log("RAG: skipped tools-rag tests: the native binding was built without tools-rag");
}

describe("RAG", () => {
  let agent: Awaited<ReturnType<ReturnType<typeof agentBuilder>["build"]>>;

  beforeAll(async () => {
    agent = await agentBuilder().build();
  });

  describe("without tools-rag", () => {
    const missingTest = test.skipIf(RAG_AVAILABLE);

    missingTest("RagService.new names the missing feature", async () => {
      await expect(RagService.new(agent)).rejects.toThrow(RAG_SERVICE_REQUIRES_FEATURE);
    });

    missingTest("RagService.withBackend names the missing feature", async () => {
      await expect(RagService.withBackend(agent)).rejects.toThrow(RAG_SERVICE_REQUIRES_FEATURE);
    });

    missingTest("RagTool.new names the missing feature", async () => {
      await expect(RagTool.new(agent)).rejects.toThrow(RAG_TOOL_REQUIRES_FEATURE);
    });
  });

  test("RagService.withBackend refuses a custom backend config", async () => {
    await expect(RagService.withBackend(agent, { backend: "redis" })).rejects.toThrow(
      RAG_CUSTOM_BACKEND_UNSUPPORTED
    );
  });

  describe("RagService", () => {
    ragTest("creates a service with every method", async () => {
      const service = await RagService.new(agent);
      expect(service).toBeInstanceOf(RagService);
      expect(
        ["indexDocument", "query", "enhancedQuery", "graphSearch", "stats", "clear"].filter(
          (name) => typeof Reflect.get(service, name) !== "function"
        )
      ).toEqual([]);
    });

    ragTest("withBackend with no config uses the default backend", async () => {
      expect(await RagService.withBackend(agent)).toBeInstanceOf(RagService);
    });

    liveRagTest(
      "indexes, counts, queries and clears a document",
      async () => {
        const service = await RagService.new(agent);
        const indexed = await service.indexDocument("Test document about AI", "doc-1");
        expect(indexed.chunk_count).toBe(1);
        expect((await service.stats()).text_count).toBe(1);

        const found = await service.query("What is AI?", 5);
        expect(found.results.length).toBe(1);

        await service.clear();
        expect((await service.stats()).text_count).toBe(0);
      },
      30000
    );
  });

  describe("RagTool", () => {
    ragTest("exposes its RagService", async () => {
      const tool = await RagTool.new(agent);
      expect(tool).toBeInstanceOf(RagTool);
      expect(tool.service()).toBeInstanceOf(RagService);
    });

    liveRagTest(
      "shares its backend with the service it returns",
      async () => {
        const tool = await RagTool.new(agent);
        const service = tool.service();
        await service.indexDocument("Shared document", "doc-1");
        expect((await tool.service().stats()).text_count).toBe(1);
      },
      30000
    );
  });
});
