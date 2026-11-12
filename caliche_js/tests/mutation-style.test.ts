import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  AgentBuilder,
  BuiltinTool,
  createApplyPatchTool,
  createCalculatorTool,
  createCodeExecutionTool,
  createFsTool,
  createHttpTool,
  MutationStyle,
} from "../src/index";

describe("mutation style", () => {
  test("mutation style factories", () => {
    expect(MutationStyle.editWrite().toString()).toBe("EditWrite");
    expect(MutationStyle.applyPatch().toString()).toBe("ApplyPatch");
    expect(MutationStyle.fromString("apply_patch").toString()).toBe("ApplyPatch");
  });

  test("fs tool apply_patch style strips write/edit", () => {
    const tool = BuiltinTool.fs(MutationStyle.applyPatch());
    expect(tool.name()).toBe("fs");
    const params = tool.parameters() as {
      properties: { op: { enum: string[] } };
    };
    const ops = params.properties.op.enum;
    expect(ops).toContain("read");
    expect(ops).not.toContain("write");
    expect(ops).not.toContain("edit");

    const result = tool.execute({ op: "write", path: "x.txt", content: "nope" });
    expect(result.success).toBe(false);
    expect(result.error ?? "").toContain("apply_patch");
  });

  test("apply_patch tool mutates file", () => {
    // A tool used outside an agent works in the current directory.
    const dir = mkdtempSync(join(tmpdir(), "caliche-js-patch-"));
    const previous = process.cwd();
    process.chdir(dir);
    try {
      const target = join(dir, "note.txt");
      const patch =
        "*** Begin Patch\n" +
        "*** Add File: note.txt\n" +
        "+alpha\n" +
        "+beta\n" +
        "*** End Patch\n";
      const tool = BuiltinTool.applyPatch();
      expect(tool.name()).toBe("apply_patch");
      const result = tool.execute({ patch });
      expect(result.success).toBe(true);
      expect(readFileSync(target, "utf8")).toBe("alpha\nbeta\n");
    } finally {
      process.chdir(previous);
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("create_* tool helpers match BuiltinTool", () => {
    expect(createCalculatorTool().name()).toBe("calculator");
    expect(createFsTool(MutationStyle.applyPatch()).name()).toBe("fs");
    expect(createApplyPatchTool().name()).toBe("apply_patch");
    expect(createHttpTool().name()).toBe("http");
    expect(createCodeExecutionTool().name()).toBe("execute_python");
  });

  test("builder mutationStyle registers apply_patch on built agent", async () => {
    const agent = await new AgentBuilder()
      .provider("lmstudio")
      .model("local-model")
      .mutationStyle(MutationStyle.applyPatch())
      .enableTool("fs")
      .enableTool("bash")
      .build();

    const tools = (await agent.listTools()).map((t) => t.name).sort();
    expect(tools).toContain("apply_patch");
    expect(tools).toContain("fs");
    expect(tools).toContain("bash");

    const registry = await agent.toolRegistry();
    expect(registry.isActive("apply_patch")).toBe(true);
    const fsSchema = registry.getToolSchemas().find((s) => s.name === "fs");
    const ops = (fsSchema?.parameters as { properties?: { op?: { enum?: string[] } } })?.properties
      ?.op?.enum;
    expect(ops).toBeDefined();
    expect(ops).not.toContain("write");
    expect(ops).not.toContain("edit");
  });
});
