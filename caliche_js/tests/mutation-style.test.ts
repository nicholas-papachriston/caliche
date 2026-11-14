import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
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

    const dir = mkdtempSync(join(tmpdir(), "caliche-js-style-"));
    try {
      const result = tool.execute({ op: "write", path: "x.txt", content: "nope" }, dir);
      expect(result.success).toBe(false);
      expect(result.error ?? "").toContain("apply_patch");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("apply_patch tool mutates a file in the workspace it is given", () => {
    const dir = mkdtempSync(join(tmpdir(), "caliche-js-patch-"));
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
      const result = tool.execute({ patch }, dir);
      expect(result.success).toBe(true);
      expect(readFileSync(target, "utf8")).toBe("alpha\nbeta\n");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("apply_patch outside the workspace it is given is refused", () => {
    const dir = mkdtempSync(join(tmpdir(), "caliche-js-patch-"));
    const outside = mkdtempSync(join(tmpdir(), "caliche-js-outside-"));
    try {
      const target = join(outside, "note.txt");
      const patch = `*** Begin Patch\n*** Add File: ${target}\n+alpha\n*** End Patch\n`;
      const result = BuiltinTool.applyPatch().execute({ patch }, dir);
      expect(result.success).toBe(false);
      expect(result.error ?? "").toStartWith("Tool 'apply_patch' was refused: ");
      expect(existsSync(target)).toBe(false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
    }
  });

  test("apply_patch without a workspace is refused and writes nothing", () => {
    const dir = mkdtempSync(join(tmpdir(), "caliche-js-patch-"));
    const previous = process.cwd();
    process.chdir(dir);
    try {
      const patch = "*** Begin Patch\n*** Add File: note.txt\n+alpha\n*** End Patch\n";
      const result = BuiltinTool.applyPatch().execute({ patch });
      expect(result.success).toBe(false);
      expect(result.error ?? "").toStartWith("Tool 'apply_patch' was refused: ");
      expect(existsSync(join(dir, "note.txt"))).toBe(false);
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
  test("a workspace must be an existing absolute directory", () => {
    const tool = BuiltinTool.fs(MutationStyle.editWrite());
    const dir = mkdtempSync(join(tmpdir(), "caliche-js-root-"));
    try {
      expect(() => tool.execute({ op: "read", path: "x.txt" }, "relative/ws")).toThrow(
        "workspace must be an absolute path: relative/ws",
      );
      const missing = join(dir, "missing");
      expect(() => tool.execute({ op: "read", path: "x.txt" }, missing)).toThrow(
        `workspace cannot be resolved: ${missing}`,
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
