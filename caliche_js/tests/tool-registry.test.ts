import { describe, expect, test } from "bun:test";
import { BuiltinTool, ToolRegistry } from "../src/index";

describe("tool registry", () => {
  test("calculator tool executes add", () => {
    const tool = BuiltinTool.calculator();
    const result = tool.execute({ operation: "add", a: 2, b: 2 });
    expect(result.success).toBe(true);
    expect(result.content).toContain("4");
  });

  test("inactive tools are hidden from schemas and execution", async () => {
    const registry = new ToolRegistry();
    registry.register(BuiltinTool.calculator());
    registry.registerDeferred(BuiltinTool.applyPatch());
    const schemas = registry.getToolSchemas();
    expect(schemas.length).toBe(1);
    expect(schemas[0]?.name).toBe("calculator");
    expect(registry.isActive("apply_patch")).toBe(false);

    const denied = await registry.execute("apply_patch", {
      patch: "*** Begin Patch\n*** Add File: /tmp/x\n+y\n*** End Patch\n",
    });
    expect(denied.success).toBe(false);
    expect(denied.error ?? "").toContain("not active");

    registry.activate(["apply_patch"]);
    expect(registry.isActive("apply_patch")).toBe(true);
    expect(registry.getToolSchemas().length).toBe(2);
    registry.deactivate(["apply_patch"]);
    expect(registry.isActive("apply_patch")).toBe(false);
    registry.activateAll();
    expect(registry.isActive("apply_patch")).toBe(true);
  });
});
