import { describe, expect, test } from "bun:test";
import {
  defaultInspectionManager,
  InspectionAction,
  PathSecurityInspector,
  PermissionMode,
  PermissionModeInspector,
  ToolInspectionManager,
} from "../src/index";

describe("tool inspection", () => {
  test("permission mode interactive factory", () => {
    expect(PermissionMode.interactive().toString()).toBe("Interactive");
    expect(PermissionMode.fromString("interactive").toString()).toBe("Interactive");
  });

  test("path security inspector denies foreign absolute", () => {
    const inspector = new PathSecurityInspector("/tmp/my-workspace");
    const allowed = inspector.inspect("fs", { op: "read", path: "src/lib.rs" }, "1");
    expect(allowed[0]?.action.kind()).toBe("allow");

    const denied = inspector.inspect("fs", { op: "read", path: "/etc/passwd" }, "2");
    expect(denied[0]?.action.kind()).toBe("deny");
  });

  test("permission mode inspector deny all", () => {
    const inspector = new PermissionModeInspector(PermissionMode.denyAll());
    const results = inspector.inspect("fs", { op: "read", path: "." }, "c1");
    expect(results[0]?.action.kind()).toBe("deny");
  });

  test("default inspection manager and tool manager", () => {
    const manager = defaultInspectionManager(PermissionMode.acceptAll(), "/tmp/ws");
    const action = manager.inspectOne("fs", { op: "read", path: "a.rs" }, "x");
    expect(action.kind()).toBe("allow");

    const built = new ToolInspectionManager();
    built.addPermissionMode(PermissionMode.denyAll());
    const denied = built.inspectOne("bash", { command: "ls" }, "b");
    expect(denied.kind()).toBe("deny");
    expect(built.inspectorNames()).toContain("permission_mode");
  });

  test("inspection action helpers", () => {
    expect(InspectionAction.allow().kind()).toBe("allow");
    expect(InspectionAction.deny("no").reason()).toBe("no");
    expect(InspectionAction.requireApproval("ask").kind()).toBe("require_approval");
  });
});
