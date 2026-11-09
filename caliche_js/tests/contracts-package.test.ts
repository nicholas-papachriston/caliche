import { describe, expect, test } from "bun:test";
import {
  parseContextRefKind,
  parsePermissionMode,
  parseSlashIntent,
} from "../packages/contracts/index.js";

describe("@caliche/contracts", () => {
  test("parseContextRefKind covers common mentions", () => {
    expect(parseContextRefKind("diff")).toEqual({ kind: "git_diff" });
    expect(parseContextRefKind("clip")).toEqual({ kind: "clipboard" });
    expect(parseContextRefKind("https://example.com")).toEqual({
      kind: "url",
      url: "https://example.com",
    });
    expect(parseContextRefKind("src/main.rs")).toEqual({
      kind: "file",
      path: "src/main.rs",
    });
    expect(parseContextRefKind("artifact:run-1")).toEqual({
      kind: "artifact",
      selector: "run-1",
    });
    expect(parseContextRefKind("resource:mcp::file://x")).toEqual({
      kind: "resource",
      server: "mcp",
      uri: "file://x",
    });
  });

  test("parseSlashIntent covers shared host intents", () => {
    expect(parseSlashIntent("/plan")).toEqual({ intent: "plan" });
    expect(parseSlashIntent("/agent research")).toEqual({
      intent: "agent",
      arg: "research",
    });
    expect(parseSlashIntent("/showplan")).toEqual({ intent: "show_plan" });
    expect(parseSlashIntent("/attach ./a.png ./b.pdf")).toEqual({
      intent: "attach",
      paths: ["./a.png", "./b.pdf"],
    });
    expect(parseSlashIntent("/help")).toBeNull();
    expect(parseSlashIntent("hello")).toBeNull();
  });

  test("parsePermissionMode accepts one spelling per mode", () => {
    expect(parsePermissionMode("Accept-Edits")).toBe("accept-edits");
    expect(parsePermissionMode("deny-all")).toBe("deny-all");
    expect(parsePermissionMode("deny_all")).toBeNull();
    expect(parsePermissionMode("acceptall")).toBeNull();
    expect(parsePermissionMode("nope")).toBeNull();
  });
});
