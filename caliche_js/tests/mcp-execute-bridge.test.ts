import { describe, expect, test } from "bun:test";
import { wrapMcpToolExecuteForNapi } from "../src/mcp_execute_bridge";

describe("wrapMcpToolExecuteForNapi", () => {
  test("sync string result", () => {
    const w = wrapMcpToolExecuteForNapi(() => "hello");
    const s = w(null, {}) as string;
    expect(JSON.parse(s)).toEqual({ success: true, content: "hello" });
  });

  test("sync object result", () => {
    const w = wrapMcpToolExecuteForNapi(() => ({
      success: true,
      content: "ok",
    }));
    const s = w(null, {}) as string;
    expect(JSON.parse(s)).toEqual({ success: true, content: "ok" });
  });

  test("async result resolves to JSON string", async () => {
    const w = wrapMcpToolExecuteForNapi(async () => "delayed");
    const out = w(null, {});
    expect(out).toBeInstanceOf(Promise);
    const s = (await out) as string;
    expect(JSON.parse(s)).toEqual({ success: true, content: "delayed" });
  });

  test("async rejection becomes error payload", async () => {
    const w = wrapMcpToolExecuteForNapi(async () => {
      throw new Error("boom");
    });
    const s = (await w(null, {})) as string;
    const parsed = JSON.parse(s) as { success: boolean; error?: string };
    expect(parsed.success).toBe(false);
    expect(parsed.error).toBe("boom");
  });

  test("sync null becomes error payload", () => {
    const w = wrapMcpToolExecuteForNapi(() => null as unknown as string);
    const s = w(null, {}) as string;
    const parsed = JSON.parse(s) as { success: boolean; error?: string };
    expect(parsed.success).toBe(false);
    expect(parsed.error).toContain("null");
  });

  test("error-first NAPI callback shape passes payload args", () => {
    let seen: Record<string, unknown> | undefined;
    const w = wrapMcpToolExecuteForNapi((args) => {
      seen = args;
      const x = Number(args.x ?? 0);
      const y = Number(args.y ?? 0);
      return { success: true, content: String(x * y) };
    });
    const s = w(null, { x: 6, y: 7 }) as string;
    expect(seen).toEqual({ x: 6, y: 7 });
    expect(JSON.parse(s)).toEqual({ success: true, content: "42" });
  });

  test("error-first NAPI callback propagates err", () => {
    const w = wrapMcpToolExecuteForNapi(() => "unused");
    const s = w(new Error("native failed"), { x: 1 }) as string;
    const parsed = JSON.parse(s) as { success: boolean; error?: string };
    expect(parsed.success).toBe(false);
    expect(parsed.error).toBe("native failed");
  });
});
