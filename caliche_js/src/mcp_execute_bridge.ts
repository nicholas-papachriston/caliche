/**
 * Bridges MCP tool `execute` functions to the NAPI ThreadsafeFunction contract:
 * the native layer accepts either a JSON string or Promise&lt;string&gt; (see `call_js_sync_bridge`).
 */

import { napiToolPayload, unwrapNapiCallbackArgs } from "./napi-callback";

export type McpToolExecuteResult = { success: boolean; content: string; error?: string } | string;

export function formatMcpToolExecuteResult(result: McpToolExecuteResult): string {
  if (result === null || result === undefined) {
    return formatMcpToolExecuteError(new Error("tool execute returned null or undefined"));
  }
  if (typeof result === "string") {
    return JSON.stringify({ success: true, content: result });
  }
  return JSON.stringify(result);
}

export function formatMcpToolExecuteError(error: unknown): string {
  return JSON.stringify({
    success: false,
    content: "",
    error: error instanceof Error ? error.message : String(error),
  });
}

/**
 * Wraps a sync or async tool executor for registration with the native MCP builder.
 *
 * NAPI-RS ThreadsafeFunctions use an error-first callback shape `(err, payload)`.
 */
export function wrapMcpToolExecuteForNapi(
  execute: (args: Record<string, unknown>) => McpToolExecuteResult | Promise<McpToolExecuteResult>
): (...callbackArgs: unknown[]) => string | Promise<string> {
  return (...callbackArgs: unknown[]) => {
    try {
      const { err } = unwrapNapiCallbackArgs(callbackArgs);
      if (err) {
        return formatMcpToolExecuteError(err);
      }
      const out = execute(napiToolPayload(callbackArgs));
      if (
        out !== null &&
        typeof out === "object" &&
        typeof (out as Promise<unknown>).then === "function"
      ) {
        return (out as Promise<McpToolExecuteResult>).then(
          (resolved) => formatMcpToolExecuteResult(resolved),
          (rejection) => formatMcpToolExecuteError(rejection)
        );
      }
      return formatMcpToolExecuteResult(out as McpToolExecuteResult);
    } catch (error) {
      return formatMcpToolExecuteError(error);
    }
  };
}
