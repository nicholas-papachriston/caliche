/**
 * Typed wrappers for common built-in tools.
 */

import { BuiltinTool } from "./rag";

/** Create a calculator tool (typed BuiltinTool wrapper). */
export function createCalculatorTool(): BuiltinTool {
  return BuiltinTool.calculator();
}

/** Create an HTTP tool. */
export function createHttpTool(): BuiltinTool {
  return BuiltinTool.http();
}

/** Create a code-execution tool. */
export function createCodeExecutionTool(): BuiltinTool {
  return BuiltinTool.codeExecution();
}
