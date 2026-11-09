/**
 * Agent lifecycle hooks for intercepting and controlling agent behavior.
 *
 * This module provides hooks that can be registered to intercept agent operations
 * at specific points in the execution lifecycle.
 */

import { native } from "./native/module";

// Native types (opaque handles from Rust)
type NativePermissionMode = any;
type NativeHookMatcher = any;

/**
 * Permission mode for automatic permission handling
 */
export class PermissionMode {
  private _mode: NativePermissionMode;

  private constructor(mode: NativePermissionMode) {
    this._mode = mode;
  }

  /**
   * Get the inner native mode (for internal use)
   */
  get inner(): NativePermissionMode {
    return this._mode;
  }

  /**
   * Default mode - use hooks for permission decisions
   */
  static default(): PermissionMode {
    if (typeof native.JsPermissionModeDefault === "function") {
      return new PermissionMode(native.JsPermissionModeDefault());
    }
    return new PermissionMode(new native.JsPermissionMode("default"));
  }

  /**
   * Auto-accept file edit operations
   */
  static acceptEdits(): PermissionMode {
    if (typeof native.JsPermissionModeAcceptEdits === "function") {
      return new PermissionMode(native.JsPermissionModeAcceptEdits());
    }
    return new PermissionMode(new native.JsPermissionMode("accept-edits"));
  }

  /**
   * Auto-accept all tool calls
   */
  static acceptAll(): PermissionMode {
    if (typeof native.JsPermissionModeAcceptAll === "function") {
      return new PermissionMode(native.JsPermissionModeAcceptAll());
    }
    return new PermissionMode(new native.JsPermissionMode("accept-all"));
  }

  /**
   * Deny all tool calls
   */
  static denyAll(): PermissionMode {
    if (typeof native.JsPermissionModeDenyAll === "function") {
      return new PermissionMode(native.JsPermissionModeDenyAll());
    }
    return new PermissionMode(new native.JsPermissionMode("deny-all"));
  }

  /**
   * Interactive mode — require approval for dangerous operations
   */
  static interactive(): PermissionMode {
    if (typeof native.JsPermissionModeInteractive === "function") {
      return new PermissionMode(native.JsPermissionModeInteractive());
    }
    return new PermissionMode(new native.JsPermissionMode("interactive"));
  }

  /**
   * Create from string
   */
  static fromString(mode: string): PermissionMode {
    return new PermissionMode(new native.JsPermissionMode(mode));
  }

  toString(): string {
    const mode = this._mode as { toString?: () => string };
    if (typeof mode.toString === "function") {
      return mode.toString();
    }
    return "PermissionMode";
  }
}

/**
 * Hook matcher for selecting which tools a hook applies to
 */
export class HookMatcher {
  private _matcher: NativeHookMatcher;

  private constructor(matcher: NativeHookMatcher) {
    this._matcher = matcher;
  }

  /**
   * Get the inner native matcher (for internal use)
   */
  get inner(): NativeHookMatcher {
    return this._matcher;
  }

  /**
   * Match a specific tool by name
   */
  static tool(toolName: string): HookMatcher {
    if (typeof native.JsHookMatcherTool === "function") {
      return new HookMatcher(native.JsHookMatcherTool(toolName));
    }
    return new HookMatcher(new native.JsHookMatcher("tool", toolName));
  }

  /**
   * Match all tools
   */
  static all(): HookMatcher {
    if (typeof native.JsHookMatcherAll === "function") {
      return new HookMatcher(native.JsHookMatcherAll());
    }
    return new HookMatcher(new native.JsHookMatcher("all"));
  }
}

/**
 * Context passed to hooks during tool execution
 */
export interface HookContext {
  /** Name of the tool being called */
  tool_name: string;
  /** Input arguments for the tool */
  tool_input: Record<string, unknown>;
  /** Unique ID for this tool use */
  tool_use_id: string;
}

/**
 * Result returned by lifecycle hooks
 */
export interface HookResult {
  /** Permission decision for the operation */
  permission_decision: {
    decision: "allow" | "deny";
    reason?: string;
  };
  /** Optional additional output from the hook */
  hook_specific_output?: unknown;
}

/**
 * Lifecycle hook interface for intercepting agent operations
 *
 * Implement this interface to create custom hooks that can intercept
 * and control tool execution.
 */
export interface LifecycleHook {
  /**
   * Called before a tool is executed
   *
   * Return a HookResult with a permission decision. If denied, the tool
   * will not be executed and an error will be returned instead.
   */
  onPreToolUse(ctx: HookContext): Promise<HookResult> | HookResult;

  /**
   * Called after a tool is executed
   *
   * This hook cannot deny the operation (it already happened), but can
   * perform side effects like logging, metrics, or modifying the result.
   */
  onPostToolUse?(
    ctx: HookContext,
    result: {
      success: boolean;
      content: string;
      error?: string;
    }
  ): Promise<HookResult> | HookResult;
}
