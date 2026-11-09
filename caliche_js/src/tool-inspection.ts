/**
 * Goose-style tool inspection (path security and permission modes).
 */

import { native } from "./native/module";

export class InspectionAction {
  private readonly _inner: any;

  private constructor(inner: unknown) {
    this._inner = inner;
  }

  static allow(): InspectionAction {
    return new InspectionAction(native.InspectionAction.allow());
  }

  static deny(reason: string): InspectionAction {
    return new InspectionAction(native.InspectionAction.deny(reason));
  }

  static requireApproval(reason: string): InspectionAction {
    return new InspectionAction(native.InspectionAction.requireApproval(reason));
  }

  /** @internal */
  static fromNative(inner: unknown): InspectionAction {
    return new InspectionAction(inner);
  }

  kind(): string {
    return this._inner.kind();
  }

  reason(): string | null {
    return this._inner.reason() ?? null;
  }

  toString(): string {
    return this._inner.toString();
  }
}

export class PathSecurityInspector {
  private readonly _inner: any;

  constructor(workspaceRoot: string) {
    this._inner = new native.PathSecurityInspector(workspaceRoot);
  }

  name(): string {
    return this._inner.name();
  }

  inspect(
    toolName: string,
    arguments_: Record<string, unknown>,
    callId: string
  ): Array<{ action: InspectionAction }> {
    const results = this._inner.inspect(toolName, arguments_, callId) as any[];
    return results.map((r) => ({
      action: InspectionAction.fromNative(r.action),
      callId: r.callId,
      inspectorName: r.inspectorName,
      reason: r.reason,
      confidence: r.confidence,
    })) as Array<{ action: InspectionAction }>;
  }
}

export class PermissionModeInspector {
  private readonly _inner: any;

  constructor(mode: { inner: unknown }) {
    this._inner = new native.PermissionModeInspector(mode.inner);
  }

  name(): string {
    return this._inner.name();
  }

  inspect(
    toolName: string,
    arguments_: Record<string, unknown>,
    callId: string
  ): Array<{ action: InspectionAction }> {
    const results = this._inner.inspect(toolName, arguments_, callId) as any[];
    return results.map((r) => ({
      action: InspectionAction.fromNative(r.action),
    }));
  }
}

export class ToolInspectionManager {
  private readonly _inner: any;

  constructor(inner?: unknown) {
    this._inner = inner ?? new native.ToolInspectionManager();
  }

  /** @internal */
  static fromNative(inner: unknown): ToolInspectionManager {
    return new ToolInspectionManager(inner);
  }

  get inner(): unknown {
    return this._inner;
  }

  addPathSecurity(workspaceRoot: string): void {
    this._inner.addPathSecurity(workspaceRoot);
  }

  addPermissionMode(mode: { inner: unknown }): void {
    this._inner.addPermissionMode(mode.inner);
  }

  inspectorNames(): string[] {
    return this._inner.inspectorNames();
  }

  inspectOne(
    toolName: string,
    arguments_: Record<string, unknown>,
    callId: string
  ): InspectionAction {
    return InspectionAction.fromNative(this._inner.inspectOne(toolName, arguments_, callId));
  }

  asHook(): InspectionHookAdapter {
    return InspectionHookAdapter.fromNative(this._inner.asHook());
  }
}

export class InspectionHookAdapter {
  private readonly _inner: any;

  private constructor(inner: unknown) {
    this._inner = inner;
  }

  static fromNative(inner: unknown): InspectionHookAdapter {
    return new InspectionHookAdapter(inner);
  }

  static fromManager(manager: ToolInspectionManager): InspectionHookAdapter {
    return new InspectionHookAdapter(new native.InspectionHookAdapter(manager.inner));
  }

  get inner(): unknown {
    return this._inner;
  }
}

export function defaultInspectionManager(
  permissionMode: { inner: unknown },
  workspaceRoot?: string | null
): ToolInspectionManager {
  return ToolInspectionManager.fromNative(
    native.defaultInspectionManager(permissionMode.inner, workspaceRoot ?? null)
  );
}
