/**
 * Thin handle over the Rust tool registry (active/deferred controls).
 */

import { native } from "./native/module";

export type ToolExecutionResult = {
  success: boolean;
  content: string;
  error?: string;
  typed_payload?: unknown;
};

export class ToolRegistry {
  private readonly _inner: any;

  constructor(inner?: unknown) {
    this._inner = inner ?? new native.ToolRegistry();
  }

  /** @internal */
  static fromNative(inner: unknown): ToolRegistry {
    return new ToolRegistry(inner);
  }

  get inner(): unknown {
    return this._inner;
  }

  register(tool: { inner?: unknown; _tool?: unknown } | unknown): void {
    const nativeTool = (tool as any)?._tool ?? (tool as any)?.inner ?? tool;
    this._inner.register(nativeTool);
  }

  registerDeferred(tool: { inner?: unknown; _tool?: unknown } | unknown): void {
    const nativeTool = (tool as any)?._tool ?? (tool as any)?.inner ?? tool;
    this._inner.registerDeferred(nativeTool);
  }

  activate(names: readonly string[]): void {
    this._inner.activate([...names]);
  }

  deactivate(names: readonly string[]): void {
    this._inner.deactivate([...names]);
  }

  setActiveTools(names?: readonly string[] | null): void {
    this._inner.setActiveTools(names == null ? null : [...names]);
  }

  activateAll(): void {
    this._inner.activateAll();
  }

  isActive(name: string): boolean {
    return this._inner.isActive(name);
  }

  activeToolNames(): string[] {
    return this._inner.activeToolNames();
  }

  async execute(name: string, arguments_: Record<string, unknown>): Promise<ToolExecutionResult> {
    return await this._inner.execute(name, arguments_);
  }

  getToolSchemas(): Array<{ name: string; description: string; parameters: unknown }> {
    return this._inner.getToolSchemas();
  }

  list(): Array<{ name: string; description: string; parameters: unknown }> {
    return this._inner.list();
  }
}
