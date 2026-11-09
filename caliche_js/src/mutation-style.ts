/**
 * File mutation style: structured fs write/edit XOR freeform apply_patch.
 */

import { native } from "./native/module";
import { BuiltinTool } from "./rag";

export class MutationStyle {
  private readonly _inner: unknown;

  private constructor(inner: unknown) {
    this._inner = inner;
  }

  get inner(): unknown {
    return this._inner;
  }

  static editWrite(): MutationStyle {
    return new MutationStyle(native.MutationStyle.editWrite());
  }

  static applyPatch(): MutationStyle {
    return new MutationStyle(native.MutationStyle.applyPatch());
  }

  static fromString(style: string): MutationStyle {
    return new MutationStyle(new native.MutationStyle(style));
  }

  toString(): string {
    const inner = this._inner as { toString(): string };
    return inner.toString();
  }
}

/** Create an fs tool, optionally with a mutation style. */
export function createFsTool(mutationStyle?: MutationStyle | null): BuiltinTool {
  return BuiltinTool.fs(mutationStyle ?? null);
}

/** Create an apply_patch tool. */
export function createApplyPatchTool(): BuiltinTool {
  return BuiltinTool.applyPatch();
}
