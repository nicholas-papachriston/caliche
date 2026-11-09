/**
 * JSONL session trees with branch restore and compaction entries.
 */

import { native } from "./native/module";

export class JsonlSessionHeader {
  private readonly _inner: any;

  constructor(id: string, cwd?: string | null) {
    this._inner = new native.JsonlSessionHeader(id, cwd ?? null);
  }

  /** @internal */
  static fromNative(inner: unknown): JsonlSessionHeader {
    const header = Object.create(JsonlSessionHeader.prototype) as JsonlSessionHeader;
    (header as any)._inner = inner;
    return header;
  }

  get inner(): unknown {
    return this._inner;
  }

  get id(): string {
    return this._inner.id;
  }

  get version(): number {
    return this._inner.version;
  }

  get cwd(): string | null {
    return this._inner.cwd ?? null;
  }

  get parentSession(): string | null {
    return this._inner.parentSession ?? null;
  }

  setParentSession(parentSession?: string | null): void {
    this._inner.setParentSession(parentSession ?? null);
  }
}

export class JsonlSessionEntry {
  private readonly _inner: any;

  private constructor(inner: unknown) {
    this._inner = inner;
  }

  static message(
    id: string,
    message: Record<string, unknown>,
    parentId?: string | null
  ): JsonlSessionEntry {
    return new JsonlSessionEntry(native.JsonlSessionEntry.message(id, message, parentId ?? null));
  }

  static branchSummary(
    id: string,
    fromId: string,
    summary: string,
    parentId?: string | null
  ): JsonlSessionEntry {
    return new JsonlSessionEntry(
      native.JsonlSessionEntry.branchSummary(id, fromId, summary, parentId ?? null)
    );
  }

  static compactionSummary(
    id: string,
    summary: string,
    parentId?: string | null,
    tokensBefore?: number | null
  ): JsonlSessionEntry {
    return new JsonlSessionEntry(
      native.JsonlSessionEntry.compactionSummary(
        id,
        summary,
        parentId ?? null,
        tokensBefore ?? null
      )
    );
  }

  static modelChange(
    id: string,
    provider: string,
    modelId: string,
    parentId?: string | null
  ): JsonlSessionEntry {
    return new JsonlSessionEntry(
      native.JsonlSessionEntry.modelChange(id, provider, modelId, parentId ?? null)
    );
  }

  /** @internal */
  static fromNative(inner: unknown): JsonlSessionEntry {
    return new JsonlSessionEntry(inner);
  }

  get inner(): unknown {
    return this._inner;
  }

  get id(): string {
    return this._inner.id;
  }

  get parentId(): string | null {
    return this._inner.parentId ?? null;
  }
}

export class JsonlSessionTree {
  private readonly _inner: any;

  constructor(header: JsonlSessionHeader | unknown) {
    if (header instanceof JsonlSessionHeader) {
      this._inner = new native.JsonlSessionTree(header.inner);
    } else {
      this._inner = header;
    }
  }

  /** @internal */
  static fromNative(inner: unknown): JsonlSessionTree {
    return new JsonlSessionTree(inner);
  }

  get header(): JsonlSessionHeader {
    return JsonlSessionHeader.fromNative(this._inner.header);
  }

  append(entry: JsonlSessionEntry): void {
    this._inner.append(entry.inner);
  }

  leaves(): string[] {
    return this._inner.leaves();
  }

  branchPath(leafId: string): unknown[] {
    return this._inner.branchPath(leafId);
  }

  restoreBranchMessages(leafId: string): Array<Record<string, unknown>> {
    return this._inner.restoreBranchMessages(leafId);
  }

  branchFrom(fromId: string, branchEntryId: string, summary: string): string {
    return this._inner.branchFrom(fromId, branchEntryId, summary);
  }

  entryCount(): number {
    return this._inner.entryCount();
  }

  entries(): JsonlSessionEntry[] {
    return (this._inner.entries() as unknown[]).map((e) => JsonlSessionEntry.fromNative(e));
  }
}

export class JsonlSessionStore {
  private readonly _inner: any;

  constructor(path: string) {
    this._inner = new native.JsonlSessionStore(path);
  }

  get path(): string {
    return this._inner.path;
  }

  create(header: JsonlSessionHeader): void {
    this._inner.create(header.inner);
  }

  appendEntry(entry: JsonlSessionEntry): void {
    this._inner.appendEntry(entry.inner);
  }

  load(): JsonlSessionTree {
    return JsonlSessionTree.fromNative(this._inner.load());
  }
}

export function parseJsonlSession(text: string): JsonlSessionTree {
  return JsonlSessionTree.fromNative(native.parseJsonlSession(text));
}

export function newEntryId(): string {
  return native.newEntryId();
}
