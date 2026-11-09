export type ContextRefKind =
  | { kind: "file"; path: string }
  | { kind: "directory"; path: string }
  | { kind: "url"; url: string }
  | { kind: "symbol"; name: string }
  | { kind: "clipboard" }
  | { kind: "git_diff" }
  | { kind: "git_status" }
  | { kind: "artifact"; selector: string }
  | { kind: "resource"; server: string | null; uri: string };

export type SlashIntent =
  | { intent: "plan" }
  | { intent: "ask" }
  | { intent: "agent"; arg?: string | null }
  | { intent: "mode" }
  | { intent: "show_plan" }
  | { intent: "resume"; id?: string | null }
  | { intent: "attach"; paths: string[] };

export type PermissionMode = "default" | "accept-edits" | "accept-all" | "deny-all" | "interactive";

/** Parse a context reference from a string without the leading `@`. */
export function parseContextRefKind(input: string): ContextRefKind | null;

/** Parse shared host slash intents; returns null outside the shared set. */
export function parseSlashIntent(input: string): SlashIntent | null;

/** Normalize a permission mode string to the wire enum. */
export function parsePermissionMode(value: string): PermissionMode | null;
