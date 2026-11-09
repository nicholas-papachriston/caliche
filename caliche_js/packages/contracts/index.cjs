"use strict";

/**
 * Pure wire helpers mirroring `caliche_contracts` (Rust).
 * No native binding dependency.
 */

/**
 * @typedef {{ kind: "file", path: string }
 *   | { kind: "directory", path: string }
 *   | { kind: "url", url: string }
 *   | { kind: "symbol", name: string }
 *   | { kind: "clipboard" }
 *   | { kind: "git_diff" }
 *   | { kind: "git_status" }
 *   | { kind: "artifact", selector: string }
 *   | { kind: "resource", server: string | null, uri: string }} ContextRefKind
 */

/**
 * @typedef {{ intent: "plan" }
 *   | { intent: "ask" }
 *   | { intent: "agent", arg?: string | null }
 *   | { intent: "mode" }
 *   | { intent: "show_plan" }
 *   | { intent: "resume", id?: string | null }
 *   | { intent: "attach", paths: string[] }} SlashIntent
 */

/**
 * @typedef {"default" | "accept-edits" | "accept-all" | "deny-all" | "interactive"} PermissionMode
 */

/**
 * Parse a context reference from a string without the leading `@`.
 * @param {string} input
 * @returns {ContextRefKind | null}
 */
function parseContextRefKind(input) {
  const trimmed = String(input ?? "").trim();
  if (!trimmed) {
    return null;
  }

  if (trimmed.startsWith("artifact:")) {
    return { kind: "artifact", selector: trimmed.slice("artifact:".length) };
  }

  if (trimmed.startsWith("resource:")) {
    return parseResourceKind(trimmed.slice("resource:".length));
  }

  switch (trimmed) {
    case "clipboard":
    case "clip":
      return { kind: "clipboard" };
    case "git:diff":
    case "diff":
      return { kind: "git_diff" };
    case "git:status":
    case "status":
      return { kind: "git_status" };
    default:
      break;
  }

  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return { kind: "url", url: trimmed };
  }

  if (trimmed.endsWith("/")) {
    return { kind: "directory", path: trimmed };
  }
  if (trimmed.includes(".") || trimmed.includes("/")) {
    return { kind: "file", path: trimmed };
  }
  return { kind: "symbol", name: trimmed };
}

/**
 * @param {string} selector
 * @returns {ContextRefKind | null}
 */
function parseResourceKind(selector) {
  const trimmed = String(selector ?? "").trim();
  if (!trimmed) {
    return null;
  }
  const sep = trimmed.indexOf("::");
  if (sep >= 0) {
    return {
      kind: "resource",
      server: trimmed.slice(0, sep),
      uri: trimmed.slice(sep + 2),
    };
  }
  return { kind: "resource", server: null, uri: trimmed };
}

/**
 * Parse shared host slash intents (`/plan`, `/ask`, `/agent`, ...).
 * Returns `null` for non-slash input or commands outside the shared set.
 * @param {string} input
 * @returns {SlashIntent | null}
 */
function parseSlashIntent(input) {
  const trimmed = String(input ?? "").trim();
  if (!trimmed.startsWith("/")) {
    return null;
  }

  const parts = trimmed.split(/\s+/).filter(Boolean);
  const raw = parts[0] ?? "";
  const command = normalizeSlashCommand(raw);
  const args = parts.slice(1);

  switch (command) {
    case "/plan":
      return { intent: "plan" };
    case "/ask":
      return { intent: "ask" };
    case "/agent":
      return { intent: "agent", arg: args[0] ?? null };
    case "/mode":
      return { intent: "mode" };
    case "/show-plan":
    case "/showplan":
      return { intent: "show_plan" };
    case "/resume":
      return { intent: "resume", id: args[0] ?? null };
    case "/attach":
      return { intent: "attach", paths: args.slice() };
    default:
      return null;
  }
}

/**
 * @param {string} raw
 * @returns {string}
 */
function normalizeSlashCommand(raw) {
  const lower = raw.toLowerCase();
  if (lower === "/showplan") {
    return "/showplan";
  }
  return lower;
}

/**
 * Normalize a permission mode string to the wire enum.
 * @param {string} value
 * @returns {PermissionMode | null}
 */
function parsePermissionMode(value) {
  switch (
    String(value ?? "")
      .trim()
      .toLowerCase()
  ) {
    case "default":
      return "default";
    case "accept-edits":
      return "accept-edits";
    case "accept-all":
      return "accept-all";
    case "deny-all":
      return "deny-all";
    case "interactive":
      return "interactive";
    default:
      return null;
  }
}

module.exports = {
  parseContextRefKind,
  parseSlashIntent,
  parsePermissionMode,
};
