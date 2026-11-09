pub const TOOL_FILESYSTEM: &str = "fs";
pub const TOOL_BASH: &str = "bash";
pub const TOOL_HTTP: &str = "http";
pub const TOOL_WEB_SEARCH: &str = "web_search";
pub const TOOL_WEB_FETCH: &str = "web_fetch";
pub const TOOL_EXECUTE_PYTHON: &str = "execute_python";
pub const TOOL_GIT: &str = "git";
pub const TOOL_SQL_EXECUTION: &str = "sql_execution";
pub const TOOL_CONTEXT_GET: &str = "context_get";
pub const TOOL_CONTEXT_SET: &str = "context_set";
pub const TOOL_WORKSPACE: &str = "workspace";
pub const TOOL_GENERATE_IMAGE: &str = "generate_image";
pub const TOOL_APPLY_PATCH: &str = "apply_patch";
#[cfg(feature = "plugins")]
pub const TOOL_PLUGIN: &str = "plugin";

/// Opt-in `--profile lean` roster: filesystem and shell only.
pub const LEAN_DEMO_TOOLS: &[&str] = &[TOOL_BASH, TOOL_FILESYSTEM];

/// Canonical session default: explore via `fs`, execute via `bash`, retrieve
/// public URL content via `web_fetch`.
///
/// `web_search` stays opt-in. `web_fetch` is not a browser.
pub const CODING_DEFAULT_TOOLS: &[&str] = &[TOOL_BASH, TOOL_FILESYSTEM, TOOL_WEB_FETCH];

/// Broader extended defaults. Includes the coding default (`web_fetch`) plus
/// HTTP, workspace orientation, and Python. `web_search` stays opt-in.
pub const EXTENDED_DEFAULT_TOOLS: &[&str] = &[
    TOOL_BASH,
    TOOL_FILESYSTEM,
    TOOL_WEB_FETCH,
    TOOL_HTTP,
    TOOL_WORKSPACE,
    TOOL_EXECUTE_PYTHON,
];

pub const FS_OP_READ: &str = "read";
pub const FS_OP_LIST: &str = "list";
pub const FS_OP_GREP: &str = "grep";
pub const FS_OP_WRITE: &str = "write";
pub const FS_OP_EDIT: &str = "edit";
pub const FS_OP_DELETE: &str = "delete";

#[must_use]
pub fn fs_op(args: &serde_json::Value) -> Option<&str> {
    args.get("op").and_then(|v| v.as_str())
}

#[must_use]
pub fn is_fs_write_op(op: &str) -> bool {
    matches!(op, FS_OP_WRITE | FS_OP_EDIT | FS_OP_DELETE)
}
