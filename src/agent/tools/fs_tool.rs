use super::apply_patch::MutationStyle;
use super::edit_file::EditFileTool;
#[cfg(feature = "tools-grep")]
use super::glob::GlobTool;
#[cfg(feature = "tools-grep")]
use super::grep::GrepTool;
use super::list_directory::ListDirectoryTool;
use super::read_file::ReadFileTool;
use super::write_file::WriteFileTool;
use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "lsp")]
use crate::lsp::LspManager;
#[cfg(feature = "lsp")]
use crate::lsp::inject_path_diagnostics;
use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::json;
#[cfg(feature = "lsp")]
use std::sync::Arc;
use std::time::Instant;

const NAME: &str = "fs";
const DESCRIPTION_EDIT_WRITE: &str = "Primary filesystem tool for codebase exploration and edits. Ops: read, list, glob, grep, write, edit, delete. Prefer glob/grep before broad reads; use read query or line ranges for large files. Do not use bash for search/list/read.";
const DESCRIPTION_APPLY_PATCH: &str = "Primary filesystem tool for codebase exploration. Ops: read, list, glob, grep, delete. File create/edit mutations use apply_patch (not fs write/edit). Prefer glob/grep before broad reads.";

/// Default filesystem tool (`MutationStyle::EditWrite`).
///
/// Prefer [`FsTool::with_mutation_style`] when constructing an XOR `apply_patch` surface.
#[derive(Debug, Default)]
pub struct FsTool {
    mutation_style: MutationStyle,
    #[cfg(feature = "lsp")]
    lsp_manager: Option<Arc<LspManager>>,
}

impl FsTool {
    /// Construct the default edit/write `fs` tool (same as [`Default`]).
    ///
    /// Bindings and call sites that historically used the unit-struct form `FsTool`
    /// should call this or [`FsTool::default`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mutation_style: MutationStyle::EditWrite,
            #[cfg(feature = "lsp")]
            lsp_manager: None,
        }
    }

    #[must_use]
    pub const fn with_mutation_style(mutation_style: MutationStyle) -> Self {
        Self {
            mutation_style,
            #[cfg(feature = "lsp")]
            lsp_manager: None,
        }
    }

    /// Attach an LSP manager so successful read/write/edit ops inject diagnostics.
    #[cfg(feature = "lsp")]
    #[must_use]
    pub fn with_lsp_manager(mut self, manager: Option<Arc<LspManager>>) -> Self {
        self.lsp_manager = manager;
        self
    }

    #[must_use]
    pub const fn mutation_style(&self) -> MutationStyle {
        self.mutation_style
    }

    const fn description_text(&self) -> &'static str {
        match self.mutation_style {
            MutationStyle::EditWrite => DESCRIPTION_EDIT_WRITE,
            MutationStyle::ApplyPatch => DESCRIPTION_APPLY_PATCH,
        }
    }

    const fn op_enum(&self) -> &'static [&'static str] {
        match self.mutation_style {
            MutationStyle::EditWrite => {
                &["read", "list", "glob", "grep", "write", "edit", "delete"]
            }
            MutationStyle::ApplyPatch => &["read", "list", "glob", "grep", "delete"],
        }
    }
}

impl ToolMeta for FsTool {
    fn description(&self) -> &str {
        self.description_text()
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(self.description_text())
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        let op_enum: Vec<&str> = self.op_enum().to_vec();
        json!({
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": op_enum,
                    "description": "Operation."
                },
                "path": {
                    "type": "string",
                    "description": "Target path."
                },
                "content": {
                    "type": "string",
                    "description": "Content for write/insert/replace operations."
                },
                "append": {
                    "type": "boolean",
                    "default": false
                },
                "force": {
                    "type": "boolean",
                    "default": false,
                    "description": "For write: skip the tiny-overwrite guard."
                },
                "create_dirs": {
                    "type": "boolean",
                    "default": true
                },
                "operations": {
                    "type": "array",
                    "description": "For edit: ordered edit operations. Prefer hashline ops when available.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["str_replace", "insert", "delete_lines", "replace_lines", "hashline_replace", "hashline_insert", "hashline_delete"],
                                "description": "Edit operation. Prefer hashline_* with hashes from fs read output."
                            },
                            "old_string": { "type": "string", "description": "For str_replace: exact text to find." },
                            "new_string": { "type": "string", "description": "For str_replace: replacement text." },
                            "line": { "type": "number", "description": "For insert: 1-indexed line to insert before." },
                            "content": { "type": "string", "description": "For insert/replace_lines: content to add." },
                            "start_line": { "type": "number", "description": "For delete_lines/replace_lines: 1-indexed start." },
                            "end_line": { "type": "number", "description": "For delete_lines/replace_lines: 1-indexed end (inclusive)." },
                            "start_hash": { "type": "string", "description": "For hashline_replace/delete: exactly 4 hex digits from `42:ab12| content` (`42:ab12` is also accepted). A legacy 2-hex hash matches only the hinted line and is not relocated." },
                            "end_hash": { "type": "string", "description": "For hashline_replace/delete: hash for end_line, same format as start_hash." },
                            "anchor_line": { "type": "number", "description": "For hashline_insert: anchor line." },
                            "anchor_hash": { "type": "string", "description": "For hashline_insert: 4-hex anchor hash from fs op=read." },
                            "position": { "type": "string", "enum": ["before", "after"], "description": "For hashline_insert." }
                        },
                        "required": ["type"]
                    }
                },
                "query": {
                    "type": "string",
                    "description": "For read: relevance query."
                },
                "max_size": {
                    "type": "number",
                    "description": "For read/grep: max bytes."
                },
                "start_line": {
                    "type": "number",
                    "description": "For read: 1-indexed first line. Numeric strings are accepted."
                },
                "end_line": {
                    "type": "number",
                    "description": "For read: 1-indexed last line, inclusive."
                },
                "limit": {
                    "type": "number",
                    "description": "For read: number of lines to return from start_line or offset."
                },
                "num_chunks": {
                    "type": "number",
                    "description": "For read query: number of chunks."
                },
                "recursive": {
                    "type": "boolean",
                    "default": false,
                    "description": "For list/grep/delete."
                },
                "max_depth": {
                    "type": "number",
                    "description": "For list recursion."
                },
                "max_entries": {
                    "type": "number",
                    "description": "For list: max entries."
                },
                "pattern": {
                    "type": "string",
                    "description": "For glob/grep: regex, or a quoted literal. Invalid regex is searched once as a literal. If wrapping quotes match nothing, one retry drops the quotes and searches the inside as a literal."
                },
                "file_pattern": {
                    "type": "string",
                    "description": "For grep: extension (`rs`, `.rs`, `*.rs`), basename, or path suffix."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "default": true,
                    "description": "For grep."
                },
                "max_matches": {
                    "type": "number",
                    "description": "For glob/grep: maximum matches."
                },
                "context_lines": {
                    "type": "number",
                    "description": "For grep: context lines."
                },
                "max_file_size": {
                    "type": "number",
                    "description": "For grep: max file size."
                },
                "include_hidden": {
                    "type": "boolean",
                    "default": false,
                    "description": "For glob/list/grep: include hidden paths. Grep still skips `.git` / `.hg` / `.svn`."
                },
                "mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "For grep: output mode."
                },
                "offset": {
                    "type": "number",
                    "description": "For read: 1-indexed first line (alias of start_line; 0 means the start of the file). For grep: skip this many matches."
                }
            },
            "required": ["op", "path"]
        })
    }
}

fn ensure_required(args: &serde_json::Value, key: &str) -> Result<()> {
    if args.get(key).is_none() || args.get(key).is_some_and(serde_json::Value::is_null) {
        bail!("Missing required parameter '{key}' for selected fs operation");
    }
    Ok(())
}

fn fs_hotpath_name(op: &str) -> &'static str {
    match op {
        "read" => "fs.read",
        "list" => "fs.list",
        "grep" => "fs.grep",
        "glob" => "fs.glob",
        "write" => "fs.write",
        "edit" => "fs.edit",
        "delete" => "fs.delete",
        _ => "fs.unknown",
    }
}

fn payload_u64(result: &ToolResult, key: &str) -> Option<u64> {
    result.typed_payload.as_ref()?.get(key)?.as_u64()
}

fn apply_fs_perf_counts(
    span: Option<&mut crate::observability::perf::PerfSpan>,
    result: &ToolResult,
) {
    let Some(span) = span else {
        return;
    };
    if let Some(count) = payload_u64(result, "files_scanned") {
        span.set_file_count(count);
    }
    if let Some(count) = payload_u64(result, "match_count") {
        span.set_match_count(count);
    } else if let Some(count) = payload_u64(result, "file_count") {
        span.set_match_count(count);
    }
    if let Some(count) = payload_u64(result, "lines_scanned") {
        span.set_line_count(count);
    }
}

fn enrich_fs_payload(
    mut result: ToolResult,
    op: &str,
    path: &str,
    started: Option<Instant>,
) -> ToolResult {
    if !result.success {
        return result;
    }
    let duration_ms =
        started.map(|started| started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64);
    match result
        .typed_payload
        .as_mut()
        .and_then(serde_json::Value::as_object_mut)
    {
        Some(obj) => {
            obj.entry("op".to_string())
                .or_insert_with(|| serde_json::Value::String(op.to_string()));
            obj.entry("path".to_string())
                .or_insert_with(|| serde_json::Value::String(path.to_string()));
            if let Some(duration_ms) = duration_ms {
                obj.insert(
                    "duration_ms".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(duration_ms)),
                );
            }
        }
        None => {
            let mut payload = json!({
                "op": op,
                "path": path,
            });
            if let Some(duration_ms) = duration_ms
                && let Some(obj) = payload.as_object_mut()
            {
                obj.insert(
                    "duration_ms".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(duration_ms)),
                );
            }
            result.typed_payload = Some(payload);
        }
    }
    result
}

#[cfg(feature = "sandbox")]
fn shell_escape_single_quoted(value: &str) -> String {
    value.replace('\'', "'\\''")
}

#[cfg(feature = "sandbox")]
async fn execute_list_sandboxed(
    arguments: serde_json::Value,
    sandbox: &dyn caliche_sandbox::SandboxHandleOps,
) -> ToolResult {
    let path_str = params::extract_string_or(&arguments, "path", ".");
    let recursive = params::extract_bool(&arguments, "recursive", false);
    let max_depth = params::extract_number_or(&arguments, "max_depth", 2) as usize;
    let escaped = shell_escape_single_quoted(&path_str);
    let script = if recursive {
        format!(
            "find '{}' -maxdepth {} -print 2>/dev/null | head -2000",
            escaped,
            max_depth.max(1)
        )
    } else {
        format!("ls -la '{escaped}'")
    };
    let cmd: Vec<&str> = vec!["sh", "-c", &script];
    match sandbox
        .exec(&cmd, None, None, std::time::Duration::from_secs(30))
        .await
    {
        Ok(result) if result.exit_code == 0 => ToolResult::success(result.stdout),
        Ok(result) => ToolResult::error(format!(
            "Sandbox list failed: exit {} - {}",
            result.exit_code, result.stderr
        )),
        Err(error) => ToolResult::error(format!("Sandbox list failed: {error}")),
    }
}

#[cfg(all(feature = "sandbox", feature = "tools-grep"))]
async fn execute_grep_sandboxed(
    arguments: serde_json::Value,
    sandbox: &dyn caliche_sandbox::SandboxHandleOps,
) -> ToolResult {
    let pattern = match params::extract_string(&arguments, "pattern", None) {
        Ok(value) => value,
        Err(error) => return ToolResult::error(error),
    };
    let attempts = match super::grep_pattern::grep_attempts(&pattern) {
        Ok(attempts) => attempts,
        Err(error) => return ToolResult::error(error),
    };
    let path_str = params::extract_string_or(&arguments, "path", ".");
    let recursive = params::extract_bool(&arguments, "recursive", true);
    let max_matches = params::extract_number_or(&arguments, "max_matches", 100) as usize;
    let escaped_path = shell_escape_single_quoted(&path_str);
    let recursive_flag = if recursive { "-r" } else { "" };
    let mut last_error = String::new();
    for (index, attempt) in attempts.iter().enumerate() {
        let mode_flag = if attempt.literal { "-F" } else { "-E" };
        let escaped_pattern = shell_escape_single_quoted(&attempt.needle);
        let script = format!(
            "grep {mode_flag} {recursive_flag} -- '{escaped_pattern}' '{escaped_path}' 2>/dev/null | head -{max_matches}"
        );
        let cmd: Vec<&str> = vec!["sh", "-c", &script];
        match sandbox
            .exec(&cmd, None, None, std::time::Duration::from_secs(60))
            .await
        {
            Ok(result) if result.exit_code == 0 || result.exit_code == 1 => {
                let retry = result.stdout.trim().is_empty() && index + 1 < attempts.len();
                if retry {
                    continue;
                }
                let mut stdout = result.stdout;
                if let Some(note) = attempt.note.as_deref() {
                    stdout.push_str("\n\n[");
                    stdout.push_str(note);
                    stdout.push(']');
                }
                return ToolResult::success(stdout);
            }
            Ok(result) => {
                last_error = format!(
                    "Sandbox grep failed: exit {} - {}. Pattern {pattern:?}. \
                     Quote a literal or escape regex metacharacters.",
                    result.exit_code, result.stderr
                );
                if index + 1 < attempts.len() {
                    continue;
                }
            }
            Err(error) => return ToolResult::error(format!("Sandbox grep failed: {error}")),
        }
    }
    ToolResult::error(if last_error.is_empty() {
        format!("Sandbox grep failed for pattern {pattern:?}")
    } else {
        last_error
    })
}

#[cfg(all(feature = "sandbox", feature = "tools-grep"))]
async fn execute_glob_sandboxed(
    arguments: serde_json::Value,
    sandbox: &dyn caliche_sandbox::SandboxHandleOps,
) -> ToolResult {
    let pattern = match params::extract_string(&arguments, "pattern", None) {
        Ok(value) => value,
        Err(error) => return ToolResult::error(error),
    };
    let path_str = params::extract_string_or(&arguments, "path", ".");
    let max_matches = params::extract_number_or(&arguments, "max_matches", 500) as usize;
    let escaped_pattern = shell_escape_single_quoted(&pattern);
    let escaped_path = shell_escape_single_quoted(&path_str);
    let script = format!(
        "find '{escaped_path}' -name '{escaped_pattern}' 2>/dev/null | head -{max_matches}"
    );
    let cmd: Vec<&str> = vec!["sh", "-c", &script];
    match sandbox
        .exec(&cmd, None, None, std::time::Duration::from_secs(60))
        .await
    {
        Ok(result) if result.exit_code == 0 => ToolResult::success(result.stdout),
        Ok(result) => ToolResult::error(format!(
            "Sandbox glob failed: exit {} - {}",
            result.exit_code, result.stderr
        )),
        Err(error) => ToolResult::error(format!("Sandbox glob failed: {error}")),
    }
}

#[cfg(feature = "sandbox")]
async fn execute_read_or_list(
    arguments: serde_json::Value,
    sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
) -> ToolResult {
    let path_str = match params::extract_string(&arguments, "path", None) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(enhance_path_error("", &e)),
    };

    // Robustness fallback for weaker model outputs:
    // when fs(op=read) targets a directory, auto-route to list.
    let is_directory = path::resolve_path_for_tool(&path_str)
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .is_some_and(|m| m.is_dir());

    if is_directory {
        if let Some(sandbox_handle) = sandbox {
            return execute_list_sandboxed(arguments, sandbox_handle).await;
        }
        let tool = ListDirectoryTool;
        return tool.execute(arguments).await;
    }

    let tool = ReadFileTool;
    if let Some(sandbox_handle) = sandbox {
        return tool
            .execute_sandboxed(arguments, Some(sandbox_handle))
            .await;
    }
    tool.execute(arguments).await
}

#[cfg(not(feature = "sandbox"))]
async fn execute_read_or_list(arguments: serde_json::Value) -> ToolResult {
    let path_str = match params::extract_string(&arguments, "path", None) {
        Ok(v) => v,
        Err(e) => return ToolResult::error(enhance_path_error("", &e)),
    };

    let is_directory = path::resolve_path_for_tool(&path_str)
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .is_some_and(|m| m.is_dir());

    if is_directory {
        let tool = ListDirectoryTool;
        return tool.execute(arguments).await;
    }

    let tool = ReadFileTool;
    tool.execute(arguments).await
}

/// Suggest similar paths when a path is not found.
/// Lists sibling entries of the parent directory and finds names similar to the target.
fn suggest_similar_paths(failed_path: &str) -> String {
    use std::path::Path;

    let path = Path::new(failed_path);
    let parent = match path.parent() {
        Some(p) if p.as_os_str().is_empty() => Path::new("."),
        Some(p) => p,
        None => return String::new(),
    };
    let target_name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n.to_lowercase(),
        None => return String::new(),
    };

    let resolved_parent = match path::resolve_path_for_tool(&parent.display().to_string()) {
        Ok(p) => p,
        Err(_) => return String::new(),
    };

    let mut suggestions = Vec::new();
    if let Ok(entries) = path::list_workspace_dir(&resolved_parent) {
        for entry in entries {
            let name = entry.name.to_string_lossy().to_lowercase();
            // Simple similarity check: contains target or vice versa
            if name.contains(&target_name) || target_name.contains(&name) {
                suggestions.push(format!(
                    "{}/{}",
                    parent.display(),
                    entry.name.to_string_lossy()
                ));
            }
        }
    }

    if suggestions.is_empty() {
        format!(
            " Try `fs op=list path={}` to see available entries.",
            parent.display()
        )
    } else {
        format!(" Did you mean: {}?", suggestions.join(", "))
    }
}

/// Enhance an error message with path suggestions if it's a "not found" error.
fn enhance_path_error(path_str: &str, error_msg: impl std::fmt::Display) -> String {
    let error_msg = error_msg.to_string();
    if error_msg.contains("No such file")
        || error_msg.contains("not found")
        || error_msg.contains("Failed to resolve path")
    {
        format!("{error_msg}{}", suggest_similar_paths(path_str))
    } else {
        error_msg
    }
}

async fn execute_delete(path_str: &str, recursive: bool) -> ToolResult {
    let file_path = match path::resolve_write_path_for_tool(path_str) {
        Ok(p) => p,
        Err(e) => return ToolResult::error(enhance_path_error(path_str, &e)),
    };
    let path_label = path_str.to_string();
    match tokio::task::spawn_blocking(move || delete_path_sync(&file_path, &path_label, recursive))
        .await
    {
        Ok(result) => result,
        Err(e) => ToolResult::error(format!("delete worker failed: {e}")),
    }
}

fn delete_path_sync(file_path: &std::path::Path, path_str: &str, recursive: bool) -> ToolResult {
    let metadata = match std::fs::metadata(file_path) {
        Ok(m) => m,
        Err(e) => {
            return ToolResult::error(format!(
                "Failed to access path '{path_str}' for delete: {e}"
            ));
        }
    };

    let is_dir = metadata.is_dir();
    let res = if is_dir {
        if !recursive {
            return ToolResult::error(format!(
                "Path '{path_str}' is a directory; set recursive=true to delete directories"
            ));
        }
        std::fs::remove_dir_all(file_path)
    } else {
        std::fs::remove_file(file_path)
    };

    match res {
        Ok(()) => ToolResult::success_with_payload(
            format!("Deleted '{path_str}'"),
            json!({
                "op": "delete",
                "path": path_str,
                "recursive": recursive,
                "deleted_directory": is_dir,
            }),
        ),
        Err(e) => ToolResult::error(format!("Failed to delete '{path_str}': {e}")),
    }
}

#[async_trait]
impl Tool for FsTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let op = match params::extract_string(&arguments, "op", None) {
            Ok(v) => v.to_lowercase(),
            Err(e) => return ToolResult::error(e),
        };
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(enhance_path_error("", &e)),
        };
        let started = crate::observability::perf::enabled().then(Instant::now);
        let mut perf = crate::observability::perf::span(fs_hotpath_name(&op));

        let result = match op.as_str() {
            "read" => {
                #[cfg(feature = "sandbox")]
                {
                    execute_read_or_list(arguments, None).await
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    execute_read_or_list(arguments).await
                }
            }
            "list" => {
                let tool = ListDirectoryTool;
                tool.execute(arguments).await
            }
            "grep" => {
                #[cfg(feature = "tools-grep")]
                {
                    if let Err(e) = ensure_required(&arguments, "pattern") {
                        return ToolResult::error(e.to_string());
                    }
                    let tool = GrepTool;
                    tool.execute(arguments).await
                }
                #[cfg(not(feature = "tools-grep"))]
                {
                    ToolResult::error(
                        "fs op=grep requires 'tools-grep' feature to be enabled".to_string(),
                    )
                }
            }
            "glob" => {
                #[cfg(feature = "tools-grep")]
                {
                    if let Err(e) = ensure_required(&arguments, "pattern") {
                        return ToolResult::error(e.to_string());
                    }
                    let tool = GlobTool;
                    tool.execute(arguments).await
                }
                #[cfg(not(feature = "tools-grep"))]
                {
                    ToolResult::error(
                        "fs op=glob requires 'tools-grep' feature to be enabled".to_string(),
                    )
                }
            }
            "write" => {
                if self.mutation_style == MutationStyle::ApplyPatch {
                    return ToolResult::error(
                        "fs op=write is disabled when mutation_style=apply_patch; use apply_patch instead",
                    );
                }
                if let Err(e) = ensure_required(&arguments, "content") {
                    return ToolResult::error(e.to_string());
                }
                let resolved = match path::resolve_write_path_for_tool(&path_str) {
                    Ok(p) => p,
                    Err(e) => return ToolResult::error(e),
                };
                let tool = WriteFileTool;
                super::file_mutation_queue::with_path_lock(&resolved, || async move {
                    tool.execute(arguments).await
                })
                .await
            }
            "edit" => {
                if self.mutation_style == MutationStyle::ApplyPatch {
                    return ToolResult::error(
                        "fs op=edit is disabled when mutation_style=apply_patch; use apply_patch instead",
                    );
                }
                if arguments.get("operations").is_none() {
                    return ToolResult::error("fs edit requires 'operations'");
                }
                // EditFileTool already serializes via the mutation queue.
                let tool = EditFileTool;
                tool.execute(arguments).await
            }
            "delete" => {
                let recursive = params::extract_bool(&arguments, "recursive", false);
                let resolved = match path::resolve_write_path_for_tool(&path_str) {
                    Ok(p) => p,
                    Err(e) => return ToolResult::error(e),
                };
                super::file_mutation_queue::with_path_lock(&resolved, || async {
                    execute_delete(&path_str, recursive).await
                })
                .await
            }
            _ => ToolResult::error(format!(
                "Unsupported fs op '{op}'. Valid ops: read, list, glob, grep, write, edit, delete"
            )),
        };
        let result = enrich_fs_payload(result, &op, &path_str, started);
        apply_fs_perf_counts(perf.as_mut(), &result);
        drop(perf);
        self.maybe_inject_lsp(&op, &path_str, result).await
    }

    #[cfg(feature = "sandbox")]
    async fn execute_sandboxed(
        &self,
        arguments: serde_json::Value,
        sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> ToolResult {
        let Some(sandbox) = sandbox else {
            return self.execute(arguments).await;
        };
        let op = match params::extract_string(&arguments, "op", None) {
            Ok(v) => v.to_lowercase(),
            Err(e) => return ToolResult::error(e),
        };
        let path_for_payload = params::extract_string_or(&arguments, "path", ".");
        let started = crate::observability::perf::enabled().then(Instant::now);
        let mut perf = crate::observability::perf::span(fs_hotpath_name(&op));

        let result = match op.as_str() {
            "read" => execute_read_or_list(arguments, Some(sandbox)).await,
            "list" => execute_list_sandboxed(arguments, sandbox).await,
            "grep" => {
                #[cfg(feature = "tools-grep")]
                {
                    execute_grep_sandboxed(arguments, sandbox).await
                }
                #[cfg(not(feature = "tools-grep"))]
                {
                    ToolResult::error(
                        "fs op=grep requires 'tools-grep' feature to be enabled".to_string(),
                    )
                }
            }
            "glob" => {
                #[cfg(feature = "tools-grep")]
                {
                    execute_glob_sandboxed(arguments, sandbox).await
                }
                #[cfg(not(feature = "tools-grep"))]
                {
                    ToolResult::error(
                        "fs op=glob requires 'tools-grep' feature to be enabled".to_string(),
                    )
                }
            }
            "write" => {
                if self.mutation_style == MutationStyle::ApplyPatch {
                    return ToolResult::error(
                        "fs op=write is disabled when mutation_style=apply_patch; use apply_patch instead",
                    );
                }
                let tool = WriteFileTool;
                tool.execute_sandboxed(arguments, Some(sandbox)).await
            }
            "edit" => {
                if self.mutation_style == MutationStyle::ApplyPatch {
                    return ToolResult::error(
                        "fs op=edit is disabled when mutation_style=apply_patch; use apply_patch instead",
                    );
                }
                let tool = EditFileTool;
                tool.execute_sandboxed(arguments, Some(sandbox)).await
            }
            "delete" => {
                let path_str = match params::extract_string(&arguments, "path", None) {
                    Ok(v) => v,
                    Err(e) => return ToolResult::error(e),
                };
                let recursive = params::extract_bool(&arguments, "recursive", false);
                let escaped = path_str.replace('\'', "'\\''");
                let script = if recursive {
                    format!("rm -rf '{escaped}'")
                } else {
                    format!("rm -f '{escaped}'")
                };
                let cmd: Vec<&str> = vec!["sh", "-c", &script];
                match sandbox
                    .exec(&cmd, None, None, std::time::Duration::from_secs(10))
                    .await
                {
                    Ok(r) if r.exit_code == 0 => {
                        ToolResult::success(format!("Deleted '{path_str}'"))
                    }
                    Ok(r) => ToolResult::error(format!(
                        "Sandbox delete failed: exit {} - {}",
                        r.exit_code, r.stderr
                    )),
                    Err(e) => ToolResult::error(format!("Sandbox delete failed: {e}")),
                }
            }
            _ => ToolResult::error(format!(
                "Unsupported fs op '{op}'. Valid ops: read, list, glob, grep, write, edit, delete"
            )),
        };
        let result = enrich_fs_payload(result, &op, &path_for_payload, started);
        apply_fs_perf_counts(perf.as_mut(), &result);
        drop(perf);
        self.maybe_inject_lsp(&op, &path_for_payload, result).await
    }
}

impl FsTool {
    #[cfg_attr(
        not(feature = "lsp"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    async fn maybe_inject_lsp(&self, op: &str, path_str: &str, result: ToolResult) -> ToolResult {
        #[cfg(feature = "lsp")]
        {
            if !matches!(op, "read" | "write" | "edit") {
                return result;
            }
            let Some(manager) = self.lsp_manager.as_ref() else {
                return result;
            };
            let Ok(resolved) = path::resolve_path_for_tool(path_str) else {
                return result;
            };
            return inject_path_diagnostics(manager, &resolved, result).await;
        }
        #[cfg(not(feature = "lsp"))]
        {
            let _ = (op, path_str);
            result
        }
    }
}

#[cfg(test)]
#[cfg(all(feature = "sandbox", feature = "tools-grep"))]
mod sandbox_grep_notice {
    use super::execute_grep_sandboxed;
    use async_trait::async_trait;
    use caliche_sandbox::{ExecResult, SandboxHandleOps};
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::time::Duration;

    struct RecordingSandbox {
        stdout: String,
        script: Mutex<String>,
    }

    #[async_trait]
    impl SandboxHandleOps for RecordingSandbox {
        async fn exec(
            &self,
            command: &[&str],
            _working_dir: Option<&str>,
            _env: Option<&HashMap<String, String>>,
            _timeout: Duration,
        ) -> anyhow::Result<ExecResult> {
            if let Some(script) = command.get(2) {
                *self.script.lock().expect("script lock") = (*script).to_string();
            }
            Ok(ExecResult {
                stdout: self.stdout.clone(),
                stderr: String::new(),
                stdout_truncated: false,
                stderr_truncated: false,
                exit_code: 0,
                duration: Duration::from_millis(1),
            })
        }

        async fn read_file(&self, _path: &str) -> anyhow::Result<Vec<u8>> {
            Ok(Vec::new())
        }

        async fn write_file(&self, _path: &str, _content: &[u8]) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn invalid_regex_notice_is_appended_to_sandbox_grep_output() {
        let sandbox = RecordingSandbox {
            stdout: "code.rs:1:foo(\n".to_string(),
            script: Mutex::new(String::new()),
        };
        let result = execute_grep_sandboxed(
            json!({
                "pattern": "foo(",
                "path": "code.rs"
            }),
            &sandbox,
        )
        .await;
        assert!(result.success, "{:?}", result.error);
        assert!(
            result.content.contains("not valid regex"),
            "sandbox grep dropped the invalid-regex notice: {}",
            result.content
        );
        assert!(
            result.content.contains("searched once as a literal"),
            "sandbox grep dropped the literal-search notice: {}",
            result.content
        );
        assert!(result.content.contains("foo("), "{}", result.content);
        let script = sandbox.script.lock().expect("script lock").clone();
        assert!(
            script.contains("-F"),
            "invalid regex must be searched as a fixed string: {script}"
        );
    }
}

#[cfg(all(test, unix))]
mod suggestion_link_tests {
    use super::suggest_similar_paths;
    use crate::agent::tools::common::set_tool_workspace_root;

    #[test]
    fn suggestions_name_workspace_files_and_not_files_behind_a_link() {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret_key.txt"), "x").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        std::fs::create_dir(ws.path().join("real")).expect("real dir");
        std::fs::write(ws.path().join("real").join("secret_key.txt"), "x").expect("inside");
        std::os::unix::fs::symlink(outside.path(), ws.path().join("linkdir")).expect("dir link");
        set_tool_workspace_root(Some(ws.path().to_path_buf()));

        let inside = suggest_similar_paths("real/secret_key");
        let linked = suggest_similar_paths("linkdir/secret_key");
        set_tool_workspace_root(None);

        assert_eq!(inside, " Did you mean: real/secret_key.txt?");
        assert_eq!(
            linked,
            " Try `fs op=list path=linkdir` to see available entries."
        );
    }
}
