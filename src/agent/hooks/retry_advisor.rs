//! Retry advisor hook: when a tool call fails, appends a short suggestion so the agent can self-correct.

use super::lifecycle::{HookContext, HookResult, LifecycleHook};
use crate::agent::hooks::permission::PermissionDecision;
use crate::agent::tools::tool::ToolResult;
use async_trait::async_trait;
use serde_json::json;

/// Hook that appends failure suggestions to tool results so the agent can retry with a fix.
#[derive(Debug)]
pub struct RetryAdvisorHook {
    enabled: bool,
}

impl Default for RetryAdvisorHook {
    fn default() -> Self {
        Self { enabled: true }
    }
}

impl RetryAdvisorHook {
    #[must_use]
    pub const fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    fn suggest(
        &self,
        tool_name: &str,
        result: &ToolResult,
        arguments: Option<&serde_json::Value>,
    ) -> Option<serde_json::Value> {
        if !self.enabled || result.success {
            return None;
        }
        let content = result.content.trim();
        let error = result.error.as_deref().unwrap_or(content);
        let suggestion = if tool_name == "bash" {
            let cmd = arguments
                .and_then(|a| a.get("command"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if Self::is_bash_file_read(cmd) {
                "Prefer fs op=read with start_line/end_line instead of bash cat/head/tail for reading files."
            } else if Self::is_bash_file_edit(cmd) {
                "Prefer fs op=edit with str_replace operations instead of bash sed/awk for editing files."
            } else if error.contains("command not found") || error.contains("not found") {
                "Check the command name and PATH; use an absolute path or install the missing tool."
            } else if error.contains("Permission denied") || error.contains("EACCES") {
                "Check file permissions or run from a directory you can write to."
            } else if error.contains("No such file") || error.contains("ENOENT") {
                "Verify the path exists; use fs op=list or workspace to find the correct path."
            } else {
                "Read the error output above and fix the command or path, then retry."
            }
        } else if tool_name == "fs" {
            if error.contains("missing 'type' field") || error.contains("fs edit requires") {
                "Each edit operation needs a 'type' field. Valid types: \
                 'str_replace' (requires 'old_string', 'new_string'), \
                 'insert' (requires 'line', 'content'), \
                 'delete_lines' (requires 'start_line', 'end_line'), \
                 'replace_lines' (requires 'start_line', 'end_line', 'content'). \
                 Example: {\"type\":\"str_replace\",\"old_string\":\"old\",\"new_string\":\"new\"}"
            } else if error.contains("Unknown operation type") {
                "Valid edit operation types: 'str_replace', 'insert', 'delete_lines', 'replace_lines'. \
                 Check the 'type' field value."
            } else if error.contains("must be a string") || error.contains("must be a number") {
                "Check required fields for the operation type. \
                 str_replace: 'old_string' and 'new_string' (strings). \
                 insert: 'line' (number) and 'content' (string). \
                 delete_lines/replace_lines: 'start_line' and 'end_line' (numbers)."
            } else if error.contains("No such file") || error.contains("not found") {
                "Use fs op=list or workspace to find the correct path, then retry with fs and the right op."
            } else {
                "Check the path and permissions, then retry."
            }
        } else {
            "Review the error above and retry with corrected arguments."
        };
        Some(json!({
            "retry_advisor": true,
            "suggestion": suggestion,
        }))
    }

    fn is_bash_file_read(cmd: &str) -> bool {
        let trimmed = cmd.trim_start();
        trimmed.starts_with("cat ")
            || trimmed.starts_with("head ")
            || trimmed.starts_with("tail ")
            || trimmed.contains("| cat")
            || trimmed.contains("| head")
            || trimmed.contains("| tail")
    }

    fn is_bash_file_edit(cmd: &str) -> bool {
        let trimmed = cmd.trim_start();
        trimmed.starts_with("sed ") || trimmed.starts_with("awk ")
    }
}

#[async_trait]
impl LifecycleHook for RetryAdvisorHook {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_post_tool_use(&self, ctx: &HookContext<'_>, result: &ToolResult) -> HookResult {
        if let Some(value) = self.suggest(ctx.tool_name.as_ref(), result, Some(ctx.tool_input)) {
            HookResult::with_output(PermissionDecision::Allow, value)
        } else {
            HookResult::allow()
        }
    }
}
