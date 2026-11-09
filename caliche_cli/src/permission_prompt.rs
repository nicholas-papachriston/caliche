//! Interactive permission prompts for CLI tool execution.

use async_trait::async_trait;
use caliche::agent::hooks::{HookContext, HookResult, LifecycleHook};
use caliche::agent::tools::names::{
    FS_OP_EDIT, FS_OP_WRITE, TOOL_EXECUTE_PYTHON, TOOL_FILESYSTEM, TOOL_GIT, TOOL_HTTP, fs_op,
    is_fs_write_op,
};
use std::collections::HashSet;
use std::io::{self, Write};
use std::sync::{Arc, RwLock};

/// Hook that prompts user for permission on dangerous operations.
///
/// Selecting "always" records a per-session allowlist for that exact tool
/// name so subsequent invocations skip the prompt for the remainder of the
/// session. The allowlist is intentionally process-local (not persisted) so
/// each run starts with a clean slate.
#[derive(Debug, Clone)]
pub struct InteractivePermissionHook {
    always_prompt: bool,
    session_allowlist: Arc<RwLock<HashSet<String>>>,
}

impl InteractivePermissionHook {
    /// Create a new interactive permission hook
    pub fn new(always_prompt: bool) -> Self {
        Self {
            always_prompt,
            session_allowlist: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    fn build_allowlist_key(ctx: &HookContext<'_>) -> String {
        match ctx.tool_name.as_ref() {
            TOOL_FILESYSTEM => {
                let op = fs_op(&ctx.tool_input).unwrap_or("?");
                format!("{}:{}", TOOL_FILESYSTEM, op)
            }
            TOOL_HTTP => {
                let method = ctx
                    .tool_input
                    .get("method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("GET")
                    .to_ascii_uppercase();
                format!("{}:{}", TOOL_HTTP, method)
            }
            TOOL_GIT => {
                let op = ctx
                    .tool_input
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("status");
                format!("{}:{}", TOOL_GIT, op)
            }
            other => other.to_string(),
        }
    }

    fn is_allowlisted(&self, key: &str) -> bool {
        self.session_allowlist
            .read()
            .map(|set| set.contains(key))
            .unwrap_or(false)
    }

    fn remember_allow(&self, key: String) {
        if let Ok(mut set) = self.session_allowlist.write() {
            set.insert(key);
        }
    }

    /// Check if a tool is considered dangerous
    fn is_dangerous_tool(ctx: &HookContext<'_>) -> bool {
        if ctx.tool_name == TOOL_FILESYSTEM {
            return fs_op(&ctx.tool_input).map(is_fs_write_op).unwrap_or(false);
        }
        if ctx.tool_name == TOOL_HTTP {
            let operation = ctx
                .tool_input
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("request");
            let method = ctx
                .tool_input
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("GET")
                .to_ascii_uppercase();
            if operation == "request" || operation == "api_call" {
                return matches!(method.as_str(), "POST" | "PUT" | "PATCH" | "DELETE");
            }
        }
        if ctx.tool_name == TOOL_GIT {
            let operation = ctx
                .tool_input
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("status");
            if operation == "commit" {
                return true;
            }
            if operation == "branch" {
                let action = ctx
                    .tool_input
                    .get("action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("list");
                return matches!(action, "create" | "delete");
            }
            return false;
        }
        matches!(ctx.tool_name.as_ref(), "bash" | TOOL_EXECUTE_PYTHON)
    }

    /// Format tool arguments for display
    fn format_tool_args(ctx: &HookContext<'_>) -> String {
        match ctx.tool_name.as_ref() {
            TOOL_FILESYSTEM => {
                if let Some(path) = ctx.tool_input.get("path").and_then(|v| v.as_str()) {
                    let op = fs_op(&ctx.tool_input).unwrap_or("?");
                    format!("Operation: {}  Path: {}", op, path)
                } else {
                    format!("Tool: {}", ctx.tool_name)
                }
            }
            "bash" => {
                if let Some(cmd) = ctx.tool_input.get("command").and_then(|v| v.as_str()) {
                    format!("Command: {}", cmd)
                } else {
                    format!("Tool: {}", ctx.tool_name)
                }
            }
            TOOL_GIT => {
                let operation = ctx
                    .tool_input
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("status");
                if operation == "commit" {
                    let message = ctx
                        .tool_input
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<missing message>");
                    format!("Operation: commit  Message: {}", message)
                } else if operation == "branch" {
                    let action = ctx
                        .tool_input
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("list");
                    let branch_name = ctx
                        .tool_input
                        .get("branch_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<none>");
                    format!(
                        "Operation: branch  Action: {}  Branch: {}",
                        action, branch_name
                    )
                } else {
                    format!("Operation: {}", operation)
                }
            }
            TOOL_HTTP => {
                let operation = ctx
                    .tool_input
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("request");
                let method = ctx
                    .tool_input
                    .get("method")
                    .and_then(|v| v.as_str())
                    .unwrap_or("GET");
                let url = ctx
                    .tool_input
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<missing url>");
                format!("Operation: {}  Method: {}  Url: {}", operation, method, url)
            }
            _ => format!("Tool: {}", ctx.tool_name),
        }
    }
}

#[async_trait]
impl LifecycleHook for InteractivePermissionHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        // If not always prompting, only prompt for dangerous tools
        if !self.always_prompt && !Self::is_dangerous_tool(ctx) {
            return HookResult::allow();
        }

        let allowlist_key = Self::build_allowlist_key(ctx);
        if self.is_allowlisted(&allowlist_key) {
            return HookResult::allow();
        }

        // Format the prompt
        let tool_info = Self::format_tool_args(ctx);
        println!("\n[PERMISSION] Tool: {}", ctx.tool_name);
        println!("  {}", tool_info);

        // Show additional context for file operations
        if ctx.tool_name == TOOL_FILESYSTEM
            && matches!(fs_op(&ctx.tool_input), Some(FS_OP_WRITE | FS_OP_EDIT))
            && let Some(path) = ctx.tool_input.get("path").and_then(|v| v.as_str())
        {
            // Check if file exists
            if std::path::Path::new(path).exists() {
                println!("  WARNING: File exists - will be overwritten");
            } else {
                println!("  New file will be created");
            }
        }

        // Prompt user
        print!("\n  [y]es / [n]o / [d]iff / [a]lways: ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            return HookResult::deny("Failed to read user input");
        }

        self.decision_from_reply(input.trim(), ctx, allowlist_key)
    }
}

impl InteractivePermissionHook {
    fn decision_from_reply(
        &self,
        reply: &str,
        ctx: &HookContext<'_>,
        allowlist_key: String,
    ) -> HookResult {
        match reply.to_lowercase().as_str() {
            "y" | "yes" => HookResult::allow(),
            "n" | "no" => HookResult::deny("User denied permission"),
            "d" | "diff" => self.decision_after_diff(ctx, allowlist_key),
            "a" | "always" => {
                self.remember_allow(allowlist_key);
                HookResult::allow()
            }
            _ => HookResult::deny("Invalid response - denied for safety"),
        }
    }

    fn decision_after_diff(&self, ctx: &HookContext<'_>, allowlist_key: String) -> HookResult {
        if ctx.tool_name == TOOL_FILESYSTEM
            && matches!(fs_op(&ctx.tool_input), Some(FS_OP_WRITE | FS_OP_EDIT))
            && let Some(path) = ctx.tool_input.get("path").and_then(|value| value.as_str())
            && let Ok(content) = std::fs::read_to_string(path)
        {
            println!("\nCurrent file content:\n{content}");
            if let Some(new_content) = ctx
                .tool_input
                .get("content")
                .and_then(|value| value.as_str())
            {
                println!("\nNew content:\n{new_content}");
            }
        }
        print!("\n  [y]es / [n]o / [a]lways: ");
        io::stdout().flush().ok();
        let mut reply = String::new();
        if io::stdin().read_line(&mut reply).is_err() {
            return HookResult::deny("Failed to read user input");
        }
        match reply.trim().to_lowercase().as_str() {
            "y" | "yes" => HookResult::allow(),
            "a" | "always" => {
                self.remember_allow(allowlist_key);
                HookResult::allow()
            }
            _ => HookResult::deny("User denied permission"),
        }
    }
}
