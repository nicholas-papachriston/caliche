//! Post-edit validation hook: runs project-appropriate check commands after fs write/edit operations
//! and injects failure output so the agent can self-correct.

use super::lifecycle::{HookContext, HookResult, LifecycleHook};
use crate::agent::hooks::permission::PermissionDecision;
use crate::agent::tools::names::{FS_OP_EDIT, FS_OP_WRITE, TOOL_FILESYSTEM, fs_op};
use crate::agent::tools::tool::ToolResult;
use async_trait::async_trait;
use serde_json::json;
use std::path::{Path, PathBuf};

/// Hook that runs a project check (e.g. cargo check, tsc --noEmit) after file edits
/// and returns the result in `hook_specific_output` so the pipeline can append it to the tool result.
#[derive(Debug)]
pub struct ValidationHook {
    /// Whether to run validation (can be disabled via config).
    enabled: bool,
    /// Override commands per extension, e.g. "rs" -> "cargo check"
    commands: std::collections::HashMap<String, String>,
}

impl Default for ValidationHook {
    fn default() -> Self {
        let mut commands = std::collections::HashMap::new();
        commands.insert("rs".to_string(), "cargo check".to_string());
        commands.insert("ts".to_string(), "npx tsc --noEmit".to_string());
        commands.insert("tsx".to_string(), "npx tsc --noEmit".to_string());
        commands.insert("js".to_string(), "node --check".to_string());
        commands.insert("py".to_string(), "python -m py_compile".to_string());
        Self {
            enabled: true,
            commands,
        }
    }
}

impl ValidationHook {
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    pub fn with_command(mut self, ext: impl Into<String>, cmd: impl Into<String>) -> Self {
        self.commands.insert(ext.into(), cmd.into());
        self
    }

    fn get_edited_path(ctx: &HookContext<'_>) -> Option<PathBuf> {
        if !Self::is_editing_tool_call(ctx) {
            return None;
        }
        let input = ctx.tool_input;
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .or_else(|| input.get("file_path").and_then(|v| v.as_str()))?;
        let p = PathBuf::from(path_str);
        if p.is_absolute() {
            Some(p)
        } else {
            std::env::current_dir().ok().map(|cwd| cwd.join(path_str))
        }
    }

    fn is_editing_tool_call(ctx: &HookContext<'_>) -> bool {
        ctx.tool_name.as_ref() == TOOL_FILESYSTEM
            && fs_op(ctx.tool_input).is_some_and(|op| matches!(op, FS_OP_WRITE | FS_OP_EDIT))
    }

    fn detect_project_check(cwd: &Path, path: &Path) -> Option<String> {
        let ext = path.extension().and_then(|e| e.to_str())?;
        if cwd.join("Cargo.toml").exists() && ext == "rs" {
            return Some("cargo check".to_string());
        }
        if cwd.join("package.json").exists() {
            if matches!(ext, "ts" | "tsx")
                && (cwd.join("tsconfig.json").exists() || cwd.join("tsconfig.tsbuildinfo").exists())
            {
                return Some("npx tsc --noEmit".to_string());
            }
            if ext == "js" {
                return Some("node --check".to_string());
            }
        }
        if (cwd.join("pyproject.toml").exists()
            || cwd.join("setup.py").exists()
            || cwd.join("requirements.txt").exists())
            && ext == "py"
        {
            return Some(format!("python -m py_compile {}", path.display()));
        }
        None
    }

    async fn run_check(cwd: &Path, command: &str) -> (bool, String, String) {
        let mut parts = command.split_whitespace();
        let program = match parts.next() {
            Some(p) => p,
            None => return (true, String::new(), String::new()),
        };
        let args: Vec<&str> = parts.collect();
        let output = tokio::process::Command::new(program)
            .args(&args)
            .current_dir(cwd)
            .env_clear()
            .envs(caliche_contracts::child_env::host_child_env())
            .output()
            .await;
        match output {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();
                (o.status.success(), stdout, stderr)
            }
            Err(e) => (false, String::new(), format!("Failed to run check: {e}")),
        }
    }
}

#[async_trait]
impl LifecycleHook for ValidationHook {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_post_tool_use(&self, ctx: &HookContext<'_>, result: &ToolResult) -> HookResult {
        if !self.enabled || !result.success {
            return HookResult::allow();
        }
        let path = match Self::get_edited_path(ctx) {
            Some(p) => p,
            None => return HookResult::allow(),
        };
        let cwd = match std::env::current_dir() {
            Ok(d) => d,
            Err(_) => return HookResult::allow(),
        };
        let command = path
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|ext| self.commands.get(ext).cloned())
            .or_else(|| Self::detect_project_check(&cwd, &path));
        let command = match command {
            Some(c) => c,
            None => return HookResult::allow(),
        };
        let (success, stdout, stderr) = Self::run_check(&cwd, &command).await;
        if success {
            return HookResult::allow();
        }
        let msg = json!({
            "validation_failed": true,
            "command": command,
            "stdout": stdout.trim(),
            "stderr": stderr.trim(),
        });
        HookResult::with_output(PermissionDecision::Allow, msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn read_operations_are_not_treated_as_edits() {
        let input = json!({
            "op": "read",
            "path": "src/main.rs"
        });
        let ctx = HookContext::new(TOOL_FILESYSTEM, &input, "call-1");

        assert!(ValidationHook::get_edited_path(&ctx).is_none());
    }

    #[test]
    fn write_and_edit_operations_are_treated_as_edits() {
        for op in [FS_OP_WRITE, FS_OP_EDIT] {
            let input = json!({
                "op": op,
                "path": "src/main.rs"
            });
            let ctx = HookContext::new(TOOL_FILESYSTEM, &input, "call-1");

            assert!(ValidationHook::get_edited_path(&ctx).is_some());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn validation_check_does_not_receive_host_credentials() {
        use crate::testing::planted_credential::{
            PLANTED_CREDENTIAL, rerun_with_planted_credential,
        };
        if rerun_with_planted_credential(
            "agent::hooks::validation::tests::validation_check_does_not_receive_host_credentials",
        ) {
            return;
        }
        let cwd = tempfile::tempdir().expect("cwd");
        let planted =
            ValidationHook::run_check(cwd.path(), &format!("printenv {PLANTED_CREDENTIAL}")).await;
        assert_eq!((planted.0, planted.1.as_str()), (false, ""));
        let path = ValidationHook::run_check(cwd.path(), "printenv PATH").await;
        assert!(path.0 && !path.1.is_empty(), "{path:?}");
    }
}
