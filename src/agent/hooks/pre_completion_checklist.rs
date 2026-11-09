//! Pre-completion checklist middleware (the "Ralph Wiggum Loop").
//!
//! Intercepts the agent before it emits a final response. If no verification command
//! (e.g. cargo test, pytest, bun test) was executed within the last N tool calls, AND
//! at least one code file was written or edited this turn, injects a system message
//! prompting verification and continues the agent loop.
//!
//! The guard on `code_files_modified` prevents the hook from firing in pure-documentation
//! sessions or in projects where no build/test tooling is available — the main source of
//! the "verify spiral" where the agent kept retrying verification commands that all failed.

use super::lifecycle::{
    CompletionContext, CompletionHookResult, HookContext, HookResult, LifecycleHook,
    SessionStartContext, SessionStartHookResult,
};
use crate::agent::tools::names::{FS_OP_EDIT, FS_OP_WRITE, TOOL_FILESYSTEM, fs_op};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;

/// Extensions that identify code files (as opposed to documentation).
const CODE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "kt", "swift", "c", "cpp", "h", "hpp",
    "cs", "rb", "php", "lua", "toml", "json", "yaml", "yml", "sh", "bash", "zsh", "fish",
];

fn is_code_file(path: &str) -> bool {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    CODE_EXTENSIONS.iter().any(|&e| e.eq_ignore_ascii_case(ext))
}

/// Config for the pre-completion checklist middleware.
#[derive(Debug, Clone)]
pub struct PreCompletionChecklistConfig {
    /// Commands that count as verification (e.g. "cargo test", "bun test").
    /// A bash tool call matches if its command contains any of these as a substring.
    pub verify_commands: Vec<String>,
    /// Require verification within the last N tool calls.
    pub require_verify_within_last_n_turns: usize,
    /// Maximum injections per session (prevents infinite loops).
    pub max_injections_per_session: usize,
}

impl Default for PreCompletionChecklistConfig {
    fn default() -> Self {
        Self {
            verify_commands: vec![
                "cargo test".to_string(),
                "cargo check".to_string(),
                "pytest".to_string(),
                "npm test".to_string(),
                "bun test".to_string(),
                "bun run test".to_string(),
                "make test".to_string(),
                "rbxtsc".to_string(),
                "go test".to_string(),
            ],
            require_verify_within_last_n_turns: 3,
            max_injections_per_session: 2,
        }
    }
}

/// Pre-completion checklist middleware.
///
/// Tracks bash tool invocations in `on_post_tool_use`. On `on_pre_completion`, if no
/// verification command was run recently AND code files were modified, injects a
/// prompt to verify before finishing.
#[derive(Debug)]
pub struct PreCompletionChecklistMiddleware {
    config: PreCompletionChecklistConfig,
    /// Recent bash commands (ring buffer).
    recent_bash_commands: Arc<RwLock<VecDeque<String>>>,
    /// Number of code files (non-markdown/non-doc) written or edited this turn.
    code_files_modified: Arc<AtomicUsize>,
}

impl PreCompletionChecklistMiddleware {
    /// Create a new middleware with the given config.
    #[must_use]
    pub fn new(config: PreCompletionChecklistConfig) -> Self {
        let cap = config.require_verify_within_last_n_turns.max(1) * 2;
        Self {
            config,
            recent_bash_commands: Arc::new(RwLock::new(VecDeque::with_capacity(cap))),
            code_files_modified: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn matches_verify(&self, command: &str) -> bool {
        let cmd_lower = command.to_lowercase();
        self.config
            .verify_commands
            .iter()
            .any(|v| cmd_lower.contains(&v.to_lowercase()))
    }

    const VERIFICATION_PROMPT: &'static str = r"Before completing, verify your solution:
1. Run the relevant tests and read the full output
2. Compare the result against the original task specification (not your own code)
3. Fix any failures before finishing";
}

#[async_trait]
impl LifecycleHook for PreCompletionChecklistMiddleware {
    async fn on_session_start(&self, _ctx: &SessionStartContext<'_>) -> SessionStartHookResult {
        self.code_files_modified.store(0, Ordering::Relaxed);
        self.recent_bash_commands.write().await.clear();
        SessionStartHookResult::none()
    }

    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_post_tool_use(
        &self,
        ctx: &HookContext<'_>,
        _result: &crate::agent::tools::tool::ToolResult,
    ) -> HookResult {
        match ctx.tool_name.as_ref() {
            "bash" => {
                let Some(command) = ctx.tool_input.get("command").and_then(|v| v.as_str()) else {
                    return HookResult::allow();
                };
                let mut guard = self.recent_bash_commands.write().await;
                if guard.len() >= self.config.require_verify_within_last_n_turns * 2 {
                    guard.pop_front();
                }
                guard.push_back(command.to_string());
            }
            TOOL_FILESYSTEM => {
                // Count code files modified so we know whether verification makes sense.
                let op = fs_op(ctx.tool_input).unwrap_or("");
                if !matches!(op, FS_OP_WRITE | FS_OP_EDIT) {
                    return HookResult::allow();
                }
                if let Some(path) = ctx.tool_input.get("path").and_then(|v| v.as_str())
                    && is_code_file(path)
                {
                    self.code_files_modified.fetch_add(1, Ordering::Relaxed);
                }
            }
            _ => {}
        }
        HookResult::allow()
    }

    async fn on_pre_completion(&self, ctx: &CompletionContext<'_>) -> CompletionHookResult {
        if ctx.injections_this_session >= self.config.max_injections_per_session {
            return CompletionHookResult::Allow;
        }

        // Only prompt verification when actual code files were modified. If the agent only
        // created documentation or did read-only work there is nothing to verify.
        if self.code_files_modified.load(Ordering::Relaxed) == 0 {
            return CompletionHookResult::Allow;
        }

        let recent: Vec<String> = {
            let guard = self.recent_bash_commands.read().await;
            guard
                .iter()
                .rev()
                .take(self.config.require_verify_within_last_n_turns)
                .cloned()
                .collect()
        };
        let any_verify = recent.iter().any(|c| self.matches_verify(c));
        if any_verify {
            return CompletionHookResult::Allow;
        }
        CompletionHookResult::InjectAndContinue {
            system_message: Self::VERIFICATION_PROMPT.to_string(),
        }
    }
}
