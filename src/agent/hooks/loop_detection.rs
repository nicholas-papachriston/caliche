//! Loop detection middleware.
//!
//! Tracks per-file edit counts and per-tool failure counts. When a file exceeds
//! the edit threshold, or the same tool call fails repeatedly, injects a context
//! note on the next LLM call to nudge the agent toward a different approach.

use super::lifecycle::{HookContext, HookResult, LifecycleHook, PreLlmContext};
use crate::agent::tools::names::{FS_OP_EDIT, FS_OP_WRITE, TOOL_FILESYSTEM, fs_op};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Config for the loop detection middleware.
#[derive(Debug, Clone)]
pub struct LoopDetectionConfig {
    /// Trigger injection after N edits to the same file (successes or failures combined).
    pub edit_threshold: usize,
    /// Stop injecting after this many warnings per file.
    pub max_warnings: usize,
    /// Trigger a failure injection after N consecutive failures for the same file/tool.
    pub failure_threshold: usize,
}

impl Default for LoopDetectionConfig {
    fn default() -> Self {
        Self {
            edit_threshold: 3,
            max_warnings: 3,
            failure_threshold: 3,
        }
    }
}

/// Per-file/tool tracking state.
#[derive(Debug, Default)]
struct EditState {
    /// Total edit attempts (success + failure).
    count: usize,
    /// Consecutive failures without an intervening success.
    consecutive_failures: usize,
    /// Number of loop-detection warnings already issued for this entry.
    warnings_issued: usize,
}

/// Loop detection middleware.
///
/// Tracks edits to files in `on_post_tool_use`. When a file exceeds the threshold,
/// or a tool call repeatedly fails, stores a pending injection for the next LLM call.
#[derive(Debug)]
pub struct LoopDetectionMiddleware {
    config: LoopDetectionConfig,
    /// Per-file edit counts and warning counts.
    edit_counts: Arc<RwLock<HashMap<PathBuf, EditState>>>,
    /// Pending context injections to prepend to the next LLM call.
    pending_injections: Arc<RwLock<Vec<String>>>,
}

impl LoopDetectionMiddleware {
    /// Create a new middleware with the given config.
    #[must_use]
    pub fn new(config: LoopDetectionConfig) -> Self {
        Self {
            config,
            edit_counts: Arc::new(RwLock::new(HashMap::new())),
            pending_injections: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Return the first known path that appears as a substring in the bash command.
    /// Returns an owned String so the read lock on `edit_counts` can be released.
    fn find_bash_path(command: &str, known_paths: &HashMap<PathBuf, EditState>) -> Option<String> {
        known_paths.keys().find_map(|p| {
            p.to_str()
                .filter(|s| command.contains(*s))
                .map(str::to_string)
        })
    }
}

#[async_trait]
impl LifecycleHook for LoopDetectionMiddleware {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_post_tool_use(
        &self,
        ctx: &HookContext<'_>,
        result: &crate::agent::tools::tool::ToolResult,
    ) -> HookResult {
        let name = ctx.tool_name.as_ref();

        // Determine the path key for this tool call.
        let path_str: Option<&str> = if name == TOOL_FILESYSTEM {
            let op = fs_op(ctx.tool_input).unwrap_or("");
            if matches!(op, FS_OP_WRITE | FS_OP_EDIT) {
                ctx.tool_input.get("path").and_then(|v| v.as_str())
            } else {
                None
            }
        } else if name == "bash" {
            let command = ctx
                .tool_input
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            // Only track bash calls if they reference a file we're already watching.
            let path_owned = {
                let counts = self.edit_counts.read().await;
                Self::find_bash_path(command, &counts)
            };
            if let Some(path_owned) = path_owned {
                let path = PathBuf::from(&path_owned);
                let mut edit_counts = self.edit_counts.write().await;
                let state = edit_counts.entry(path).or_default();
                state.count += 1;
                if result.success {
                    state.consecutive_failures = 0;
                } else {
                    state.consecutive_failures += 1;
                }
                let should_warn_edit = state.count >= self.config.edit_threshold
                    && state.warnings_issued < self.config.max_warnings;
                let should_warn_fail = state.consecutive_failures >= self.config.failure_threshold
                    && state.warnings_issued < self.config.max_warnings;
                if should_warn_edit || should_warn_fail {
                    state.warnings_issued += 1;
                    let msg = if should_warn_fail {
                        format!(
                            "Warning: You have attempted to modify '{}' {} times via bash, \
                             with {} consecutive failures. Stop retrying the same strategy. \
                             Re-read the task specification and consider a completely different approach.",
                            path_owned, state.count, state.consecutive_failures
                        )
                    } else {
                        format!(
                            "Note: You have edited '{}' {} times (including bash). \
                             If the same approach keeps failing, consider:\n\
                             - Re-reading the original task specification from scratch\n\
                             - Using a completely different implementation strategy\n\
                             - Checking whether a dependency or configuration is the real issue",
                            path_owned, state.count
                        )
                    };
                    drop(edit_counts);
                    self.pending_injections.write().await.push(msg);
                }
            }
            return HookResult::allow();
        } else {
            return HookResult::allow();
        };

        let Some(path_str) = path_str else {
            return HookResult::allow();
        };

        let path = PathBuf::from(path_str);
        let mut counts = self.edit_counts.write().await;
        let state = counts.entry(path).or_default();
        state.count += 1;

        if result.success {
            state.consecutive_failures = 0;
        } else {
            state.consecutive_failures += 1;
        }

        let should_warn_edit = state.count >= self.config.edit_threshold
            && state.warnings_issued < self.config.max_warnings;
        let should_warn_fail = state.consecutive_failures >= self.config.failure_threshold
            && state.warnings_issued < self.config.max_warnings;

        if should_warn_edit || should_warn_fail {
            state.warnings_issued += 1;
            let msg = if should_warn_fail {
                format!(
                    "Warning: The last {} attempts to edit '{}' have all failed. \
                     Stop retrying the same strategy. Re-read the task specification and \
                     consider a completely different approach — the current method is not working.",
                    state.consecutive_failures, path_str
                )
            } else {
                format!(
                    "Note: You have edited '{}' {} times. If the same approach keeps failing, consider:\n\
                     - Re-reading the original task specification from scratch\n\
                     - Using a completely different implementation strategy\n\
                     - Checking whether a dependency or configuration is the real issue",
                    path_str, state.count
                )
            };
            drop(counts);
            self.pending_injections.write().await.push(msg);
        }
        HookResult::allow()
    }

    async fn on_pre_llm_call(&self, _ctx: &PreLlmContext<'_>) -> Option<String> {
        let mut guard = self.pending_injections.write().await;
        let injections = std::mem::take(&mut *guard);
        if injections.is_empty() {
            None
        } else {
            Some(injections.join("\n\n"))
        }
    }
}
