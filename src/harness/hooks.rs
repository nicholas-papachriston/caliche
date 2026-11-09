//! Tool lifecycle hooks for intercepting and observing tool execution.
//!
//! These hooks allow the harness to:
//! - Emit events for tool execution
//! - Track file modifications automatically
//! - Record undo/rollback state
//! - Implement permission prompts

use crate::agent::hooks::{HookContext, HookResult, LifecycleHook};
use crate::agent::tools::{ToolCall, ToolResult};
use crate::harness::events::{AgentEvent, EventStream};
use crate::harness::formatter::FormatterRule;
use crate::harness::state::StateManager;
use async_trait::async_trait;
use beef::lean::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::RwLock;

/// Tool execution hook that gets called before and after tool execution
pub struct ToolHooks {
    event_stream: EventStream,
    state_manager: Arc<RwLock<StateManager>>,
    current_turn: AtomicUsize,
    formatter_rules: Vec<FormatterRule>,
    started_at: Mutex<HashMap<String, Instant>>,
    /// Provider tool-use id to UUID v7. Shared with the agent stream loop when set.
    action_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl ToolHooks {
    /// Create a new tool hooks instance
    pub fn new(event_stream: EventStream, state_manager: Arc<RwLock<StateManager>>) -> Self {
        Self {
            event_stream,
            state_manager,
            current_turn: AtomicUsize::new(0),
            formatter_rules: vec![],
            started_at: Mutex::new(HashMap::new()),
            action_ids: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Share the agent's tool-action map so stream events and hook events use one id.
    #[must_use]
    pub fn with_action_ids(mut self, action_ids: Arc<Mutex<HashMap<String, String>>>) -> Self {
        self.action_ids = action_ids;
        self
    }

    /// Attach formatter rules loaded from config.
    pub fn with_formatters(mut self, rules: Vec<FormatterRule>) -> Self {
        self.formatter_rules = rules;
        self
    }

    /// Set the current turn number
    pub fn set_turn(&self, turn: usize) {
        self.current_turn.store(turn, Ordering::Relaxed);
    }

    fn mark_tool_start(&self, tool_id: &str) {
        if let Ok(mut started) = self.started_at.lock() {
            started.insert(tool_id.to_string(), Instant::now());
        }
    }

    fn lock_action_ids(&self) -> std::sync::MutexGuard<'_, HashMap<String, String>> {
        self.action_ids
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn action_id_for(&self, tool_use_id: &str) -> String {
        let mut ids = self.lock_action_ids();
        if let Some(existing) = ids.get(tool_use_id) {
            return existing.clone();
        }
        let action_id = crate::utils::time_id::generate_time_id().to_string();
        ids.insert(tool_use_id.to_string(), action_id.clone());
        action_id
    }

    fn take_action_id(&self, tool_use_id: &str) -> String {
        self.lock_action_ids()
            .remove(tool_use_id)
            .unwrap_or_else(|| crate::utils::time_id::generate_time_id().to_string())
    }

    fn take_tool_duration_ms(&self, tool_id: &str) -> u64 {
        self.started_at
            .lock()
            .ok()
            .and_then(|mut started| started.remove(tool_id))
            .map(|started| u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX))
            .unwrap_or(0)
    }

    fn tool_call_from_context(ctx: &HookContext<'_>) -> ToolCall {
        ToolCall {
            name: Cow::owned(ctx.tool_name.as_ref().to_owned()),
            arguments: ctx.tool_input.clone(),
        }
    }

    /// Called before a tool is executed
    pub async fn before_tool(&self, tool_id: &str, call: &ToolCall) {
        self.mark_tool_start(tool_id);
        let action_id = self.action_id_for(tool_id);
        self.event_stream.emit(AgentEvent::ToolExecuting {
            tool_id: tool_id.to_string(),
            tool_name: call.name.as_ref().to_string(),
            progress: Some(0.0),
            action_id,
        });

        // If this is an fs write/edit op, capture the old content
        if call.name == "fs"
            && matches!(
                call.arguments.get("op").and_then(|v| v.as_str()),
                Some("write" | "edit")
            )
            && let Some(path_str) = call.arguments.get("path").and_then(|v| v.as_str())
        {
            let path = PathBuf::from(path_str);
            if let Ok(old_content) = tokio::fs::read_to_string(&path).await {
                let mut state = self.state_manager.write().await;
                state.set_turn(self.current_turn.load(Ordering::Relaxed));
                // Pre-record the file state before modification
                state.record_file_write(&path, Some(&old_content), "");
            }
        }
    }

    /// Called after a tool is executed
    pub async fn after_tool(
        &self,
        tool_id: &str,
        call: &ToolCall,
        result: &ToolResult,
        duration_ms: u64,
    ) {
        let duration_ms = if duration_ms == 0 {
            self.take_tool_duration_ms(tool_id)
        } else {
            duration_ms
        };
        let action_id = self.take_action_id(tool_id);
        if result.success {
            self.event_stream.emit(AgentEvent::tool_complete(
                tool_id,
                call.name.as_ref(),
                result,
                duration_ms,
                action_id,
            ));
        } else {
            self.event_stream.emit(AgentEvent::ToolError {
                tool_id: tool_id.to_string(),
                tool_name: call.name.as_ref().to_string(),
                error: result
                    .error
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string()),
                timestamp: chrono::Utc::now(),
                action_id,
            });
        }

        // Track file operations
        self.track_file_operation(call, result).await;
    }

    /// Run the first matching formatter for `path`, logging a warning if it fails.
    /// No-ops when no formatter rules are configured or no rule matches the extension.
    async fn maybe_run_formatter(&self, path: &Path) {
        if self.formatter_rules.is_empty() {
            return;
        }
        if let Some(warning) =
            crate::harness::formatter::run_formatter_if_match(&self.formatter_rules, path).await
        {
            tracing::warn!(formatter_warning = %warning);
        }
    }

    /// Track file operations for undo/rollback
    async fn track_file_operation(&self, call: &ToolCall, result: &ToolResult) {
        if !result.success {
            return;
        }

        if call.name.as_ref() == "fs" {
            let op = call
                .arguments
                .get("op")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if let Some(path_str) = call.arguments.get("path").and_then(|v| v.as_str()) {
                let path = PathBuf::from(path_str);
                match op {
                    "read" => {
                        self.event_stream.emit(AgentEvent::FileRead {
                            path,
                            lines: None,
                            size_bytes: result.content.len(),
                        });
                    }
                    "write" => {
                        let is_new = !path.exists();
                        self.event_stream.emit(AgentEvent::file_write(
                            path.clone(),
                            None, // Could add diff preview here
                            is_new,
                            None,
                        ));
                        self.maybe_run_formatter(&path).await;
                    }
                    "edit" => {
                        self.event_stream.emit(AgentEvent::file_write(
                            path.clone(),
                            None,
                            false,
                            None,
                        ));
                        self.maybe_run_formatter(&path).await;
                    }
                    "list" => {
                        let file_count = result.content.lines().count();
                        self.event_stream
                            .emit(AgentEvent::DirectoryListed { path, file_count });
                    }
                    _ => {}
                }
            }
        }
    }
}

impl std::fmt::Debug for ToolHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolHooks")
            .field("current_turn", &self.current_turn.load(Ordering::Relaxed))
            .finish()
    }
}

#[async_trait]
impl LifecycleHook for ToolHooks {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        let call = Self::tool_call_from_context(ctx);
        self.before_tool(ctx.tool_use_id.as_ref(), &call).await;
        HookResult::allow()
    }

    async fn on_post_tool_use(&self, ctx: &HookContext<'_>, result: &ToolResult) -> HookResult {
        let call = Self::tool_call_from_context(ctx);
        self.after_tool(ctx.tool_use_id.as_ref(), &call, result, 0)
            .await;
        HookResult::allow()
    }
}

/// Represents the result of analyzing tool calls in a response
#[derive(Debug, Clone)]
pub struct ToolCallAnalysis {
    /// Tool calls that were requested
    pub tool_calls: Vec<ToolCall>,
    /// Whether there are file modifications
    pub has_file_modifications: bool,
    /// Files that would be modified
    pub files_to_modify: Vec<PathBuf>,
    /// Whether there are dangerous operations (bash, delete, etc.)
    pub has_dangerous_ops: bool,
}

impl ToolCallAnalysis {
    /// Analyze a list of tool calls
    pub fn analyze(calls: &[ToolCall]) -> Self {
        let mut files_to_modify = Vec::new();
        let mut has_dangerous_ops = false;

        for call in calls {
            match call.name.as_ref() {
                "fs" => {
                    let op = call
                        .arguments
                        .get("op")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if matches!(op, "write" | "edit" | "delete")
                        && let Some(path_str) = call.arguments.get("path").and_then(|v| v.as_str())
                    {
                        files_to_modify.push(PathBuf::from(path_str));
                    }
                }
                "bash" | "execute_python" => {
                    has_dangerous_ops = true;
                }
                _ => {}
            }
        }

        Self {
            tool_calls: calls.to_vec(),
            has_file_modifications: !files_to_modify.is_empty(),
            files_to_modify,
            has_dangerous_ops,
        }
    }

    /// Check if permission should be requested
    pub fn needs_permission(&self) -> bool {
        self.has_file_modifications || self.has_dangerous_ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::HookContext;
    use serde_json::json;

    #[tokio::test]
    async fn lifecycle_hooks_emit_tool_and_file_events() {
        let stream = EventStream::new();
        let mut rx = stream.subscribe();
        let hooks = ToolHooks::new(stream, Arc::new(RwLock::new(StateManager::new())));
        let args = json!({"op": "read", "path": "/tmp/example.rs"});
        let ctx = HookContext::new("fs", &args, "call-1");

        hooks.on_pre_tool_use(&ctx).await;
        let result = ToolResult::success("fn main() {}");
        hooks.on_post_tool_use(&ctx, &result).await;

        let executing = rx.try_recv().expect("ToolExecuting");
        let start_id = match &executing {
            AgentEvent::ToolExecuting {
                tool_name,
                tool_id,
                action_id,
                ..
            } => {
                assert_eq!(tool_name, "fs");
                assert_eq!(tool_id, "call-1");
                assert_time_id(action_id);
                action_id.clone()
            }
            other => panic!("expected ToolExecuting, got {other:?}"),
        };
        let complete = rx.try_recv().expect("ToolComplete");
        match &complete {
            AgentEvent::ToolComplete {
                tool_name,
                action_id,
                success: true,
                ..
            } => {
                assert_eq!(tool_name, "fs");
                assert_eq!(action_id, &start_id);
            }
            other => panic!("expected ToolComplete, got {other:?}"),
        }
        let read = rx.try_recv().expect("FileRead");
        match read {
            AgentEvent::FileRead { path, .. } => {
                assert_eq!(path, PathBuf::from("/tmp/example.rs"));
            }
            other => panic!("expected FileRead, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn lifecycle_hooks_emit_bash_events() {
        let stream = EventStream::new();
        let mut rx = stream.subscribe();
        let hooks = ToolHooks::new(stream, Arc::new(RwLock::new(StateManager::new())));
        let args = json!({"command": "cargo test"});
        let ctx = HookContext::new("bash", &args, "call-bash");

        hooks.on_pre_tool_use(&ctx).await;
        let result = ToolResult::success("ok");
        hooks.on_post_tool_use(&ctx, &result).await;

        let start = rx.try_recv().expect("ToolExecuting");
        let AgentEvent::ToolExecuting {
            tool_name,
            action_id: start_id,
            ..
        } = &start
        else {
            panic!("expected ToolExecuting, got {start:?}");
        };
        assert_eq!(tool_name, "bash");
        assert_time_id(start_id);
        let end = rx.try_recv().expect("ToolComplete");
        let AgentEvent::ToolComplete {
            tool_name,
            action_id: end_id,
            success: true,
            ..
        } = &end
        else {
            panic!("expected ToolComplete, got {end:?}");
        };
        assert_eq!(tool_name, "bash");
        assert_eq!(end_id, start_id);
    }

    #[tokio::test]
    async fn hooks_reuse_a_published_action_id() {
        let stream = EventStream::new();
        let mut rx = stream.subscribe();
        let published = "01936b8e-7c3a-7000-8000-000000000001".to_string();
        let ids = Arc::new(Mutex::new(HashMap::from([(
            "call-1".to_string(),
            published.clone(),
        )])));
        let hooks = ToolHooks::new(stream, Arc::new(RwLock::new(StateManager::new())))
            .with_action_ids(Arc::clone(&ids));
        let args = json!({"command": "true"});
        let ctx = HookContext::new("bash", &args, "call-1");

        hooks.on_pre_tool_use(&ctx).await;
        let start = rx.try_recv().expect("ToolExecuting");
        let AgentEvent::ToolExecuting { action_id, .. } = &start else {
            panic!("expected ToolExecuting, got {start:?}");
        };
        assert_eq!(action_id, &published);

        hooks
            .on_post_tool_use(&ctx, &ToolResult::success("ok"))
            .await;
        let end = rx.try_recv().expect("ToolComplete");
        let AgentEvent::ToolComplete { action_id, .. } = &end else {
            panic!("expected ToolComplete, got {end:?}");
        };
        assert_eq!(action_id, &published);
        assert!(
            ids.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get("call-1")
                .is_none()
        );
    }

    fn assert_time_id(id: &str) {
        let parsed = uuid::Uuid::parse_str(id).expect("action id is a uuid");
        assert_eq!(parsed.get_version(), Some(uuid::Version::SortRand));
    }
}
