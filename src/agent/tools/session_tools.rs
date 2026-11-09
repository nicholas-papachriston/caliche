//! Agent-to-agent session tools (`sessions_list`, `sessions_history`, `sessions_send`, `sessions_spawn`).

use crate::agent::session::live_registry::{LiveSessionRegistry, SessionMessage};
use crate::agent::tools::tool::ToolResult;
use crate::messages::TokenUsage;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

thread_local! {
    static CALLER_SESSION_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Set the current caller session id (used by `AgentService` when entering chat). Cleared when the guard is dropped.
#[cfg(feature = "session-tools")]
pub(crate) fn set_caller_session_id(id: Option<String>) {
    CALLER_SESSION_ID.with(|c| *c.borrow_mut() = id);
}

/// Get the current caller session id for session tools (e.g. `SessionsSendTool`).
#[cfg(feature = "session-tools")]
pub(crate) fn get_caller_session_id() -> Option<String> {
    CALLER_SESSION_ID.with(|c| c.borrow().clone())
}

/// Guard that sets caller session id on creation and clears it on drop. Use at the start of `chat/stream_chat`.
#[cfg(feature = "session-tools")]
pub(crate) struct CallerSessionIdGuard;

#[cfg(feature = "session-tools")]
impl CallerSessionIdGuard {
    pub fn new(id: impl Into<String>) -> Self {
        set_caller_session_id(Some(id.into()));
        Self
    }
}

#[cfg(feature = "session-tools")]
impl Drop for CallerSessionIdGuard {
    fn drop(&mut self) {
        set_caller_session_id(None);
    }
}

/// Wraps a stream and holds `CallerSessionIdGuard` so the id is set for the duration of the stream.
#[cfg(feature = "session-tools")]
pub(crate) struct StreamWithCallerIdGuard<S> {
    _guard: CallerSessionIdGuard,
    inner: S,
}

#[cfg(feature = "session-tools")]
impl<S, Item> StreamWithCallerIdGuard<S>
where
    S: futures_util::Stream<Item = Item> + Unpin,
{
    pub fn new(id: impl Into<String>, inner: S) -> Self {
        Self {
            _guard: CallerSessionIdGuard::new(id),
            inner,
        }
    }
}

#[cfg(feature = "session-tools")]
impl<S, Item> futures_util::Stream for StreamWithCallerIdGuard<S>
where
    S: futures_util::Stream<Item = Item> + Unpin,
{
    type Item = Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

const NAME: &str = "sessions_list";
const DESCRIPTION: &str = "List all active agent sessions. Returns session IDs, models, status, and tags. Optional filter by tag or status.";
/// Lists all active sessions with their metadata.
#[derive(Debug)]
pub struct SessionsListTool;

impl crate::agent::tools::tool::ToolMeta for SessionsListTool {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "tag": { "type": "string", "description": "Filter by tag (optional)" },
                "status": { "type": "string", "enum": ["active", "idle", "processing"], "description": "Filter by status (optional)" }
            }
        })
    }
}

#[async_trait]
impl crate::agent::tools::tool::Tool for SessionsListTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let registry = LiveSessionRegistry::global();
        let tag = arguments.get("tag").and_then(|v| v.as_str());
        let sessions = match tag {
            Some(t) => registry.list_by_tag(t),
            None => registry.list(),
        };
        let status_filter = arguments.get("status").and_then(|v| v.as_str());
        let sessions: Vec<_> = match status_filter {
            Some("active") => sessions
                .into_iter()
                .filter(|s| {
                    s.status == crate::agent::session::live_registry::SessionLiveStatus::Active
                })
                .collect(),
            Some("idle") => sessions
                .into_iter()
                .filter(|s| {
                    s.status == crate::agent::session::live_registry::SessionLiveStatus::Idle
                })
                .collect(),
            Some("processing") => sessions
                .into_iter()
                .filter(|s| {
                    s.status == crate::agent::session::live_registry::SessionLiveStatus::Processing
                })
                .collect(),
            _ => sessions,
        };
        let out = match serde_json::to_string_pretty(&sessions) {
            Ok(s) => s,
            Err(e) => return ToolResult::error(e.to_string()),
        };
        ToolResult::success(out)
    }
}

/// Fetches conversation history from another session (placeholder: live registry does not store history).
#[derive(Debug)]
pub struct SessionsHistoryTool;

impl crate::agent::tools::tool::ToolMeta for SessionsHistoryTool {
    fn name(&self) -> &'static str {
        "sessions_history"
    }

    fn description(&self) -> &'static str {
        "Fetch conversation history from another session. Requires session_id. Optional limit (default 20)."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("sessions_history")
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(
            "Fetch conversation history from another session. Requires session_id. Optional limit (default 20).",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string", "description": "Target session ID" },
                "limit": { "type": "integer", "description": "Max messages to return (default 20)" }
            },
            "required": ["session_id"]
        })
    }
}

#[async_trait]
impl crate::agent::tools::tool::Tool for SessionsHistoryTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let session_id = match arguments.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("session_id is required"),
        };
        let limit = arguments
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(20) as usize;
        let registry = LiveSessionRegistry::global();
        let Some(entries) = registry.history_snapshot(session_id, limit) else {
            return ToolResult::error(format!("Session {session_id} not found"));
        };
        match serde_json::to_string_pretty(&entries) {
            Ok(out) => ToolResult::success(out),
            Err(err) => ToolResult::error(format!("Serialize history: {err}")),
        }
    }
}

/// Sends a message to another session with optional reply-back.
#[derive(Debug)]
pub struct SessionsSendTool;

impl crate::agent::tools::tool::ToolMeta for SessionsSendTool {
    fn name(&self) -> &'static str {
        "sessions_send"
    }

    fn description(&self) -> &'static str {
        "Send a message to another session. If reply_back is true, waits for the target's response (timeout 120s)."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("sessions_send")
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(
            "Send a message to another session. If reply_back is true, waits for the target's response (timeout 120s).",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string", "description": "Target session ID" },
                "message": { "type": "string", "description": "Message content" },
                "reply_back": { "type": "boolean", "description": "If true, wait for response (default false)" }
            },
            "required": ["session_id", "message"]
        })
    }
}

#[async_trait]
impl crate::agent::tools::tool::Tool for SessionsSendTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let session_id = match arguments.get("session_id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolResult::error("session_id is required"),
        };
        let message = match arguments.get("message").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => return ToolResult::error("message is required"),
        };
        let reply_back = arguments
            .get("reply_back")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let registry = LiveSessionRegistry::global();
        let (reply_tx, reply_rx) = if reply_back {
            let (tx, rx) = oneshot::channel();
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let from_session_id = get_caller_session_id().unwrap_or_else(|| "caller".to_string());
        let msg = SessionMessage {
            from_session_id,
            content: message,
            reply_back,
            reply_tx,
        };

        if let Err(e) = registry.send_message(&session_id, msg).await {
            return ToolResult::error(e.to_string());
        }

        if let Some(rx) = reply_rx {
            let timeout = Duration::from_secs(120);
            match tokio::time::timeout(timeout, rx).await {
                Ok(Ok(resp)) => {
                    let out = format!(
                        "Response from {}: {} (usage: {:?})",
                        resp.session_id, resp.content, resp.usage
                    );
                    ToolResult::success(out)
                }
                Ok(Err(_)) => ToolResult::error("Target session did not reply"),
                Err(_) => ToolResult::error("Timeout waiting for reply (120s)"),
            }
        } else {
            ToolResult::success("Message sent")
        }
    }
}

/// Spawner for new sub-sessions (injected by the service when session-tools are enabled).
#[async_trait]
pub trait SessionSpawner: Send + Sync + std::fmt::Debug {
    /// Update the model/provider used when spawn args omit them.
    fn set_selection(&self, _selection: crate::agent::subagent::SubagentModelSelection) {}

    /// Spawn a new session with the given message and optional config; return `session_id`, response text, and usage.
    async fn spawn(
        &self,
        model: Option<&str>,
        tools: Option<&[String]>,
        message: &str,
        tags: Option<&[String]>,
        workspace: Option<&std::path::Path>,
    ) -> Result<(String, String, TokenUsage)>;
}

/// Spawns a new sub-session (no-op when no spawner is configured).
#[derive(Debug)]
pub struct SessionsSpawnTool {
    spawner: Option<Arc<dyn SessionSpawner>>,
}

impl SessionsSpawnTool {
    #[must_use]
    pub fn new(spawner: Option<Arc<dyn SessionSpawner>>) -> Self {
        Self { spawner }
    }
}

impl crate::agent::tools::tool::ToolMeta for SessionsSpawnTool {
    fn name(&self) -> &'static str {
        "sessions_spawn"
    }

    fn description(&self) -> &'static str {
        "Spawn a new sub-session with optional model, tools, message, tags, and workspace. Returns session_id and response."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("sessions_spawn")
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(
            "Spawn a new sub-session with optional model, tools, message, tags, and workspace. Returns session_id and response.",
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "model": { "type": "string", "description": "Model name (optional)" },
                "tools": { "type": "array", "items": { "type": "string" }, "description": "Tool names (optional)" },
                "message": { "type": "string", "description": "Initial message" },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags (optional)" },
                "workspace": { "type": "string", "description": "Workspace path (optional)" }
            },
            "required": ["message"]
        })
    }
}

#[async_trait]
impl crate::agent::tools::tool::Tool for SessionsSpawnTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let Some(ref spawner) = self.spawner else {
            return ToolResult::error("Sessions spawn not configured (no spawner)");
        };
        let message = match arguments.get("message").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return ToolResult::error("message is required"),
        };
        let model = arguments.get("model").and_then(|v| v.as_str());
        let tools: Option<Vec<String>> =
            arguments.get("tools").and_then(|v| v.as_array()).map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });
        let tags: Option<Vec<String>> = arguments.get("tags").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        });
        let workspace = arguments
            .get("workspace")
            .and_then(|v| v.as_str())
            .map(std::path::PathBuf::from);

        let tools_ref = tools.as_deref();
        let tags_ref = tags.as_deref();
        let workspace_ref = workspace.as_deref();

        match spawner
            .spawn(model, tools_ref, message, tags_ref, workspace_ref)
            .await
        {
            Ok((session_id, response_text, usage)) => {
                let out = json!({
                    "session_id": session_id,
                    "response": response_text,
                    "usage": usage
                });
                ToolResult::success(serde_json::to_string_pretty(&out).unwrap_or_default())
            }
            Err(e) => ToolResult::error(e),
        }
    }
}

/// Configuration for session tools (e.g. spawner for `sessions_spawn`).
/// Use with `AgentBuilder::session_tools_config()` to wire session tools from a single config.
#[derive(Debug, Clone)]
pub struct SessionToolsConfig {
    /// Optional spawner for the `sessions_spawn` tool.
    pub spawner: Option<Arc<dyn SessionSpawner>>,
}

/// Builder for `SessionToolsConfig`.
#[derive(Debug, Default)]
pub struct SessionToolsConfigBuilder {
    spawner: Option<Arc<dyn SessionSpawner>>,
}

impl SessionToolsConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawner(mut self, s: Arc<dyn SessionSpawner>) -> Self {
        self.spawner = Some(s);
        self
    }

    #[must_use]
    pub fn build(self) -> SessionToolsConfig {
        SessionToolsConfig {
            spawner: self.spawner,
        }
    }
}

impl SessionToolsConfig {
    /// Fluent builder for session tools config.
    #[must_use]
    pub fn builder() -> SessionToolsConfigBuilder {
        SessionToolsConfigBuilder::new()
    }
}
