//! Streamable HTTP transport primitives for MCP.
//!
//! This module provides transport-agnostic session management for MCP over HTTP.
//! HTTP framework integration (Axum handlers/routes) is implemented in API crates.

use crate::mcp::handler::McpHandler;
use crate::mcp::protocol::{JsonRpcRequest, JsonRpcResponse};
use crate::mcp::server::McpServer;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};
use uuid::Uuid;

struct SessionEntry {
    handler: Arc<Mutex<McpHandler>>,
    last_seen: Instant,
}

/// MCP Streamable HTTP session manager.
///
/// Keeps one handler per caller-supplied key. One key serializes every request
/// on that key for the whole call.
///
/// The stateless HTTP route does not use this type. Protocol `2026-07-28` has
/// no session id, so each HTTP POST builds its own handler. Stdio keeps one
/// handler for the life of the connection.
pub struct McpSessionManager {
    sessions: DashMap<String, SessionEntry>,
    ttl: Duration,
}

impl std::fmt::Debug for McpSessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpSessionManager")
            .field("session_count", &self.sessions.len())
            .field("ttl", &self.ttl)
            .finish()
    }
}

impl Default for McpSessionManager {
    fn default() -> Self {
        Self::new(Duration::from_secs(15 * 60))
    }
}

impl McpSessionManager {
    /// Create a new session manager with the provided TTL.
    pub fn new(ttl: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            ttl,
        }
    }

    /// Returns the configured session TTL.
    pub fn ttl(&self) -> Duration {
        self.ttl
    }

    /// Remove a session. Returns true when a session existed and was removed.
    pub fn remove_session(&self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }

    /// Returns true when a session exists in the table.
    ///
    /// Expired entries are cleaned asynchronously by [`Self::start_reaper`];
    /// this method intentionally stays lock-light on the hot path.
    pub fn has_session(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    /// Look up a session without touching its `last_seen` timestamp.
    pub fn session_handler(&self, session_id: &str) -> Option<Arc<Mutex<McpHandler>>> {
        self.sessions
            .get(session_id)
            .map(|entry| Arc::clone(&entry.handler))
    }

    /// Get or create a handler for `session_id`, spawning a fresh handler
    /// from `server` when the entry is absent. Also refreshes `last_seen`.
    pub fn get_or_create(&self, server: &McpServer, session_id: &str) -> Arc<Mutex<McpHandler>> {
        let mut entry = self
            .sessions
            .entry(session_id.to_string())
            .or_insert_with(|| SessionEntry {
                handler: Arc::new(Mutex::new(server.clone().create_handler())),
                last_seen: Instant::now(),
            });
        entry.last_seen = Instant::now();
        Arc::clone(&entry.handler)
    }

    /// Process one JSON-RPC request in the context of a streamable HTTP session.
    ///
    /// Returns the resolved session id and an optional JSON-RPC response
    /// (`None` for notifications).
    pub async fn handle_request(
        &self,
        server: &McpServer,
        session_id: Option<&str>,
        request: JsonRpcRequest,
    ) -> (String, Option<JsonRpcResponse>) {
        let resolved_session_id = session_id
            .map(ToString::to_string)
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let handler = self.get_or_create(server, &resolved_session_id);
        let mut guard = handler.lock().await;
        let response = guard.handle_request(request).await;
        (resolved_session_id, response)
    }

    /// Spawn a background task that evicts expired sessions at a cadence of
    /// `ttl / 4`. The returned [`tokio::task::JoinHandle`] can be dropped or
    /// aborted to stop the reaper.
    pub fn start_reaper(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let interval_period = self.ttl.max(Duration::from_secs(60)) / 4;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_period);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                self.evict_expired();
            }
        })
    }

    /// Evict sessions whose `last_seen` has exceeded the configured TTL.
    ///
    /// Exposed so callers can trigger a reaper manually in tests; the hot
    /// request path relies on [`Self::start_reaper`] instead.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        self.sessions
            .retain(|_, entry| now.duration_since(entry.last_seen) < self.ttl);
    }
}
