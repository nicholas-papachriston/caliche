/// JSON-RPC protocol version on every request, response, and notification.
pub const JSONRPC_VERSION: &str = "2.0";

pub mod methods {
    pub const CHAT_SEND: &str = "chat.send";
    pub const CHAT_STREAM: &str = "chat.stream";
    pub const CHAT_STREAM_EVENT: &str = "chat.stream.event";
    pub const SETTINGS_GET: &str = "settings.get";
    pub const SETTINGS_UPDATE: &str = "settings.update";
    pub const TOOLS_LIST: &str = "tools.list";
    pub const MODELS_LIST: &str = "models.list";
    pub const MODELS_LIST_BY_PROVIDER: &str = "models.listByProvider";

    pub const MCP_GET_CONFIG: &str = "mcp.getConfig";
    pub const MCP_UPDATE_CONFIG: &str = "mcp.updateConfig";
    pub const MCP_TEST_CONFIG: &str = "mcp.testConfig";

    pub const SESSIONS_CREATE: &str = "sessions.create";
    pub const SESSIONS_GET: &str = "sessions.get";
    pub const SESSIONS_DELETE: &str = "sessions.delete";
    pub const SESSIONS_UNDO: &str = "sessions.undo";
    pub const SESSIONS_REDO: &str = "sessions.unrevert";
    pub const SESSIONS_CLEAR_HISTORY: &str = "sessions.clearHistory";
    pub const SESSIONS_CHECKPOINT: &str = "sessions.checkpoint";
    pub const SESSIONS_ROLLBACK: &str = "sessions.rollback";
    pub const SESSIONS_GET_MODE: &str = "sessions.getMode";
    pub const SESSIONS_SET_MODE: &str = "sessions.setMode";
    pub const SESSIONS_GET_CONTEXT: &str = "sessions.getContext";
    pub const SESSIONS_DIFF: &str = "sessions.diff";
    pub const SESSIONS_GET_PLAN: &str = "sessions.getPlan";
    pub const SESSIONS_TRACK_FILE: &str = "sessions.trackFile";
    /// Notification. One harness event for a subscribed session.
    ///
    /// Params: `{ "session_id": string, "event": object }`.
    pub const SESSIONS_EVENT: &str = "sessions.event";
    /// Notification. This session subscription missed `skipped` broadcasts.
    ///
    /// Params: `{ "session_id": string, "skipped": number }`. On receipt, re-read
    /// session history and the latest checkpoint before acting on later
    /// `sessions.event` notifications. `skipped` counts missed broadcasts.
    /// Resume a task or chat stream with `after` set to the last durable event
    /// id the client stored. The session event socket is live only and does
    /// not honor `Last-Event-ID`. If a subscription send fails, the forwarder
    /// stops instead of delivering later events without this notification.
    pub const SESSIONS_RESYNC: &str = "sessions.resync";

    pub const CODING_RUN: &str = "coding.run";
}
