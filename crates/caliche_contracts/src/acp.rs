//! ACP protocol version 1 wire types.
//!
//! The structs and enums in this module are the official
//! [`agent-client-protocol-schema`](https://crates.io/crates/agent-client-protocol-schema)
//! 1.10.2 v1 types, re-exported so Caliche clients and servers share one
//! pin. This crate does not enable `unstable_protocol_v2`. Version 2 stays
//! off the wire: agents answer `initialize` with protocol version 1.
//!
//! The higher-level `agent-client-protocol` runtime crate is not a dependency.
//! It pins schema 1.9.1, and the stdio loops in `caliche_server` and
//! `caliche_acp_client` already own transport. Pulling that runtime in
//! would duplicate the schema types.
//!
//! Official [`NewSessionRequest`] and [`LoadSessionRequest`] require
//! `mcpServers`. Callers send the field. An empty list means no servers.
//! The agent still validates the list and does not connect those servers.
//!
//! Official [`ToolCall`] omits the default kind (`other`) and the default
//! status (`pending`). Updates still send those fields when they are set.
//! [`AgentCapabilities`] always includes `auth`. An empty object means this
//! agent advertises no auth capability, and `authMethods` stays empty.

pub use agent_client_protocol_schema::ProtocolVersion;
pub use agent_client_protocol_schema::v1::{
    AGENT_METHOD_NAMES, AgentCapabilities, AgentMethodNames, AuthMethod, AuthMethodAgent,
    AuthMethodId, AuthMethodTerminal, CLIENT_METHOD_NAMES, CancelNotification, ClientCapabilities,
    ClientMethodNames, ContentBlock, ContentChunk, EmbeddedResource, EmbeddedResourceResource,
    FileSystemCapabilities, Implementation, InitializeRequest, InitializeResponse,
    ListSessionsRequest, ListSessionsResponse, LoadSessionRequest, McpCapabilities, McpServer,
    McpServerHttp, McpServerSse, McpServerStdio, NewSessionRequest, NewSessionResponse,
    PermissionOption, PermissionOptionId, PermissionOptionKind, PromptCapabilities, PromptRequest,
    PromptResponse, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionCapabilities, SessionId, SessionInfo,
    SessionListCapabilities, SessionNotification, SessionUpdate, StopReason, TextContent, ToolCall,
    ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields, ToolKind,
};

/// JSON-RPC method names from the official schema constants.
pub mod methods {
    use super::{AGENT_METHOD_NAMES, CLIENT_METHOD_NAMES};

    pub const INITIALIZE: &str = AGENT_METHOD_NAMES.initialize;
    pub const AUTHENTICATE: &str = AGENT_METHOD_NAMES.authenticate;
    pub const SESSION_NEW: &str = AGENT_METHOD_NAMES.session_new;
    pub const SESSION_LOAD: &str = AGENT_METHOD_NAMES.session_load;
    pub const SESSION_RESUME: &str = AGENT_METHOD_NAMES.session_resume;
    pub const SESSION_PROMPT: &str = AGENT_METHOD_NAMES.session_prompt;
    pub const SESSION_CANCEL: &str = AGENT_METHOD_NAMES.session_cancel;
    pub const SESSION_CLOSE: &str = AGENT_METHOD_NAMES.session_close;
    pub const SESSION_LIST: &str = AGENT_METHOD_NAMES.session_list;
    pub const SESSION_SET_MODE: &str = AGENT_METHOD_NAMES.session_set_mode;
    pub const SESSION_UPDATE: &str = CLIENT_METHOD_NAMES.session_update;
    pub const SESSION_REQUEST_PERMISSION: &str = CLIENT_METHOD_NAMES.session_request_permission;
}

#[cfg(test)]
mod tests {
    use super::{InitializeResponse, NewSessionRequest, ProtocolVersion, methods};
    use serde_json::json;

    #[test]
    fn initialize_response_is_version_one_with_empty_auth() {
        let value = serde_json::to_value(InitializeResponse::new(ProtocolVersion::V1))
            .expect("encode initialize");
        assert_eq!(value["protocolVersion"], 1);
        assert_eq!(value["authMethods"], json!([]));
        assert_eq!(methods::INITIALIZE, "initialize");
        assert_eq!(methods::SESSION_UPDATE, "session/update");
        assert_eq!(
            methods::SESSION_REQUEST_PERMISSION,
            "session/request_permission"
        );
    }

    #[test]
    fn new_session_requires_mcp_servers() {
        let missing = serde_json::from_value::<NewSessionRequest>(json!({"cwd": "/tmp"}));
        assert!(missing.is_err(), "mcpServers is required");
        let parsed: NewSessionRequest =
            serde_json::from_value(json!({"cwd": "/tmp", "mcpServers": []})).expect("decode");
        assert!(parsed.mcp_servers.is_empty());
        assert_eq!(parsed.cwd.as_os_str(), "/tmp");
    }
}
