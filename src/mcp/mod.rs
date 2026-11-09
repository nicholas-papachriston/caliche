//! Model Context Protocol (MCP) server implementation.
//!
//! This module provides a complete MCP server implementation that exposes
//! caliche's tools, templates (as prompts), and resources (RAG collections)
//! through the Model Context Protocol.
//!
//! # Features
//!
//! - **Tools**: Expose all registered tools via MCP tools capability
//! - **Prompts**: Expose agent templates as MCP prompts
//! - **Resources**: Expose RAG document collections as MCP resources
//! - **Multiple Transports**: `stdio` and streamable HTTP.
//!
//! # Example
//!
//! ```rust,no_run
//! use caliche::mcp::McpServer;
//! use caliche::ToolRegistry;
//! use caliche::templates::TemplateRegistry;
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let tool_registry = Arc::new(ToolRegistry::new());
//! let template_registry = TemplateRegistry::new();
//!
//! let server = McpServer::new(tool_registry, template_registry);
//! // Use server with a transport...
//! # let _ = server;
//! # Ok(())
//! # }
//! ```

pub mod capabilities;
#[cfg(feature = "mcp-client")]
pub mod client;
#[cfg(feature = "code-mode")]
pub mod code_mode;
pub mod collection_metadata;
mod engine_providers;
pub mod handler;
pub mod in_process_server;
pub mod policy_gate;
pub mod protocol;
pub mod server;
#[cfg(feature = "mcp-client")]
pub mod stdio_allowlist;
pub mod transports;

#[cfg(feature = "mcp-client")]
pub use client::{McpClient, McpClientConfig, McpClientTransportConfig, McpInputRequired};
#[cfg(feature = "code-mode")]
pub use code_mode::{
    CodeModeAuth, CodeModeMcpServerBuilder, CodeModeOptions, CodeModeSandboxConfig, CodeModeSchema,
    OperationDef,
};
pub use collection_metadata::CollectionMetadata;

pub use caliche_mcp::revision::{modern_header_error, protocol_header_is_modern};
pub use handler::McpHandler;
// Explicit exports from protocol module (MCP protocol types)
pub use in_process_server::{InProcessMcpServer, InProcessMcpServerBuilder};
pub use protocol::{
    CancelledNotificationParams, ClientCapabilities, ClientInfo, Cursor, InitializeParams,
    InitializeResult, JsonRpcError, JsonRpcRequest, JsonRpcResponse, LATEST_PROTOCOL_VERSION,
    LogLevel, McpTool, PingParams, PingResult, ProgressNotificationParams, ProgressToken,
    PromptsCapability, PromptsListParams, PromptsListResult, RequestId, ResourceTemplate,
    ResourceTemplatesListResult, ResourcesCapability, ResourcesListParams, ResourcesListResult,
    SUPPORTED_PROTOCOL_VERSIONS, ServerCapabilities, ServerInfo, SetLevelParams, ToolsCallParams,
    ToolsCallResult, ToolsCapability, ToolsListParams, ToolsListResult,
};
pub use server::{McpServer, McpServerBuilder, McpServerConfig, McpServerConfigBuilder};
#[cfg(feature = "mcp-client")]
pub use stdio_allowlist::{
    command_allowed_by_allowlist, stdio_command_allowlist_from_env, validate_stdio_command_against,
    validate_stdio_command_for_library, validate_stdio_command_required,
};
pub use transports::McpSessionManager;
