//! Standalone Model Context Protocol (MCP) crate.
//!
//! This crate is intentionally free of the Caliche engine. Use it to:
//! - speak MCP JSON-RPC types ([`protocol`])
//! - paginate list endpoints ([`pagination`])
//! - parse protocol version dates ([`negotiate`])
//! - plug host registries in through [`providers`] traits
//!
//! The root `caliche` crate re-exports these types from `caliche::mcp`
//! and supplies adapters for tool registries, templates, and RAG.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cow_serde;
pub mod negotiate;
pub mod pagination;
pub mod protocol;
pub mod providers;
pub mod revision;
pub mod router;

pub use negotiate::parse_protocol_date;
pub use pagination::{
    DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, Page, PaginationError, cmp_name, decode_cursor,
    encode_cursor, paginate, paginate_owned,
};
pub use protocol::{
    CancelledNotificationParams, ClientCapabilities, ClientInfo, ContentPart, Cursor, Icon,
    InitializeParams, InitializeResult, JsonRpcError, JsonRpcRequest, JsonRpcResponse,
    LATEST_PROTOCOL_VERSION, LogLevel, McpPrompt, McpResource, McpTool, ProgressNotificationParams,
    ProgressToken, PromptArgument, PromptMessage, PromptsGetParams, PromptsGetResult,
    PromptsListParams, PromptsListResult, ResourceContents, ResourceReference, ResourceTemplate,
    ResourceTemplatesListParams, ResourceTemplatesListResult, ResourcesListParams,
    ResourcesListResult, ResourcesReadParams, ResourcesReadResult, SUPPORTED_PROTOCOL_VERSIONS,
    ServerCapabilities, ServerInfo, SetLevelParams, ToolAnnotations, ToolsCallParams,
    ToolsCallResult, ToolsCapability, ToolsListParams, ToolsListResult,
};
pub use providers::{
    NoPrompts, NoResources, PromptProvider, ResourceProvider, ToolDescriptor, ToolInvokeResult,
    ToolProvider,
};
pub use revision::{
    CacheHint, HEADER_MISMATCH, META_PROTOCOL_VERSION, REQUEST_CANCELLED,
    UNSUPPORTED_PROTOCOL_VERSION, admit, cache_hint, cancelled_request_key, expected_mcp_name,
    listen_response, meta_protocol_version, modern_header_error, protocol_header_is_modern,
    request_cancelled, request_id_key, shape_result, unsupported_protocol,
};
pub use router::{ProviderRouter, ProviderRouterConfig};
