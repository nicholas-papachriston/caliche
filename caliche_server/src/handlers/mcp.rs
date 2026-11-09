//! MCP API handlers for HTTP endpoints.

#[cfg(feature = "mcp")]
pub mod http;
#[cfg(feature = "mcp")]
pub mod streamable_http;

#[cfg(feature = "mcp")]
pub use http::{capabilities_handler, prompts_handler, resources_handler, tools_handler};
#[cfg(feature = "mcp")]
pub use streamable_http::{
    StreamableHttpState, streamable_http_delete_handler, streamable_http_get_handler,
    streamable_http_post_handler,
};
