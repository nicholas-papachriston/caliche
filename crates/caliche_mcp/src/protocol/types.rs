//! MCP (Model Context Protocol) protocol types and JSON-RPC 2.0 structures.
//!
//! This module defines the core protocol structures for MCP communication,
//! including JSON-RPC 2.0 message formats and MCP-specific request/response types.

use crate::cow_serde::{
    deserialize_cow_str, deserialize_option_cow_str, serialize_cow_str, serialize_option_cow_str,
};
use beef::lean::Cow;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (must be "2.0")
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub jsonrpc: Cow<'static, str>,
    /// Method name
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub method: Cow<'static, str>,
    /// Request parameters (can be object, array, or null)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
    /// Request ID (can be string, number, or null)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

/// JSON-RPC 2.0 response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (must be "2.0")
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub jsonrpc: Cow<'static, str>,
    /// Result (present on success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error (present on failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    /// Request ID (matches request ID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

impl JsonRpcResponse {
    /// Create a success response
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: Cow::borrowed("2.0"),
            result: Some(result),
            error: None,
            id,
        }
    }

    /// Create an error response
    pub fn error(id: Option<Value>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: Cow::borrowed("2.0"),
            result: None,
            error: Some(error),
            id,
        }
    }
}

/// JSON-RPC 2.0 error structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub message: Cow<'static, str>,
    /// Optional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    /// Parse error (-32700)
    pub fn parse_error(data: Option<Value>) -> Self {
        Self {
            code: -32700,
            message: Cow::borrowed("Parse error"),
            data,
        }
    }

    /// Invalid request (-32600)
    pub fn invalid_request(data: Option<Value>) -> Self {
        Self {
            code: -32600,
            message: Cow::borrowed("Invalid Request"),
            data,
        }
    }

    /// Method not found (-32601)
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: Cow::borrowed("Method not found"),
            data: Some(Value::String(method.to_string())),
        }
    }

    /// Invalid params (-32602)
    pub fn invalid_params(data: Option<Value>) -> Self {
        Self {
            code: -32602,
            message: Cow::borrowed("Invalid params"),
            data,
        }
    }

    /// Internal error (-32603)
    pub fn internal_error(data: Option<Value>) -> Self {
        Self {
            code: -32603,
            message: Cow::borrowed("Internal error"),
            data,
        }
    }

    /// Server error (custom, -32000 to -32099)
    pub fn server_error(code: i32, message: String, data: Option<Value>) -> Self {
        Self {
            code,
            message: Cow::owned(message),
            data,
        }
    }
}

/// Protocol revisions this server speaks.
///
/// Pre-release servers speak only `2026-07-28`. A client puts that version in
/// `_meta` on every request and calls `server/discover`. The outbound client
/// still speaks `initialize` when it calls an older server.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2026-07-28"];

/// Latest MCP protocol version supported by this server.
///
/// Derived from the last entry of [`SUPPORTED_PROTOCOL_VERSIONS`] so the two
/// stay in sync automatically.
pub const LATEST_PROTOCOL_VERSION: &str =
    SUPPORTED_PROTOCOL_VERSIONS[SUPPORTED_PROTOCOL_VERSIONS.len() - 1];

/// Initialize request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    /// Protocol version
    pub protocol_version: String,
    /// Client capabilities
    pub capabilities: ClientCapabilities,
    /// Client information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_info: Option<ClientInfo>,
}

/// Client capabilities
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    /// Roots capability (workspace folders the client exposes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub roots: Option<Value>,
    /// Sampling capability (server-initiated LLM calls).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Value>,
    /// Elicitation capability (server-initiated user input).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elicitation: Option<Value>,
    /// Experimental task-augmented requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tasks: Option<Value>,
    /// Experimental capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Value>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientInfo {
    /// Client name
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    /// Client version
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_option_cow_str",
        deserialize_with = "deserialize_option_cow_str"
    )]
    pub version: Option<Cow<'static, str>>,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Optional icons (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Icon>>,
}

/// Initialize result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// Protocol version
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub protocol_version: Cow<'static, str>,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Server information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info: Option<ServerInfo>,
    /// Instructions describing how to use the server and its features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Server capabilities
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ServerCapabilities {
    /// Tools capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    /// Prompts capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    /// Resources capability
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
    /// Logging capability (`logging/setLevel` + `notifications/message`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logging: Option<Value>,
    /// Completions capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completions: Option<Value>,
    /// Experimental task-augmented requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tasks: Option<Value>,
    /// Experimental capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Value>,
}

/// Tools capability metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCapability {
    /// List of available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Prompts capability metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsCapability {
    /// List of available prompts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resources capability metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesCapability {
    /// Subscribe to resource changes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// List of available resources
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    /// Server name
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    /// Server version
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_option_cow_str",
        deserialize_with = "deserialize_option_cow_str"
    )]
    pub version: Option<Cow<'static, str>>,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Optional description (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional icons (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Icon>>,
}

/// Icon metadata (MCP 2025-11-25).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Icon {
    /// Icon URI (https or data).
    pub src: String,
    /// Optional MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional size hints such as `48x48`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sizes: Option<Vec<String>>,
    /// Optional theme (`light` / `dark`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
}

/// Ping request (no parameters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingParams {}

/// Ping result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResult {}

/// Tool definition for MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpTool {
    /// Tool name
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    /// Tool description
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub description: Cow<'static, str>,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tool input schema (JSON Schema)
    pub input_schema: Value,
    /// Optional output schema (JSON Schema) for structured results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Optional tool annotations (readOnlyHint, destructiveHint, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<ToolAnnotations>,
    /// Optional icons (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Icon>>,
    /// Extension metadata.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl McpTool {
    /// Build a tool descriptor with only the required fields filled.
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        description: impl Into<Cow<'static, str>>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            title: None,
            input_schema,
            output_schema: None,
            annotations: None,
            icons: None,
            meta: None,
        }
    }
}

/// Tool behavior annotations (MCP tool annotations).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolAnnotations {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub read_only_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub destructive_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idempotent_hint: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_world_hint: Option<bool>,
}

/// Cursor for pagination
pub type Cursor = String;

/// Tools list parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsListParams {
    /// Optional cursor for pagination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<Cursor>,
}

/// Tools list result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsListResult {
    /// List of available tools
    pub tools: Vec<McpTool>,
    /// Optional cursor for next page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,
}

/// Tools call parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCallParams {
    /// Tool name
    pub name: String,
    /// Tool arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// Tools call result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsCallResult {
    /// Tool result content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ContentPart>>,
    /// Structured tool output (MCP 2025-06-18+). Prefer this when an
    /// `outputSchema` was advertised on the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    /// Whether the tool execution failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Extension metadata.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

impl ToolsCallResult {
    /// Build a call result that satisfies the structured-content compatibility rule.
    ///
    /// When `structured` is present, the JSON is also included as a text content
    /// part (MCP SHOULD) unless `text` is already that exact JSON string.
    pub fn from_text(text: impl Into<String>, structured: Option<Value>, is_error: bool) -> Self {
        let text = text.into();
        let mut content = Vec::with_capacity(1 + usize::from(structured.is_some()));
        content.push(ContentPart::Text { text: text.clone() });
        if let Some(ref value) = structured {
            if let Ok(json) = serde_json::to_string(value) {
                if json != text {
                    content.push(ContentPart::Text { text: json });
                }
            }
        }
        Self {
            content: Some(content),
            structured_content: structured,
            is_error: Some(is_error),
            meta: None,
        }
    }
}

/// Content part for tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "audio")]
    Audio {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "resource")]
    Resource { resource: ResourceReference },
    #[serde(rename = "resource_link")]
    ResourceLink {
        uri: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(rename = "mimeType", skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
    },
}

/// Resource reference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceReference {
    /// Resource URI
    pub uri: String,
    /// Resource text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Resource MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Prompt definition for MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPrompt {
    /// Prompt name
    pub name: String,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Prompt description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional icons (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Icon>>,
    /// Prompt arguments schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<PromptArgument>>,
    /// Extension metadata.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Prompt argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    /// Argument name
    pub name: String,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Argument description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether argument is required
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
}

/// Prompts list parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsListParams {
    /// Optional cursor for pagination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<Cursor>,
}

/// Prompts list result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptsListResult {
    /// List of available prompts
    pub prompts: Vec<McpPrompt>,
    /// Optional cursor for next page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,
}

/// Prompts get parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsGetParams {
    /// Prompt name
    pub name: String,
    /// Prompt arguments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

/// Prompts get result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptsGetResult {
    /// Prompt messages
    pub messages: Vec<PromptMessage>,
    /// Prompt description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Prompt message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMessage {
    /// Message role
    #[serde(
        serialize_with = "serialize_cow_str",
        deserialize_with = "deserialize_cow_str"
    )]
    pub role: Cow<'static, str>,
    /// Message content
    pub content: Value,
}

/// Resource definition for MCP
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpResource {
    /// Resource URI
    pub uri: String,
    /// Resource name (spec-required; optional on the wire for older peers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Resource description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Resource MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Optional icons (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Icon>>,
    /// Extension metadata.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Resources list parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesListParams {
    /// Optional cursor for pagination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<Cursor>,
}

/// Resources list result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourcesListResult {
    /// List of available resources
    pub resources: Vec<McpResource>,
    /// Optional cursor for next page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,
}

/// Resources read parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesReadParams {
    /// Resource URI
    pub uri: String,
}

/// Resources read result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcesReadResult {
    /// Resource contents
    pub contents: Vec<ResourceContents>,
}

/// Resource contents
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceContents {
    /// Resource URI
    pub uri: String,
    /// Resource MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Resource text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Resource blob (base64 encoded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

/// Progress token for progress notifications (string or number in the wire schema).
pub type ProgressToken = Value;

/// Request ID type
pub type RequestId = Value;

/// Cancelled notification parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelledNotificationParams {
    /// The ID of the request to cancel
    pub request_id: RequestId,
    /// Optional reason for cancellation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Progress notification parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressNotificationParams {
    /// Progress token from request `_meta.progressToken`
    pub progress_token: ProgressToken,
    /// Progress value so far (required by the schema)
    pub progress: f64,
    /// Total progress value when known
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
    /// Human-readable progress message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Logging level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

/// Set level request parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetLevelParams {
    /// Logging level
    pub level: LogLevel,
}

/// Resource template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceTemplate {
    /// Template URI
    pub uri_template: String,
    /// Template name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional human title (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Template description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Template MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Optional icons (2025-11-25).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icons: Option<Vec<Icon>>,
    /// Extension metadata.
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Resource templates list parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceTemplatesListParams {
    /// Optional cursor for pagination
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cursor: Option<Cursor>,
}

/// Resource templates list result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceTemplatesListResult {
    /// List of resource templates
    pub resource_templates: Vec<ResourceTemplate>,
    /// Optional cursor for the next page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<Cursor>,
}
