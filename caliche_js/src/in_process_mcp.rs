//! JavaScript bindings for in-process MCP servers

use async_trait::async_trait;
use caliche::agent::tools::{Tool, ToolMeta, ToolResult};
use caliche::mcp::in_process_server::{
    InProcessMcpServer as RustInProcessMcpServer,
    InProcessMcpServerBuilder as RustInProcessMcpServerBuilder,
};
use caliche::mcp::protocol::ContentPart;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use serde_json::Value;
use std::sync::Arc;

use crate::call_js_sync_bridge;

/// JavaScript tool wrapper for in-process MCP
struct JsInProcessMcpTool {
    name: Arc<str>,
    description: Arc<str>,
    parameters: Value,
    execute_fn: ThreadsafeFunction<Value>,
}

impl std::fmt::Debug for JsInProcessMcpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsInProcessMcpTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("execute_fn", &"<ThreadsafeFunction>")
            .finish()
    }
}

impl ToolMeta for JsInProcessMcpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }
}

#[async_trait]
impl Tool for JsInProcessMcpTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        // Call the JS function using sync bridge
        let result_str = match call_js_sync_bridge(&self.execute_fn, arguments).await {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::error(format!("JavaScript callback error: {e}"));
            }
        };

        // Parse JSON string
        let result_value: Value = match serde_json::from_str(&result_str) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult::error(format!("Failed to parse JSON string: {e}"));
            }
        };

        // Try to parse as ToolResult first
        if let Ok(tool_result) = serde_json::from_value::<ToolResult>(result_value.clone()) {
            return tool_result;
        }

        // Try to parse as a dict with success/content/error
        if let Some(dict) = result_value.as_object() {
            let success = dict
                .get("success")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true);
            let content = dict
                .get("content")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string)
                .unwrap_or_default();
            let error = dict
                .get("error")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string);

            return ToolResult {
                success,
                content,
                typed_payload: dict.get("payload").cloned(),
                error,
                usage: None,
                warnings: None,
            };
        }

        // Fallback: treat as string content
        let content = result_value.as_str().map_or_else(
            || result_value.to_string(),
            std::string::ToString::to_string,
        );
        ToolResult::success(content)
    }
}

#[napi(object)]
pub struct InProcessMcpTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// In-process MCP Server Builder
#[napi]
pub struct InProcessMcpServerBuilder {
    name: String,
    version: String,
    tools: Vec<Arc<dyn Tool>>,
}

#[napi]
impl InProcessMcpServerBuilder {
    /// Create a new in-process MCP server builder
    #[napi(constructor, catch_unwind)]
    pub fn new(name: String) -> Self {
        Self {
            name,
            version: env!("CARGO_PKG_VERSION").to_string(),
            tools: Vec::new(),
        }
    }

    /// Set the server version
    #[napi(catch_unwind)]
    pub fn version(&mut self, version: String) {
        self.version = version;
    }

    /// Add a tool to the builder
    #[napi(catch_unwind)]
    pub fn add_tool(&mut self, tool: InProcessMcpTool, execute_fn: ThreadsafeFunction<Value>) {
        let js_tool = Arc::new(JsInProcessMcpTool {
            name: Arc::from(tool.name.as_str()),
            description: Arc::from(tool.description.as_str()),
            parameters: tool.parameters,
            execute_fn,
        }) as Arc<dyn Tool>;

        self.tools.push(js_tool);
    }

    /// Build the in-process MCP server
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn build(&self) -> napi::Result<InProcessMcpServer> {
        let mut builder =
            RustInProcessMcpServerBuilder::new(self.name.clone()).version(self.version.clone());

        // Add all tools at once
        if !self.tools.is_empty() {
            builder = builder.add_tools(self.tools.clone());
        }

        let server = builder.build().map_err(|e| {
            napi::Error::from_reason(format!("Failed to build in-process MCP server: {e}"))
        })?;

        Ok(InProcessMcpServer {
            inner: Arc::new(server),
        })
    }
}

/// In-process MCP Server
#[napi]
pub struct InProcessMcpServer {
    pub(crate) inner: Arc<RustInProcessMcpServer>,
}

#[napi]
impl InProcessMcpServer {
    #[napi(getter, catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[napi(getter, catch_unwind)]
    pub fn version(&self) -> String {
        self.inner.version().to_string()
    }

    /// List all available tools as MCP tools
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn list_tools(&self) -> Vec<Value> {
        let tools = self.inner.list_tools();
        tools
            .into_iter()
            .map(|mcp_tool| {
                serde_json::json!({
                    "name": mcp_tool.name,
                    "description": mcp_tool.description,
                    "input_schema": mcp_tool.input_schema
                })
            })
            .collect()
    }

    /// Call a tool by name with arguments
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn call_tool(&self, name: String, arguments: Value) -> napi::Result<Value> {
        let result = self
            .inner
            .call_tool(&name, arguments)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Tool execution failed: {e}")))?;

        // Convert result to JSON Value
        let mut result_dict = serde_json::Map::new();

        if let Some(ref content) = result.content {
            let content_list: Vec<Value> = content
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => {
                        serde_json::json!({
                            "type": "text",
                            "text": text
                        })
                    }
                    ContentPart::Resource { resource } => {
                        let mut resource_dict = serde_json::Map::new();
                        resource_dict.insert(
                            "uri".to_string(),
                            serde_json::Value::String(resource.uri.clone()),
                        );
                        if let Some(ref text) = resource.text {
                            resource_dict.insert(
                                "text".to_string(),
                                serde_json::Value::String(text.clone()),
                            );
                        }
                        if let Some(ref mime_type) = resource.mime_type {
                            resource_dict.insert(
                                "mimeType".to_string(),
                                serde_json::Value::String(mime_type.clone()),
                            );
                        }
                        serde_json::json!({
                            "type": "resource",
                            "resource": resource_dict
                        })
                    }
                    ContentPart::Image { data, mime_type } => {
                        serde_json::json!({
                            "type": "image",
                            "data": data,
                            "mimeType": mime_type
                        })
                    }
                    ContentPart::Audio { data, mime_type } => {
                        serde_json::json!({
                            "type": "audio",
                            "data": data,
                            "mimeType": mime_type
                        })
                    }
                    ContentPart::ResourceLink {
                        uri,
                        name,
                        description,
                        mime_type,
                    } => {
                        let mut map = serde_json::Map::new();
                        map.insert(
                            "type".to_string(),
                            serde_json::Value::String("resource_link".to_string()),
                        );
                        map.insert("uri".to_string(), serde_json::Value::String(uri.clone()));
                        if let Some(name) = name {
                            map.insert("name".to_string(), serde_json::Value::String(name.clone()));
                        }
                        if let Some(description) = description {
                            map.insert(
                                "description".to_string(),
                                serde_json::Value::String(description.clone()),
                            );
                        }
                        if let Some(mime_type) = mime_type {
                            map.insert(
                                "mimeType".to_string(),
                                serde_json::Value::String(mime_type.clone()),
                            );
                        }
                        serde_json::Value::Object(map)
                    }
                })
                .collect();
            result_dict.insert(
                "content".to_string(),
                serde_json::Value::Array(content_list),
            );
        }

        if let Some(is_error) = result.is_error {
            result_dict.insert("is_error".to_string(), serde_json::Value::Bool(is_error));
        }

        Ok(serde_json::Value::Object(result_dict))
    }
}

impl InProcessMcpServer {
    /// Get the inner server (for internal use)
    pub(crate) fn get_inner(&self) -> Arc<RustInProcessMcpServer> {
        Arc::clone(&self.inner)
    }
}
