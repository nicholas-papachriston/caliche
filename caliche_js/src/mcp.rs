//! JavaScript bindings for MCP server functionality
#![expect(
    clippy::unused_async,
    reason = "the NAPI build callback must preserve its Promise-returning JavaScript ABI"
)]

use async_trait::async_trait;
use caliche::agent::tools::{Tool, ToolMeta, ToolResult};
use caliche::mcp::server::{McpServer as RustMcpServer, McpServerBuilder as RustMcpServerBuilder};
use caliche::mcp::transports::stdio::run_stdio_transport;
use caliche::templates::TemplateRegistry;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use serde_json::Value;
use std::sync::Arc;
use std::sync::RwLock;

/// Helper function to handle `RwLock` read operations with proper error handling
fn rwlock_read<T>(lock: &RwLock<T>) -> napi::Result<std::sync::RwLockReadGuard<'_, T>> {
    lock.read()
        .map_err(|e| napi::Error::from_reason(format!("Failed to acquire read lock: {e}")))
}

/// Helper function to handle `RwLock` write operations with proper error handling
fn rwlock_write<T>(lock: &RwLock<T>) -> napi::Result<std::sync::RwLockWriteGuard<'_, T>> {
    lock.write()
        .map_err(|e| napi::Error::from_reason(format!("Failed to acquire write lock: {e}")))
}
use tokio::sync::Mutex;

/// JavaScript tool wrapper for MCP
// Note: Cannot derive Debug because ThreadsafeFunction doesn't implement Debug
struct JsMcpTool {
    name: Arc<str>,
    description: Arc<str>,
    parameters: Value,
    execute_fn: ThreadsafeFunction<Value>,
}

impl std::fmt::Debug for JsMcpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsMcpTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("execute_fn", &"<ThreadsafeFunction>")
            .finish()
    }
}

impl ToolMeta for JsMcpTool {
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
impl Tool for JsMcpTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        // Use sync bridge to call JavaScript function
        // This avoids thread safety issues by using blocking calls with oneshot channels
        use crate::call_js_sync_bridge;
        let json_string = match call_js_sync_bridge(&self.execute_fn, arguments).await {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::error(format!("JavaScript callback error: {e}"));
            }
        };

        // Parse JSON string
        let result_value: Value = match serde_json::from_str(&json_string) {
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
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// MCP Server Builder
#[napi]
pub struct McpServerBuilder {
    inner: RwLock<RustMcpServerBuilder>,
}

fn build_mcp_server_internal(
    builder: caliche::mcp::server::McpServerBuilder,
) -> napi::Result<McpServer> {
    let server = builder
        .build()
        .map_err(|e| napi::Error::from_reason(format!("Failed to build MCP server: {e}")))?;
    Ok(McpServer {
        inner: Arc::new(Mutex::new(server)),
    })
}

fn update_mcp_builder(
    lock: &RwLock<RustMcpServerBuilder>,
    update: impl FnOnce(RustMcpServerBuilder) -> RustMcpServerBuilder,
) -> napi::Result<()> {
    let mut inner = rwlock_write(lock)?;
    let builder = std::mem::take(&mut *inner);
    *inner = update(builder);
    drop(inner);
    Ok(())
}

#[napi]
impl McpServerBuilder {
    /// Create a new MCP server builder
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(caliche::mcp::server::McpServerBuilder::new()),
        }
    }

    /// Set server name
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn set_name(&self, name: String) -> napi::Result<()> {
        update_mcp_builder(&self.inner, |builder| builder.name(name))
    }

    /// Set server version
    #[napi(catch_unwind)]
    pub fn set_version(&self, version: String) -> napi::Result<()> {
        update_mcp_builder(&self.inner, |builder| builder.version(version))
    }

    /// Enable or disable tools capability
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn enable_tools(&self, enabled: bool) -> napi::Result<()> {
        update_mcp_builder(&self.inner, |builder| builder.enable_tools(enabled))
    }

    /// Enable or disable prompts capability
    #[napi(catch_unwind)]
    pub fn enable_prompts(&self, enabled: bool) -> napi::Result<()> {
        update_mcp_builder(&self.inner, |builder| builder.enable_prompts(enabled))
    }

    /// Enable or disable resources capability
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn enable_resources(&self, enabled: bool) -> napi::Result<()> {
        update_mcp_builder(&self.inner, |builder| builder.enable_resources(enabled))
    }

    /// Add a tool to the builder
    #[napi(catch_unwind)]
    pub fn add_tool(
        &self,
        tool: McpTool,
        execute_fn: ThreadsafeFunction<Value>,
    ) -> napi::Result<()> {
        let js_tool = Arc::new(JsMcpTool {
            name: Arc::from(tool.name),
            description: Arc::from(tool.description),
            parameters: tool.parameters,
            execute_fn,
        }) as Arc<dyn Tool>;

        update_mcp_builder(&self.inner, |builder| builder.add_tool(js_tool))
    }

    /// Build the MCP server
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn build(&self) -> napi::Result<McpServer> {
        // Clone builder state before async to avoid holding RefCell across await
        // Use a block to ensure all borrows are dropped before await
        let builder = {
            let inner = rwlock_read(&self.inner)?;
            let mut new_builder = caliche::mcp::server::McpServerBuilder::new();

            // Clone to owned strings - builder.name/version require Cow<'static, str>
            let name = inner.get_name().to_string();
            if !name.is_empty() {
                new_builder = new_builder.name(name);
            }
            if let Some(version) = inner.get_version() {
                new_builder = new_builder.version(version.to_string());
            }
            new_builder = new_builder
                .enable_tools(inner.tools_enabled())
                .enable_prompts(inner.prompts_enabled())
                .enable_resources(inner.resources_enabled());

            if let Some(registry) = inner.get_tool_registry() {
                new_builder = new_builder.tool_registry(Arc::clone(registry));
            }

            if let Some(_registry) = inner.get_template_registry() {
                new_builder = new_builder.template_registry(TemplateRegistry::new());
            }

            #[cfg(feature = "tools-rag")]
            {
                if let Some(rag_tool) = inner.get_rag_tool() {
                    new_builder = new_builder.rag_tool(Some(Arc::clone(rag_tool)));
                }
            }

            if let Some(collections) = inner.get_document_collections() {
                new_builder = new_builder.document_collections(Some(Arc::clone(collections)));
            }

            drop(inner);
            new_builder
        };

        build_mcp_server_internal(builder)
    }
}

/// MCP Server
#[napi]
pub struct McpServer {
    inner: Arc<Mutex<RustMcpServer>>,
}

#[napi]
impl McpServer {
    /// Run the server with stdio transport
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn run_stdio(&self) -> napi::Result<()> {
        let server_clone = {
            let inner = self.inner.lock().await;
            inner.clone()
        };

        let handler = server_clone.create_handler();
        run_stdio_transport(handler)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to run stdio transport: {e}")))
    }
}
