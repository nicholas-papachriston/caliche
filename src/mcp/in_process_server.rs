//! In-process MCP server for direct tool registration.
//!
//! This module provides an in-process MCP server implementation that wraps
//! Rust tools directly, eliminating the need for subprocess management and
//! IPC overhead. Tools are executed in the same process as the agent.

use crate::agent::hooks::HookRegistry;
use crate::agent::tools::{Tool, ToolRegistry};
use crate::mcp::protocol::{McpTool, ToolsCallParams, ToolsCallResult};
use anyhow::Result;
use beef::lean::Cow;
use serde_json::Value;
use std::sync::Arc;

/// In-process MCP server for tool registration
///
/// Unlike external MCP servers that run in separate processes, this server
/// executes tools directly in-process, providing better performance and
/// simpler deployment.
#[derive(Clone)]
pub struct InProcessMcpServer {
    /// Server name
    name: Cow<'static, str>,
    /// Server version
    version: Cow<'static, str>,
    /// Tool registry containing registered tools
    tool_registry: Arc<ToolRegistry>,
    /// Pre-tool hooks. Empty means only the compiled policy floor applies.
    hooks: HookRegistry,
}

impl InProcessMcpServer {
    /// Create a new in-process MCP server builder
    pub fn builder(name: impl Into<Cow<'static, str>>) -> InProcessMcpServerBuilder {
        InProcessMcpServerBuilder::new(name)
    }

    /// Get the server name
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    /// Get the server version
    pub fn version(&self) -> &str {
        self.version.as_ref()
    }

    /// Get the tool registry
    pub fn tool_registry(&self) -> &Arc<ToolRegistry> {
        &self.tool_registry
    }

    /// List all available tools as MCP tools
    pub fn list_tools(&self) -> Vec<McpTool> {
        self.tool_registry
            .list()
            .into_iter()
            .map(|(name, description, parameters)| McpTool::new(name, description, parameters))
            .collect()
    }

    /// Call a tool by name with arguments
    ///
    /// This executes the tool directly in-process without subprocess overhead.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<ToolsCallResult> {
        if let Some(reason) =
            crate::mcp::policy_gate::pre_tool_block(&self.hooks, name, &arguments).await
        {
            return Ok(ToolsCallResult::from_text(reason, None, true));
        }

        let tool = self
            .tool_registry
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", name))?;

        let result = tool.execute(arguments).await;

        let text = if result.success {
            result.content
        } else {
            format!(
                "Error: {}",
                result.error.as_deref().unwrap_or("Unknown error")
            )
        };
        Ok(ToolsCallResult::from_text(
            text,
            result.typed_payload,
            !result.success,
        ))
    }

    /// Handle MCP tools/call request
    pub async fn handle_tools_call(&self, params: ToolsCallParams) -> Result<ToolsCallResult> {
        let arguments = params.arguments.unwrap_or_else(|| serde_json::json!({}));
        self.call_tool(&params.name, arguments).await
    }
}

/// Builder for creating in-process MCP servers
pub struct InProcessMcpServerBuilder {
    name: Cow<'static, str>,
    version: Cow<'static, str>,
    tools: Vec<Arc<dyn Tool>>,
    hooks: HookRegistry,
}

impl InProcessMcpServerBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            version: Cow::borrowed(env!("CARGO_PKG_VERSION")),
            tools: Vec::new(),
            hooks: HookRegistry::new(),
        }
    }

    /// Attach pre-tool hooks. A deny or an approval request blocks the tool.
    pub fn hooks(mut self, hooks: HookRegistry) -> Self {
        self.hooks = hooks;
        self
    }

    /// Set the server version
    pub fn version(mut self, version: impl Into<Cow<'static, str>>) -> Self {
        self.version = version.into();
        self
    }

    /// Add a tool to the server
    pub fn add_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Add multiple tools to the server
    pub fn add_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Build the in-process MCP server
    pub fn build(self) -> Result<InProcessMcpServer> {
        let registry = ToolRegistry::new();

        for tool in self.tools {
            registry.register(tool);
        }

        Ok(InProcessMcpServer {
            name: self.name,
            version: self.version,
            tool_registry: Arc::new(registry),
            hooks: self.hooks,
        })
    }
}

impl Default for InProcessMcpServerBuilder {
    fn default() -> Self {
        Self::new("in-process-server")
    }
}
