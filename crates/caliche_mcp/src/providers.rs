//! Host-agnostic provider traits for MCP capabilities.
//!
//! Engine crates implement these traits against their own registries so this
//! package stays free of Caliche tool / template / RAG types.

use crate::protocol::{
    ContentPart, McpPrompt, McpResource, McpTool, PromptMessage, ResourceContents, ResourceTemplate,
};
use async_trait::async_trait;
use serde_json::Value;

/// Descriptor used when listing tools through a [`ToolProvider`].
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    pub tool: McpTool,
}

/// Result of invoking a tool through a [`ToolProvider`].
#[derive(Debug, Clone)]
pub struct ToolInvokeResult {
    pub content: Vec<ContentPart>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
}

impl ToolInvokeResult {
    pub fn text_ok(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::Text { text: text.into() }],
            structured_content: None,
            is_error: false,
        }
    }

    pub fn text_err(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::Text { text: text.into() }],
            structured_content: None,
            is_error: true,
        }
    }

    pub fn with_structured(mut self, value: Value) -> Self {
        self.structured_content = Some(value);
        self
    }
}

/// Lists and invokes tools without depending on a host registry type.
#[async_trait]
pub trait ToolProvider: Send + Sync {
    async fn list_tools(&self) -> anyhow::Result<Vec<ToolDescriptor>>;
    async fn call_tool(&self, name: &str, arguments: Value) -> anyhow::Result<ToolInvokeResult>;
}

/// Prompt list / get provider.
#[async_trait]
pub trait PromptProvider: Send + Sync {
    async fn list_prompts(&self) -> anyhow::Result<Vec<McpPrompt>>;
    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> anyhow::Result<(Vec<PromptMessage>, Option<String>)>;
}

/// Resource list / read / template provider.
#[async_trait]
pub trait ResourceProvider: Send + Sync {
    async fn list_resources(&self) -> anyhow::Result<Vec<McpResource>>;
    async fn list_templates(&self) -> anyhow::Result<Vec<ResourceTemplate>>;
    async fn read_resource(&self, uri: &str) -> anyhow::Result<Vec<ResourceContents>>;
}

/// Empty resource provider for servers that do not expose resources.
pub struct NoResources;

#[async_trait]
impl ResourceProvider for NoResources {
    async fn list_resources(&self) -> anyhow::Result<Vec<McpResource>> {
        Ok(Vec::new())
    }

    async fn list_templates(&self) -> anyhow::Result<Vec<ResourceTemplate>> {
        Ok(Vec::new())
    }

    async fn read_resource(&self, uri: &str) -> anyhow::Result<Vec<ResourceContents>> {
        anyhow::bail!("resource not found: {uri}")
    }
}

/// Empty prompt provider for tools-only servers.
pub struct NoPrompts;

#[async_trait]
impl PromptProvider for NoPrompts {
    async fn list_prompts(&self) -> anyhow::Result<Vec<McpPrompt>> {
        Ok(Vec::new())
    }

    async fn get_prompt(
        &self,
        name: &str,
        _arguments: Option<Value>,
    ) -> anyhow::Result<(Vec<PromptMessage>, Option<String>)> {
        anyhow::bail!("prompt not found: {name}")
    }
}
