use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::mcp::client::McpClient;
use crate::mcp::protocol::ContentPart;

#[derive(Debug)]
pub struct RemoteMcpTool {
    name: String,
    description: String,
    parameters: Value,
    remote_tool_name: String,
    client: Arc<McpClient>,
}

impl RemoteMcpTool {
    pub fn new(
        name: String,
        description: String,
        parameters: Value,
        remote_tool_name: String,
        client: Arc<McpClient>,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            remote_tool_name,
            client,
        }
    }
}

impl ToolMeta for RemoteMcpTool {
    fn description(&self) -> &str {
        &self.description
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> Value {
        if self.parameters.is_null() {
            json!({
                "type": "object",
                "properties": {}
            })
        } else {
            self.parameters.clone()
        }
    }
}

#[async_trait]
impl Tool for RemoteMcpTool {
    async fn execute(&self, arguments: Value) -> ToolResult {
        match self
            .client
            .call_tool(&self.remote_tool_name, arguments)
            .await
        {
            Ok(result) => {
                if result.is_error.unwrap_or(false) {
                    return ToolResult::error("Remote MCP tool returned an error");
                }
                let mut content_parts = Vec::new();
                if let Some(content) = result.content {
                    for part in content {
                        match part {
                            ContentPart::Text { text } => content_parts.push(text),
                            ContentPart::Resource { resource } => {
                                if let Some(text) = resource.text {
                                    content_parts.push(text);
                                } else {
                                    content_parts.push(format!("resource://{}", resource.uri));
                                }
                            }
                            ContentPart::Image { mime_type, .. } => {
                                content_parts.push(format!("[image:{mime_type}]"));
                            }
                            ContentPart::Audio { mime_type, .. } => {
                                content_parts.push(format!("[audio:{mime_type}]"));
                            }
                            ContentPart::ResourceLink { uri, name, .. } => {
                                content_parts.push(match name {
                                    Some(name) => format!("{name} ({uri})"),
                                    None => uri,
                                });
                            }
                        }
                    }
                }
                let mut tool_result = ToolResult::success(content_parts.join("\n"));
                tool_result.typed_payload = result.structured_content;
                tool_result
            }
            Err(e) => ToolResult::error(format!("Remote MCP call failed: {}", e)),
        }
    }
}
