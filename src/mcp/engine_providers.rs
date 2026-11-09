//! Engine registries behind the shared MCP provider router.

use crate::agent::hooks::HookRegistry;
use crate::agent::tools::ToolRegistry;
use crate::mcp::capabilities::prompts::{collect_prompts, handle_prompts_get};
use crate::mcp::capabilities::resources::{collect_resource_templates, collect_resources};
use crate::mcp::capabilities::tools::{collect_tools, execute_mcp_tool};
use crate::mcp::protocol::{
    ContentPart, McpResource, PromptsGetParams, ResourceContents, ResourceTemplate, ToolsCallResult,
};
use crate::templates::TemplateRegistry;
use async_trait::async_trait;
use caliche_mcp::providers::{
    PromptProvider, ResourceProvider, ToolDescriptor, ToolInvokeResult, ToolProvider,
};
use dashmap::DashMap;
use serde_json::Value;
use std::sync::Arc;

pub(crate) struct EngineTools {
    pub(crate) registry: Arc<ToolRegistry>,
    pub(crate) hooks: HookRegistry,
}

pub(crate) struct EnginePrompts {
    pub(crate) registry: Arc<TemplateRegistry>,
}

pub(crate) struct EngineResources {
    pub(crate) collections: Option<Arc<DashMap<String, crate::mcp::CollectionMetadata>>>,
    #[cfg(feature = "tools-rag")]
    pub(crate) rag: Option<Arc<crate::agent::tools::RagTool>>,
}

#[async_trait]
impl ToolProvider for EngineTools {
    async fn list_tools(&self) -> anyhow::Result<Vec<ToolDescriptor>> {
        Ok(collect_tools(&self.registry)
            .into_iter()
            .map(|tool| ToolDescriptor { tool })
            .collect())
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> anyhow::Result<ToolInvokeResult> {
        let result = execute_mcp_tool(&self.registry, &self.hooks, name, arguments).await;
        Ok(invoke_from_call(result))
    }
}

#[async_trait]
impl PromptProvider for EnginePrompts {
    async fn list_prompts(&self) -> anyhow::Result<Vec<crate::mcp::protocol::McpPrompt>> {
        Ok(collect_prompts(&self.registry))
    }

    async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<Value>,
    ) -> anyhow::Result<(Vec<crate::mcp::protocol::PromptMessage>, Option<String>)> {
        let result = handle_prompts_get(
            &self.registry,
            PromptsGetParams {
                name: name.to_string(),
                arguments,
            },
        )
        .await?;
        Ok((result.messages, result.description))
    }
}

#[async_trait]
impl ResourceProvider for EngineResources {
    async fn list_resources(&self) -> anyhow::Result<Vec<McpResource>> {
        Ok(collect_resources(self.collections.as_ref()))
    }

    async fn list_templates(&self) -> anyhow::Result<Vec<ResourceTemplate>> {
        Ok(collect_resource_templates())
    }

    #[cfg(feature = "tools-rag")]
    async fn read_resource(&self, uri: &str) -> anyhow::Result<Vec<ResourceContents>> {
        read_engine_resource(self, uri).await
    }

    #[cfg(not(feature = "tools-rag"))]
    async fn read_resource(&self, uri: &str) -> anyhow::Result<Vec<ResourceContents>> {
        anyhow::bail!("resources/read is unavailable for {uri}")
    }
}

#[cfg(feature = "tools-rag")]
async fn read_engine_resource(
    resources: &EngineResources,
    uri: &str,
) -> anyhow::Result<Vec<ResourceContents>> {
    use crate::mcp::protocol::ResourcesReadParams;

    let result = crate::mcp::capabilities::resources::handle_resources_read(
        resources.rag.as_ref(),
        ResourcesReadParams {
            uri: uri.to_string(),
        },
    )
    .await?;
    Ok(result.contents)
}

fn invoke_from_call(result: ToolsCallResult) -> ToolInvokeResult {
    let text = result
        .content
        .unwrap_or_default()
        .into_iter()
        .find_map(|part| match part {
            ContentPart::Text { text } => Some(text),
            ContentPart::Image { .. }
            | ContentPart::Audio { .. }
            | ContentPart::Resource { .. }
            | ContentPart::ResourceLink { .. } => None,
        })
        .unwrap_or_default();
    let mut invoke = if result.is_error == Some(true) {
        ToolInvokeResult::text_err(text)
    } else {
        ToolInvokeResult::text_ok(text)
    };
    if let Some(value) = result.structured_content {
        invoke = invoke.with_structured(value);
    }
    invoke
}
