//! Schema-defined tool adapter: implements Tool + ToolMeta by delegating to StepRuntime.

use crate::agent::tools::Tool;
use crate::agent::tools::tool::{ToolCapability, ToolMeta, ToolResult};
use crate::plugins::runtime::{AgentCapabilities, StepRuntime};
use crate::plugins::schema::{PluginDefinition, PluginSpec};
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug)]
pub struct SchemaToolAdapter {
    definition: Arc<PluginDefinition>,
    spec: crate::plugins::schema::ToolPluginSpec,
    runtime: Arc<StepRuntime>,
    capabilities: Arc<AgentCapabilities>,
}

impl SchemaToolAdapter {
    pub fn new(
        definition: Arc<PluginDefinition>,
        runtime: Arc<StepRuntime>,
        capabilities: Arc<AgentCapabilities>,
    ) -> Option<Self> {
        let spec = match &definition.spec {
            PluginSpec::Tool(s) => s.clone(),
            _ => return None,
        };
        Some(Self {
            definition,
            spec,
            runtime,
            capabilities,
        })
    }
}

impl ToolMeta for SchemaToolAdapter {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn description(&self) -> &str {
        &self.definition.description
    }

    fn parameters(&self) -> serde_json::Value {
        if self.spec.parameters.is_null() {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        } else {
            self.spec.parameters.clone()
        }
    }

    fn is_readonly(&self) -> bool {
        self.spec.is_readonly
    }

    fn capabilities(&self) -> Vec<ToolCapability> {
        self.spec.capabilities.clone()
    }
}

#[async_trait]
impl Tool for SchemaToolAdapter {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        match self
            .runtime
            .execute(&self.spec, arguments, &self.capabilities)
            .await
        {
            Ok(value) => {
                let content = serde_json::to_string(&value).unwrap_or_else(|_| value.to_string());
                ToolResult::success(content)
            }
            Err(e) => ToolResult::error(format!("Plugin execution failed: {}", e)),
        }
    }
}
