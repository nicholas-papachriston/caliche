//! The `plugin` tool: create, update, delete, list, get, export, import plugins.

use crate::agent::tools::registry::ToolRegistry;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::plugins::adapters::tool_adapter::SchemaToolAdapter;
use crate::plugins::runtime::{AgentCapabilities, StepRuntime};
use crate::plugins::schema::{PluginDefinition, PluginSpec};
use crate::plugins::validation::ValidationContext;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use uuid::Uuid;

const NAME: &str = "plugin";
const DESCRIPTION: &str = "Manage schema-based plugins: create tools, skills, hooks, embedding providers, evaluators, document loaders, stream backends, or output parsers at runtime from JSON definitions, or list/get/export/import/delete existing ones. \
Always set 'op' to one of: create, update, delete, list, get, export, import. \
For create, update, import: pass 'definition' (full plugin JSON). For delete, get, export: pass 'name' (plugin name). For list: no other fields needed. \
Plugin types: tool (step DSL, callable immediately), skill (applied at build, in discovery catalog), hook (lifecycle guardrails; matcher 'all' or tool name), embedding (HTTP API for embeddings; first one used as agent embedding provider), evaluator (step DSL for scoring; use with evaluate_dataset), document_loader (extension/mime + parsing steps; use SchemaDocumentLoader::from_content for proprietary formats), stream_backend (HTTP endpoints for append/read_after/last_id/delete_stream; first one used as durable_stream_backend when connection is inline), output_parser (extraction steps + format_instructions; parse() returns Value). \
Document loader: extension, mime_types?, parsing_steps (args: content), documents_path?, content_path?, metadata_path?. Stream backend: connection (inline url), endpoints (append, read_after, last_id, delete_stream), id_path?, events_path?, event_id_path?. Output parser: extraction_steps (args: text), format_instructions?, output_path?. \
Definition must include: name, version, description, plugin_type (tool|provider|backend|skill|source|hook|embedding|evaluator|document_loader|stream_backend|output_parser), spec (type-specific). \
Plugins are validated and persisted; tool plugins become callable immediately; hook/embedding/skill/stream_backend apply on next build.";

#[derive(Debug)]
pub struct PluginTool {
    pub registry: Arc<crate::plugins::registry::PluginRegistry>,
    pub runtime: Arc<StepRuntime>,
    pub tool_registry: Arc<ToolRegistry>,
    pub capabilities: Arc<AgentCapabilities>,
}

impl PluginTool {
    pub fn new(
        registry: Arc<crate::plugins::registry::PluginRegistry>,
        runtime: Arc<StepRuntime>,
        tool_registry: Arc<ToolRegistry>,
        capabilities: Arc<AgentCapabilities>,
    ) -> Self {
        Self {
            registry,
            runtime,
            tool_registry,
            capabilities,
        }
    }

    fn validation_ctx(&self) -> ValidationContext {
        ValidationContext {
            sandbox_policy: self.capabilities.sandbox_policy,
            domain_allowlist: self.capabilities.domain_allowlist.clone(),
            known_connection_refs: None,
        }
    }

    fn stamp_definition_ids(def_value: &mut serde_json::Value) {
        let now = chrono::Utc::now().to_rfc3339();
        if let Some(obj) = def_value.as_object_mut() {
            obj.insert("id".to_string(), json!(Uuid::now_v7().to_string()));
            obj.insert("created_at".to_string(), json!(now));
            obj.insert("updated_at".to_string(), json!(now));
        }
    }

    fn parse_definition(def_value: serde_json::Value) -> Result<PluginDefinition, String> {
        serde_json::from_value(def_value).map_err(|e| format!("invalid definition: {}", e))
    }

    fn register_tool_adapter(&self, def: PluginDefinition) {
        if let PluginSpec::Tool(_) = &def.spec
            && let Some(adapter) = SchemaToolAdapter::new(
                Arc::new(def),
                Arc::clone(&self.runtime),
                Arc::clone(&self.capabilities),
            )
        {
            self.tool_registry.register(Arc::new(adapter));
        }
    }

    async fn execute_create(&self, arguments: &serde_json::Value) -> ToolResult {
        let mut def_value = match arguments.get("definition").cloned() {
            Some(d) => d,
            None => return ToolResult::error("op=create requires definition".to_string()),
        };
        Self::stamp_definition_ids(&mut def_value);
        let def = match Self::parse_definition(def_value) {
            Ok(d) => d,
            Err(e) => return ToolResult::error(e),
        };
        let name = def.name.clone();
        match self
            .registry
            .register(def.clone(), &self.validation_ctx())
            .await
        {
            Ok(()) => {
                self.register_tool_adapter(def);
                ToolResult::success(format!("Plugin '{}' created.", name))
            }
            Err(e) => ToolResult::error(format!("create failed: {}", e)),
        }
    }

    async fn execute_update(&self, arguments: &serde_json::Value) -> ToolResult {
        let def_value = match arguments.get("definition").cloned() {
            Some(d) => d,
            None => return ToolResult::error("op=update requires definition".to_string()),
        };
        let def = match Self::parse_definition(def_value) {
            Ok(d) => d,
            Err(e) => return ToolResult::error(e),
        };
        let name = def.name.clone();
        match self
            .registry
            .update(def.clone(), &self.validation_ctx())
            .await
        {
            Ok(()) => {
                if let PluginSpec::Tool(_) = &def.spec {
                    self.tool_registry.unregister(&name);
                    self.register_tool_adapter(def);
                }
                ToolResult::success(format!("Plugin '{}' updated.", name))
            }
            Err(e) => ToolResult::error(format!("update failed: {}", e)),
        }
    }

    async fn execute_delete(&self, arguments: &serde_json::Value) -> ToolResult {
        let name = match arguments.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => return ToolResult::error("op=delete requires name".to_string()),
        };
        match self.registry.delete(&name).await {
            Ok(()) => {
                self.tool_registry.unregister(&name);
                ToolResult::success(format!("Plugin '{}' deleted.", name))
            }
            Err(e) => ToolResult::error(format!("delete failed: {}", e)),
        }
    }

    fn execute_list(&self) -> ToolResult {
        let plugins = self.registry.list();
        let summary: Vec<String> = plugins
            .iter()
            .map(|p| format!("{} ({})", p.name, p.version))
            .collect();
        ToolResult::success(summary.join("\n"))
    }

    fn execute_get(&self, arguments: &serde_json::Value) -> ToolResult {
        let name = match arguments.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return ToolResult::error("op=get requires name".to_string()),
        };
        match self.registry.get(name) {
            Some(def) => {
                let json =
                    serde_json::to_string_pretty(def.as_ref()).unwrap_or_else(|_| "{}".to_string());
                ToolResult::success(json)
            }
            None => ToolResult::error(format!("plugin not found: {}", name)),
        }
    }

    async fn execute_export(&self, arguments: &serde_json::Value) -> ToolResult {
        let name = match arguments.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return ToolResult::error("op=export requires name".to_string()),
        };
        match self.registry.export(name).await {
            Ok(json) => ToolResult::success(json),
            Err(e) => ToolResult::error(format!("export failed: {}", e)),
        }
    }

    async fn execute_import(&self, arguments: &serde_json::Value) -> ToolResult {
        let mut def_value = match arguments.get("definition").cloned() {
            Some(d) => d,
            None => return ToolResult::error("op=import requires definition".to_string()),
        };
        Self::stamp_definition_ids(&mut def_value);
        let def = match Self::parse_definition(def_value) {
            Ok(d) => d,
            Err(e) => return ToolResult::error(e),
        };
        match self
            .registry
            .register(def.clone(), &self.validation_ctx())
            .await
        {
            Ok(()) => {
                self.register_tool_adapter(def);
                ToolResult::success("Plugin imported.".to_string())
            }
            Err(e) => ToolResult::error(format!("import failed: {}", e)),
        }
    }
}

impl ToolMeta for PluginTool {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["create", "update", "delete", "list", "get", "export", "import"],
                    "description": "Operation: create (register new plugin, requires definition), update (replace existing, requires definition), delete (remove by name, requires name), list (return all plugin names/versions), get (full definition by name, requires name), export (JSON string by name, requires name), import (load from definition, requires definition)."
                },
                "definition": {
                    "type": "object",
                    "description": "Required for create, update, import. Full plugin JSON: name, version, description, plugin_type, spec. Types: tool, skill, hook, embedding, evaluator, document_loader (extension, mime_types?, parsing_steps; args: content), stream_backend (connection inline url, endpoints append/read_after/last_id/delete_stream), output_parser (extraction_steps; args: text, format_instructions?, output_path?). id, created_at, updated_at filled if omitted."
                },
                "name": {
                    "type": "string",
                    "description": "Required for delete, get, export. The unique plugin name (machine-readable identifier)."
                }
            },
            "required": ["op"]
        })
    }
}

#[async_trait]
impl Tool for PluginTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let op = arguments
            .get("op")
            .and_then(|v| v.as_str())
            .unwrap_or("list");
        match op {
            "create" => self.execute_create(&arguments).await,
            "update" => self.execute_update(&arguments).await,
            "delete" => self.execute_delete(&arguments).await,
            "list" => self.execute_list(),
            "get" => self.execute_get(&arguments),
            "export" => self.execute_export(&arguments).await,
            "import" => self.execute_import(&arguments).await,
            _ => ToolResult::error(format!("unknown op: {}", op)),
        }
    }
}
