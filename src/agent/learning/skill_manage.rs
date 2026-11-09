//! `skill_manage` tool: lets the agent create, patch, list, and delete skills
//! at runtime via the plugin store.

use crate::agent::learning::LearningLoop;
use crate::agent::tools::tool::{Tool, ToolResult};
use crate::plugins::PluginStore;
use crate::plugins::schema::{PluginDefinition, PluginSpec, PluginType, SkillPluginSpec};
use async_trait::async_trait;
use std::sync::Arc;

/// Tool that the agent can invoke to manage its own skill library.
///
/// Registered automatically when the learning loop is enabled.
#[derive(Debug)]
pub struct SkillManageTool {
    learning_loop: Arc<LearningLoop>,
}

impl SkillManageTool {
    pub fn new(learning_loop: Arc<LearningLoop>) -> Self {
        Self { learning_loop }
    }

    fn store(&self) -> Option<&Arc<PluginStore>> {
        self.learning_loop.plugin_store.as_ref()
    }
}

impl crate::agent::tools::tool::ToolMeta for SkillManageTool {
    fn name(&self) -> &str {
        "skill_manage"
    }

    fn description(&self) -> &str {
        "Create, patch, list, or delete reusable skill definitions. \
         Skills are stored persistently and loaded in future sessions."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "patch", "list", "delete"],
                    "description": "The action to perform on the skill library."
                },
                "name": {
                    "type": "string",
                    "description": "Skill name (required for create; optional for patch/delete when skill_id is supplied)."
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the skill (create/patch)."
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt template for the skill (create/patch). \
                                    Should include procedure, pitfalls, and verification steps."
                },
                "required_tools": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Tool names required by this skill (create/patch)."
                },
                "skill_id": {
                    "type": "string",
                    "description": "Skill ID (preferred for patch/delete)."
                }
            },
            "required": ["action"]
        })
    }

    fn is_readonly(&self) -> bool {
        false
    }
}

#[async_trait]
impl Tool for SkillManageTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let action = match args.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => return ToolResult::error("Missing required field: action"),
        };

        match action {
            "create" => self.handle_create(&args).await,
            "patch" => self.handle_patch(&args).await,
            "list" => self.handle_list().await,
            "delete" => self.handle_delete(&args).await,
            other => ToolResult::error(format!(
                "Unknown action: {other}. Valid actions: create, patch, list, delete"
            )),
        }
    }
}

impl SkillManageTool {
    async fn handle_create(&self, args: &serde_json::Value) -> ToolResult {
        let Some(store) = self.store() else {
            return ToolResult::error(
                "skill_manage: plugin store is not configured on the learning loop",
            );
        };
        let name = match args.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.trim(),
            None => return ToolResult::error("Missing required field: name"),
        };
        if name.is_empty() {
            return ToolResult::error("Field `name` must be non-empty");
        }
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("Auto-created skill");
        let system_prompt = args
            .get("system_prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let required_tools: Vec<String> = args
            .get("required_tools")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let now = chrono::Utc::now();
        let definition = PluginDefinition {
            id: format!("skill-{}", uuid::Uuid::now_v7()),
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: description.to_string(),
            author: Some("agent".to_string()),
            tags: vec!["agent-created".to_string()],
            plugin_type: PluginType::Skill,
            spec: PluginSpec::Skill(SkillPluginSpec {
                system_prompt_template: system_prompt.to_string(),
                parameters: std::collections::HashMap::new(),
                required_tools,
                restrict_tools: false,
                max_iterations: None,
            }),
            created_at: now,
            updated_at: now,
        };

        if let Err(err) = store.put(&definition).await {
            return ToolResult::error(format!("failed to persist skill: {err}"));
        }

        self.learning_loop.mark_skill_created().await;

        let summary = serde_json::json!({
            "status": "created",
            "skill_id": definition.id,
            "name": definition.name,
            "description": definition.description,
        });

        ToolResult::success(serde_json::to_string_pretty(&summary).unwrap_or_default())
    }

    async fn handle_patch(&self, args: &serde_json::Value) -> ToolResult {
        let Some(store) = self.store() else {
            return ToolResult::error(
                "skill_manage: plugin store is not configured on the learning loop",
            );
        };
        let skill_id = args.get("skill_id").and_then(|v| v.as_str());
        let name = args.get("name").and_then(|v| v.as_str());

        let mut existing = match (skill_id, name) {
            (Some(id), _) => match store.get_by_id(id).await {
                Ok(Some(def)) => def,
                Ok(None) => {
                    return ToolResult::error(format!("no skill found with id `{id}`"));
                }
                Err(err) => {
                    return ToolResult::error(format!("failed to load skill: {err}"));
                }
            },
            (None, Some(n)) => match store.get_by_name(n).await {
                Ok(Some(def)) => def,
                Ok(None) => {
                    return ToolResult::error(format!("no skill found with name `{n}`"));
                }
                Err(err) => {
                    return ToolResult::error(format!("failed to load skill: {err}"));
                }
            },
            (None, None) => {
                return ToolResult::error("patch requires either `skill_id` or `name`");
            }
        };

        if !matches!(existing.plugin_type, PluginType::Skill) {
            return ToolResult::error(format!(
                "plugin `{}` is not a skill (type={:?})",
                existing.name, existing.plugin_type
            ));
        }

        if let Some(desc) = args.get("description").and_then(|v| v.as_str()) {
            existing.description = desc.to_string();
        }
        if let PluginSpec::Skill(ref mut spec) = existing.spec {
            if let Some(prompt) = args.get("system_prompt").and_then(|v| v.as_str()) {
                spec.system_prompt_template = prompt.to_string();
            }
            if let Some(tools) = args.get("required_tools").and_then(|v| v.as_array()) {
                let mut deduped: Vec<String> = tools
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                deduped.sort();
                deduped.dedup();
                if !deduped.is_empty() {
                    spec.required_tools = deduped;
                }
            }
        }
        existing.updated_at = chrono::Utc::now();

        if let Err(err) = store.put(&existing).await {
            return ToolResult::error(format!("failed to persist patched skill: {err}"));
        }

        let out = serde_json::json!({
            "status": "patched",
            "skill_id": existing.id,
            "name": existing.name,
            "description": existing.description,
        });
        ToolResult::success(serde_json::to_string_pretty(&out).unwrap_or_default())
    }

    async fn handle_list(&self) -> ToolResult {
        let Some(store) = self.store() else {
            let tracker = self.learning_loop.episode_tracker.read().await;
            let summary = serde_json::json!({
                "skills_created_this_session": tracker.skills_created(),
                "note": "plugin store not configured; only session-local counters are available."
            });
            return ToolResult::success(serde_json::to_string_pretty(&summary).unwrap_or_default());
        };
        let all = match store.load_all().await {
            Ok(defs) => defs,
            Err(err) => {
                return ToolResult::error(format!("failed to list skills: {err}"));
            }
        };
        let skills: Vec<serde_json::Value> = all
            .into_iter()
            .filter(|def| matches!(def.plugin_type, PluginType::Skill))
            .map(|def| {
                let tools: Vec<String> = match &def.spec {
                    PluginSpec::Skill(spec) => spec.required_tools.clone(),
                    _ => Vec::new(),
                };
                serde_json::json!({
                    "id": def.id,
                    "name": def.name,
                    "description": def.description,
                    "version": def.version,
                    "required_tools": tools,
                })
            })
            .collect();
        let out = serde_json::json!({
            "count": skills.len(),
            "skills": skills,
        });
        ToolResult::success(serde_json::to_string_pretty(&out).unwrap_or_default())
    }

    async fn handle_delete(&self, args: &serde_json::Value) -> ToolResult {
        let Some(store) = self.store() else {
            return ToolResult::error(
                "skill_manage: plugin store is not configured on the learning loop",
            );
        };
        let skill_id = args.get("skill_id").and_then(|v| v.as_str());
        let name = args.get("name").and_then(|v| v.as_str());

        let target_name = match (skill_id, name) {
            (Some(id), _) => match store.get_by_id(id).await {
                Ok(Some(def)) => def.name,
                Ok(None) => {
                    return ToolResult::error(format!("no skill found with id `{id}`"));
                }
                Err(err) => {
                    return ToolResult::error(format!("failed to resolve skill: {err}"));
                }
            },
            (None, Some(n)) => n.to_string(),
            (None, None) => {
                return ToolResult::error("delete requires either `skill_id` or `name`");
            }
        };

        if let Err(err) = store.delete_by_name(&target_name).await {
            return ToolResult::error(format!("failed to delete skill: {err}"));
        }

        let out = serde_json::json!({
            "status": "deleted",
            "name": target_name,
        });
        ToolResult::success(serde_json::to_string_pretty(&out).unwrap_or_default())
    }
}
