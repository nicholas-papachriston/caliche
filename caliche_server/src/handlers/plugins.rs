use crate::AppState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::plugins::schema::{PluginDefinition, PluginType};
use caliche::plugins::validation::{ValidationContext, validate_plugin};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;
use uuid::Uuid;

type ApiResult<T> = Result<Json<T>, (StatusCode, String)>;

#[derive(Debug, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub version: String,
    pub description: String,
    pub plugin_type: String,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct PluginListResponse {
    pub plugins: Vec<PluginSummary>,
}

#[derive(Debug, Serialize)]
pub struct PluginGuide {
    pub plugin_type: String,
    pub title: String,
    pub summary: String,
    pub when_to_use: String,
    pub notes: Vec<String>,
    pub starter_definition: Value,
}

#[derive(Debug, Serialize)]
pub struct PluginCatalogResponse {
    pub guides: Vec<PluginGuide>,
}

#[derive(Debug, Deserialize)]
pub struct PluginDefinitionRequest {
    pub definition: Value,
}

#[derive(Debug, Serialize)]
pub struct PluginValidationResponse {
    pub valid: bool,
    pub normalized_definition: Value,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct PluginSaveResponse {
    pub created: bool,
    pub summary: PluginSummary,
    pub definition: PluginDefinition,
}

#[derive(Debug, Serialize)]
pub struct PluginExportResponse {
    pub name: String,
    pub definition: String,
}

fn registry(
    state: &AppState,
) -> Result<Arc<caliche::plugins::registry::PluginRegistry>, (StatusCode, String)> {
    state.plugin_registry.clone().ok_or_else(|| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "Plugin registry is not available".to_string(),
        )
    })
}

fn validation_ctx() -> ValidationContext {
    ValidationContext::default()
}

fn plugin_summary(def: &PluginDefinition) -> PluginSummary {
    PluginSummary {
        name: def.name.clone(),
        version: def.version.clone(),
        description: def.description.clone(),
        plugin_type: serde_json::to_value(def.plugin_type)
            .ok()
            .and_then(|v| v.as_str().map(ToString::to_string))
            .unwrap_or_else(|| format!("{:?}", def.plugin_type).to_lowercase()),
        author: def.author.clone(),
        tags: def.tags.clone(),
        updated_at: def.updated_at.to_rfc3339(),
    }
}

fn normalize_definition(
    mut value: Value,
    existing: Option<&PluginDefinition>,
) -> anyhow::Result<PluginDefinition> {
    let now = chrono::Utc::now();
    let Some(obj) = value.as_object_mut() else {
        anyhow::bail!("definition must be a JSON object");
    };

    if !obj.contains_key("id") {
        let id = existing
            .map(|def| def.id.clone())
            .unwrap_or_else(|| Uuid::now_v7().to_string());
        obj.insert("id".to_string(), json!(id));
    }
    if !obj.contains_key("created_at") {
        let created_at = existing
            .map(|def| def.created_at.to_rfc3339())
            .unwrap_or_else(|| now.to_rfc3339());
        obj.insert("created_at".to_string(), json!(created_at));
    }
    obj.insert("updated_at".to_string(), json!(now.to_rfc3339()));

    let def: PluginDefinition = serde_json::from_value(value)?;
    validate_plugin(&def, &validation_ctx())?;
    Ok(def)
}

fn starter_definition(plugin_type: PluginType) -> Value {
    match plugin_type {
        PluginType::Tool => json!({
            "name": "fetch_release_notes",
            "version": "0.1.0",
            "description": "Fetches release notes from a JSON API and returns a short summary.",
            "plugin_type": "tool",
            "spec": {
                "type": "tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": { "type": "string", "description": "Product name to query" }
                    },
                    "required": ["product"]
                },
                "steps": [
                    {
                        "name": "fetch",
                        "type": "http",
                        "method": "GET",
                        "url": "https://example.com/releases?product={{args.product}}",
                        "headers": {},
                        "body": null,
                        "auth_ref": null,
                        "extract": "$"
                    },
                    {
                        "name": "done",
                        "type": "return",
                        "value": "steps.fetch"
                    }
                ],
                "is_readonly": true,
                "capabilities": ["read_only"],
                "timeout_secs": 30
            }
        }),
        PluginType::Provider => json!({
            "name": "custom_chat_provider",
            "version": "0.1.0",
            "description": "Maps an OpenAI-compatible chat API into Caliche.",
            "plugin_type": "provider",
            "spec": {
                "type": "provider",
                "base_url": "https://api.example.com",
                "auth": { "type": "bearer", "ref_key": "CUSTOM_API_KEY" },
                "models_endpoint": { "path": "/v1/models", "method": "GET" },
                "completion_endpoint": { "path": "/v1/chat/completions", "method": "POST" },
                "streaming": true,
                "supports_tools": true,
                "supports_vision": false,
                "request_mapping": {
                    "messages_path": "messages",
                    "max_tokens_path": "max_tokens",
                    "temperature_path": "temperature",
                    "tools_path": "tools",
                    "extra": {}
                },
                "response_mapping": {
                    "content_path": "choices.0.message.content",
                    "finish_reason_path": "choices.0.finish_reason",
                    "usage_path": "usage",
                    "tool_calls_path": "choices.0.message.tool_calls",
                    "extra": {}
                }
            }
        }),
        PluginType::Backend => json!({
            "name": "custom_vector_backend",
            "version": "0.1.0",
            "description": "Connects Caliche to a remote vector backend.",
            "plugin_type": "backend",
            "spec": {
                "type": "backend",
                "connection": { "kind": "inline", "url": "https://vectors.example.com" },
                "dimension": 1536,
                "endpoints": {
                    "upsert": { "path": "/upsert", "method": "POST" },
                    "search": { "path": "/search", "method": "POST" },
                    "delete": { "path": "/delete", "method": "POST" },
                    "count": { "path": "/count", "method": "GET" }
                },
                "result_mapping": {
                    "id_path": "id",
                    "score_path": "score",
                    "content_path": "content",
                    "metadata_path": "metadata"
                },
                "capabilities": {
                    "native_hybrid": false,
                    "native_filter": true,
                    "supports_namespaces": true,
                    "max_dimension": 1536,
                    "max_batch": 100,
                    "supports_streaming": false
                }
            }
        }),
        PluginType::Skill => json!({
            "name": "incident_triage",
            "version": "0.1.0",
            "description": "Reusable incident triage operating mode.",
            "plugin_type": "skill",
            "spec": {
                "type": "skill",
                "system_prompt_template": "You are the incident triage specialist. Summarize impact, suspected cause, and next action.",
                "parameters": {
                    "service": {
                        "description": "Service being investigated",
                        "type": "string",
                        "default": "core-api"
                    }
                },
                "required_tools": ["fs", "grep"],
                "restrict_tools": false,
                "max_iterations": 8
            }
        }),
        PluginType::Source => json!({
            "name": "runbook_source",
            "version": "0.1.0",
            "description": "Resolves a custom runbook:// URI into text content.",
            "plugin_type": "source",
            "spec": {
                "type": "source",
                "uri_scheme": "runbook",
                "resolver_steps": [
                    {
                        "name": "fetch",
                        "type": "http",
                        "method": "GET",
                        "url": "https://runbooks.example.com/{{args.uri}}",
                        "headers": {},
                        "body": null,
                        "auth_ref": null,
                        "extract": "$"
                    },
                    {
                        "name": "done",
                        "type": "return",
                        "value": "steps.fetch"
                    }
                ],
                "content_transform": null
            }
        }),
        PluginType::Hook => json!({
            "name": "dangerous_command_guard",
            "version": "0.1.0",
            "description": "Requires approval before destructive shell commands run.",
            "plugin_type": "hook",
            "spec": {
                "type": "hook",
                "matcher": "bash",
                "pre_tool_use_steps": [
                    {
                        "name": "decide",
                        "type": "return",
                        "value": "{\"decision\":\"require_approval\",\"reason\":\"Review destructive commands before execution.\"}"
                    }
                ],
                "decision_path": "decision",
                "reason_path": "reason",
                "amendment_path": "proposed_amendment"
            }
        }),
        PluginType::Embedding => json!({
            "name": "custom_embeddings",
            "version": "0.1.0",
            "description": "Maps a remote embedding API into Caliche.",
            "plugin_type": "embedding",
            "spec": {
                "type": "embedding",
                "base_url": "https://api.example.com",
                "auth": { "type": "bearer", "ref_key": "CUSTOM_API_KEY" },
                "embed_endpoint": { "path": "/v1/embeddings", "method": "POST" },
                "request_mapping": {
                    "texts_path": "input",
                    "model_path": "model"
                },
                "response_mapping": {
                    "embeddings_path": "data",
                    "usage_path": "usage",
                    "model_path": "model"
                }
            }
        }),
        PluginType::Evaluator => json!({
            "name": "answer_quality_score",
            "version": "0.1.0",
            "description": "Scores output quality using a small step pipeline.",
            "plugin_type": "evaluator",
            "spec": {
                "type": "evaluator",
                "steps": [
                    {
                        "name": "score",
                        "type": "return",
                        "value": "{\"score\":0.9,\"passed\":true,\"details\":{\"reason\":\"Helpful and correct.\"}}"
                    }
                ],
                "score_path": "score",
                "passed_path": "passed",
                "details_path": "details"
            }
        }),
        PluginType::DocumentLoader => json!({
            "name": "csv_loader",
            "version": "0.1.0",
            "description": "Parses custom CSV-like files into documents.",
            "plugin_type": "document_loader",
            "spec": {
                "type": "document_loader",
                "extension": "csvx",
                "mime_types": ["text/csv"],
                "parsing_steps": [
                    {
                        "name": "documents",
                        "type": "return",
                        "value": "{\"documents\":[{\"content\":\"{{args.content}}\",\"metadata\":{\"source\":\"csvx\"}}]}"
                    }
                ],
                "documents_path": "documents",
                "content_path": "content",
                "metadata_path": "metadata"
            }
        }),
        PluginType::StreamBackend => json!({
            "name": "http_stream_backend",
            "version": "0.1.0",
            "description": "Durable stream backend backed by an HTTP service.",
            "plugin_type": "stream_backend",
            "spec": {
                "type": "stream_backend",
                "connection": { "kind": "inline", "url": "https://streams.example.com" },
                "endpoints": {
                    "append": { "path": "/streams/append", "method": "POST" },
                    "read_after": { "path": "/streams/read_after", "method": "POST" },
                    "last_id": { "path": "/streams/last_id", "method": "GET" },
                    "delete_stream": { "path": "/streams/delete", "method": "POST" }
                },
                "id_path": "id",
                "events_path": "events",
                "event_id_path": "id"
            }
        }),
        PluginType::OutputParser => json!({
            "name": "json_ticket_parser",
            "version": "0.1.0",
            "description": "Parses model output into a structured ticket object.",
            "plugin_type": "output_parser",
            "spec": {
                "type": "output_parser",
                "extraction_steps": [
                    {
                        "name": "parsed",
                        "type": "return",
                        "value": "{\"ticket\":{\"summary\":\"{{args.text}}\"}}"
                    }
                ],
                "format_instructions": "Return JSON with a top-level 'ticket' object.",
                "output_path": "ticket"
            }
        }),
        PluginType::ContextEngine => json!({
            "name": "passthrough_context_engine",
            "version": "0.1.0",
            "description": "Context engine that passes messages through unchanged.",
            "plugin_type": "context_engine",
            "spec": {
                "type": "context_engine",
                "assemble_steps": [
                    {
                        "name": "assembled",
                        "type": "return",
                        "value": "{\"messages\": {{args.messages}}, \"estimated_tokens\": 0}"
                    }
                ],
                "compact_steps": [],
                "owns_compaction": false,
                "system_prompt_addition": null
            }
        }),
    }
}

fn plugin_guides() -> Vec<PluginGuide> {
    [
        (
            PluginType::Tool,
            "Tool",
            "Callable capability implemented by the step DSL.",
            "Use for reusable actions the agent can invoke immediately after registration.",
            vec![
                "Must include at least one step; multi-step pipelines must end with a return step.",
                "Read-only tools should set is_readonly and limit capabilities.",
            ],
        ),
        (
            PluginType::Provider,
            "Provider",
            "Maps an external chat-completions API into Caliche.",
            "Use when you need a custom model provider or OpenAI-compatible gateway.",
            vec![
                "Requires completion endpoint plus request/response path mappings.",
                "Auth is referenced by key name, not embedded credentials.",
            ],
        ),
        (
            PluginType::Backend,
            "Vector Backend",
            "Remote vector store adapter definition.",
            "Use to integrate a vector database that is not already built into dripstone.",
            vec![
                "Dimension is required and must match your embedding model output.",
                "Endpoints and result mappings should align to the backend response shape.",
            ],
        ),
        (
            PluginType::Skill,
            "Skill",
            "Reusable prompt+tool operating mode.",
            "Use to package repeatable behavior and expose it in discovery.",
            vec![
                "Skills are applied at build time and added to the discovery catalog.",
                "Use parameters for configurable prompts instead of hardcoding every variant.",
            ],
        ),
        (
            PluginType::Source,
            "Source Resolver",
            "Custom URI scheme loader backed by the step DSL.",
            "Use when you want `scheme://...` references to resolve into context.",
            vec![
                "Resolver steps should return content or a structured payload you later transform.",
                "Optional content_transform lets you normalize the resolved output.",
            ],
        ),
        (
            PluginType::Hook,
            "Hook",
            "Lifecycle guardrail executed before tool use.",
            "Use to deny, allow, or require approval for risky operations.",
            vec![
                "matcher can be `all` or a specific tool name.",
                "Final output should expose decision/reason/amendment fields or equivalent mapped paths.",
            ],
        ),
        (
            PluginType::Embedding,
            "Embedding Provider",
            "Maps a remote embeddings API into Caliche.",
            "Use for custom embedding infrastructure or third-party providers.",
            vec![
                "First embedding plugin loaded becomes the embedding provider.",
                "texts_path and embeddings_path are required and should point at array fields.",
            ],
        ),
        (
            PluginType::Evaluator,
            "Evaluator",
            "Scores model outputs using a DSL pipeline.",
            "Use for benchmark, rubric, or dataset evaluation flows.",
            vec![
                "The final output should expose score, passed, and optional details fields.",
                "Evaluators operate on output, expected, and context args.",
            ],
        ),
        (
            PluginType::DocumentLoader,
            "Document Loader",
            "Parses custom file formats into `Document` records.",
            "Use for proprietary or uncommon document formats in ingestion pipelines.",
            vec![
                "extension is required; mime_types are optional but recommended.",
                "Parsing steps receive raw content as `args.content`.",
            ],
        ),
        (
            PluginType::StreamBackend,
            "Stream Backend",
            "Durable stream adapter mapped to HTTP endpoints.",
            "Use when event streams should persist outside the default runtime.",
            vec![
                "At least one endpoint is required; append/read_after/last_id/delete_stream are supported.",
                "The first inline stream backend loaded can become the durable stream backend.",
            ],
        ),
        (
            PluginType::OutputParser,
            "Output Parser",
            "Extracts structured JSON from model text output.",
            "Use to package reusable parsing logic and format instructions.",
            vec![
                "Extraction steps receive `args.text`.",
                "format_instructions should tell the model the exact output shape you expect.",
            ],
        ),
    ]
    .into_iter()
    .map(|(plugin_type, title, summary, when_to_use, notes)| PluginGuide {
        plugin_type: serde_json::to_value(plugin_type)
            .ok()
            .and_then(|v| v.as_str().map(ToString::to_string))
            .unwrap_or_default(),
        title: title.to_string(),
        summary: summary.to_string(),
        when_to_use: when_to_use.to_string(),
        notes: notes.into_iter().map(ToString::to_string).collect(),
        starter_definition: starter_definition(plugin_type),
    })
    .collect()
}

pub async fn list_plugins_handler(State(state): State<AppState>) -> ApiResult<PluginListResponse> {
    let registry = registry(&state)?;
    let mut plugins: Vec<_> = registry
        .list()
        .into_iter()
        .map(|def| plugin_summary(def.as_ref()))
        .collect();
    plugins.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Json(PluginListResponse { plugins }))
}

pub async fn plugin_catalog_handler(
    State(_state): State<AppState>,
) -> ApiResult<PluginCatalogResponse> {
    Ok(Json(PluginCatalogResponse {
        guides: plugin_guides(),
    }))
}

pub async fn get_plugin_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<PluginDefinition> {
    let registry = registry(&state)?;
    let plugin = registry
        .get(&name)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Plugin not found: {}", name)))?;
    Ok(Json(plugin.as_ref().clone()))
}

pub async fn export_plugin_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> ApiResult<PluginExportResponse> {
    let registry = registry(&state)?;
    let definition = registry.export(&name).await.map_err(|e| {
        if e.to_string().contains("not found") {
            (StatusCode::NOT_FOUND, e.to_string())
        } else {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        }
    })?;
    Ok(Json(PluginExportResponse { name, definition }))
}

pub async fn validate_plugin_handler(
    State(state): State<AppState>,
    Json(request): Json<PluginDefinitionRequest>,
) -> ApiResult<PluginValidationResponse> {
    let registry = registry(&state)?;
    let existing_name = request
        .definition
        .get("name")
        .and_then(Value::as_str)
        .and_then(|name| registry.get(name));

    match normalize_definition(request.definition, existing_name.as_deref()) {
        Ok(def) => Ok(Json(PluginValidationResponse {
            valid: true,
            normalized_definition: serde_json::to_value(def).unwrap_or(Value::Null),
            errors: Vec::new(),
        })),
        Err(err) => Ok(Json(PluginValidationResponse {
            valid: false,
            normalized_definition: Value::Null,
            errors: vec![err.to_string()],
        })),
    }
}

pub async fn save_plugin_handler(
    State(state): State<AppState>,
    Json(request): Json<PluginDefinitionRequest>,
) -> ApiResult<PluginSaveResponse> {
    let registry = registry(&state)?;
    let existing_name = request
        .definition
        .get("name")
        .and_then(Value::as_str)
        .and_then(|name| registry.get(name));

    let def = normalize_definition(request.definition, existing_name.as_deref()).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("invalid plugin definition: {}", e),
        )
    })?;
    let created = existing_name.is_none();

    if created {
        registry
            .register(def.clone(), &validation_ctx())
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("failed to create plugin: {}", e),
                )
            })?;
    } else {
        registry
            .update(def.clone(), &validation_ctx())
            .await
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("failed to update plugin: {}", e),
                )
            })?;
    }

    Ok(Json(PluginSaveResponse {
        created,
        summary: plugin_summary(&def),
        definition: def,
    }))
}

pub async fn delete_plugin_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let registry = registry(&state)?;
    registry.delete(&name).await.map_err(|e| {
        if e.to_string().contains("delete from store") {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
        } else {
            (StatusCode::BAD_REQUEST, e.to_string())
        }
    })?;
    Ok(StatusCode::NO_CONTENT)
}
