//! Plugin schema definitions: envelope and type-specific specs.
//!
//! All plugin types (Tool, Provider, Backend, Skill, Source) share a common
//! `PluginDefinition` envelope and dispatch via `PluginSpec`.

use crate::agent::tools::tool::ToolCapability;
use crate::plugins::step_dsl::Step;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Universal plugin envelope: identity, type, and type-specific spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDefinition {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub plugin_type: PluginType,
    pub spec: PluginSpec,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginType {
    Tool,
    Provider,
    Backend,
    Skill,
    Source,
    Hook,
    Embedding,
    Evaluator,
    DocumentLoader,
    StreamBackend,
    OutputParser,
    ContextEngine,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginSpec {
    Tool(ToolPluginSpec),
    Provider(ProviderPluginSpec),
    Backend(BackendPluginSpec),
    Skill(SkillPluginSpec),
    Source(SourcePluginSpec),
    Hook(HookPluginSpec),
    Embedding(EmbeddingPluginSpec),
    Evaluator(EvaluatorPluginSpec),
    DocumentLoader(DocumentLoaderPluginSpec),
    StreamBackend(StreamBackendPluginSpec),
    OutputParser(OutputParserPluginSpec),
    ContextEngine(ContextEnginePluginSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPluginSpec {
    pub parameters: serde_json::Value,
    pub steps: Vec<Step>,
    #[serde(default)]
    pub is_readonly: bool,
    #[serde(default)]
    pub capabilities: Vec<ToolCapability>,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderPluginSpec {
    pub base_url: String,
    pub auth: ProviderAuth,
    pub models_endpoint: Option<EndpointMapping>,
    pub completion_endpoint: EndpointMapping,
    pub streaming: bool,
    #[serde(default)]
    pub supports_tools: bool,
    #[serde(default)]
    pub supports_vision: bool,
    pub request_mapping: RequestMapping,
    pub response_mapping: ResponseMapping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderAuth {
    None,
    Bearer { ref_key: String },
    Header { name: String, ref_key: String },
    QueryParam { name: String, ref_key: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointMapping {
    pub path: String,
    pub method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMapping {
    pub messages_path: Option<String>,
    pub max_tokens_path: Option<String>,
    pub temperature_path: Option<String>,
    pub tools_path: Option<String>,
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMapping {
    pub content_path: String,
    pub finish_reason_path: Option<String>,
    pub usage_path: Option<String>,
    pub tool_calls_path: Option<String>,
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

/// Backend capabilities for schema-defined vector backends.
/// Mirrors dripstone::core::Capabilities for serialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendCapabilities {
    #[serde(default)]
    pub native_hybrid: bool,
    #[serde(default)]
    pub native_full_text: bool,
    #[serde(default)]
    pub native_hybrid_full_text: bool,
    #[serde(default)]
    pub native_filter: bool,
    #[serde(default = "default_true")]
    pub supports_namespaces: bool,
    /// Late-interaction MaxSim on the remote store (default false for HTTP schema backends).
    #[serde(default)]
    pub native_maxsim: bool,
    pub max_dimension: Option<usize>,
    pub max_batch: Option<usize>,
    #[serde(default)]
    pub supports_streaming: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendPluginSpec {
    pub connection: ConnectionSpec,
    pub dimension: usize,
    pub endpoints: BackendEndpoints,
    pub result_mapping: ResultMapping,
    #[serde(default)]
    pub capabilities: BackendCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConnectionSpec {
    Named { ref_key: String },
    Inline { url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendEndpoints {
    pub upsert: Option<EndpointMapping>,
    pub search: Option<EndpointMapping>,
    pub delete: Option<EndpointMapping>,
    pub count: Option<EndpointMapping>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultMapping {
    pub id_path: Option<String>,
    pub score_path: Option<String>,
    pub content_path: Option<String>,
    pub metadata_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPluginSpec {
    pub system_prompt_template: String,
    #[serde(default)]
    pub parameters: HashMap<String, SkillParam>,
    #[serde(default)]
    pub required_tools: Vec<String>,
    #[serde(default)]
    pub restrict_tools: bool,
    pub max_iterations: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillParam {
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub param_type: Option<String>,
    pub default: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcePluginSpec {
    pub uri_scheme: String,
    pub resolver_steps: Vec<Step>,
    pub content_transform: Option<Step>,
}

/// Hook plugin: step DSL runs with tool context; final output maps to Allow/Deny/RequireApproval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookPluginSpec {
    /// "all" or a specific tool name.
    pub matcher: String,
    /// Steps run on pre_tool_use with args: tool_name, tool_input, tool_use_id.
    pub pre_tool_use_steps: Vec<Step>,
    /// Optional path to decision in final step output; default "decision" or root string "allow".
    #[serde(default)]
    pub decision_path: Option<String>,
    /// Optional path to reason (for deny/require_approval); default "reason".
    #[serde(default)]
    pub reason_path: Option<String>,
    /// Optional path to proposed_amendment (for require_approval); default "proposed_amendment".
    #[serde(default)]
    pub amendment_path: Option<String>,
}

/// Embedding plugin: HTTP API mapped to EmbeddingProvider (texts -> embeddings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingPluginSpec {
    pub base_url: String,
    pub auth: ProviderAuth,
    pub embed_endpoint: EndpointMapping,
    pub request_mapping: EmbeddingRequestMapping,
    pub response_mapping: EmbeddingResponseMapping,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequestMapping {
    pub texts_path: String,
    #[serde(default)]
    pub model_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponseMapping {
    pub embeddings_path: String,
    #[serde(default)]
    pub usage_path: Option<String>,
    #[serde(default)]
    pub model_path: Option<String>,
}

/// Evaluator plugin: step DSL runs with output, expected, context; final output has score, passed, details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorPluginSpec {
    pub steps: Vec<Step>,
    /// Path to score in final step output; default "score".
    #[serde(default)]
    pub score_path: Option<String>,
    /// Path to passed in final step output; default "passed".
    #[serde(default)]
    pub passed_path: Option<String>,
    /// Path to details object; default "details".
    #[serde(default)]
    pub details_path: Option<String>,
}

/// Document loader plugin: file extension or MIME to handle, parsing steps (step DSL), output shape.
/// Steps receive args: content (string). Final step output is mapped to Vec<Document>.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentLoaderPluginSpec {
    /// File extension this loader handles (e.g. "xyz", "json").
    pub extension: String,
    /// Optional MIME types (e.g. ["application/x-custom"]).
    #[serde(default)]
    pub mime_types: Vec<String>,
    /// Parsing steps; args: content (raw file content as string).
    pub parsing_steps: Vec<Step>,
    /// Path to array of documents in final step output; default "documents".
    #[serde(default)]
    pub documents_path: Option<String>,
    /// Path to document content within each item; default "content".
    #[serde(default)]
    pub content_path: Option<String>,
    /// Path to document metadata within each item; default "metadata".
    #[serde(default)]
    pub metadata_path: Option<String>,
}

/// Stream backend plugin: map append, read_after, last_id, delete_stream to HTTP or DB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamBackendPluginSpec {
    pub connection: ConnectionSpec,
    pub endpoints: StreamBackendEndpoints,
    /// Path to event ID in append response; default "id".
    #[serde(default)]
    pub id_path: Option<String>,
    /// Path to array of (id, event) in read_after response; default "events".
    #[serde(default)]
    pub events_path: Option<String>,
    /// Path to event ID within each event item; default "id".
    #[serde(default)]
    pub event_id_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamBackendEndpoints {
    pub append: Option<EndpointMapping>,
    pub read_after: Option<EndpointMapping>,
    pub last_id: Option<EndpointMapping>,
    pub delete_stream: Option<EndpointMapping>,
}

/// Output parser plugin: extraction steps (step DSL), format instructions, output path.
/// Steps receive args: text (LLM output string). Parser output is serde_json::Value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputParserPluginSpec {
    /// Extraction steps; args: text.
    pub extraction_steps: Vec<Step>,
    /// Instructions to include in the prompt for the LLM.
    #[serde(default)]
    pub format_instructions: String,
    /// Path to parsed value in final step output; if missing, entire output is used.
    #[serde(default)]
    pub output_path: Option<String>,
}

/// Context engine plugin: pluggable context assembly / compaction lifecycle.
///
/// An exclusive slot: only one context engine is active at a time.
/// Steps in `assemble_steps` receive `messages` (JSON array) and `token_budget` (number)
/// and must return `{ "messages": [...], "estimated_tokens": N }`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEnginePluginSpec {
    /// Steps to assemble context before each LLM call.
    pub assemble_steps: Vec<Step>,
    /// Steps to compact context when over budget.
    #[serde(default)]
    pub compact_steps: Vec<Step>,
    /// Whether this engine owns compaction entirely (disables built-in compaction).
    #[serde(default)]
    pub owns_compaction: bool,
    /// Optional additional system prompt fragment injected by the engine.
    #[serde(default)]
    pub system_prompt_addition: Option<String>,
}
