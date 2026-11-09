//! Request and response models for API endpoints.

use caliche::loaders::Document;
use caliche::messages::TokenUsage;
use caliche::orchestration::{PendingInterrupt, WorkflowEdge, WorkflowNode, WorkflowState};
use caliche::rag::{RagFusionMode, RagSearchMode};
use caliche::reasoning::ReasoningStep;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===== Chat API Models =====

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatEstimateRequest {
    pub message: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    #[serde(default)]
    pub context_size: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ChatEstimateResponse {
    pub estimated_cost_usd: f64,
    pub breakdown: caliche::cost::CostBreakdown,
    pub confidence: caliche::cost::ConfidenceLevel,
    pub request_id: String,
}

// ===== Reasoning API Models =====

#[derive(Debug, Deserialize, Serialize)]
pub struct ReasoningRequest {
    pub message: String,
    #[serde(default)]
    pub max_iterations: Option<usize>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct ReasoningResponse {
    pub final_answer: String,
    pub steps: Vec<ReasoningStep>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    pub request_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TreeOfThoughtsRequest {
    pub problem: String,
    #[serde(default)]
    pub num_branches: Option<usize>,
    #[serde(default)]
    pub depth: Option<usize>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct TreeOfThoughtsResponse {
    pub final_answer: String,
    pub explored_paths: Vec<Vec<String>>,
    pub best_path: Vec<String>,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct ReasoningPatternInfo {
    pub name: String,
    pub description: String,
    pub default_max_iterations: usize,
}

#[derive(Debug, Serialize)]
pub struct ReasoningPatternsResponse {
    pub patterns: Vec<ReasoningPatternInfo>,
}

// ===== Chain API Models =====

#[derive(Debug, Deserialize, Serialize)]
pub struct LLMChainRequest {
    pub prompt_template: String,
    pub variables: serde_json::Value,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct ChainResponse {
    pub output: serde_json::Value,
    pub intermediate_steps: Vec<(String, serde_json::Value)>,
    pub request_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SequentialChainRequest {
    pub chains: Vec<ChainStep>,
    pub input: serde_json::Value,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChainStep {
    pub name: String,
    pub prompt_template: String,
    pub output_key: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MapReduceRequest {
    pub map_template: String,
    pub reduce_template: String,
    pub items: Vec<serde_json::Value>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RetrievalChainRequest {
    pub query: String,
    #[serde(default)]
    pub collection: Option<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub search_mode: Option<ApiSearchMode>,
    #[serde(default)]
    pub fusion_mode: Option<ApiFusionMode>,
    #[serde(default)]
    pub dense_weight: Option<f32>,
    #[serde(default)]
    pub sparse_weight: Option<f32>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HybridSearchRequest {
    pub query: String,
    #[serde(default)]
    pub collection: Option<String>,
    #[serde(default)]
    pub vector_weight: Option<f32>,
    #[serde(default)]
    pub keyword_weight: Option<f32>,
    #[serde(default)]
    pub search_mode: Option<ApiSearchMode>,
    #[serde(default)]
    pub fusion_mode: Option<ApiFusionMode>,
    #[serde(default)]
    pub dense_weight: Option<f32>,
    #[serde(default)]
    pub sparse_weight: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentMapReduceRequest {
    pub document: String,
    pub map_template: String,
    pub reduce_template: String,
    #[serde(default)]
    pub chunk_size: Option<usize>,
    #[serde(default)]
    pub chunk_overlap: Option<usize>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

// ===== Document API Models =====

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentLoadRequest {
    pub source: DocumentSource,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DocumentSource {
    Csv {
        #[serde(flatten)]
        location: DocumentLocation,
        #[serde(default)]
        content_column: Option<String>,
        #[serde(default)]
        metadata_columns: Option<Vec<String>>,
    },
    Json {
        #[serde(flatten)]
        location: DocumentLocation,
        #[serde(default)]
        content_key: Option<String>,
    },
    Markdown {
        #[serde(flatten)]
        location: DocumentLocation,
    },
    Html {
        #[serde(flatten)]
        location: DocumentLocation,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum DocumentLocation {
    Path { path: String },
    Content { content: String },
}

#[derive(Debug, Serialize)]
pub struct DocumentLoadResponse {
    pub documents: Vec<DocumentDto>,
    pub count: usize,
    pub request_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentDto {
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl From<Document> for DocumentDto {
    fn from(doc: Document) -> Self {
        Self {
            content: doc.content,
            metadata: doc.metadata,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentSplitRequest {
    pub documents: Vec<DocumentDto>,
    pub chunk_size: usize,
    #[serde(default)]
    pub chunk_overlap: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct DocumentSplitResponse {
    pub chunks: Vec<ChunkDto>,
    pub total_chunks: usize,
    pub request_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkDto {
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub chunk_index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentIndexRequest {
    pub documents: Vec<DocumentDto>,
    pub collection_name: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub embedding_model: Option<String>,
    #[serde(default)]
    pub vector_storage: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DocumentIndexResponse {
    pub collection_name: String,
    pub indexed_count: usize,
    pub ids: Vec<String>,
    pub request_id: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentBatchIndexRequest {
    pub documents: Vec<DocumentBatchItem>,
    #[serde(default)]
    pub collection_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentBatchItem {
    pub doc_id: String,
    pub document: String,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct DocumentBatchIndexResponse {
    pub total_chunks: usize,
    pub total_documents: usize,
    pub failures: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct CollectionsResponse {
    pub collections: Vec<CollectionInfo>,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub document_count: usize,
    pub created_at: Option<String>,
    /// Last registry update (RFC3339), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ApiSearchMode {
    Dense,
    Hybrid,
    FullText,
    HybridFullText,
}

impl ApiSearchMode {
    pub const fn to_rag(self) -> RagSearchMode {
        match self {
            Self::Dense => RagSearchMode::Dense,
            Self::Hybrid => RagSearchMode::HybridDenseSparse,
            Self::FullText => RagSearchMode::FullText,
            Self::HybridFullText => RagSearchMode::HybridDenseFullText,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Hybrid => "hybrid",
            Self::FullText => "full_text",
            Self::HybridFullText => "hybrid_full_text",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub enum ApiFusionMode {
    #[serde(rename = "weighted")]
    Weighted,
    #[serde(rename = "rrf")]
    ReciprocalRank,
}

impl ApiFusionMode {
    pub const fn to_rag(self) -> RagFusionMode {
        match self {
            Self::Weighted => RagFusionMode::Weighted,
            Self::ReciprocalRank => RagFusionMode::ReciprocalRank,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::ReciprocalRank => "rrf",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DocumentSearchRequest {
    pub query: String,
    pub collection_name: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_similarity: Option<f32>,
    #[serde(default)]
    pub filter: Option<serde_json::Value>,
    #[serde(default)]
    pub search_mode: Option<ApiSearchMode>,
    #[serde(default)]
    pub fusion_mode: Option<ApiFusionMode>,
    #[serde(default)]
    pub dense_weight: Option<f32>,
    #[serde(default)]
    pub sparse_weight: Option<f32>,
    #[serde(default)]
    pub vector_storage: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DocumentSearchResponse {
    pub results: Vec<SearchResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fusion_mode: Option<String>,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub content: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

// ===== Orchestration API Models =====

#[derive(Debug, Deserialize, Serialize)]
pub struct WorkflowRequest {
    pub nodes: Vec<WorkflowNode>,
    pub edges: Vec<WorkflowEdge>,
    pub input: serde_json::Value,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct WorkflowResponse {
    pub workflow_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_id: Option<String>,
    pub state: WorkflowState,
    pub results: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct WorkflowStatusResponse {
    pub workflow_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_id: Option<String>,
    pub state: WorkflowState,
    pub current_node: Option<String>,
    pub completed_nodes: Vec<String>,
    pub results: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MultiAgentRequest {
    pub pattern: MultiAgentPattern,
    pub agents: Vec<AgentConfig>,
    pub task: String,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MultiAgentPattern {
    Pipeline,
    Debate,
    Hierarchical,
    Supervisor,
    #[serde(rename = "fan_out_collect")]
    FanOutCollect,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResumeWorkflowRequest {
    pub interrupt_id: String,
    pub feedback: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct InterruptsListResponse {
    pub interrupts: Vec<PendingInterrupt>,
}

#[derive(Debug, Serialize)]
pub struct InterruptResponse {
    pub interrupt: PendingInterrupt,
}

#[derive(Debug, Serialize)]
pub struct ResumeWorkflowResponse {
    pub success: bool,
    pub execution_id: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub capabilities: Option<Vec<String>>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct MultiAgentResponse {
    pub pattern: String,
    pub final_result: String,
    pub agent_results: Vec<AgentResult>,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct AgentResult {
    pub agent_id: String,
    pub agent_name: String,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct CoordinatorRequest {
    pub task: String,
    pub required_capabilities: Vec<String>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Run the methodical investigate/verify loop.
    #[serde(default)]
    pub complex: bool,
    /// Override provider for spawned subagents (`anthropic`, `openai`, ...).
    #[serde(default)]
    pub subagent_provider: Option<String>,
    /// Override model for spawned subagents. Accepts `provider/model`.
    #[serde(default)]
    pub subagent_model: Option<String>,
    /// Maximum investigate/verify rounds when `complex` is true.
    #[serde(default)]
    pub max_rounds: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct CoordinatorResponse {
    pub selected_agent: String,
    pub result: String,
    pub capabilities_matched: Vec<String>,
    pub request_id: String,
}

// ===== Common Error Response =====

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    pub request_id: String,
}

impl ErrorResponse {
    pub fn new(error: String, request_id: String) -> Self {
        Self {
            error,
            details: None,
            request_id,
        }
    }

    pub fn with_details(error: String, details: String, request_id: String) -> Self {
        Self {
            error,
            details: Some(details),
            request_id,
        }
    }
}
