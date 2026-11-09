//! Document ingestion, indexing, and RAG search types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentDto {
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDto {
    pub content: String,
    pub metadata: std::collections::HashMap<String, String>,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSplitRequest {
    pub documents: Vec<DocumentDto>,
    pub chunk_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_overlap: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSplitResponse {
    pub chunks: Vec<ChunkDto>,
    pub total_chunks: usize,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentIndexRequest {
    pub documents: Vec<DocumentDto>,
    pub collection_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentIndexResponse {
    pub collection_name: String,
    pub indexed_count: usize,
    pub ids: Vec<String>,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub document_count: usize,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionsResponse {
    pub collections: Vec<CollectionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSearchRequest {
    pub query: String,
    pub collection_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub content: String,
    pub score: f32,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSearchResponse {
    pub results: Vec<SearchResult>,
    pub request_id: String,
}
