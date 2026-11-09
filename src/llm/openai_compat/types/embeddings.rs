//! OpenAI-compatible embedding types.

use serde::{Deserialize, Serialize};

use super::responses::OpenAICompatUsage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatEmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatEmbeddingResponse {
    pub object: String,
    pub data: Vec<OpenAICompatEmbeddingData>,
    pub model: String,
    pub usage: OpenAICompatUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatEmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}
