//! OpenAI-compatible response types.

use serde::{Deserialize, Serialize};

use super::messages::OpenAICompatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAICompatChoice>,
    pub usage: Option<OpenAICompatUsage>,
    /// Provider-specific citations (e.g. Perplexity Sonar).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatChoice {
    pub index: u32,
    pub message: OpenAICompatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatUsage {
    pub prompt_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<OpenAIPromptTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIPromptTokensDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}
