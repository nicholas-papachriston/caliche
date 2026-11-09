//! OpenAI-compatible streaming types.

use beef::lean::Cow;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAICompatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<super::responses::OpenAICompatUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatStreamChoice {
    pub index: u32,
    pub delta: OpenAICompatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAICompatStreamToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatStreamToolCall {
    pub index: u32,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::utils::beef_helpers::serialize_option_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_option_cow_str"
    )]
    pub id: Option<Cow<'static, str>>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: OpenAICompatStreamFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatStreamFunction {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::utils::beef_helpers::serialize_option_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_option_cow_str"
    )]
    pub name: Option<Cow<'static, str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
