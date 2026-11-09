//! OpenAI-compatible request types.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

use super::messages::OpenAICompatMessage;
use super::tools::OpenAICompatTool;

fn serialize_arc_vec<S, T>(v: &Arc<Vec<T>>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    (**v).serialize(s)
}

fn deserialize_arc_vec<'de, D, T>(d: D) -> Result<Arc<Vec<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let vec = Vec::deserialize(d)?;
    Ok(Arc::new(vec))
}

#[expect(
    clippy::ref_option,
    reason = "serde serialize_with callback ABI requires a reference to the field type"
)]
fn serialize_option_arc_vec<S, T>(v: &Option<Arc<Vec<T>>>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    T: Serialize,
{
    match v {
        None => s.serialize_none(),
        Some(arc) => (**arc).serialize(s),
    }
}

fn deserialize_option_arc_vec<'de, D, T>(d: D) -> Result<Option<Arc<Vec<T>>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    let opt = Option::<Vec<T>>::deserialize(d)?;
    Ok(opt.map(Arc::new))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatRequest {
    pub model: String,
    #[serde(
        serialize_with = "serialize_arc_vec",
        deserialize_with = "deserialize_arc_vec"
    )]
    pub messages: Arc<Vec<OpenAICompatMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_option_arc_vec",
        deserialize_with = "deserialize_option_arc_vec"
    )]
    pub tools: Option<Arc<Vec<OpenAICompatTool>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "usage")]
    pub usage_config: Option<serde_json::Value>,
    /// Routes related requests onto one OpenAI prompt-cache partition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}
