//! OpenAI-compatible message and content types.

use beef::lean::Cow;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

fn serialize_arc_str<S>(v: &Arc<String>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    s.serialize_str(v.as_str())
}

fn deserialize_arc_str<'de, D>(d: D) -> Result<Arc<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Ok(Arc::new(s))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatMessage {
    #[serde(
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub role: Cow<'static, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAICompatContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAICompatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        alias = "reasoning",
        serialize_with = "serialize_option_arc_str",
        deserialize_with = "deserialize_option_arc_str"
    )]
    pub reasoning_content: Option<Arc<String>>,
}

#[expect(
    clippy::ref_option,
    reason = "serde serialize_with callback ABI requires a reference to the field type"
)]
fn serialize_option_arc_str<S>(v: &Option<Arc<String>>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match v {
        None => s.serialize_none(),
        Some(arc) => s.serialize_str(arc.as_str()),
    }
}

fn deserialize_option_arc_str<'de, D>(d: D) -> Result<Option<Arc<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(d)?;
    Ok(opt.map(Arc::new))
}

/// OpenAI-compatible content (can be string or array of content parts)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAICompatContent {
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    String(Arc<String>),
    Array(Vec<serde_json::Value>),
}

impl OpenAICompatContent {
    /// Extract text content, concatenating text parts from arrays
    #[must_use]
    pub fn extract_text(&self) -> String {
        match self {
            Self::String(s) => s.as_ref().clone(),
            Self::Array(parts) => parts
                .iter()
                .filter_map(|part| {
                    part.get("type")
                        .and_then(|t| t.as_str())
                        .filter(|t| *t == "text")
                        .and_then(|_| part.get("text"))
                        .and_then(|t| t.as_str())
                        .map(std::string::ToString::to_string)
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatToolCall {
    #[serde(
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub id: Cow<'static, str>,
    #[serde(
        rename = "type",
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub call_type: Cow<'static, str>,
    pub function: OpenAICompatFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatFunctionCall {
    #[serde(
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub arguments: Arc<String>,
}
