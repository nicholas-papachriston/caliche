//! OpenAI-compatible tool schema types.

use beef::lean::Cow;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatTool {
    #[serde(
        rename = "type",
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub tool_type: Cow<'static, str>,
    pub function: OpenAICompatFunctionSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatFunctionSchema {
    #[serde(
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub name: Cow<'static, str>,
    #[serde(
        serialize_with = "crate::utils::beef_helpers::serialize_cow_str",
        deserialize_with = "crate::utils::beef_helpers::deserialize_cow_str"
    )]
    pub description: Cow<'static, str>,
    pub parameters: serde_json::Value,
}
