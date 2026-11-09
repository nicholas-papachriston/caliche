//! Utilities for OpenAI-compatible providers.

use crate::messages::{MessageRole, ToolChoice};

/// Convert `MessageRole` to a static string slice for OpenAI-compatible APIs.
///
/// This is a zero-cost conversion that returns static string slices.
/// Most OpenAI-compatible APIs use the same role names: "system", "user", "assistant", "tool".
#[must_use]
pub const fn role_to_str(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Convert `ToolChoice` to OpenAI-compatible JSON format.
///
/// Returns a JSON value that can be used in OpenAI-compatible API requests.
/// Uses lowercase values: "auto", "none", "required" for the standard choices.
#[must_use]
pub fn convert_tool_choice_openai(tool_choice: &ToolChoice) -> Option<serde_json::Value> {
    match tool_choice {
        ToolChoice::Auto => Some(serde_json::json!("auto")),
        ToolChoice::None => Some(serde_json::json!("none")),
        ToolChoice::Required => Some(serde_json::json!("required")),
        ToolChoice::Specific(name) => Some(serde_json::json!({
            "type": "function",
            "function": { "name": name }
        })),
    }
}

/// Convert `ToolChoice` to Anthropic API format.
///
/// Only includes a value when tools are present; Auto is omitted (API default).
/// Returns object shape: `{ type: "none" }`, `{ type: "any" }`, or `{ type: "tool", name }`.
#[must_use]
pub fn convert_tool_choice_anthropic(
    tool_choice: Option<&ToolChoice>,
    has_tools: bool,
) -> Option<serde_json::Value> {
    if !has_tools {
        return None;
    }
    tool_choice.and_then(|choice| match choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(serde_json::json!({"type": "none"})),
        ToolChoice::Required => Some(serde_json::json!({"type": "any"})),
        ToolChoice::Specific(name) => Some(serde_json::json!({
            "type": "tool",
            "name": name
        })),
    })
}

/// Convert `ToolChoice` to Cohere-compatible JSON format.
///
/// Returns a JSON value that can be used in Cohere API requests.
/// Uses uppercase values: "AUTO", "NONE", "ANY" for the standard choices.
#[must_use]
pub fn convert_tool_choice_cohere(tool_choice: &ToolChoice) -> Option<serde_json::Value> {
    match tool_choice {
        ToolChoice::Auto => Some(serde_json::json!("AUTO")),
        ToolChoice::None => Some(serde_json::json!("NONE")),
        ToolChoice::Required => Some(serde_json::json!("ANY")),
        ToolChoice::Specific(name) => Some(serde_json::json!({
            "type": "function",
            "name": name
        })),
    }
}
