//! Shared utilities for tool schema conversion across LLM providers.

use crate::llm::cache::ConversionCache;
use std::sync::Arc;

/// Convert a tool schema to the standard JSON object format used by most providers.
/// Returns `{"type": ..., "properties": ..., "required": ...}`.
#[must_use]
pub fn tool_schema_to_json_value(tool: &crate::messages::ToolSchema) -> serde_json::Value {
    serde_json::json!({
        "type": tool.parameters.schema_type,
        "properties": tool.parameters.properties,
        "required": tool.parameters.required,
    })
}

/// Cache tool schema conversions keyed by tool list hash.
///
/// Providers store a `ConversionCache<Vec<T>>` and pass a per-provider mapping
/// closure. Cache hits return `Arc` without recomputing or cloning the tool list.
pub fn cached_tool_mapping<T, F>(
    cache: &ConversionCache<Vec<T>>,
    tools: &[crate::messages::ToolSchema],
    compute: F,
) -> Arc<Vec<T>>
where
    F: FnOnce(&[crate::messages::ToolSchema]) -> Vec<T>,
    T: Send + Sync,
{
    cache.get_or_compute_with_hash(tools, crate::llm::cache::hash_tools_for_cache, compute)
}
