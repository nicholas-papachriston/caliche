//! Query result formatters for RAG tool (shared between query and `advanced_query` operations).

use crate::agent::tools::ToolResult;
use crate::agent::tools::context_manager::ContextStats;
use crate::messages::TokenUsage;
use serde_json::json;

/// Build `ToolResult` from JSON content with optional usage and warnings.
pub fn tool_result_from_usage_warnings(
    content: String,
    usage: Option<TokenUsage>,
    warnings: Vec<String>,
) -> ToolResult {
    match (usage, warnings.is_empty()) {
        (Some(u), true) => ToolResult::success_with_usage(content, u),
        (Some(u), false) => ToolResult::success_with_usage_and_warnings(content, u, warnings),
        (None, true) => ToolResult::success(content),
        (None, false) => ToolResult::success_with_warnings(content, warnings),
    }
}

/// Add `embedding_tokens_used` and `context_stats` to a result JSON object.
pub fn add_usage_and_context_to_json(
    result: &mut serde_json::Value,
    usage: Option<&TokenUsage>,
    context_stats: Option<&ContextStats>,
) {
    if let Some(usage_val) = usage {
        result["embedding_tokens_used"] = json!(usage_val.embedding_tokens);
    }
    if let Some(stats) = context_stats {
        result["context_stats"] = json!({
            "total_chunks": stats.total_chunks,
            "included_chunks": stats.included_chunks,
            "omitted_chunks": stats.omitted_chunks,
            "truncated": stats.truncated
        });
    }
}

/// Format a "no results" query response and return as `ToolResult`.
#[allow(clippy::too_many_arguments)]
pub fn format_no_results(
    query: &str,
    top_similarity: Option<f32>,
    min_similarity: f32,
    message_suffix: &str,
    usage: Option<TokenUsage>,
    warnings: Vec<String>,
    context_stats: Option<&ContextStats>,
    extra: &serde_json::Value,
) -> ToolResult {
    let debug_info = if let Some(top_sim) = top_similarity {
        format!(" (highest similarity was {top_sim:.3}, below threshold {min_similarity:.3})")
    } else {
        String::new()
    };
    let mut result = json!({
        "status": "no_results",
        "message": format!(
            "No relevant documents found for query: '{}'{}. {}",
            query, debug_info, message_suffix
        ),
        "results_count": 0,
        "top_similarity": top_similarity
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            result[k.as_str()] = v.clone();
        }
    }
    if !warnings.is_empty() {
        result["warnings"] = json!(warnings);
    }
    add_usage_and_context_to_json(&mut result, usage.as_ref(), context_stats);
    tool_result_from_usage_warnings(result.to_string(), usage, warnings)
}

/// Format a successful query response (with results) and return as `ToolResult`.
#[allow(clippy::too_many_arguments)]
pub fn format_query_success(
    formatted_results: Vec<String>,
    query: &str,
    top_k: usize,
    min_similarity: f32,
    usage: Option<TokenUsage>,
    warnings: Vec<String>,
    context_stats: Option<&ContextStats>,
    extra: &serde_json::Value,
) -> ToolResult {
    let mut result = json!({
        "status": "success",
        "results": formatted_results,
        "results_count": formatted_results.len(),
        "query": query,
        "top_k": top_k,
        "min_similarity": min_similarity
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            result[k.as_str()] = v.clone();
        }
    }
    if !warnings.is_empty() {
        result["warnings"] = json!(warnings);
    }
    add_usage_and_context_to_json(&mut result, usage.as_ref(), context_stats);
    tool_result_from_usage_warnings(result.to_string(), usage, warnings)
}
