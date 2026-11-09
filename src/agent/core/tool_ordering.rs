//! Shared tool-schema ordering helper used by both the blocking `chat_loop`
//! path and the streaming `stream_chat` path so that native tool calls see
//! the same ordering heuristics regardless of response mode.

use crate::agent::tools::embedding_order::order_tool_schemas_by_embedding_relevance;
use crate::agent::tools::metadata::order_tool_schemas_by_relevance;
use crate::llm::EmbeddingProvider;
use crate::messages::ToolSchema;
use std::sync::Arc;

/// Tool-count threshold under which ordering is skipped (fewer than this and
/// the LLM is trivially going to consider all schemas).
pub const TOOL_ORDERING_THRESHOLD: usize = 3;

/// Order tool schemas by relevance to the query, preferring embedding-based
/// ordering if an embedding provider is supplied, falling back to the
/// lexical heuristic. Returns the input unchanged when at or below
/// [`TOOL_ORDERING_THRESHOLD`] or when the query is empty.
pub async fn order_tools(
    tool_schemas: &Arc<Vec<ToolSchema>>,
    query: &str,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
) -> Arc<Vec<ToolSchema>> {
    if tool_schemas.len() <= TOOL_ORDERING_THRESHOLD {
        return Arc::clone(tool_schemas);
    }
    let query_trimmed = query.trim();
    if query_trimmed.is_empty() {
        return Arc::clone(tool_schemas);
    }

    if let Some(emb) = embedding_provider {
        match order_tool_schemas_by_embedding_relevance(tool_schemas, query_trimmed, emb.as_ref())
            .await
        {
            Ok(ordered) => ordered,
            Err(err) => {
                tracing::debug!(
                    error = %err,
                    "embedding-based tool ordering failed; falling back to lexical heuristic"
                );
                order_tool_schemas_by_relevance(tool_schemas, query_trimmed)
            }
        }
    } else {
        order_tool_schemas_by_relevance(tool_schemas, query_trimmed)
    }
}
