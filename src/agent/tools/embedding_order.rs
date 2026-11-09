//! Optional embedding-based tool schema ordering.
//! When an embedding provider is set, tools are ranked by similarity to the user query.

use crate::llm::{EmbeddingProvider, EmbeddingRequest};
use crate::messages::ToolSchema;
use crate::utils::math::cosine_similarity;
use std::sync::Arc;

/// Order tool schemas by embedding similarity to the query.
/// Embeds the query and each tool description, then ranks by cosine similarity.
/// On failure (e.g. embedding API error), returns the original order.
pub async fn order_tool_schemas_by_embedding_relevance(
    schemas: &Arc<Vec<ToolSchema>>,
    query: &str,
    embedding_provider: &(dyn EmbeddingProvider + Send + Sync),
) -> anyhow::Result<Arc<Vec<ToolSchema>>> {
    if query.trim().is_empty() || schemas.is_empty() {
        return Ok(Arc::clone(schemas));
    }

    let texts: Vec<String> = std::iter::once(query.to_string())
        .chain(
            schemas
                .iter()
                .map(|s| format!("{}: {}", s.name.as_ref(), s.description.as_ref())),
        )
        .collect();

    let response = embedding_provider
        .embed(EmbeddingRequest {
            texts: texts.clone(),
            model: None,
        })
        .await?;

    let embeddings = response.embeddings;
    if embeddings.len() != texts.len() || embeddings.is_empty() {
        return Ok(Arc::clone(schemas));
    }

    let query_emb = &embeddings[0];
    let mut indices: Vec<(f32, usize)> = embeddings[1..]
        .iter()
        .enumerate()
        .map(|(i, emb)| (cosine_similarity(query_emb, emb), i))
        .collect();

    indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let is_same_order = indices
        .iter()
        .enumerate()
        .all(|(pos, (_, idx))| pos == *idx);
    if is_same_order {
        return Ok(Arc::clone(schemas));
    }

    let out: Vec<ToolSchema> = indices
        .iter()
        .map(|(_, idx)| schemas[*idx].clone())
        .collect();
    Ok(Arc::new(out))
}
