//! Query-aware context selection and relevance scoring.

use crate::agent::memory::token_counter::TokenCounter;
use crate::llm::{EmbeddingProvider, EmbeddingRequest};
use crate::messages::Message;
use crate::utils::math::cosine_similarity;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Relevance score for a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceScore {
    /// Overall relevance score (0.0 - 1.0)
    pub score: f32,
    /// Semantic similarity score
    pub semantic_score: f32,
    /// Recency score (newer messages score higher)
    pub recency_score: f32,
    /// Importance score (based on message content)
    pub importance_score: f32,
    /// Message timestamp
    pub timestamp: Option<DateTime<Utc>>,
}

impl RelevanceScore {
    /// Create a new relevance score.
    #[must_use]
    pub fn new(
        semantic_score: f32,
        recency_score: f32,
        importance_score: f32,
        timestamp: Option<DateTime<Utc>>,
    ) -> Self {
        // Weighted combination: semantic (50%), recency (30%), importance (20%)
        let score = importance_score.mul_add(0.2, (semantic_score * 0.5) + (recency_score * 0.3));

        Self {
            score: score.clamp(0.0f32, 1.0f32),
            semantic_score: semantic_score.clamp(0.0f32, 1.0f32),
            recency_score: recency_score.clamp(0.0f32, 1.0f32),
            importance_score: importance_score.clamp(0.0f32, 1.0f32),
            timestamp,
        }
    }
}

/// Scored message with relevance information.
#[derive(Debug, Clone)]
pub struct ScoredMessage {
    pub message: Message,
    pub score: RelevanceScore,
}

/// Relevance scorer for messages.
#[derive(Clone)]
pub struct RelevanceScorer {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    pub token_counter: Arc<dyn TokenCounter>,
    /// Decay factor for recency (higher = faster decay)
    pub recency_decay_factor: f32,
    /// Minimum relevance threshold for inclusion
    pub min_relevance_threshold: f32,
}

impl RelevanceScorer {
    /// Create a new relevance scorer.
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        token_counter: Arc<dyn TokenCounter>,
    ) -> Self {
        Self {
            embedding_provider,
            token_counter,
            recency_decay_factor: 0.1,
            min_relevance_threshold: 0.3,
        }
    }

    /// Score messages for relevance to a query.
    pub async fn score_messages(
        &self,
        messages: &[Message],
        query: &str,
    ) -> Result<Vec<ScoredMessage>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        // Get query embedding
        let query_embedding = self
            .embedding_provider
            .embed(EmbeddingRequest {
                texts: vec![query.to_string()],
                model: None,
            })
            .await?
            .embeddings
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated for query"))?
            .clone();

        // Score each message
        let mut scored = Vec::with_capacity(messages.len());
        let now = Utc::now();
        let query_embedding_vec: Vec<f32> = query_embedding.clone();

        for message in messages {
            let semantic_score = self
                .compute_semantic_similarity(message, &query_embedding_vec)
                .await?;
            let recency_score = self.compute_recency_score(message, now);
            let importance_score = self.compute_importance_score(message);

            let timestamp = message.metadata.as_ref().and_then(|m| {
                m.get("timestamp")
                    .and_then(|v| v.as_str())
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc))
            });

            let relevance =
                RelevanceScore::new(semantic_score, recency_score, importance_score, timestamp);

            scored.push(ScoredMessage {
                message: message.clone(),
                score: relevance,
            });
        }

        Ok(scored)
    }

    /// Select top-k most relevant messages within token budget.
    pub async fn select_relevant(
        &self,
        messages: &[Message],
        query: &str,
        max_tokens: usize,
        top_k: Option<usize>,
    ) -> Result<Vec<Message>> {
        let scored = self.score_messages(messages, query).await?;

        // Filter by minimum relevance threshold
        let filtered: Vec<_> = scored
            .into_iter()
            .filter(|sm| sm.score.score >= self.min_relevance_threshold)
            .collect();

        // Sort by relevance score (descending)
        let mut sorted = filtered;
        sorted.sort_by(|a, b| {
            b.score
                .score
                .partial_cmp(&a.score.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply top_k limit if specified
        let candidates = if let Some(k) = top_k {
            sorted.into_iter().take(k).collect()
        } else {
            sorted
        };

        // Select messages within token budget
        let mut selected = Vec::new();
        let mut total_tokens = 0;

        for scored_msg in candidates {
            let msg_tokens = self
                .token_counter
                .estimate_message_tokens(&scored_msg.message);
            if total_tokens + msg_tokens <= max_tokens {
                selected.push(scored_msg.message);
                total_tokens += msg_tokens;
            } else {
                break;
            }
        }

        Ok(selected)
    }

    /// Compute semantic similarity between message and query embedding.
    async fn compute_semantic_similarity(
        &self,
        message: &Message,
        query_embedding: &[f32],
    ) -> Result<f32> {
        let message_text = message.content.extract_text();
        if message_text.is_empty() {
            return Ok(0.0);
        }

        let message_text_clone = message_text.clone();
        let embedding_response = self
            .embedding_provider
            .embed(EmbeddingRequest {
                texts: vec![message_text_clone],
                model: None,
            })
            .await?;

        let message_embedding = embedding_response
            .embeddings
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding generated for message"))?
            .clone();

        Ok(cosine_similarity(query_embedding, &message_embedding))
    }

    /// Compute recency score based on message age.
    fn compute_recency_score(&self, message: &Message, now: DateTime<Utc>) -> f32 {
        let timestamp = message.metadata.as_ref().and_then(|m| {
            m.get("timestamp")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc))
        });

        if let Some(ts) = timestamp {
            let age_seconds = (now - ts).num_seconds().max(0) as f32;
            // Exponential decay: score = e^(-decay_factor * age_hours)
            let age_hours = age_seconds / 3600.0;
            let score: f32 = (-self.recency_decay_factor * age_hours).exp();
            score.clamp(0.0f32, 1.0f32)
        } else {
            // No timestamp, assume recent
            0.8
        }
    }

    /// Compute importance score based on message content characteristics.
    fn compute_importance_score(&self, message: &Message) -> f32 {
        let text = message.content.extract_text();
        if text.is_empty() {
            return 0.0;
        }

        let mut score: f32 = 0.5; // Base importance

        // System messages are more important
        if matches!(message.role, crate::messages::MessageRole::System) {
            score += 0.3;
        }

        // Longer messages may contain more information
        let word_count = text.split_whitespace().count();
        if word_count > 100 {
            score += 0.1;
        }

        // Messages with structured content (tool calls) are important
        if matches!(
            message.content,
            crate::messages::MessageContent::Structured { .. }
        ) {
            score += 0.2;
        }

        // Check for important keywords
        let important_keywords = [
            "error",
            "important",
            "note",
            "remember",
            "key",
            "critical",
            "summary",
        ];
        let lower_text = text.to_lowercase();
        for keyword in &important_keywords {
            if lower_text.contains(keyword) {
                score += 0.05;
            }
        }

        score.clamp(0.0f32, 1.0f32)
    }
}

/// Cosine similarity between two vectors.
/// Returns 0.0 for empty slices, mismatched lengths, or zero-norm vectors (undefined angle).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevance_score_combination() {
        let score = RelevanceScore::new(0.8, 0.6, 0.4, None);
        assert!(score.score > 0.0 && score.score <= 1.0);
        assert_eq!(score.semantic_score, 0.8);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let c = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &c), 0.0);
    }
}
