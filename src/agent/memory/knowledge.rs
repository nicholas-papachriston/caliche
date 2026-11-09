//! Cross-session knowledge store for persistent semantic memory.

use crate::agent::memory::base::{MemoryPersistence, MemorySnapshot};
use crate::agent::memory::token_counter::TokenCounter;
use crate::llm::{EmbeddingProvider, EmbeddingRequest};
use crate::messages::Message;
use crate::utils::math::cosine_similarity;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Knowledge entry representing a fact or piece of information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// Unique identifier
    pub id: String,
    /// The knowledge content
    pub content: String,
    /// Category or type of knowledge
    pub category: String,
    /// Embedding vector for semantic search
    pub embedding: Option<Vec<f32>>,
    /// Importance score (0.0 - 1.0)
    pub importance: f32,
    /// Number of times this knowledge has been accessed
    pub access_count: usize,
    /// When this knowledge was first learned
    pub learned_at: DateTime<Utc>,
    /// Last time this knowledge was accessed
    pub last_accessed: DateTime<Utc>,
    /// Source conversation or context
    pub source: Option<String>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry.
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        category: impl Into<String>,
        importance: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            content: content.into(),
            category: category.into(),
            embedding: None,
            importance: importance.clamp(0.0f32, 1.0f32),
            access_count: 0,
            learned_at: now,
            last_accessed: now,
            source: None,
            metadata: serde_json::json!({}),
        }
    }

    /// Record an access to this knowledge.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Utc::now();
    }
}

/// Knowledge store for cross-session persistent memory.
pub struct KnowledgeStore {
    entries: Vec<KnowledgeEntry>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    token_counter: Arc<dyn TokenCounter>,
    max_entries: usize,
}

impl KnowledgeStore {
    /// Create a new knowledge store.
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        token_counter: Arc<dyn TokenCounter>,
        max_entries: usize,
    ) -> Self {
        Self {
            entries: Vec::new(),
            embedding_provider,
            token_counter,
            max_entries,
        }
    }

    /// Add knowledge from a conversation message.
    pub async fn learn_from_message(
        &mut self,
        message: &Message,
        category: String,
        importance: f32,
        source: Option<String>,
    ) -> Result<String> {
        let content = message.content.extract_text();
        if content.is_empty() {
            return Err(anyhow::anyhow!("Message content is empty"));
        }

        let id = format!("knowledge:{}", uuid::Uuid::now_v7());
        let mut entry = KnowledgeEntry::new(id.clone(), content, category, importance);
        entry.source = source;

        // Generate embedding
        let embedding_response = self
            .embedding_provider
            .embed(EmbeddingRequest {
                texts: vec![entry.content.clone()],
                model: None,
            })
            .await?;

        if let Some(embedding) = embedding_response.embeddings.first() {
            entry.embedding = Some(embedding.clone());
        }

        self.entries.push(entry);
        self.enforce_limits();

        Ok(id)
    }

    /// Extract and learn key facts from messages.
    pub async fn extract_and_learn(
        &mut self,
        messages: &[Message],
        source: Option<String>,
    ) -> Result<Vec<String>> {
        let mut learned_ids = Vec::new();

        for message in messages {
            // Simple heuristic: system messages and assistant messages with facts
            let should_extract = matches!(
                message.role,
                crate::messages::MessageRole::System | crate::messages::MessageRole::Assistant
            );

            if should_extract {
                let content = message.content.extract_text();
                // Extract if message contains factual information
                if self.is_factual(&content) {
                    let category = self.categorize_content(&content);
                    let importance = self.estimate_importance(&content);
                    let id = self
                        .learn_from_message(message, category, importance, source.clone())
                        .await?;
                    learned_ids.push(id);
                }
            }
        }

        Ok(learned_ids)
    }

    /// Search for relevant knowledge by semantic similarity.
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> Result<Vec<(&KnowledgeEntry, f32)>> {
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
            .ok_or_else(|| anyhow::anyhow!("No embedding generated"))?
            .clone();

        // Score all entries
        let mut scored: Vec<(&KnowledgeEntry, f32)> = self
            .entries
            .iter()
            .filter_map(|entry| {
                entry.embedding.as_ref().map(|emb| {
                    let similarity = cosine_similarity(&query_embedding, emb);
                    (entry, similarity)
                })
            })
            .filter(|(_, score)| *score >= min_similarity)
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        scored.truncate(top_k);

        // Record access would require interior mutability
        // For now, we skip recording access during search to avoid borrowing issues
        let _ = scored.len(); // Use scored to avoid unused variable warning

        Ok(scored)
    }

    /// Search for relevant knowledge, trimming results to fit within a token budget.
    pub async fn search_budgeted(
        &self,
        query: &str,
        top_k: usize,
        min_similarity: f32,
        max_tokens: usize,
    ) -> Result<Vec<(&KnowledgeEntry, f32)>> {
        let all = self.search(query, top_k, min_similarity).await?;
        let mut budget_remaining = max_tokens;
        let mut results = Vec::new();
        for (entry, score) in all {
            let tokens = self.token_counter.estimate_text_tokens(&entry.content);
            if tokens > budget_remaining {
                break;
            }
            budget_remaining -= tokens;
            results.push((entry, score));
        }
        Ok(results)
    }

    /// Get knowledge entries by category.
    #[must_use]
    pub fn get_by_category(&self, category: &str) -> Vec<&KnowledgeEntry> {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Get all entries.
    #[must_use]
    pub fn all_entries(&self) -> &[KnowledgeEntry] {
        &self.entries
    }

    /// Remove an entry by ID.
    pub fn remove(&mut self, id: &str) -> Option<KnowledgeEntry> {
        let pos = self.entries.iter().position(|e| e.id == id)?;
        Some(self.entries.remove(pos))
    }

    /// Clear all knowledge.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Serialize all entries to a `MemorySnapshot` for persistence.
    pub fn to_snapshot(&self) -> Result<MemorySnapshot> {
        Ok(MemorySnapshot {
            messages: vec![],
            metadata: serde_json::to_value(&self.entries)?,
        })
    }

    /// Restore entries from a persisted snapshot.
    pub fn from_snapshot(&mut self, snapshot: &MemorySnapshot) -> Result<()> {
        if !snapshot.metadata.is_null() {
            self.entries = serde_json::from_value(snapshot.metadata.clone())?;
        }
        Ok(())
    }

    /// Save to a `MemoryPersistence` backend.
    pub async fn save(&self, persistence: &dyn MemoryPersistence, key: &str) -> Result<()> {
        let snapshot = self.to_snapshot()?;
        persistence.save_snapshot(key, &snapshot).await
    }

    /// Load from a `MemoryPersistence` backend.
    pub async fn load(&mut self, persistence: &dyn MemoryPersistence, key: &str) -> Result<()> {
        if let Some(snapshot) = persistence.load_snapshot(key).await? {
            self.from_snapshot(&snapshot)?;
        }
        Ok(())
    }

    /// Enforce maximum entry limit by removing least important/accessed entries.
    fn enforce_limits(&mut self) {
        if self.entries.len() <= self.max_entries {
            return;
        }

        // Sort by importance and access count
        self.entries.sort_by(|a, b| {
            let score_a = (a.access_count as f32).mul_add(0.01, a.importance);
            let score_b = (b.access_count as f32).mul_add(0.01, b.importance);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top entries
        self.entries.truncate(self.max_entries);
    }

    /// Check if content plausibly contains a factual assertion.
    ///
    /// Short-circuits on trivial inputs (too short, pure questions) before
    /// scanning tokens for declarative cue words using whole-word matching.
    fn is_factual(&self, content: &str) -> bool {
        const MIN_LEN: usize = 20;
        let trimmed = content.trim();
        if trimmed.len() < MIN_LEN {
            return false;
        }
        // A lone question almost never yields a fact worth memorizing.
        if trimmed.ends_with('?') && !trimmed.contains('.') {
            return false;
        }

        const FACTUAL_CUES: &[&str] = &[
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "fact",
            "facts",
            "knows",
            "known",
            "remember",
            "note",
            "because",
            "therefore",
        ];
        iter_lowercase_words(trimmed).any(|word| FACTUAL_CUES.contains(&word.as_str()))
    }

    /// Categorize content based on whole-word keyword matches.
    fn categorize_content(&self, content: &str) -> String {
        const PERSON: &[&str] = &["user", "person", "name", "they", "he", "she"];
        const ORG: &[&str] = &["company", "organization", "team", "corp", "inc"];
        const LOCATION: &[&str] = &["location", "place", "city", "country", "address"];
        const PREFERENCE: &[&str] = &[
            "preference",
            "like",
            "likes",
            "dislike",
            "dislikes",
            "favorite",
        ];

        let mut person = false;
        let mut org = false;
        let mut location = false;
        let mut preference = false;

        for word in iter_lowercase_words(content) {
            if !person && PERSON.contains(&word.as_str()) {
                person = true;
            } else if !org && ORG.contains(&word.as_str()) {
                org = true;
            } else if !location && LOCATION.contains(&word.as_str()) {
                location = true;
            } else if !preference && PREFERENCE.contains(&word.as_str()) {
                preference = true;
            }
            if person && org && location && preference {
                break;
            }
        }

        if preference {
            "preference".to_string()
        } else if location {
            "location".to_string()
        } else if org {
            "organization".to_string()
        } else if person {
            "person".to_string()
        } else {
            "general".to_string()
        }
    }

    /// Estimate importance based on content characteristics.
    fn estimate_importance(&self, content: &str) -> f32 {
        const HIGH_SIGNAL: &[&str] = &["important", "critical", "key", "must", "required"];
        const NOTE_SIGNAL: &[&str] = &["remember", "note", "warning", "caution"];

        let mut importance: f32 = 0.5;
        let mut high = false;
        let mut note = false;
        for word in iter_lowercase_words(content) {
            if !high && HIGH_SIGNAL.contains(&word.as_str()) {
                high = true;
            } else if !note && NOTE_SIGNAL.contains(&word.as_str()) {
                note = true;
            }
            if high && note {
                break;
            }
        }
        if high {
            importance += 0.3;
        }
        if note {
            importance += 0.2;
        }
        if content.len() > 100 {
            importance += 0.1;
        }

        importance.clamp(0.0f32, 1.0f32)
    }
}

/// Iterate over lowercase word tokens in `text`, keeping alphanumeric runs only.
/// Allocates one `String` per word, which is cheap compared to a full clone of
/// the document and avoids regex compilation overhead on this hot path.
fn iter_lowercase_words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_entry() {
        let entry = KnowledgeEntry::new(
            "id1".to_string(),
            "Test content".to_string(),
            "test".to_string(),
            0.8,
        );
        assert_eq!(entry.importance, 0.8);
        assert_eq!(entry.access_count, 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
    }
}
