use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{
    agent::memory::{
        base::{
            MemoryInspection, MemorySnapshot, MemoryStats, MemoryStrategy, render_conversation,
        },
        buffer::BufferMemory,
        config::{BufferMemoryConfig, EntityMemoryConfig},
        token_counter::TokenCounter,
    },
    llm::{EmbeddingProvider, EmbeddingRequest, Provider},
    messages::{ArcMessage, CompletionRequest, Message, MessageRole},
    utils::math::cosine_similarity,
};

/// Structured representation of an extracted entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    pub id: Uuid,
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub embedding: Vec<f32>,
    pub importance: f32,
    pub last_updated: DateTime<Utc>,
}

impl EntityRecord {
    fn merge(&mut self, incoming: EntityCandidate, embedding: Vec<f32>) {
        // Update with the most recent description while preserving key facts.
        if incoming.description.len() > self.description.len() {
            self.description = incoming.description;
        }
        self.embedding = embedding;
        self.importance = self.importance.max(incoming.importance);
        self.last_updated = Utc::now();
    }

    fn to_line(&self) -> String {
        format!(
            "{} ({}) - {}",
            self.name.trim(),
            self.entity_type.trim(),
            self.description.trim()
        )
    }
}

/// Candidate entity extracted from conversation messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCandidate {
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub importance: f32,
}

impl EntityCandidate {
    #[must_use]
    pub fn key(&self) -> String {
        format!(
            "{}::{}",
            self.name.to_lowercase().trim(),
            self.entity_type.to_lowercase().trim()
        )
    }

    #[must_use]
    pub fn embed_text(&self) -> String {
        format!(
            "{} ({}) - {}",
            self.name.trim(),
            self.entity_type.trim(),
            self.description.trim()
        )
    }
}

/// Extract entities from a batch of messages.
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    async fn extract(&self, messages: &[Message]) -> Result<Vec<EntityCandidate>>;
}

/// Memory strategy that tracks entities mentioned in conversation and injects relevant context.
pub struct EntityMemory {
    config: EntityMemoryConfig,
    token_counter: Arc<dyn TokenCounter>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    extractor: Arc<dyn EntityExtractor>,
    buffer: BufferMemory,
    entities: Vec<EntityRecord>,
    /// Store `ArcMessage` to avoid cloning until extraction is needed
    pending_messages: Vec<ArcMessage>,
    last_updated: Option<DateTime<Utc>>,
}

impl EntityMemory {
    /// Create a new entity memory implementation.
    pub fn new(
        config: EntityMemoryConfig,
        token_counter: Arc<dyn TokenCounter>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        extractor: Arc<dyn EntityExtractor>,
    ) -> Self {
        let buffer_config = BufferMemoryConfig {
            max_tokens: config.max_tokens,
            ..BufferMemoryConfig::default()
        };

        Self {
            config,
            token_counter: Arc::clone(&token_counter),
            embedding_provider,
            extractor,
            buffer: BufferMemory::new(buffer_config, token_counter),
            entities: Vec::new(),
            pending_messages: Vec::new(),
            last_updated: None,
        }
    }

    /// Create entity memory from forked state (same config, forked buffer, cloned entities).
    pub fn from_fork(
        config: EntityMemoryConfig,
        token_counter: Arc<dyn TokenCounter>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        extractor: Arc<dyn EntityExtractor>,
        buffer: BufferMemory,
        entities: Vec<EntityRecord>,
        last_updated: Option<DateTime<Utc>>,
    ) -> Self {
        Self {
            config,
            token_counter,
            embedding_provider,
            extractor,
            buffer,
            entities,
            pending_messages: Vec::new(),
            last_updated,
        }
    }

    fn entity_map(&self) -> HashMap<String, usize> {
        self.entities
            .iter()
            .enumerate()
            .map(|(index, entity)| {
                (
                    format!(
                        "{}::{}",
                        entity.name.to_lowercase(),
                        entity.entity_type.to_lowercase()
                    ),
                    index,
                )
            })
            .collect()
    }

    fn build_entity_context(&self) -> Option<Message> {
        if self.entities.is_empty() {
            return None;
        }

        // Create sorted indices instead of cloning entire entities
        let mut sorted_indices: Vec<usize> = (0..self.entities.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.entities[b]
                .last_updated
                .cmp(&self.entities[a].last_updated)
        });

        let mut context = String::from("Known entities and facts:\n");
        let mut tokens = self.token_counter.estimate_text_tokens(&context);

        let mut lines_added = 0usize;

        for &idx in sorted_indices.iter().take(self.config.max_entities) {
            let entity = &self.entities[idx];
            let line = format!("- {}\n", entity.to_line());
            let line_tokens = self.token_counter.estimate_text_tokens(&line);

            if tokens.saturating_add(line_tokens) > self.config.max_tokens {
                break;
            }

            context.push_str(&line);
            tokens = tokens.saturating_add(line_tokens);
            lines_added += 1;
        }

        if lines_added == 0 {
            return None;
        }

        Some(Message::new(
            MessageRole::System,
            context.trim().to_string(),
        ))
    }

    async fn process_pending(&mut self) -> Result<()> {
        if self.pending_messages.is_empty() {
            return Ok(());
        }

        // Convert ArcMessages to Messages only when needed for extraction
        let batch: Vec<Message> = self
            .pending_messages
            .iter()
            .map(|arc| (**arc).clone())
            .collect();
        std::mem::take(&mut self.pending_messages); // Clear pending after extraction
        let candidates = self.extractor.extract(&batch).await?;

        if candidates.is_empty() {
            return Ok(());
        }

        let embeddings = self
            .embedding_provider
            .embed(EmbeddingRequest {
                texts: candidates.iter().map(EntityCandidate::embed_text).collect(),
                model: None,
            })
            .await?;

        let mut map = self.entity_map();

        for (candidate, embedding) in candidates.into_iter().zip(embeddings.embeddings) {
            let key = candidate.key();
            if let Some(index) = map.get(&key).copied() {
                if let Some(existing) = self.entities.get_mut(index) {
                    existing.merge(candidate, embedding);
                }
            } else {
                // Enforce max_entities at insertion time: evict least recently updated if at cap.
                if self.entities.len() >= self.config.max_entities
                    && let Some((idx, _)) = self
                        .entities
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.last_updated)
                {
                    let evicted = self.entities.remove(idx);
                    let evicted_key = format!(
                        "{}::{}",
                        evicted.name.to_lowercase().trim(),
                        evicted.entity_type.to_lowercase().trim()
                    );
                    map.remove(&evicted_key);
                    for (_, i) in &mut map {
                        if *i > idx {
                            *i -= 1;
                        }
                    }
                }
                let record = EntityRecord {
                    id: Uuid::now_v7(),
                    name: candidate.name,
                    entity_type: candidate.entity_type,
                    description: candidate.description,
                    embedding,
                    importance: candidate.importance,
                    last_updated: Utc::now(),
                };
                self.entities.push(record);
                map.insert(key, self.entities.len() - 1);
            }
        }

        self.entities
            .sort_by(|a, b| b.last_updated.cmp(&a.last_updated));

        Ok(())
    }

    fn total_token_estimate(&self, context: &[Message]) -> usize {
        self.token_counter.estimate_messages_tokens(context)
    }

    fn trim_context_to_budget(&self, context: &mut Vec<Message>) {
        while !context.is_empty() && self.total_token_estimate(context) > self.config.max_tokens {
            let removal_index = match context.first() {
                Some(message)
                    if matches!(message.role, MessageRole::System) && context.len() > 1 =>
                {
                    1
                }
                _ => 0,
            };

            if context.len() > removal_index {
                context.remove(removal_index);
            } else {
                break;
            }
        }
    }
}

#[async_trait]
impl MemoryStrategy for EntityMemory {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        // Convert to ArcMessage once, then share between buffer and pending
        let arc_messages: Vec<ArcMessage> = messages.into_iter().map(Arc::new).collect();

        // Clone Arcs (cheap) for buffer - BufferMemory will convert to ArcMessage internally anyway
        // We still need to clone here because BufferMemory's add_messages takes Vec<Message>
        // but we avoid double-cloning by reusing the ArcMessages
        let buffer_messages: Vec<Message> =
            arc_messages.iter().map(|arc| (**arc).clone()).collect();
        self.buffer.add_messages(buffer_messages).await?;

        // Store Arc refs for pending (no additional message clone - reuse ArcMessages)
        self.pending_messages.extend(arc_messages);

        if self.pending_messages.len() >= self.config.extraction_interval {
            self.process_pending().await?;
        }

        self.last_updated = Some(Utc::now());

        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        let mut context = Vec::new();

        if let Some(entity_message) = self.build_entity_context() {
            context.push(entity_message);
        }

        // Get buffer context and append (move instead of clone where possible)
        let buffer_context = self.buffer.get_context().await?;
        context.extend(buffer_context);

        if self.total_token_estimate(&context) > self.config.max_tokens {
            self.trim_context_to_budget(&mut context);
        }

        Ok(context)
    }

    async fn clear(&mut self) -> Result<()> {
        self.buffer.clear().await?;
        self.entities.clear();
        self.pending_messages.clear();
        self.last_updated = Some(Utc::now());
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        let buffer_stats = self.buffer.stats();
        MemoryStats {
            total_messages: buffer_stats.total_messages,
            total_tokens_estimated: buffer_stats.total_tokens_estimated,
            total_summaries: buffer_stats.total_summaries,
            total_entities: self.entities.len(),
            last_updated: self.last_updated.or(buffer_stats.last_updated),
        }
    }

    async fn save(
        &self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(store) = persistence {
            let snapshot = self.save_state().await?;
            store.save_snapshot(conversation_id, &snapshot).await?;
        }

        Ok(())
    }

    async fn load(
        &mut self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(store) = persistence
            && let Some(snapshot) = store.load_snapshot(conversation_id).await?
        {
            self.load_state(snapshot).await?;
        }

        Ok(())
    }

    async fn save_state(&self) -> Result<MemorySnapshot> {
        let mut metadata = Map::new();
        metadata.insert(
            "entities".to_string(),
            serde_json::to_value(&self.entities)?,
        );
        let pending_messages: Vec<Message> = self
            .pending_messages
            .iter()
            .map(|arc| (**arc).clone())
            .collect();
        metadata.insert(
            "pending_messages".to_string(),
            serde_json::to_value(&pending_messages)?,
        );
        if let Some(summary) = self.buffer.summary() {
            metadata.insert("summary".to_string(), serde_json::to_value(summary)?);
        }
        if let Some(updated) = self.last_updated {
            metadata.insert(
                "last_updated".to_string(),
                Value::String(updated.to_rfc3339()),
            );
        }
        Ok(MemorySnapshot {
            messages: self.buffer.messages(),
            metadata: Value::Object(metadata),
        })
    }

    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        self.entities = snapshot
            .metadata
            .get("entities")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();

        let pending_messages: Vec<Message> = snapshot
            .metadata
            .get("pending_messages")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        self.pending_messages = pending_messages.into_iter().map(Arc::new).collect();

        if let Some(updated) = snapshot
            .metadata
            .get("last_updated")
            .and_then(|value| value.as_str())
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        {
            self.last_updated = Some(updated.with_timezone(&Utc));
        }

        let summary_message = snapshot
            .metadata
            .get("summary")
            .and_then(|value| serde_json::from_value::<Message>(value.clone()).ok());

        self.buffer.clear().await?;
        if !snapshot.messages.is_empty() {
            self.buffer.add_messages(snapshot.messages).await?;
        }
        self.buffer.set_summary_message(summary_message);
        Ok(())
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        let entities: Vec<String> = self.entities.iter().map(EntityRecord::to_line).collect();
        Ok(MemoryInspection {
            strategy_name: "entity".to_string(),
            messages: self.get_context().await?,
            summaries: vec![],
            entities,
            knowledge: vec![],
            stats: self.stats(),
        })
    }

    async fn search(&self, query: &str) -> Result<Vec<(Message, f32)>> {
        if query.is_empty() {
            return self
                .get_context()
                .await
                .map(|ms| ms.into_iter().map(|m| (m, 1.0)).collect());
        }
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
        let mut scored: Vec<(Message, f32)> = self
            .entities
            .iter()
            .filter_map(|e| {
                let sim = cosine_similarity(&query_embedding, &e.embedding);
                if sim >= self.config.similarity_threshold {
                    let msg = Message::new(
                        MessageRole::System,
                        format!("{} ({}) - {}", e.name, e.entity_type, e.description),
                    );
                    Some((msg, sim))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if scored.is_empty() {
            return self
                .get_context()
                .await
                .map(|ms| ms.into_iter().map(|m| (m, 1.0)).collect());
        }
        Ok(scored)
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        let buffer = self.buffer.fork_concrete().await?;
        let forked = Self::from_fork(
            self.config.clone(),
            Arc::clone(&self.token_counter),
            Arc::clone(&self.embedding_provider),
            Arc::clone(&self.extractor),
            buffer,
            self.entities.clone(),
            self.last_updated,
        );
        Ok(Box::new(forked))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

const ENTITY_EXTRACTION_SYSTEM_PROMPT: &str = "You identify and track entities mentioned in conversations. Extract people, organizations, locations, facts, and key concepts. Return JSON with an \"entities\" array. Each entity must include: name, entity_type, description, and importance (0.0-1.0).";

/// LLM-backed entity extractor that produces structured entity candidates.
pub struct ProviderEntityExtractor {
    provider: Arc<dyn Provider>,
    max_tokens: u32,
    temperature: f32,
}

impl ProviderEntityExtractor {
    pub fn new(provider: Arc<dyn Provider>, max_tokens: u32) -> Self {
        Self {
            provider,
            max_tokens: max_tokens.clamp(256, 1024),
            temperature: 0.2,
        }
    }
}

#[derive(Debug, Deserialize)]
struct EntityExtractionEnvelope {
    #[serde(default)]
    entities: Vec<EntityExtractionItem>,
}

#[derive(Debug, Deserialize)]
struct EntityExtractionItem {
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default, alias = "type", alias = "entity_type")]
    entity_type: Option<String>,
    #[serde(default)]
    importance: Option<f32>,
}

#[async_trait]
impl EntityExtractor for ProviderEntityExtractor {
    async fn extract(&self, messages: &[Message]) -> Result<Vec<EntityCandidate>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let prompt_messages = vec![
            Message::system(ENTITY_EXTRACTION_SYSTEM_PROMPT),
            Message::user(render_conversation(messages)),
        ];

        let request = CompletionRequest {
            messages: Arc::new(prompt_messages),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = self.provider.complete(request).await?;
        // Use as_text() to avoid unnecessary clone
        let raw = response.content.as_text().to_string();

        let envelope: EntityExtractionEnvelope = if let Ok(parsed) = serde_json::from_str(&raw) {
            parsed
        } else {
            let wrapped = format!("{{\"entities\":{raw}}}");
            match serde_json::from_str(&wrapped) {
                Ok(parsed) => parsed,
                Err(err) => {
                    tracing::warn!("Failed to parse entity extraction response: {err}");
                    return Ok(Vec::new());
                }
            }
        };

        let mut candidates = Vec::new();
        for entity in envelope.entities {
            let name = entity.name.trim();
            let description = entity.description.trim();
            if name.is_empty() || description.is_empty() {
                continue;
            }

            let entity_type = entity
                .entity_type
                .unwrap_or_else(|| "unknown".to_string())
                .trim()
                .to_string();

            let importance = entity.importance.unwrap_or(0.5).clamp(0.0, 1.0);

            candidates.push(EntityCandidate {
                name: name.to_string(),
                entity_type,
                description: description.to_string(),
                importance,
            });
        }

        Ok(candidates)
    }
}
