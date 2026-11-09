//! Integration tests for memory strategies.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use caliche::llm::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use caliche::memory::{
    MemoryStrategy,
    buffer::BufferMemory,
    config::{BufferMemoryConfig, EntityMemoryConfig, SummaryMemoryConfig},
    entity::{EntityCandidate, EntityExtractor, EntityMemory},
    persistent::{BackendMemoryPersistence, PersistentMemory},
    summary::{ConversationSummarizer, SummaryMemory},
    token_counter::{ApproximateTokenCounter, TokenCounter},
};
use caliche::messages::{Message, MessageRole};
use tokio::sync::Mutex;

fn token_counter() -> Arc<dyn TokenCounter> {
    Arc::new(ApproximateTokenCounter {
        characters_per_token: 3.0,
        minimum_tokens: 1,
    })
}

fn msg(role: MessageRole, content: &str) -> Message {
    Message::new(role, content)
}

#[tokio::test]
async fn buffer_memory_eviction() {
    let config = BufferMemoryConfig {
        max_tokens: 20,
        min_messages: 1,
        preserve_system: true,
        summary_placeholder: false,
    };
    let mut memory = BufferMemory::new(config, token_counter());

    memory
        .add_messages(vec![
            msg(MessageRole::System, "Rules"),
            msg(MessageRole::User, "Message one"),
            msg(MessageRole::Assistant, "Message two"),
            msg(MessageRole::User, "Message three"),
        ])
        .await
        .expect("Adding messages should succeed");

    let context = memory
        .get_context()
        .await
        .expect("Getting context should succeed");

    // Should always keep system prompt plus the most recent turns within budget.
    assert!(
        context
            .iter()
            .any(|m| matches!(m.role, MessageRole::System))
    );
    assert!(context.last().is_some_and(
        |m| matches!(m.role, MessageRole::User) && m.content.extract_text() == "Message three"
    ));
    assert!(context.len() <= 4);
}

#[derive(Clone)]
struct StaticSummarizer;

#[async_trait]
impl ConversationSummarizer for StaticSummarizer {
    async fn summarize(&self, messages: &[Message]) -> Result<String> {
        let summary = messages
            .iter()
            .map(|m| m.content.extract_text())
            .collect::<Vec<_>>()
            .join(" | ");
        Ok(format!("SUMMARY: {summary}"))
    }
}

#[tokio::test]
async fn summary_memory_creates_summary() {
    let config = SummaryMemoryConfig {
        max_tokens: 60,
        max_summary_tokens: None,
        summarize_threshold_pct: 0.5,
        chunk_size: 1,
        max_summaries: 3,
    };
    let summarizer = Arc::new(StaticSummarizer);
    let mut memory = SummaryMemory::new(config, token_counter(), summarizer);

    let long_text = "This is a long message to trigger summarization.".repeat(5);

    memory
        .add_messages(vec![
            msg(MessageRole::User, &long_text),
            msg(MessageRole::Assistant, "First reply"),
            msg(MessageRole::User, &long_text),
        ])
        .await
        .expect("Adding messages should succeed");

    let context = memory
        .get_context()
        .await
        .expect("Getting context should succeed");
    assert!(!context.is_empty());
    let has_summary = context.iter().any(|m| {
        matches!(m.role, MessageRole::System)
            && m.content
                .extract_text()
                .starts_with("Summary of earlier conversation")
    });
    assert!(has_summary, "expected summarizer to produce system summary");
}

#[derive(Debug)]
struct DummyEmbeddingProvider;

#[async_trait]
impl EmbeddingProvider for DummyEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        Ok(EmbeddingResponse {
            embeddings: request.texts.iter().map(|_| vec![0.0_f32; 3]).collect(),
            usage: None,
            model: "dummy".to_string(),
        })
    }

    fn name(&self) -> &str {
        "dummy"
    }
}

struct StaticEntityExtractor;

#[async_trait]
impl EntityExtractor for StaticEntityExtractor {
    async fn extract(&self, _messages: &[Message]) -> Result<Vec<EntityCandidate>> {
        Ok(vec![EntityCandidate {
            name: "Nova".to_string(),
            entity_type: "Person".to_string(),
            description: "Owns analytics pipeline".to_string(),
            importance: 0.8,
        }])
    }
}

#[tokio::test]
async fn entity_memory_adds_entity_context() {
    let config = EntityMemoryConfig {
        max_tokens: 200,
        max_entities: 10,
        extraction_interval: 1,
        similarity_threshold: 0.5,
    };

    let mut memory = EntityMemory::new(
        config,
        token_counter(),
        Arc::new(DummyEmbeddingProvider),
        Arc::new(StaticEntityExtractor),
    );

    memory
        .add_messages(vec![msg(
            MessageRole::User,
            "Remember that Nova owns the analytics pipeline.",
        )])
        .await
        .expect("Adding messages should succeed");

    let context = memory
        .get_context()
        .await
        .expect("Getting context should succeed");
    assert!(context.first().is_some_and(
        |m| matches!(m.role, MessageRole::System) && m.content.extract_text().contains("Nova")
    ));
}

/// Extractor that returns many distinct entities to test max_entities enforcement at insertion time.
struct ManyEntitiesExtractor(usize);

#[async_trait]
impl EntityExtractor for ManyEntitiesExtractor {
    async fn extract(&self, _messages: &[Message]) -> Result<Vec<EntityCandidate>> {
        let n = self.0;
        Ok((0..n)
            .map(|i| EntityCandidate {
                name: format!("Entity{i}"),
                entity_type: "Test".to_string(),
                description: format!("Description {i}"),
                importance: 0.5,
            })
            .collect())
    }
}

#[tokio::test]
async fn entity_memory_enforces_max_entities_during_insertion() {
    let max_entities = 3;
    let config = EntityMemoryConfig {
        max_tokens: 500,
        max_entities,
        extraction_interval: 1,
        similarity_threshold: 0.5,
    };

    let mut memory = EntityMemory::new(
        config,
        token_counter(),
        Arc::new(DummyEmbeddingProvider),
        Arc::new(ManyEntitiesExtractor(10)),
    );

    memory
        .add_messages(vec![msg(
            MessageRole::User,
            "Trigger extraction with many entities.",
        )])
        .await
        .expect("Adding messages should succeed");

    let stats = memory.stats();
    assert!(
        stats.total_entities <= max_entities,
        "entity count {} must not exceed max_entities {}",
        stats.total_entities,
        max_entities
    );
}

#[derive(Clone, Debug, Default)]
struct InMemoryBackend {
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

#[async_trait]
impl caliche::persistence::PersistenceBackend for InMemoryBackend {
    async fn set(&self, key: &str, value: &[u8], _ttl: Option<u64>) -> Result<()> {
        self.data
            .lock()
            .await
            .insert(key.to_string(), value.to_vec());
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.lock().await.get(key).cloned())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.data.lock().await.remove(key);
        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.data.lock().await.contains_key(key))
    }

    async fn expire(&self, _key: &str, _ttl: u64) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn persistent_memory_round_trip() {
    let backend = Arc::new(InMemoryBackend::default());
    let persistence = Arc::new(BackendMemoryPersistence::new(
        backend.clone(),
        "memory:",
        None,
    ));

    let config = BufferMemoryConfig {
        max_tokens: 200,
        min_messages: 1,
        preserve_system: true,
        summary_placeholder: false,
    };

    let memory = BufferMemory::new(config.clone(), token_counter());
    let mut persisted =
        PersistentMemory::new(memory, persistence, "conversation-1").with_autosave(false);

    persisted
        .add_messages(vec![msg(MessageRole::User, "Persist me!")])
        .await
        .expect("Adding messages should succeed");
    persisted
        .flush()
        .await
        .expect("Flushing persisted memory should succeed");

    let memory2 = BufferMemory::new(config, token_counter());
    let mut restored = PersistentMemory::new(
        memory2,
        Arc::new(BackendMemoryPersistence::new(backend, "memory:", None)),
        "conversation-1",
    )
    .with_autosave(false);
    restored
        .hydrate()
        .await
        .expect("Hydrating restored memory should succeed");

    let context = restored
        .get_context()
        .await
        .expect("Getting context should succeed");
    assert_eq!(context.len(), 1);
    assert_eq!(context[0].content.extract_text(), "Persist me!");
}
