use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
#[cfg(feature = "vector-memory")]
use dripstone::{Backend, Chunk, Metric, SearchQuery};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::sync::OnceCell;
use uuid::Uuid;

use crate::agent::memory::{
    base::{MemoryInspection, MemorySnapshot, MemoryStats, MemoryStrategy},
    buffer::BufferMemory,
    config::{BufferMemoryConfig, VectorMemoryConfig},
    token_counter::TokenCounter,
};
use crate::llm::{EmbeddingProvider, EmbeddingRequest};
use crate::messages::Message;

/// Represents a conversation turn (user message + assistant response) stored as an embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TurnRecord {
    /// Unique identifier for this turn
    id: String,
    /// User message text
    user_message: String,
    /// Assistant response text
    assistant_message: String,
    /// Timestamp when this turn was created
    created_at: DateTime<Utc>,
    /// Token count estimate for this turn
    token_estimate: usize,
}

impl TurnRecord {
    /// Create a new turn record from user and assistant messages.
    fn new(user_message: Message, assistant_message: Message, token_estimate: usize) -> Self {
        Self {
            id: format!("turn:{}", Uuid::now_v7()),
            user_message: user_message.content.extract_text(),
            assistant_message: assistant_message.content.extract_text(),
            created_at: Utc::now(),
            token_estimate,
        }
    }

    /// Format the turn as text for embedding.
    fn to_embedding_text(&self) -> String {
        format!(
            "User: {}\nAssistant: {}",
            self.user_message.trim(),
            self.assistant_message.trim()
        )
    }

    /// Convert back to messages for context.
    fn to_messages(&self) -> Vec<Message> {
        vec![
            Message::user(self.user_message.clone()),
            Message::assistant(self.assistant_message.clone()),
        ]
    }

    /// Serialize to JSON for metadata storage.
    fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "created_at": self.created_at.to_rfc3339(),
            "token_estimate": self.token_estimate,
        })
    }

    /// Deserialize from JSON metadata.
    fn from_json(value: &Value) -> Option<Self> {
        let id = value.get("id")?.as_str()?.to_string();
        let user_message = value.get("user_message")?.as_str()?.to_string();
        let assistant_message = value.get("assistant_message")?.as_str()?.to_string();
        let created_at_str = value.get("created_at")?.as_str()?;
        let created_at = DateTime::parse_from_rfc3339(created_at_str)
            .ok()?
            .with_timezone(&Utc);
        let token_estimate = value.get("token_estimate")?.as_u64()? as usize;

        Some(Self {
            id,
            user_message,
            assistant_message,
            created_at,
            token_estimate,
        })
    }
}

/// Memory strategy that stores conversation turns as embeddings and retrieves semantically relevant context.
pub struct VectorMemory {
    config: VectorMemoryConfig,
    token_counter: Arc<dyn TokenCounter>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    vector_backend: Arc<dyn Backend + Send + Sync>,
    buffer: BufferMemory,
    namespace: String,
    last_updated: Option<DateTime<Utc>>,
    /// Cached count of stored memories (atomic for thread-safe access)
    memory_count: Arc<AtomicUsize>,
    /// One-shot lazy initializer that reconciles `memory_count` with the backend.
    /// Using `OnceCell` ensures concurrent callers share the same init future and
    /// that a failing `count()` leaves the cell unset so the next caller retries.
    count_initialized: Arc<OnceCell<()>>,
}

impl VectorMemory {
    /// Create a new vector memory instance.
    pub fn new(
        config: VectorMemoryConfig,
        token_counter: Arc<dyn TokenCounter>,
        embedding_provider: Arc<dyn EmbeddingProvider>,
        vector_backend: Arc<dyn Backend + Send + Sync>,
        conversation_id: Option<String>,
    ) -> Self {
        let namespace = config.namespace.clone().unwrap_or_else(|| {
            conversation_id.unwrap_or_else(|| format!("agent:{}", Uuid::now_v7()))
        });

        let buffer_config = BufferMemoryConfig {
            max_tokens: config.buffer_tokens,
            min_messages: 2,
            preserve_system: true,
            summary_placeholder: false,
        };

        Self {
            config,
            token_counter: Arc::clone(&token_counter),
            embedding_provider,
            vector_backend,
            buffer: BufferMemory::new(buffer_config, token_counter),
            namespace,
            last_updated: None,
            memory_count: Arc::new(AtomicUsize::new(0)),
            count_initialized: Arc::new(OnceCell::new()),
        }
    }

    /// Ensure the memory count is initialized from the backend.
    /// Concurrent callers share the same init future; on error the cell stays
    /// unset so the next caller retries instead of cementing `memory_count == 0`.
    async fn ensure_count_initialized(&self) -> Result<()> {
        self.count_initialized
            .get_or_try_init(|| async {
                let count = self
                    .vector_backend
                    .count(Some(&self.namespace))
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to count embeddings: {e}"))?;
                self.memory_count.store(count, Ordering::Release);
                Ok::<_, anyhow::Error>(())
            })
            .await?;
        Ok(())
    }

    /// Group messages into conversation turns (user + assistant pairs).
    /// Returns owned Messages since they need to be stored.
    fn group_into_turns(&self, messages: &[Message]) -> Vec<(Message, Message)> {
        let mut turns = Vec::new();
        let mut current_user: Option<&Message> = None;

        for message in messages {
            match message.role {
                crate::messages::MessageRole::User => {
                    // If we have a pending user message without assistant response, skip it
                    // (incomplete turn)
                    current_user = Some(message);
                }
                crate::messages::MessageRole::Assistant => {
                    if let Some(user_msg) = current_user.take() {
                        // Clone only when we have a complete turn
                        turns.push((user_msg.clone(), message.clone()));
                    }
                }
                _ => {
                    // System or Tool messages don't form turns, reset
                    current_user = None;
                }
            }
        }

        turns
    }

    /// Store a conversation turn as an embedding.
    async fn store_turn(&self, turn: &TurnRecord) -> Result<()> {
        // Generate embedding for the turn
        let embedding_text = turn.to_embedding_text();
        let request = EmbeddingRequest {
            texts: vec![embedding_text],
            model: None,
        };

        let response = self.embedding_provider.embed(request).await?;
        let embedding = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned from provider"))?;

        // Create chunk with metadata
        let metadata = turn.to_json();
        let chunk = Chunk::new(
            turn.id.clone(),
            0, // chunk_index
            turn.to_embedding_text(),
            embedding,
            0, // start_offset
            0, // end_offset
            1, // total_chunks
            metadata,
            Some(self.namespace.clone()),
        );

        // Ensure count is initialized before storing
        self.ensure_count_initialized().await?;

        // Store in vector backend
        self.vector_backend
            .upsert(&[chunk])
            .await
            .map_err(|e| anyhow::anyhow!("Failed to store turn embedding: {e}"))?;

        // Increment the cached count
        let new_count = self.memory_count.fetch_add(1, Ordering::AcqRel) + 1;

        // Enforce max_memories limit by removing oldest if needed
        // Only check if we're over the limit (optimization)
        if new_count > self.config.max_memories {
            self.enforce_memory_limit().await?;
        }

        Ok(())
    }

    /// Enforce the maximum number of stored memories.
    /// Optimized to avoid loading all embeddings into memory at once.
    async fn enforce_memory_limit(&self) -> Result<()> {
        // Fast path: check cached count first (avoid expensive list operation)
        let current_count = self.memory_count.load(Ordering::Acquire);
        if current_count <= self.config.max_memories {
            return Ok(());
        }

        // Get actual count from backend (cheap operation)
        let actual_count = self
            .vector_backend
            .count(Some(&self.namespace))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to count embeddings: {e}"))?;

        // Update cached count to match actual count (handles race conditions)
        self.memory_count.store(actual_count, Ordering::Release);

        if actual_count <= self.config.max_memories {
            return Ok(());
        }

        // Calculate how many we need to delete
        let to_delete = actual_count - self.config.max_memories;

        // List all chunks in the namespace to sort by creation time
        let all_chunks = self
            .vector_backend
            .list(Some(&self.namespace), Some(actual_count))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to retrieve records for cleanup: {e}"))?;

        // Extract IDs with timestamps. Chunks whose `created_at` is missing or
        // unparseable are treated as oldest (epoch) so enforcement is deterministic
        // and malformed records are pruned first.
        let epoch = DateTime::<Utc>::from_timestamp(0, 0).unwrap_or_default();
        let mut turns_with_ids: Vec<(String, DateTime<Utc>)> = Vec::with_capacity(all_chunks.len());
        for chunk in all_chunks {
            let created_at = chunk
                .metadata
                .get("created_at")
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map_or(epoch, |dt| dt.with_timezone(&Utc));
            turns_with_ids.push((chunk.id, created_at));
        }

        // Sort by creation time (oldest first); ties fall back to id for stability.
        turns_with_ids.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        // Delete oldest turns until we're under the limit
        let mut deleted_count = 0;
        for (id, _) in turns_with_ids.iter().take(to_delete) {
            self.vector_backend
                .delete(id, Some(&self.namespace))
                .await
                .map_err(|e| anyhow::anyhow!("Failed to delete old turn: {e}"))?;
            deleted_count += 1;
        }

        // Update cached count after deletions
        if deleted_count > 0 {
            self.memory_count.fetch_sub(deleted_count, Ordering::AcqRel);
        }

        Ok(())
    }

    /// Retrieve semantically relevant memories based on the query.
    async fn retrieve_memories(&self, query_text: &str) -> Result<Vec<TurnRecord>> {
        // Generate embedding for the query
        let request = EmbeddingRequest {
            texts: vec![query_text.to_string()],
            model: None,
        };

        let response = self.embedding_provider.embed(request).await?;
        let query_vector = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned for query"))?;

        // Search for similar turns
        let query = SearchQuery::new(query_vector.clone())
            .limit(self.config.top_k)
            .metric(Metric::Cosine)
            .namespace(self.namespace.clone());

        let results = self
            .vector_backend
            .search(&query)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to search memories: {e}"))?;

        // Filter by similarity threshold and extract turn records
        let mut memories = Vec::new();
        for result in results {
            if result.score < self.config.similarity_threshold {
                continue;
            }

            let metadata = &result.chunk.metadata;
            if let Some(turn) = TurnRecord::from_json(metadata) {
                memories.push(turn);
            }
        }

        // Sort by recency (most recent first) and relevance
        memories.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        Ok(memories)
    }

    /// Get IDs of messages currently in the buffer to avoid duplicates.
    /// Uses a hash-based approach to avoid cloning message content.
    fn get_buffer_message_ids(&self) -> HashSet<String> {
        // Extract text from buffer messages to create simple IDs
        // Use as_text() for zero-copy when possible
        let messages = self.buffer.messages();
        messages
            .iter()
            .map(|msg| {
                // Create a simple ID from message content
                let text = msg
                    .content
                    .as_text()
                    .map(std::string::ToString::to_string)
                    .unwrap_or_else(|| msg.content.extract_text());
                format!("{:?}:{}", msg.role, text)
            })
            .collect()
    }

    /// Estimate tokens for a list of messages.
    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        self.token_counter.estimate_messages_tokens(messages)
    }
}

#[async_trait]
impl MemoryStrategy for VectorMemory {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        // Add to buffer first (requires clone for buffer's internal storage)
        self.buffer.add_messages(messages.clone()).await?;

        // Group into turns and store each turn as an embedding
        let turns = self.group_into_turns(&messages);
        for (user_msg, assistant_msg) in turns {
            // Estimate tokens using references
            let turn_messages = [&user_msg, &assistant_msg];
            let token_estimate = turn_messages
                .iter()
                .map(|msg| self.token_counter.estimate_message_tokens(msg))
                .sum();
            let turn = TurnRecord::new(user_msg, assistant_msg, token_estimate);
            self.store_turn(&turn).await?;
        }

        self.last_updated = Some(Utc::now());
        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        let mut context = Vec::new();

        // Get recent messages from buffer
        let buffer_messages = self.buffer.get_context().await?;
        let buffer_tokens = self.estimate_tokens(&buffer_messages);

        // Get buffer message IDs for deduplication
        let buffer_ids = self.get_buffer_message_ids();

        // Determine how many tokens we have left for retrieved memories
        let memory_token_budget: usize = self.config.max_tokens.saturating_sub(buffer_tokens);

        // If we have a user message in the buffer, use it for semantic search
        let query_text = buffer_messages
            .iter()
            .rev()
            .find(|msg| matches!(msg.role, crate::messages::MessageRole::User))
            .map_or_else(
                || "conversation".to_string(),
                |msg| msg.content.extract_text(),
            );

        // Retrieve semantically relevant memories
        let memories = self.retrieve_memories(&query_text).await?;

        // Add memories to context, avoiding duplicates and respecting token budget
        let mut memory_tokens: usize = 0;
        for memory in memories {
            let memory_messages = memory.to_messages();
            let turn_tokens: usize = self.estimate_tokens(&memory_messages);

            // Check if this memory is already in buffer (simple deduplication)
            let memory_id = format!(
                "{:?}:{}",
                crate::messages::MessageRole::User,
                memory.user_message
            );
            if buffer_ids.contains(&memory_id) {
                continue;
            }

            // Check token budget
            let new_total: usize = memory_tokens.saturating_add(turn_tokens);
            if new_total > memory_token_budget {
                break;
            }

            context.extend(memory_messages);
            memory_tokens = new_total;
        }

        // Combine: retrieved memories (oldest first) + buffer messages (recent first)
        // This ensures chronological order: older relevant memories, then recent context
        let mut final_context = context;
        final_context.extend(buffer_messages);

        // Final token check and trim if needed
        let total_tokens = self.estimate_tokens(&final_context);
        if total_tokens > self.config.max_tokens {
            // Trim from the beginning (oldest memories) if we exceed budget
            let mut trimmed = Vec::new();
            let mut current_tokens: usize = 0;

            for message in final_context.into_iter().rev() {
                let msg_tokens: usize = self.estimate_tokens(std::slice::from_ref(&message));
                let new_total: usize = current_tokens.saturating_add(msg_tokens);
                if new_total > self.config.max_tokens {
                    break;
                }
                trimmed.push(message);
                current_tokens = new_total;
            }

            trimmed.reverse();
            Ok(trimmed)
        } else {
            Ok(final_context)
        }
    }

    async fn clear(&mut self) -> Result<()> {
        self.buffer.clear().await?;

        // Delete all embeddings in the namespace
        // Using delete_namespace is efficient for clearing all data
        self.vector_backend
            .delete_namespace(&self.namespace)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to clear namespace: {e}"))?;

        // Reset counters after clearing. `OnceCell` has no reset primitive,
        // so we swap in a fresh cell which is safe under `&mut self`.
        self.memory_count.store(0, Ordering::Release);
        self.count_initialized = Arc::new(OnceCell::new());

        self.last_updated = Some(Utc::now());
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        let buffer_stats = self.buffer.stats();

        // Return the cached memory count (atomic read, no async needed)
        let memory_count = self.memory_count.load(Ordering::Acquire);

        MemoryStats {
            total_messages: buffer_stats.total_messages,
            total_tokens_estimated: buffer_stats.total_tokens_estimated,
            total_summaries: buffer_stats.total_summaries,
            total_entities: memory_count,
            last_updated: self.last_updated.or(buffer_stats.last_updated),
        }
    }

    async fn save(
        &self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        self.buffer.save(persistence, conversation_id).await?;
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
        self.buffer.load(persistence, conversation_id).await?;
        if let Some(store) = persistence
            && let Some(snapshot) = store.load_snapshot(conversation_id).await?
        {
            self.load_state(snapshot).await?;
        }

        Ok(())
    }

    async fn save_state(&self) -> Result<MemorySnapshot> {
        // Vector data lives in the backend; the snapshot only records buffer state
        // plus metadata (namespace, count, last_updated) so the vectors can be
        // re-bound to this namespace on load.
        self.ensure_count_initialized().await?;
        let mut metadata = Map::new();
        metadata.insert(
            "namespace".to_string(),
            Value::String(self.namespace.clone()),
        );
        let memory_count = self.memory_count.load(Ordering::Acquire);
        metadata.insert(
            "memory_count".to_string(),
            Value::Number(serde_json::Number::from(memory_count)),
        );
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
        if let Some(namespace_str) = snapshot
            .metadata
            .get("namespace")
            .and_then(|value| value.as_str())
            && namespace_str != self.namespace
        {
            tracing::warn!(
                "Namespace mismatch: expected {}, got {}",
                self.namespace,
                namespace_str
            );
        }
        if let Some(updated) = snapshot
            .metadata
            .get("last_updated")
            .and_then(|value| value.as_str())
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        {
            self.last_updated = Some(updated.with_timezone(&Utc));
        }
        if let Some(count) = snapshot
            .metadata
            .get("memory_count")
            .and_then(serde_json::Value::as_u64)
        {
            self.memory_count.store(count as usize, Ordering::Release);
            // Seed the OnceCell so the cached count is treated as authoritative.
            let _ = self.count_initialized.set(());
        }
        // Restore buffer messages (snapshot.messages mirrors what save_state captured).
        self.buffer.clear().await?;
        if !snapshot.messages.is_empty() {
            self.buffer.add_messages(snapshot.messages).await?;
        }
        Ok(())
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        let count = self.memory_count.load(Ordering::Acquire);
        let knowledge = if count > 0 {
            vec![format!("{} stored conversation turns", count)]
        } else {
            vec![]
        };
        Ok(MemoryInspection {
            strategy_name: "vector".to_string(),
            messages: self.get_context().await?,
            summaries: vec![],
            entities: vec![],
            knowledge,
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
        let request = EmbeddingRequest {
            texts: vec![query.to_string()],
            model: None,
        };
        let response = self.embedding_provider.embed(request).await?;
        let query_vector = response
            .embeddings
            .first()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?;
        let search_query = SearchQuery::new(query_vector.clone())
            .limit(self.config.top_k)
            .metric(Metric::Cosine)
            .namespace(self.namespace.clone());
        let results = self
            .vector_backend
            .search(&search_query)
            .await
            .map_err(|e| anyhow::anyhow!("Search failed: {e}"))?;
        let mut out = Vec::new();
        for result in results {
            if result.score < self.config.similarity_threshold {
                continue;
            }
            if let Some(turn) = TurnRecord::from_json(&result.chunk.metadata) {
                for msg in turn.to_messages() {
                    out.push((msg, result.score));
                }
            }
        }
        if out.is_empty() {
            return self
                .get_context()
                .await
                .map(|ms| ms.into_iter().map(|m| (m, 1.0)).collect());
        }
        Ok(out)
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        let buffer = self.buffer.fork_concrete().await?;
        let new_ns = format!("{}-fork-{}", self.namespace, Uuid::now_v7());
        let mut config = self.config.clone();
        config.namespace = Some(new_ns.clone());
        Ok(Box::new(Self {
            config,
            token_counter: Arc::clone(&self.token_counter),
            embedding_provider: Arc::clone(&self.embedding_provider),
            vector_backend: Arc::clone(&self.vector_backend),
            buffer,
            namespace: new_ns,
            last_updated: self.last_updated,
            memory_count: Arc::new(AtomicUsize::new(0)),
            count_initialized: Arc::new(OnceCell::new()),
        }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
