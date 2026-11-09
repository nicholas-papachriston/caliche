use std::fmt::Debug;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::{Message, MessageRole};

/// Render conversation turns as `Role: text` lines for LLM summarization /
/// entity-extraction prompts.
pub(crate) fn render_conversation(messages: &[Message]) -> String {
    let mut output = String::new();
    for message in messages {
        let role = match message.role {
            MessageRole::System => "System",
            MessageRole::User => "User",
            MessageRole::Assistant => "Assistant",
            MessageRole::Tool => "Tool",
        };

        output.push_str(role);
        output.push_str(": ");
        if let Some(text) = message.content.as_text() {
            output.push_str(text);
        } else {
            output.push_str(&message.content.extract_text());
        }
        output.push('\n');
    }
    output
}

/// Snapshot of memory suitable for persistence backends.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemorySnapshot {
    /// Ordered list of conversation messages that should be restored into memory.
    #[serde(default)]
    pub messages: Vec<Message>,
    /// Arbitrary metadata that a memory strategy may wish to persist.
    #[serde(default)]
    pub metadata: Value,
}

/// High-level statistics describing the state of a memory strategy.
///
/// Metrics are computed independently per strategy; they may not sum exactly (e.g. `total_tokens_estimated`
/// may differ from the sum of per-message token counts) and are intended for diagnostics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of messages currently retained in memory.
    pub total_messages: usize,
    /// Estimated token count for all retained messages.
    pub total_tokens_estimated: usize,
    /// Number of summary entries maintained by the strategy.
    pub total_summaries: usize,
    /// Number of tracked entities (if applicable).
    pub total_entities: usize,
    /// Timestamp of the most recent update.
    pub last_updated: Option<DateTime<Utc>>,
}

impl MemoryStats {
    /// Construct stats for strategies that only track message counts.
    #[must_use]
    pub fn from_message_count(total_messages: usize) -> Self {
        Self {
            total_messages,
            ..Self::default()
        }
    }
}

/// A window into what the memory currently holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInspection {
    /// Name of the strategy (e.g. "buffer", "summary", "entity", "vector").
    pub strategy_name: String,
    /// Messages currently in context.
    pub messages: Vec<Message>,
    /// Summary strings (for `SummaryMemory`).
    pub summaries: Vec<String>,
    /// Entity names/descriptions (for `EntityMemory`).
    pub entities: Vec<String>,
    /// Knowledge entries (for `EnhancedMemory` / `KnowledgeStore`).
    pub knowledge: Vec<String>,
    /// Current stats.
    pub stats: MemoryStats,
}

/// Abstraction over durable storage for persisting memory snapshots.
#[async_trait]
pub trait MemoryPersistence: Send + Sync + Debug {
    /// Save a snapshot for the given conversation identifier.
    async fn save_snapshot(&self, conversation_id: &str, snapshot: &MemorySnapshot) -> Result<()>;

    /// Load a snapshot for the given conversation identifier.
    async fn load_snapshot(&self, conversation_id: &str) -> Result<Option<MemorySnapshot>>;

    /// Remove any persisted data for the given conversation identifier.
    async fn clear_snapshot(&self, conversation_id: &str) -> Result<()>;
}

/// Core trait implemented by all memory strategies.
#[async_trait]
pub trait MemoryStrategy: Send + Sync {
    /// Add new messages to the memory store.
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()>;

    /// Retrieve messages to include in the next LLM context.
    async fn get_context(&self) -> Result<Vec<Message>>;

    /// Retrieve messages as an Arc for zero-copy sharing.
    ///
    /// Default implementation wraps `get_context()` in an Arc.
    /// Implementations can override this to cache the Arc and avoid cloning.
    async fn get_context_arc(&self) -> Result<Arc<Vec<Message>>> {
        Ok(Arc::new(self.get_context().await?))
    }

    /// Remove all stored memory for the current conversation.
    async fn clear(&mut self) -> Result<()>;

    /// Gather statistics about the memory strategy.
    fn stats(&self) -> MemoryStats;

    /// Persist the memory state using the provided backend.
    ///
    /// The default implementation is a no-op, allowing strategies
    /// that do not support persistence to opt out gracefully.
    async fn save(
        &self,
        _persistence: Option<&dyn MemoryPersistence>,
        _conversation_id: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Restore the memory state using the provided backend.
    ///
    /// The default implementation is a no-op, allowing strategies
    /// that do not support persistence to opt out gracefully.
    async fn load(
        &mut self,
        _persistence: Option<&dyn MemoryPersistence>,
        _conversation_id: &str,
    ) -> Result<()> {
        Ok(())
    }

    /// Capture this strategy's state as a portable snapshot suitable for
    /// embedding inside a `SessionSnapshot`. The default implementation returns
    /// only the current context messages; strategy-specific state (summaries,
    /// entities, knowledge, child snapshots) must be overridden.
    async fn save_state(&self) -> Result<MemorySnapshot> {
        Ok(MemorySnapshot {
            messages: self.get_context().await?,
            metadata: Value::Null,
        })
    }

    /// Restore this strategy's state from a snapshot produced by `save_state`.
    /// The default implementation clears and repopulates via `add_messages`,
    /// which is appropriate for strategies that only retain conversational
    /// context. Strategies with richer state should override.
    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        self.clear().await?;
        if !snapshot.messages.is_empty() {
            self.add_messages(snapshot.messages).await?;
        }
        Ok(())
    }

    /// Return a full inspection of current memory contents.
    async fn inspect(&self) -> Result<MemoryInspection> {
        Ok(MemoryInspection {
            strategy_name: "unknown".to_string(),
            messages: self.get_context().await?,
            summaries: vec![],
            entities: vec![],
            knowledge: vec![],
            stats: self.stats(),
        })
    }

    /// Search memory for messages relevant to a query.
    /// Returns matching messages with relevance scores (default: all messages with score 1.0).
    async fn search(&self, _query: &str) -> Result<Vec<(Message, f32)>> {
        let messages = self.get_context().await?;
        Ok(messages.into_iter().map(|m| (m, 1.0)).collect())
    }

    /// Create a forked copy of this memory strategy for parallel work.
    /// Returns a new independent strategy initialized with the current state.
    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>>;

    /// Merge state from another strategy instance into this one.
    /// Default: appends the other strategy's context as messages to this one.
    async fn merge_from(&mut self, other: &dyn MemoryStrategy) -> Result<()> {
        let messages = other.get_context().await?;
        self.add_messages(messages).await
    }

    /// Concrete strategy, so callers can read configuration the trait does not expose.
    fn as_any(&self) -> &dyn std::any::Any;
}

#[async_trait]
impl MemoryStrategy for Box<dyn MemoryStrategy> {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        (**self).add_messages(messages).await
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        (**self).get_context().await
    }

    async fn get_context_arc(&self) -> Result<Arc<Vec<Message>>> {
        (**self).get_context_arc().await
    }

    async fn clear(&mut self) -> Result<()> {
        (**self).clear().await
    }

    fn stats(&self) -> MemoryStats {
        (**self).stats()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        (**self).as_any()
    }

    async fn save(
        &self,
        persistence: Option<&dyn MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        (**self).save(persistence, conversation_id).await
    }

    async fn load(
        &mut self,
        persistence: Option<&dyn MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        (**self).load(persistence, conversation_id).await
    }

    async fn save_state(&self) -> Result<MemorySnapshot> {
        (**self).save_state().await
    }

    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        (**self).load_state(snapshot).await
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        (**self).inspect().await
    }

    async fn search(&self, query: &str) -> Result<Vec<(Message, f32)>> {
        (**self).search(query).await
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        (**self).fork().await
    }

    async fn merge_from(&mut self, other: &dyn MemoryStrategy) -> Result<()> {
        (**self).merge_from(other).await
    }
}
