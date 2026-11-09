use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::{Map, Value};

use crate::agent::memory::{
    base::{MemoryInspection, MemorySnapshot, MemoryStats, MemoryStrategy},
    config::BufferMemoryConfig,
    container::RingBuffer,
    token_counter::TokenCounter,
};
use crate::messages::{ArcMessage, Message, MessageRole};

/// Sliding window memory that retains the most recent messages within a token budget.
pub struct BufferMemory {
    config: BufferMemoryConfig,
    token_counter: Arc<dyn TokenCounter>,
    messages: RingBuffer<ArcMessage>,
    summary: Option<ArcMessage>,
    last_updated: Option<DateTime<Utc>>,
    /// Cached Arc-wrapped context for zero-copy access (thread-safe cache)
    cached_context: Mutex<Option<Arc<Vec<Message>>>>,
}

impl BufferMemory {
    /// Create a new buffer memory instance with the provided configuration and token counter.
    /// Validates that `min_messages` fits within `max_tokens`; clamps and logs a warning if not.
    pub fn new(mut config: BufferMemoryConfig, token_counter: Arc<dyn TokenCounter>) -> Self {
        if config.validate_min_messages_vs_max_tokens() {
            tracing::warn!(
                max_tokens = config.max_tokens,
                min_messages = config.min_messages,
                "BufferMemoryConfig: min_messages clamped to fit within max_tokens"
            );
        }
        Self {
            config,
            token_counter,
            messages: RingBuffer::new(),
            summary: None,
            last_updated: None,
            cached_context: Mutex::new(None),
        }
    }

    /// Configured token cap for this buffer.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        self.config.max_tokens
    }

    /// Set or replace the summary message injected before the retained history.
    pub fn with_summary(mut self, summary_text: impl Into<String>) -> Self {
        self.summary = Some(Arc::new(Message::new(MessageRole::System, summary_text)));
        self
    }

    /// Update the summary message at runtime.
    pub fn set_summary(&mut self, summary_text: Option<String>) {
        self.summary = summary_text.map(|text| Arc::new(Message::new(MessageRole::System, text)));
        self.last_updated = Some(Utc::now());
        if let Ok(cache) = self.cached_context.get_mut() {
            *cache = None; // Invalidate cache
        }
    }

    pub(crate) fn set_summary_message(&mut self, summary: Option<Message>) {
        self.summary = summary.map(Arc::new);
        self.last_updated = Some(Utc::now());
        if let Ok(cache) = self.cached_context.get_mut() {
            *cache = None; // Invalidate cache
        }
    }

    /// Access currently retained messages (clones for API compatibility).
    /// For internal use, prefer working with `ArcMessage` directly to avoid cloning.
    pub fn messages(&self) -> Vec<Message> {
        self.messages.iter().map(|arc| (**arc).clone()).collect()
    }

    /// Access the optional summary message (clones for API compatibility).
    pub fn summary(&self) -> Option<Message> {
        self.summary.as_ref().map(|arc| (**arc).clone())
    }

    /// Build the context Vec from current state (used by `get_context` and cache rebuild).
    fn build_context_vec(&self) -> Vec<Message> {
        let mut context =
            Vec::with_capacity(self.summary.as_ref().map_or(0, |_| 1) + self.messages.len());
        if let Some(summary) = &self.summary {
            context.push((**summary).clone());
        }
        context.extend(self.messages.iter().map(|arc| (**arc).clone()));
        context
    }

    fn total_token_estimate(&self) -> usize {
        // Use iterator-based counting - dereference Arc to get &Message without cloning
        use crate::agent::memory::token_counter::estimate_messages_tokens_iter;
        let mut total: usize = estimate_messages_tokens_iter(
            self.token_counter.as_ref(),
            self.messages.iter().map(std::convert::AsRef::as_ref),
        );
        if let Some(summary) = &self.summary {
            total = total.saturating_add(self.token_counter.estimate_message_tokens(summary));
        }

        total
    }

    fn enforce_limits(&mut self) {
        if self.total_token_estimate() <= self.config.max_tokens {
            return;
        }
        let before = self.messages.len();
        let owned: Vec<Message> = self
            .messages
            .iter()
            .map(|message| (**message).clone())
            .collect();
        let kept = crate::agent::core::tool_history::trim_history_preserving_tool_pairs(
            owned,
            self.token_counter.as_ref(),
            self.config.max_tokens,
            self.config.min_messages,
        );
        let truncated = before.saturating_sub(kept.len());
        self.messages.clear();
        self.messages.extend(kept.into_iter().map(Arc::new));

        if truncated > 0 && self.config.summary_placeholder && self.summary.is_none() {
            let text = if truncated == 1 {
                "Earlier conversation truncated to satisfy context window.".to_string()
            } else {
                format!("{truncated} earlier messages truncated to satisfy context window.")
            };
            self.summary = Some(Arc::new(Message::new(MessageRole::System, text)));
        }

        // Invalidate cache when messages are modified
        if let Ok(cache) = self.cached_context.get_mut() {
            *cache = None;
        }
    }
}

#[async_trait]
impl MemoryStrategy for BufferMemory {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        // Convert to ArcMessages for efficient storage
        let arc_messages: Vec<ArcMessage> = messages.into_iter().map(Arc::new).collect();
        self.messages.extend(arc_messages);
        self.enforce_limits();
        self.last_updated = Some(Utc::now());

        // Rebuild cache immediately instead of invalidating to avoid full clone on next get_context_arc().
        let context = self.build_context_vec();
        if let Ok(cache) = self.cached_context.get_mut() {
            *cache = Some(Arc::new(context));
        }

        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        Ok(self.build_context_vec())
    }

    async fn get_context_arc(&self) -> Result<Arc<Vec<Message>>> {
        // Check cache first (fast path)
        {
            let cache = self
                .cached_context
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex poisoned: {e}"))?;
            if let Some(ref cached) = *cache {
                return Ok(Arc::clone(cached));
            }
        }

        // Build context
        let context = self.get_context().await?;
        let context_arc = Arc::new(context);

        // Cache it for next time
        {
            let mut cache = self
                .cached_context
                .lock()
                .map_err(|e| anyhow::anyhow!("Mutex poisoned: {e}"))?;
            *cache = Some(Arc::clone(&context_arc));
        }

        Ok(context_arc)
    }

    async fn clear(&mut self) -> Result<()> {
        self.messages.clear();
        self.summary = None;
        self.last_updated = Some(Utc::now());
        if let Ok(cache) = self.cached_context.get_mut() {
            *cache = None; // Invalidate cache
        }
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_messages: self.messages.len(),
            total_tokens_estimated: self.total_token_estimate(),
            total_summaries: usize::from(self.summary.is_some()),
            total_entities: 0,
            last_updated: self.last_updated,
        }
    }

    async fn save(
        &self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(store) = persistence {
            let mut metadata = Map::new();

            if let Some(summary) = &self.summary {
                metadata.insert("summary".to_string(), serde_json::to_value(summary)?);
            }

            super::persistence_helpers::serialize_last_updated(&mut metadata, self.last_updated);

            // Convert ArcMessages back to Messages for serialization
            let messages: Vec<Message> = self.messages.iter().map(|arc| (**arc).clone()).collect();
            let snapshot = MemorySnapshot {
                messages,
                metadata: Value::Object(metadata),
            };

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
        if let Some(summary) = &self.summary {
            metadata.insert("summary".to_string(), serde_json::to_value(summary)?);
        }
        super::persistence_helpers::serialize_last_updated(&mut metadata, self.last_updated);
        let messages: Vec<Message> = self.messages.iter().map(|arc| (**arc).clone()).collect();
        Ok(MemorySnapshot {
            messages,
            metadata: Value::Object(metadata),
        })
    }

    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        let arc_messages: Vec<ArcMessage> = snapshot.messages.into_iter().map(Arc::new).collect();
        self.messages.clear();
        self.messages.extend(arc_messages);
        let summary = snapshot
            .metadata
            .get("summary")
            .and_then(|value| serde_json::from_value::<Message>(value.clone()).ok());
        if let Some(metadata) = snapshot.metadata.as_object() {
            self.last_updated = super::persistence_helpers::deserialize_last_updated(metadata);
        }
        self.set_summary_message(summary);
        self.enforce_limits();
        Ok(())
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        Ok(MemoryInspection {
            strategy_name: "buffer".to_string(),
            messages: self.get_context().await?,
            summaries: vec![],
            entities: vec![],
            knowledge: vec![],
            stats: self.stats(),
        })
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        Ok(Box::new(self.fork_concrete().await?))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl BufferMemory {
    /// Fork into a concrete `BufferMemory` (for use by wrappers like `EntityMemory`).
    pub async fn fork_concrete(&self) -> Result<Self> {
        let messages = self.get_context().await?;
        let mut forked = Self::new(self.config.clone(), Arc::clone(&self.token_counter));
        forked.add_messages(messages).await?;
        Ok(forked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::core::tool_history::tool_history_is_provider_safe;
    use crate::agent::memory::{ApproximateTokenCounter, BufferMemoryConfig};
    use crate::messages::{MessageContent, ToolCallInfo};

    fn parallel_turn(user: &str, id_a: &str, id_b: &str, body: &str) -> Vec<Message> {
        vec![
            Message::user(user),
            Message {
                role: MessageRole::Assistant,
                content: MessageContent::Structured {
                    text: None,
                    tool_calls: Some(vec![
                        ToolCallInfo::from_json(id_a, "fs", r#"{"path":"a.rs"}"#),
                        ToolCallInfo::from_json(id_b, "fs", r#"{"path":"b.rs"}"#),
                    ]),
                    tool_call_id: None,
                },
                metadata: None,
            },
            Message {
                role: MessageRole::Tool,
                content: MessageContent::Structured {
                    text: Some(std::sync::Arc::new(body.to_string())),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned(id_a.to_string())),
                },
                metadata: None,
            },
            Message {
                role: MessageRole::Tool,
                content: MessageContent::Structured {
                    text: Some(std::sync::Arc::new(body.to_string())),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned(id_b.to_string())),
                },
                metadata: None,
            },
        ]
    }

    #[tokio::test]
    async fn buffer_drops_whole_parallel_tool_turns_at_8192() {
        let body = "z".repeat(8_000);
        let mut stored = parallel_turn("turn one", "c1", "c2", &body);
        stored.extend(parallel_turn("turn two", "c3", "c4", &body));
        stored.extend(parallel_turn("turn three", "c5", "c6", &body));
        let counter = Arc::new(ApproximateTokenCounter::default());
        let mut memory = BufferMemory::new(
            BufferMemoryConfig::default().with_max_tokens(8_192),
            counter,
        );
        memory.add_messages(stored).await.expect("retain history");
        let context = memory.get_context().await.expect("context");
        assert!(memory.stats().total_tokens_estimated <= 8_192);
        assert!(tool_history_is_provider_safe(&context));
        assert!(context.iter().any(|message| {
            message
                .content
                .as_tool_calls()
                .is_some_and(|calls| calls.len() == 2)
        }));
    }
}
