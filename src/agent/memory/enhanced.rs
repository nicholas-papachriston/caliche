//! Enhanced memory strategy that integrates budget allocation, relevance scoring,
//! compression, working memory, and knowledge store.
//!
//! Do not wrap another `EnhancedMemory` as the base strategy; that creates a circular
//! dependency and can cause infinite loops or stack overflow in fork/merge.

use crate::agent::memory::base::{MemoryInspection, MemoryStats, MemoryStrategy};
use crate::agent::memory::budget::ContextBudgetAllocator;
use crate::agent::memory::compression::ToolOutputCompressor;
use crate::agent::memory::knowledge::KnowledgeStore;
use crate::agent::memory::relevance::RelevanceScorer;
use crate::agent::memory::working::{WorkingMemory, WorkingMemoryManager};
use crate::messages::Message;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// Enhanced memory strategy that combines multiple context management techniques.
pub struct EnhancedMemory {
    /// Base memory strategy (e.g., `BufferMemory`, `SummaryMemory`)
    base_strategy: Box<dyn MemoryStrategy>,
    /// Context budget allocator
    budget_allocator: ContextBudgetAllocator,
    /// Relevance scorer for query-aware selection
    relevance_scorer: RelevanceScorer,
    /// Tool output compressor
    tool_compressor: Option<Arc<ToolOutputCompressor>>,
    /// Working memory manager
    working_memory_manager: WorkingMemoryManager,
    /// Knowledge store for cross-session learning
    knowledge_store: Option<Arc<tokio::sync::RwLock<KnowledgeStore>>>,
    /// Current query for context-aware retrieval
    current_query: Option<String>,
    last_updated: Option<DateTime<Utc>>,
}

impl EnhancedMemory {
    /// Create a new enhanced memory strategy.
    #[must_use]
    pub fn new(
        base_strategy: Box<dyn MemoryStrategy>,
        budget_allocator: ContextBudgetAllocator,
        relevance_scorer: RelevanceScorer,
        working_memory_manager: WorkingMemoryManager,
    ) -> Self {
        Self {
            base_strategy,
            budget_allocator,
            relevance_scorer,
            tool_compressor: None,
            working_memory_manager,
            knowledge_store: None,
            current_query: None,
            last_updated: None,
        }
    }

    /// Set tool output compressor.
    #[must_use]
    pub fn with_tool_compressor(mut self, compressor: Arc<ToolOutputCompressor>) -> Self {
        self.tool_compressor = Some(compressor);
        self
    }

    /// Set knowledge store.
    pub fn with_knowledge_store(
        mut self,
        knowledge_store: Arc<tokio::sync::RwLock<KnowledgeStore>>,
    ) -> Self {
        self.knowledge_store = Some(knowledge_store);
        self
    }

    /// Set current query for context-aware retrieval.
    pub fn set_query(&mut self, query: impl Into<String>) {
        self.current_query = Some(query.into());
    }

    /// Get working memory for current task.
    pub fn get_working_memory(&mut self, task_id: &str) -> &mut WorkingMemory {
        self.working_memory_manager
            .get_or_create(task_id.to_string())
    }

    /// Compress tool outputs before adding to memory.
    async fn compress_tool_outputs(&self, messages: Vec<Message>) -> Result<Vec<Message>> {
        if let Some(ref compressor) = self.tool_compressor {
            let mut compressed = Vec::new();
            for message in messages {
                // Check if this is a tool output
                let is_tool_output = matches!(message.role, crate::messages::MessageRole::Tool)
                    || matches!(
                        message.content,
                        crate::messages::MessageContent::Structured { .. }
                    );

                if is_tool_output {
                    // Get budget for tools
                    let budget = self
                        .budget_allocator
                        .allocate(self.current_query.as_deref());
                    let compressed_msg = compressor
                        .compress_if_needed(&message, budget.tools_tokens)
                        .await?;
                    compressed.push(compressed_msg);
                } else {
                    compressed.push(message);
                }
            }
            Ok(compressed)
        } else {
            Ok(messages)
        }
    }
}

#[async_trait]
impl MemoryStrategy for EnhancedMemory {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        // Compress tool outputs if compressor is available
        let messages = self.compress_tool_outputs(messages).await?;

        // Extract knowledge from messages if knowledge store is available
        if let Some(ref knowledge_store) = self.knowledge_store {
            let mut store = knowledge_store.write().await;
            if let Err(e) = store.extract_and_learn(&messages, None).await {
                tracing::warn!("Failed to extract knowledge: {}", e);
            }
        }

        // Add to base strategy
        self.base_strategy.add_messages(messages).await?;
        self.last_updated = Some(Utc::now());

        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        // Get base context
        let mut context = self.base_strategy.get_context().await?;

        // Allocate budget
        let budget = self
            .budget_allocator
            .allocate(self.current_query.as_deref());

        // If we have a query, use relevance scoring to select most relevant messages
        if let Some(ref query) = self.current_query {
            let relevant = self
                .relevance_scorer
                .select_relevant(&context, query, budget.history_tokens, None)
                .await?;
            context = relevant;
        } else {
            // No query, use base context but enforce budget
            let token_counter = &self.relevance_scorer.token_counter;
            let total_tokens = token_counter.estimate_messages_tokens(&context);
            if total_tokens > budget.history_tokens {
                // Trim oldest messages while avoiding O(n^2) repeated front-removals.
                let mut drop_count = 0usize;
                let mut running_tokens = total_tokens;
                while running_tokens > budget.history_tokens && drop_count < context.len() {
                    running_tokens = running_tokens.saturating_sub(
                        token_counter.estimate_message_tokens(&context[drop_count]),
                    );
                    drop_count += 1;
                }
                if drop_count > 0 {
                    context.drain(..drop_count);
                }
            }
        }

        // Add relevant knowledge from knowledge store
        if let Some(ref knowledge_store) = self.knowledge_store
            && let Some(ref query) = self.current_query
        {
            let store = knowledge_store.read().await;
            if let Ok(knowledge) = store.search(query, 3, 0.7).await {
                for (entry, _score) in knowledge {
                    let knowledge_msg = Message::new(
                        crate::messages::MessageRole::System,
                        format!("[Knowledge] {}", entry.content),
                    );
                    let token_counter = &self.relevance_scorer.token_counter;
                    let knowledge_tokens = token_counter.estimate_message_tokens(&knowledge_msg);
                    if token_counter.estimate_messages_tokens(&context) + knowledge_tokens
                        <= budget.history_tokens
                    {
                        context.push(knowledge_msg);
                    }
                }
            }
        }

        Ok(context)
    }

    async fn get_context_arc(&self) -> Result<Arc<Vec<Message>>> {
        Ok(Arc::new(self.get_context().await?))
    }

    async fn clear(&mut self) -> Result<()> {
        self.base_strategy.clear().await?;
        self.working_memory_manager.clear_all();
        self.current_query = None;
        self.last_updated = Some(Utc::now());
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        let mut base_stats = self.base_strategy.stats();
        // Could enhance stats with knowledge store info, working memory info, etc.
        base_stats.last_updated = self.last_updated.or(base_stats.last_updated);
        base_stats
    }

    async fn save(
        &self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        self.base_strategy
            .save(persistence, conversation_id)
            .await?;
        if let (Some(p), Some(knowledge_store)) = (persistence, &self.knowledge_store) {
            let store = knowledge_store.read().await;
            let key = format!("{conversation_id}:knowledge");
            store.save(p, &key).await?;
        }
        Ok(())
    }

    async fn load(
        &mut self,
        persistence: Option<&dyn crate::agent::memory::MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        self.base_strategy
            .load(persistence, conversation_id)
            .await?;
        if let (Some(p), Some(knowledge_store)) = (persistence, &self.knowledge_store) {
            let mut store = knowledge_store.write().await;
            let key = format!("{conversation_id}:knowledge");
            store.load(p, &key).await?;
        }
        Ok(())
    }

    async fn save_state(&self) -> Result<crate::agent::memory::base::MemorySnapshot> {
        let base_snapshot = self.base_strategy.save_state().await?;
        let knowledge_snapshot = if let Some(ref knowledge_store) = self.knowledge_store {
            let store = knowledge_store.read().await;
            Some(store.to_snapshot()?)
        } else {
            None
        };
        let mut metadata = serde_json::Map::new();
        metadata.insert("base".to_string(), base_snapshot.metadata);
        if let Some(ks) = knowledge_snapshot {
            metadata.insert("knowledge".to_string(), serde_json::to_value(ks)?);
        }
        if let Some(updated) = self.last_updated {
            metadata.insert(
                "last_updated".to_string(),
                serde_json::Value::String(updated.to_rfc3339()),
            );
        }
        Ok(crate::agent::memory::base::MemorySnapshot {
            messages: base_snapshot.messages,
            metadata: serde_json::Value::Object(metadata),
        })
    }

    async fn load_state(
        &mut self,
        snapshot: crate::agent::memory::base::MemorySnapshot,
    ) -> Result<()> {
        let base_meta = snapshot
            .metadata
            .get("base")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let base_snapshot = crate::agent::memory::base::MemorySnapshot {
            messages: snapshot.messages,
            metadata: base_meta,
        };
        self.base_strategy.load_state(base_snapshot).await?;

        if let (Some(knowledge_value), Some(knowledge_store)) = (
            snapshot.metadata.get("knowledge"),
            self.knowledge_store.as_ref(),
        ) && let Ok(ks) = serde_json::from_value::<crate::agent::memory::base::MemorySnapshot>(
            knowledge_value.clone(),
        ) {
            let mut store = knowledge_store.write().await;
            store.from_snapshot(&ks)?;
        }

        if let Some(updated) = snapshot
            .metadata
            .get("last_updated")
            .and_then(|value| value.as_str())
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
        {
            self.last_updated = Some(updated.with_timezone(&Utc));
        }
        Ok(())
    }

    async fn inspect(&self) -> Result<MemoryInspection> {
        let mut inspection = self.base_strategy.inspect().await?;
        inspection.strategy_name = "enhanced".to_string();
        if let Some(ref knowledge_store) = self.knowledge_store {
            let store = knowledge_store.read().await;
            inspection.knowledge = store
                .all_entries()
                .iter()
                .map(|e| format!("{}: {}", e.category, e.content))
                .collect();
        }
        Ok(inspection)
    }

    async fn search(&self, query: &str) -> Result<Vec<(Message, f32)>> {
        let mut results = self.base_strategy.search(query).await?;
        if let Some(ref knowledge_store) = self.knowledge_store
            && !query.is_empty()
            && let Ok(knowledge) = knowledge_store.read().await.search(query, 5, 0.5).await
        {
            for (entry, score) in knowledge {
                let msg = Message::new(
                    crate::messages::MessageRole::System,
                    format!("[Knowledge] {}", entry.content),
                );
                results.push((msg, score));
            }
        }
        Ok(results)
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        let forked_base = self.base_strategy.fork().await?;
        let inspection = forked_base.inspect().await?;
        if inspection.strategy_name == "enhanced" {
            return Err(anyhow!(
                "EnhancedMemory must not wrap another EnhancedMemory: circular strategy detected in fork"
            ));
        }
        Ok(Box::new(Self {
            base_strategy: forked_base,
            budget_allocator: self.budget_allocator.clone(),
            relevance_scorer: self.relevance_scorer.clone(),
            tool_compressor: self.tool_compressor.clone(),
            working_memory_manager: self.working_memory_manager.clone(),
            knowledge_store: self.knowledge_store.clone(),
            current_query: self.current_query.clone(),
            last_updated: self.last_updated,
        }))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
