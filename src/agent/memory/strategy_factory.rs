use crate::agent::memory::budget::ContextBudgetAllocator;
use crate::agent::memory::relevance::RelevanceScorer;
use crate::agent::memory::{
    ApproximateTokenCounter, BufferMemory, BufferMemoryConfig, CompositeMemory,
    ConversationSummarizer, EnhancedMemory, EntityMemory, EntityMemoryConfig, MemoryStrategy,
    ProviderEntityExtractor, ProviderSummarizer, SummaryMemory, SummaryMemoryConfig,
    WorkingMemoryManager,
};
use crate::llm::{EmbeddingProvider, Provider};
use crate::messages::Message;
use anyhow::Result;
use std::str::FromStr;
use std::sync::Arc;

/// Available memory strategy types selectable via string configuration.
///
/// Vector-backed memory is intentionally not represented here because it
/// requires an explicit backend handle; build it via
/// [`AgentBuilder::with_vector_memory_backend`] instead of this factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategyType {
    Buffer,
    Summary,
    Entity,
    /// Composite: buffer (recent) + summary (older).
    Composite,
    Enhanced,
}

impl FromStr for MemoryStrategyType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "buffer" => Ok(Self::Buffer),
            "summary" => Ok(Self::Summary),
            "entity" => Ok(Self::Entity),
            "composite" => Ok(Self::Composite),
            "enhanced" => Ok(Self::Enhanced),
            "vector" => Err(anyhow::anyhow!(
                "Vector memory requires an explicit vector backend. \
                 Use AgentBuilder::with_vector_memory_backend(...) instead."
            )),
            _ => Err(anyhow::anyhow!("Unknown memory strategy type: {s}")),
        }
    }
}

impl MemoryStrategyType {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Summary => "summary",
            Self::Entity => "entity",
            Self::Composite => "composite",
            Self::Enhanced => "enhanced",
        }
    }
}

/// Build a configured memory strategy and optionally hydrate with existing history.
pub async fn create_memory_strategy(
    strategy_type: MemoryStrategyType,
    max_tokens: usize,
    provider_arc: Arc<dyn Provider>,
    history: Vec<Message>,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
) -> Result<Box<dyn MemoryStrategy>> {
    let token_counter: Arc<dyn crate::agent::memory::TokenCounter> =
        Arc::new(ApproximateTokenCounter::default());

    let mut memory: Box<dyn MemoryStrategy> = match strategy_type {
        MemoryStrategyType::Buffer => {
            let config = BufferMemoryConfig::default().with_max_tokens(max_tokens);
            Box::new(BufferMemory::new(config, token_counter))
        }
        MemoryStrategyType::Summary => {
            let config = SummaryMemoryConfig::default()
                .with_max_tokens(max_tokens)
                .with_summarize_threshold(0.8);
            let summarizer: Arc<dyn ConversationSummarizer> =
                Arc::new(ProviderSummarizer::new(provider_arc, 512));
            Box::new(SummaryMemory::new(config, token_counter, summarizer))
        }
        MemoryStrategyType::Entity => {
            let config = EntityMemoryConfig::default().with_max_tokens(max_tokens);
            let entity_tokens = (max_tokens.min(u32::MAX as usize)) as u32;
            let extractor = Arc::new(ProviderEntityExtractor::new(
                provider_arc.clone(),
                entity_tokens.max(256),
            ));
            let embedding_provider = embedding_provider.ok_or_else(|| {
                anyhow::anyhow!(
                    "Entity memory requires an embedding-capable provider. \
                     The current provider does not support embeddings."
                )
            })?;
            Box::new(EntityMemory::new(
                config,
                token_counter,
                embedding_provider,
                extractor,
            ))
        }
        MemoryStrategyType::Composite => {
            let primary_budget = (max_tokens as f32 * 0.6).round() as usize;
            let buffer_config = BufferMemoryConfig::default().with_max_tokens(primary_budget);
            let primary = Box::new(BufferMemory::new(buffer_config, Arc::clone(&token_counter)));
            let summary_config = SummaryMemoryConfig::default()
                .with_max_tokens(max_tokens)
                .with_summarize_threshold(0.8);
            let summarizer: Arc<dyn ConversationSummarizer> =
                Arc::new(ProviderSummarizer::new(provider_arc, 512));
            let secondary = Box::new(SummaryMemory::new(
                summary_config,
                Arc::clone(&token_counter),
                summarizer,
            ));
            Box::new(CompositeMemory::new(
                primary,
                vec![secondary],
                token_counter,
                max_tokens,
                0.6,
            ))
        }
        MemoryStrategyType::Enhanced => {
            if let Some(emb) = embedding_provider {
                let budget_allocator = ContextBudgetAllocator::new(max_tokens);
                let relevance_scorer = RelevanceScorer::new(emb, Arc::clone(&token_counter));
                let working_mgr =
                    WorkingMemoryManager::new(Arc::clone(&token_counter), max_tokens / 10);
                let base_config = BufferMemoryConfig::default().with_max_tokens(max_tokens);
                let base = Box::new(BufferMemory::new(base_config, Arc::clone(&token_counter)));
                Box::new(EnhancedMemory::new(
                    base,
                    budget_allocator,
                    relevance_scorer,
                    working_mgr,
                ))
            } else {
                tracing::warn!(
                    "Enhanced memory requested without an embedding provider; falling back to SummaryMemory"
                );
                let config = SummaryMemoryConfig::default()
                    .with_max_tokens(max_tokens)
                    .with_summarize_threshold(0.8);
                let summarizer: Arc<dyn ConversationSummarizer> =
                    Arc::new(ProviderSummarizer::new(provider_arc, 512));
                Box::new(SummaryMemory::new(config, token_counter, summarizer))
            }
        }
    };

    if !history.is_empty() {
        memory.add_messages(history).await?;
    }

    Ok(memory)
}

/// Format memory stats for interactive UIs.
pub async fn format_memory_stats(
    memory: Option<Arc<tokio::sync::RwLock<Box<dyn MemoryStrategy>>>>,
) -> String {
    if let Some(memory_arc) = memory {
        let memory_guard = memory_arc.read().await;
        let stats = memory_guard.stats();
        format!(
            "Messages: {} | Tokens: ~{} | Summaries: {} | Entities: {}",
            stats.total_messages,
            stats.total_tokens_estimated,
            stats.total_summaries,
            stats.total_entities
        )
    } else {
        "No memory configured".to_string()
    }
}

/// Result details from an auto-compaction pass.
#[derive(Debug, Clone)]
pub struct AutoCompactResult {
    pub kept_messages: Vec<Message>,
    pub summary_text: String,
}

/// Summarize older conversation state and keep the most recent tail.
pub async fn maybe_auto_compact_memory(
    service: &mut crate::AgentService,
    max_context_tokens: usize,
    should_compact: bool,
) -> Result<Option<AutoCompactResult>> {
    if !should_compact {
        return Ok(None);
    }

    let memory_arc = service
        .memory()
        .ok_or_else(|| anyhow::anyhow!("No memory configured"))?;
    let history: Vec<Message> = {
        let memory = memory_arc.read().await;
        memory.get_context().await?
    };

    if history.len() < 3 {
        return Ok(None);
    }

    let max_keep = 15.min(history.len().saturating_sub(1));
    let min_keep = 5.min(max_keep);
    let keep_count = (history.len() / 3).clamp(min_keep, max_keep);
    let to_summarize = &history[..history.len().saturating_sub(keep_count)];
    let to_keep = &history[history.len().saturating_sub(keep_count)..];

    if to_summarize.is_empty() {
        return Ok(None);
    }

    let summarizer_tokens = (max_context_tokens / 10)
        .clamp(512, 2048)
        .min(u32::MAX as usize);
    let summarizer = Arc::new(ProviderSummarizer::new(
        service.llm_provider(),
        summarizer_tokens as u32,
    ));
    let summary_text: String = summarizer.summarize(to_summarize).await?;
    let summary_message =
        Message::system(format!("[Previous conversation summary]\n{summary_text}"));

    let mut messages_to_add = vec![summary_message];
    messages_to_add.extend(to_keep.iter().cloned());
    {
        let mut memory = memory_arc.write().await;
        memory.clear().await?;
        memory.add_messages(messages_to_add).await?;
    }

    #[cfg(feature = "tools-rag")]
    if let Some(workspace_root) = service.workspace_root() {
        let body = format!(
            "## Auto Compact Summary\n\n{}\n\n## Context\n- kept_messages: {}\n",
            summary_text,
            to_keep.len()
        );
        if let Err(e) = crate::workspace::write_workspace_memory_markdown(
            &workspace_root,
            "auto-compact-summary",
            &body,
        )
        .await
        {
            tracing::debug!("Failed to persist markdown memory summary: {}", e);
        }
    }

    Ok(Some(AutoCompactResult {
        kept_messages: to_keep.to_vec(),
        summary_text,
    }))
}
