use serde::{Deserialize, Serialize};

use super::budget::AllocationStrategy;
use super::compression::CompressionStrategy;

/// Session transcript budget when the model context window is unknown.
pub const DEFAULT_SESSION_MEMORY_TOKENS: usize = 64_000;

/// Divide a known context window by this to size the session transcript.
pub const SESSION_MEMORY_WINDOW_DIVISOR: usize = 2;

/// Env override for [`session_memory_token_budget`].
pub const SESSION_MEMORY_TOKENS_ENV: &str = "CALICHE_SESSION_MEMORY_TOKENS";

/// Token budget for a session transcript.
///
/// An explicit `configured` value wins. Otherwise half of a known context
/// window is used, which leaves room for the live request. When the window
/// is unknown, [`DEFAULT_SESSION_MEMORY_TOKENS`] is the tool-heavy default.
#[must_use]
pub fn resolve_session_memory_tokens(
    context_window: Option<usize>,
    configured: Option<usize>,
) -> usize {
    if let Some(tokens) = configured.filter(|tokens| *tokens > 0) {
        return tokens;
    }
    match context_window {
        Some(window) if window > 0 => (window / SESSION_MEMORY_WINDOW_DIVISOR).clamp(1, window),
        _ => DEFAULT_SESSION_MEMORY_TOKENS,
    }
}

/// Session transcript budget, including the env override.
#[must_use]
pub fn session_memory_token_budget(context_window: Option<usize>) -> usize {
    let configured = std::env::var(SESSION_MEMORY_TOKENS_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok());
    resolve_session_memory_tokens(context_window, configured)
}

/// Configuration for buffer-based conversation memory.
/// `max_tokens` should not exceed the LLM model's context window; it will be capped at build time when known.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferMemoryConfig {
    /// Maximum number of tokens to retain in memory. Should be less than or equal to the model's context size.
    pub max_tokens: usize,
    /// Minimum number of messages to keep regardless of token limits.
    pub min_messages: usize,
    /// Preserve the initial system message if encountered.
    pub preserve_system: bool,
    /// Inject a placeholder summary when older messages are truncated.
    pub summary_placeholder: bool,
}

impl Default for BufferMemoryConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            min_messages: 4,
            preserve_system: true,
            summary_placeholder: true,
        }
    }
}

impl BufferMemoryConfig {
    /// Create a new buffer memory config with custom max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the minimum number of messages to keep
    #[must_use]
    pub const fn with_min_messages(mut self, min_messages: usize) -> Self {
        self.min_messages = min_messages;
        self
    }

    /// Clamp `max_tokens` to the model context limit. Returns true if `max_tokens` was reduced.
    /// Call this when the model's context window is known; `max_tokens` should not exceed it.
    pub const fn clamp_to_context_limit(&mut self, limit: usize) -> bool {
        if self.max_tokens > limit {
            self.max_tokens = limit;
            true
        } else {
            false
        }
    }

    /// Ensure `min_messages` fits within `max_tokens` using a conservative token-per-message estimate.
    /// Clamps `min_messages` if necessary. Returns true if `min_messages` was reduced.
    /// Call from the buffer memory constructor or when applying config.
    pub const fn validate_min_messages_vs_max_tokens(&mut self) -> bool {
        const MIN_TOKENS_PER_MESSAGE: usize = 4;
        if self.max_tokens == 0 {
            return false;
        }
        let max_min_messages = self.max_tokens / MIN_TOKENS_PER_MESSAGE;
        if self.min_messages > max_min_messages {
            self.min_messages = max_min_messages;
            true
        } else {
            false
        }
    }
}

/// Configuration for summarization-based memory.
/// `max_tokens` should not exceed the LLM model's context window; it will be capped at build time when known.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMemoryConfig {
    /// Maximum token budget for the combined prompt. Should be less than or equal to the model's context size.
    pub max_tokens: usize,
    /// Maximum tokens per summary entry. If None, defaults to `max_tokens` / 4. Used to cap runaway summarizer output.
    pub max_summary_tokens: Option<usize>,
    /// Percentage (0.0 - 1.0) at which summarization should be triggered.
    pub summarize_threshold_pct: f32,
    /// Number of messages to include in each summarization batch.
    pub chunk_size: usize,
    /// Maximum number of summaries to retain.
    pub max_summaries: usize,
}

impl SummaryMemoryConfig {
    /// Create a new summary memory config with custom max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the threshold percentage at which summarization is triggered (0.0 - 1.0)
    #[must_use]
    pub const fn with_summarize_threshold(mut self, threshold: f32) -> Self {
        self.summarize_threshold_pct = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the chunk size for summarization batches
    #[must_use]
    pub const fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Clamp `max_tokens` to the model context limit. Returns true if `max_tokens` was reduced.
    pub const fn clamp_to_context_limit(&mut self, limit: usize) -> bool {
        if self.max_tokens > limit {
            self.max_tokens = limit;
            true
        } else {
            false
        }
    }
}

impl Default for SummaryMemoryConfig {
    fn default() -> Self {
        Self {
            max_tokens: 8192,
            max_summary_tokens: None,
            summarize_threshold_pct: 0.8,
            chunk_size: 12,
            max_summaries: 8,
        }
    }
}

/// Configuration for entity-tracking memory.
/// `max_tokens` should not exceed the LLM model's context window; it will be capped at build time when known.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMemoryConfig {
    /// Maximum token budget including entity context. Should be less than or equal to the model's context size.
    pub max_tokens: usize,
    /// Maximum number of tracked entities.
    pub max_entities: usize,
    /// Extract entities every N messages.
    pub extraction_interval: usize,
    /// Minimum relevance threshold for injecting entities into context.
    pub similarity_threshold: f32,
}

impl EntityMemoryConfig {
    /// Create a new entity memory config with custom max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the maximum number of entities to track
    #[must_use]
    pub const fn with_max_entities(mut self, max_entities: usize) -> Self {
        self.max_entities = max_entities;
        self
    }

    /// Set the extraction interval (extract entities every N messages)
    #[must_use]
    pub const fn with_extraction_interval(mut self, interval: usize) -> Self {
        self.extraction_interval = interval;
        self
    }

    /// Clamp `max_tokens` to the model context limit. Returns true if `max_tokens` was reduced.
    pub const fn clamp_to_context_limit(&mut self, limit: usize) -> bool {
        if self.max_tokens > limit {
            self.max_tokens = limit;
            true
        } else {
            false
        }
    }
}

impl Default for EntityMemoryConfig {
    fn default() -> Self {
        Self {
            max_tokens: 6144,
            max_entities: 128,
            extraction_interval: 3,
            similarity_threshold: 0.65,
        }
    }
}

/// Configuration for vector-based semantic memory.
/// Only available when the `vector-memory` feature is enabled.
#[cfg(feature = "vector-memory")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMemoryConfig {
    /// Maximum token budget for the combined context (buffer + retrieved memories).
    pub max_tokens: usize,
    /// Maximum number of conversation turns to store as embeddings.
    pub max_memories: usize,
    /// Number of most relevant memories to retrieve.
    pub top_k: usize,
    /// Minimum similarity threshold (0.0-1.0) for retrieving memories.
    pub similarity_threshold: f32,
    /// Token budget reserved for recent buffer messages.
    pub buffer_tokens: usize,
    /// Optional namespace for isolating memories in the vector backend.
    pub namespace: Option<String>,
}

#[cfg(feature = "vector-memory")]
impl VectorMemoryConfig {
    /// Create a new vector memory config with custom max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the maximum number of memories to store
    #[must_use]
    pub const fn with_max_memories(mut self, max_memories: usize) -> Self {
        self.max_memories = max_memories;
        self
    }

    /// Set the number of top memories to retrieve
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the similarity threshold (0.0 - 1.0)
    #[must_use]
    pub const fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the namespace for isolating memories
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }
}

#[cfg(feature = "vector-memory")]
impl Default for VectorMemoryConfig {
    fn default() -> Self {
        Self {
            max_tokens: 8192,
            max_memories: 1000,
            top_k: 5,
            similarity_threshold: 0.7,
            buffer_tokens: 4096,
            namespace: None,
        }
    }
}

/// Configuration for composite memory (primary + secondaries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeMemoryConfig {
    /// Primary strategy (e.g. buffer) for recent context.
    pub primary: Box<MemoryType>,
    /// Secondary strategies (e.g. vector) for supplemental retrieval.
    pub secondaries: Vec<MemoryType>,
    /// Max tokens for the combined context.
    pub max_tokens: usize,
    /// Fraction of budget (0.0–1.0) for primary; rest for secondaries.
    pub primary_budget_pct: f32,
}

impl Default for CompositeMemoryConfig {
    fn default() -> Self {
        Self {
            primary: Box::new(MemoryType::Buffer(BufferMemoryConfig::default())),
            secondaries: vec![],
            max_tokens: 16384,
            primary_budget_pct: 0.6,
        }
    }
}

impl CompositeMemoryConfig {
    /// Set max tokens for the combined context.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set fraction of budget for primary (0.0–1.0).
    #[must_use]
    pub const fn primary_budget_pct(mut self, pct: f32) -> Self {
        self.primary_budget_pct = pct.clamp(0.0, 1.0);
        self
    }

    /// Add a secondary strategy.
    #[must_use]
    pub fn add_secondary(mut self, secondary: MemoryType) -> Self {
        self.secondaries.push(secondary);
        self
    }
}

/// Configuration for enhanced memory (base + budget, relevance, compression, working memory, knowledge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMemoryConfig {
    /// Base memory strategy (e.g. buffer, summary).
    pub base: Box<MemoryType>,
    /// Max tokens for context budget.
    pub max_tokens: usize,
    /// Budget allocation strategy.
    pub budget_strategy: AllocationStrategy,
    /// Enable relevance scoring for context selection.
    pub enable_relevance_scoring: bool,
    /// Enable tool output compression.
    pub enable_tool_compression: bool,
    /// Compression strategy for tool outputs.
    pub compression_strategy: CompressionStrategy,
    /// Enable knowledge store for cross-session learning.
    pub enable_knowledge_store: bool,
    /// Max entries in knowledge store.
    pub knowledge_max_entries: usize,
    /// Token budget for working memory per task.
    pub working_memory_tokens: usize,
}

impl Default for EnhancedMemoryConfig {
    fn default() -> Self {
        Self {
            base: Box::new(MemoryType::Buffer(BufferMemoryConfig::default())),
            max_tokens: 16384,
            budget_strategy: AllocationStrategy::Fixed {
                system_pct: 0.1,
                history_pct: 0.5,
                rag_pct: 0.3,
                tools_pct: 0.1,
            },
            enable_relevance_scoring: true,
            enable_tool_compression: true,
            compression_strategy: CompressionStrategy::Hybrid,
            enable_knowledge_store: true,
            knowledge_max_entries: 256,
            working_memory_tokens: 2048,
        }
    }
}

impl EnhancedMemoryConfig {
    /// Set max tokens.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set budget strategy.
    #[must_use]
    pub const fn budget_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.budget_strategy = strategy;
        self
    }

    /// Enable or disable relevance scoring.
    #[must_use]
    pub const fn with_relevance_scoring(mut self, enable: bool) -> Self {
        self.enable_relevance_scoring = enable;
        self
    }

    /// Enable or disable tool compression.
    #[must_use]
    pub const fn with_tool_compression(mut self, enable: bool) -> Self {
        self.enable_tool_compression = enable;
        self
    }

    /// Set compression strategy.
    #[must_use]
    pub const fn compression_strategy(mut self, strategy: CompressionStrategy) -> Self {
        self.compression_strategy = strategy;
        self
    }

    /// Enable or disable knowledge store.
    #[must_use]
    pub const fn with_knowledge_store(mut self, enable: bool) -> Self {
        self.enable_knowledge_store = enable;
        self
    }

    /// Set knowledge max entries.
    #[must_use]
    pub const fn knowledge_max_entries(mut self, max_entries: usize) -> Self {
        self.knowledge_max_entries = max_entries;
        self
    }

    /// Set working memory token budget.
    #[must_use]
    pub const fn working_memory_tokens(mut self, tokens: usize) -> Self {
        self.working_memory_tokens = tokens;
        self
    }
}

/// Declarative configuration for selecting a memory strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryType {
    Buffer(BufferMemoryConfig),
    Summary(SummaryMemoryConfig),
    Entity(EntityMemoryConfig),
    #[cfg(feature = "vector-memory")]
    Vector(VectorMemoryConfig),
    Composite(CompositeMemoryConfig),
    Enhanced(EnhancedMemoryConfig),
}

impl Default for MemoryType {
    fn default() -> Self {
        Self::Buffer(BufferMemoryConfig::default())
    }
}

/// Fluent builder for creating memory configurations.
///
/// Provides a convenient API for configuring memory strategies with sensible defaults.
///
/// # Example
///
/// ```rust,no_run
/// use caliche::AgentBuilder;
/// use caliche::agent::memory::MemoryBuilder;
///
/// let builder = AgentBuilder::new()
///     .memory(MemoryBuilder::summary()
///         .max_tokens(16384)
///         .summarize_at(0.75)
///         .build());
/// ```
pub struct MemoryBuilder {
    memory_type: MemoryType,
}

impl MemoryBuilder {
    /// Create a buffer memory builder with default settings
    #[must_use]
    pub fn buffer() -> Self {
        Self {
            memory_type: MemoryType::Buffer(BufferMemoryConfig::default()),
        }
    }

    /// Create a summary memory builder with default settings
    #[must_use]
    pub fn summary() -> Self {
        Self {
            memory_type: MemoryType::Summary(SummaryMemoryConfig::default()),
        }
    }

    /// Create an entity memory builder with default settings
    #[must_use]
    pub fn entity() -> Self {
        Self {
            memory_type: MemoryType::Entity(EntityMemoryConfig::default()),
        }
    }

    /// Create a vector memory builder with default settings
    #[cfg(feature = "vector-memory")]
    #[must_use]
    pub fn vector() -> Self {
        Self {
            memory_type: MemoryType::Vector(VectorMemoryConfig::default()),
        }
    }

    /// Create a composite memory builder (primary + secondaries).
    #[must_use]
    pub fn composite(primary: MemoryType) -> Self {
        Self {
            memory_type: MemoryType::Composite(CompositeMemoryConfig {
                primary: Box::new(primary),
                secondaries: vec![],
                max_tokens: 16384,
                primary_budget_pct: 0.6,
            }),
        }
    }

    /// Create an enhanced memory builder (base + budget, relevance, compression, working memory, knowledge).
    #[must_use]
    pub fn enhanced(base: MemoryType) -> Self {
        Self {
            memory_type: MemoryType::Enhanced(EnhancedMemoryConfig {
                base: Box::new(base),
                ..EnhancedMemoryConfig::default()
            }),
        }
    }

    /// Enable or disable relevance scoring (enhanced memory).
    #[must_use]
    pub const fn with_relevance_scoring(mut self, enable: bool) -> Self {
        if let MemoryType::Enhanced(ref mut config) = self.memory_type {
            config.enable_relevance_scoring = enable;
        }
        self
    }

    /// Enable or disable tool compression (enhanced memory).
    #[must_use]
    pub const fn with_tool_compression(mut self, enable: bool) -> Self {
        if let MemoryType::Enhanced(ref mut config) = self.memory_type {
            config.enable_tool_compression = enable;
        }
        self
    }

    /// Enable or disable knowledge store (enhanced memory).
    #[must_use]
    pub const fn with_knowledge_store(mut self, enable: bool) -> Self {
        if let MemoryType::Enhanced(ref mut config) = self.memory_type {
            config.enable_knowledge_store = enable;
        }
        self
    }

    /// Set budget strategy (enhanced memory).
    #[must_use]
    pub const fn budget_strategy(mut self, strategy: AllocationStrategy) -> Self {
        if let MemoryType::Enhanced(ref mut config) = self.memory_type {
            config.budget_strategy = strategy;
        }
        self
    }

    /// Add a secondary strategy to composite memory.
    #[must_use]
    pub fn add_secondary(mut self, secondary: MemoryType) -> Self {
        if let MemoryType::Composite(ref mut config) = self.memory_type {
            config.secondaries.push(secondary);
        }
        self
    }

    /// Set primary budget fraction (0.0–1.0) for composite memory.
    #[must_use]
    pub const fn primary_budget(mut self, pct: f32) -> Self {
        if let MemoryType::Composite(ref mut config) = self.memory_type {
            config.primary_budget_pct = pct.clamp(0.0, 1.0);
        }
        self
    }

    /// Configure buffer memory with custom max tokens.
    #[must_use]
    pub const fn buffer_with_tokens(mut self, max_tokens: usize) -> Self {
        if let MemoryType::Buffer(ref mut config) = self.memory_type {
            config.max_tokens = max_tokens;
        }
        self
    }

    /// Set max tokens for the current memory type.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: usize) -> Self {
        match &mut self.memory_type {
            MemoryType::Buffer(config) => config.max_tokens = max_tokens,
            MemoryType::Summary(config) => config.max_tokens = max_tokens,
            MemoryType::Entity(config) => config.max_tokens = max_tokens,
            #[cfg(feature = "vector-memory")]
            MemoryType::Vector(config) => config.max_tokens = max_tokens,
            MemoryType::Composite(config) => config.max_tokens = max_tokens,
            MemoryType::Enhanced(config) => config.max_tokens = max_tokens,
        }
        self
    }

    /// Configure summary memory threshold (0.0 - 1.0).
    #[must_use]
    pub const fn summarize_at(mut self, threshold: f32) -> Self {
        if let MemoryType::Summary(ref mut config) = self.memory_type {
            config.summarize_threshold_pct = threshold;
        }
        self
    }

    /// Configure vector memory `top_k`.
    #[cfg(feature = "vector-memory")]
    #[must_use]
    pub const fn top_k(mut self, top_k: usize) -> Self {
        if let MemoryType::Vector(ref mut config) = self.memory_type {
            config.top_k = top_k;
        }
        self
    }

    /// Build the memory configuration
    #[must_use]
    pub fn build(self) -> MemoryType {
        self.memory_type
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_SESSION_MEMORY_TOKENS, resolve_session_memory_tokens};

    #[test]
    fn session_budget_uses_half_the_window_or_the_tool_heavy_default() {
        assert_eq!(resolve_session_memory_tokens(Some(200_000), None), 100_000);
        assert_eq!(
            resolve_session_memory_tokens(None, None),
            DEFAULT_SESSION_MEMORY_TOKENS
        );
        assert_eq!(
            resolve_session_memory_tokens(Some(200_000), Some(24_000)),
            24_000
        );
        assert_eq!(resolve_session_memory_tokens(Some(1), None), 1);
    }
}
