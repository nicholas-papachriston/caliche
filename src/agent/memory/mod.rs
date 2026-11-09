pub mod base;
pub mod budget;
pub mod buffer;
pub mod composite;
pub mod compression;
pub mod config;
pub mod container;
pub mod enhanced;
pub mod entity;
pub mod knowledge;
pub(crate) mod persistence_helpers;
pub mod persistent;
pub mod relevance;
pub mod strategy_factory;
pub mod summary;
pub mod token_counter;
#[cfg(feature = "vector-memory")]
pub mod vector;
pub mod working;

pub use base::{MemoryInspection, MemoryPersistence, MemorySnapshot, MemoryStats, MemoryStrategy};
pub use budget::{
    AllocationStrategy, BudgetComponent, ContextBudget, ContextBudgetAllocator, QueryComplexity,
};
pub use buffer::BufferMemory;
pub use composite::CompositeMemory;
pub use compression::{
    CompressedMessage, CompressionStrategy, LlmMessageCompressor, MessageCompressor,
    ToolOutputCompressor,
};
#[cfg(feature = "vector-memory")]
pub use config::VectorMemoryConfig;
pub use config::{
    BufferMemoryConfig, CompositeMemoryConfig, DEFAULT_SESSION_MEMORY_TOKENS, EnhancedMemoryConfig,
    EntityMemoryConfig, MemoryBuilder, MemoryType, SESSION_MEMORY_TOKENS_ENV,
    SESSION_MEMORY_WINDOW_DIVISOR, SummaryMemoryConfig, resolve_session_memory_tokens,
    session_memory_token_budget,
};
pub use enhanced::EnhancedMemory;
pub use entity::{
    EntityCandidate, EntityExtractor, EntityMemory, EntityRecord, ProviderEntityExtractor,
};
pub use knowledge::{KnowledgeEntry, KnowledgeStore};
pub use persistent::{
    BackendMemoryPersistence, ConversationStoreMemoryPersistence, PersistentMemory,
};
pub use relevance::{RelevanceScore, RelevanceScorer, ScoredMessage};
pub use strategy_factory::{
    AutoCompactResult, MemoryStrategyType, create_memory_strategy, format_memory_stats,
    maybe_auto_compact_memory,
};
pub use summary::{ConversationSummarizer, ProviderSummarizer, SummaryMemory};
#[cfg(feature = "token-tiktoken")]
pub use token_counter::TiktokenCounter;
pub use token_counter::{
    ApproximateTokenCounter, TokenCountEstimate, TokenCounter, TokenCounterFactory,
};
#[cfg(feature = "vector-memory")]
pub use vector::VectorMemory;
pub use working::{WorkingMemory, WorkingMemoryManager};
