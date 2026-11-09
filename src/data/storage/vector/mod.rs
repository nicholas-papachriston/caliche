//! Vector store configuration and integration with `Dripstone` backends.
//!
//! This module provides configuration for different vector store backends,
//! allowing Caliche to use various vector databases for RAG and retrieval chains.

pub mod chunking;
pub mod config;
pub mod indexed;

pub use config::{
    VectorBackendConfig, VectorBackendConfigWrapper, tier_config_from_json_value,
    tiered_tiers_from_json_str, tiered_tiers_from_json_value,
};

pub use indexed::{
    deferred_ann_index_enabled_from_env, flush_deferred_ann_index, indexed_backend_config_from_env,
    resolve_ann_index_dimension, vector_dimension_from_env, wrap_with_deferred_ann_index,
    wrap_workspace_backend_with_ann_index,
};

// Re-export dripstone types for convenience
pub use dripstone::{
    Backend,
    Chunk,
    FullTextIndexedBackendConfig,
    IndexUpdateMode,
    IndexedBackendConfig,
    MemoryBackend,
    Metric,
    SearchResult,
    // Tiered backend
    SearchStrategy,
    TierConfig,
    WriteStrategy,
    // Utilities
    index::{HnswConfig, HnswGraphSpace, HnswIndex},
    metrics::{NoOpMetrics, SharedMetrics, SimpleMetrics, VectorMetrics},
    quantization::ProductQuantizer,
    search::{Filter, FilterBuilder},
};

#[cfg(feature = "vector-store-mmap")]
pub use dripstone::{MmapBackendConfig, MmapVectorEncoding};
