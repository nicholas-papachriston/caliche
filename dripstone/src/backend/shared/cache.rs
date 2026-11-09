//! Cache interface and common implementations
//!
//! Provides a unified cache trait that can be implemented by different
//! cache backends (in-memory, Redis, etc.)

use crate::embedding::EmbeddingRecord;

/// Statistics about cache performance
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current cache size in bytes
    pub current_size_bytes: usize,
    /// Maximum cache size in bytes
    pub max_size_bytes: usize,
    /// Number of entries in cache
    pub entry_count: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

/// Trait for cache implementations
pub trait Cache: Send + Sync {
    /// Put a record into the cache
    fn put(&self, record: EmbeddingRecord);

    /// Get a record from the cache
    fn get(&self, id: &str, namespace: Option<&str>) -> Option<EmbeddingRecord>;

    /// Remove a record from the cache
    fn remove(&self, id: &str);

    /// Clear all records in a namespace
    fn clear_namespace(&self, namespace: Option<&str>);

    /// Get cache statistics
    fn stats(&self) -> CacheStats;

    /// Estimate size of a record (for eviction decisions)
    fn estimate_size(&self, record: &EmbeddingRecord) -> usize {
        super::estimate_record_size(record)
    }
}
