//! Cache performance metrics for tracking hit/miss rates.
//!
//! This module provides metrics tracking for the provider and tool registry caches
//! to help identify performance bottlenecks and optimize cache settings.

use std::sync::atomic::{AtomicU64, Ordering};

/// Cache performance statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    hits: AtomicU64,
    /// Number of cache misses
    misses: AtomicU64,
}

impl CacheStats {
    /// Create a new cache stats tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a cache hit
    pub fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a cache miss
    pub fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of cache hits
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get the number of cache misses
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get the total number of cache accesses
    pub fn total(&self) -> u64 {
        self.hits() + self.misses()
    }

    /// Get the cache hit rate as a percentage (0.0 - 100.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.hits() as f64 / total as f64) * 100.0
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

/// Aggregated cache metrics for all cache types
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Provider cache statistics
    pub provider_cache: CacheStats,
    /// Tool registry cache statistics
    pub tool_registry_cache: CacheStats,
}

impl CacheMetrics {
    /// Create a new cache metrics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot of current metrics
    pub fn snapshot(&self) -> CacheMetricsSnapshot {
        CacheMetricsSnapshot {
            provider_hits: self.provider_cache.hits(),
            provider_misses: self.provider_cache.misses(),
            provider_hit_rate: self.provider_cache.hit_rate(),
            registry_hits: self.tool_registry_cache.hits(),
            registry_misses: self.tool_registry_cache.misses(),
            registry_hit_rate: self.tool_registry_cache.hit_rate(),
        }
    }

    /// Log a warning if hit rate drops below threshold
    pub fn check_hit_rate(&self, threshold: f64) {
        let provider_rate = self.provider_cache.hit_rate();
        let registry_rate = self.tool_registry_cache.hit_rate();

        if self.provider_cache.total() > 10 && provider_rate < threshold {
            tracing::warn!(
                "Provider cache hit rate low: {:.1}% (threshold: {:.1}%)",
                provider_rate,
                threshold
            );
        }

        if self.tool_registry_cache.total() > 10 && registry_rate < threshold {
            tracing::warn!(
                "Tool registry cache hit rate low: {:.1}% (threshold: {:.1}%)",
                registry_rate,
                threshold
            );
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.provider_cache.reset();
        self.tool_registry_cache.reset();
    }
}

/// Serializable snapshot of cache metrics
#[derive(Debug, Clone, serde::Serialize)]
pub struct CacheMetricsSnapshot {
    pub provider_hits: u64,
    pub provider_misses: u64,
    pub provider_hit_rate: f64,
    pub registry_hits: u64,
    pub registry_misses: u64,
    pub registry_hit_rate: f64,
}
