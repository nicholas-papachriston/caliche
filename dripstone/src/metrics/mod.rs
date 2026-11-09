//! Unified metrics interface for telemetry collection across backends.
//!
//! Provides a standardized way to collect performance metrics, operational
//! statistics, and monitoring data from vector operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

/// Trait for collecting vector operation metrics
pub trait VectorMetrics: Send + Sync {
    /// Record search operation latency
    fn record_search_latency(&self, duration: Duration);

    /// Record index build operation latency
    fn record_index_build_latency(&self, duration: Duration);

    /// Record upsert operation latency
    fn record_upsert_latency(&self, duration: Duration);

    /// Record the current index size (number of vectors)
    fn record_index_size(&self, size: usize);

    /// Increment cache hit counter
    fn increment_cache_hit(&self);

    /// Increment cache miss counter
    fn increment_cache_miss(&self);

    /// Record search result count
    fn record_search_result_count(&self, count: usize);

    /// Record vector dimension
    fn record_vector_dimension(&self, dimension: usize);

    /// Record error occurrence
    fn record_error(&self, error_type: &str);

    /// Record batch operation size
    fn record_batch_size(&self, size: usize);

    /// Record memory usage (in bytes)
    fn record_memory_usage(&self, bytes: usize);

    /// Record quantization compression ratio
    fn record_compression_ratio(&self, ratio: f32);
}

/// No-op metrics implementation (default, does nothing)
#[derive(Debug, Clone, Default)]
#[must_use]
pub struct NoOpMetrics;

impl VectorMetrics for NoOpMetrics {
    fn record_search_latency(&self, _duration: Duration) {}
    fn record_index_build_latency(&self, _duration: Duration) {}
    fn record_upsert_latency(&self, _duration: Duration) {}
    fn record_index_size(&self, _size: usize) {}
    fn increment_cache_hit(&self) {}
    fn increment_cache_miss(&self) {}
    fn record_search_result_count(&self, _count: usize) {}
    fn record_vector_dimension(&self, _dimension: usize) {}
    fn record_error(&self, _error_type: &str) {}
    fn record_batch_size(&self, _size: usize) {}
    fn record_memory_usage(&self, _bytes: usize) {}
    fn record_compression_ratio(&self, _ratio: f32) {}
}

/// Maximum number of samples retained in a bounded latency reservoir.
///
/// `SimpleMetrics` is meant for development and tests; we keep the reservoir
/// small enough that long-running uses can't grow unbounded.
const LATENCY_RESERVOIR_CAPACITY: usize = 4096;
const LATENCY_RESERVOIR_CAPACITY_U32: u32 = 4096;

/// Bounded latency reservoir using first-fill-then-overwrite semantics.
///
/// Keeps the most recent `LATENCY_RESERVOIR_CAPACITY` samples. This is
/// deterministic and cheap, and good enough for averaging in tests. For
/// production telemetry, implement `VectorMetrics` against a real metrics
/// backend (Prometheus, OpenTelemetry, etc.).
#[derive(Debug, Default)]
struct BoundedReservoir {
    samples: parking_lot::Mutex<Vec<Duration>>,
    cursor: AtomicUsize,
}

impl BoundedReservoir {
    fn record(&self, duration: Duration) {
        let mut samples = self.samples.lock();
        if samples.len() < LATENCY_RESERVOIR_CAPACITY {
            samples.push(duration);
            return;
        }
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed) % LATENCY_RESERVOIR_CAPACITY;
        samples[idx] = duration;
    }

    fn average(&self) -> Option<Duration> {
        let samples = self.samples.lock();
        if samples.is_empty() {
            return None;
        }
        let total: Duration = samples.iter().sum();
        let sample_count = u32::try_from(samples.len()).unwrap_or(LATENCY_RESERVOIR_CAPACITY_U32);
        drop(samples);
        Some(total / sample_count)
    }

    fn count(&self) -> usize {
        self.samples.lock().len()
    }
}

/// Simple in-memory metrics collector for testing and development.
///
/// All counters use atomics; latency samples use bounded reservoirs. For
/// real telemetry export use a production `VectorMetrics` implementation.
#[derive(Debug, Default)]
#[must_use]
pub struct SimpleMetrics {
    search_latencies: Arc<BoundedReservoir>,
    index_build_latencies: Arc<BoundedReservoir>,
    upsert_latencies: Arc<BoundedReservoir>,
    index_sizes: Arc<parking_lot::Mutex<Vec<usize>>>,
    cache_hits: Arc<AtomicU64>,
    cache_misses: Arc<AtomicU64>,
    search_result_counts: Arc<parking_lot::Mutex<Vec<usize>>>,
    errors: Arc<parking_lot::Mutex<HashMap<String, u64>>>,
    last_dimension: Arc<AtomicUsize>,
    total_batches: Arc<AtomicU64>,
    total_batched_items: Arc<AtomicU64>,
    last_memory_bytes: Arc<AtomicUsize>,
    compression_ratio_bits: Arc<AtomicU64>,
}

impl Clone for SimpleMetrics {
    fn clone(&self) -> Self {
        Self {
            search_latencies: Arc::clone(&self.search_latencies),
            index_build_latencies: Arc::clone(&self.index_build_latencies),
            upsert_latencies: Arc::clone(&self.upsert_latencies),
            index_sizes: Arc::clone(&self.index_sizes),
            cache_hits: Arc::clone(&self.cache_hits),
            cache_misses: Arc::clone(&self.cache_misses),
            search_result_counts: Arc::clone(&self.search_result_counts),
            errors: Arc::clone(&self.errors),
            last_dimension: Arc::clone(&self.last_dimension),
            total_batches: Arc::clone(&self.total_batches),
            total_batched_items: Arc::clone(&self.total_batched_items),
            last_memory_bytes: Arc::clone(&self.last_memory_bytes),
            compression_ratio_bits: Arc::clone(&self.compression_ratio_bits),
        }
    }
}

impl SimpleMetrics {
    /// Create a new `SimpleMetrics` instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Get average search latency across the retained reservoir.
    #[must_use]
    pub fn average_search_latency(&self) -> Option<Duration> {
        self.search_latencies.average()
    }

    /// Get average index-build latency across the retained reservoir.
    #[must_use]
    pub fn average_index_build_latency(&self) -> Option<Duration> {
        self.index_build_latencies.average()
    }

    /// Get average upsert latency across the retained reservoir.
    #[must_use]
    pub fn average_upsert_latency(&self) -> Option<Duration> {
        self.upsert_latencies.average()
    }

    /// Number of search latency samples currently retained.
    #[must_use]
    pub fn search_latency_sample_count(&self) -> usize {
        self.search_latencies.count()
    }

    /// Get current index size (last recorded value).
    #[must_use]
    pub fn current_index_size(&self) -> Option<usize> {
        self.index_sizes.lock().last().copied()
    }

    /// Get cache hit rate.
    #[must_use]
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        u64_to_f64(hits) / u64_to_f64(total)
    }

    /// Total cache hits recorded.
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Relaxed)
    }

    /// Total cache misses recorded.
    #[must_use]
    pub fn cache_misses(&self) -> u64 {
        self.cache_misses.load(Ordering::Relaxed)
    }

    /// Snapshot of error counts keyed by error type.
    #[must_use]
    pub fn error_counts(&self) -> HashMap<String, u64> {
        self.errors.lock().clone()
    }

    /// Get average search result count.
    #[must_use]
    pub fn average_search_result_count(&self) -> Option<f64> {
        let counts = self.search_result_counts.lock();
        if counts.is_empty() {
            return None;
        }
        let sum = counts.iter().copied().fold(0usize, usize::saturating_add);
        Some(usize_to_f64(sum) / usize_to_f64(counts.len()))
    }

    /// Last recorded vector dimension, if any.
    #[must_use]
    pub fn last_vector_dimension(&self) -> Option<usize> {
        match self.last_dimension.load(Ordering::Relaxed) {
            0 => None,
            dim => Some(dim),
        }
    }

    /// Total number of recorded batch operations.
    #[must_use]
    pub fn total_batch_operations(&self) -> u64 {
        self.total_batches.load(Ordering::Relaxed)
    }

    /// Total number of items across all recorded batches.
    #[must_use]
    pub fn total_batched_items(&self) -> u64 {
        self.total_batched_items.load(Ordering::Relaxed)
    }

    /// Average batch size across all recorded batches.
    #[must_use]
    pub fn average_batch_size(&self) -> Option<f64> {
        let batches = self.total_batches.load(Ordering::Relaxed);
        if batches == 0 {
            return None;
        }
        Some(u64_to_f64(self.total_batched_items.load(Ordering::Relaxed)) / u64_to_f64(batches))
    }

    /// Last recorded memory usage in bytes.
    #[must_use]
    pub fn last_memory_usage_bytes(&self) -> Option<usize> {
        match self.last_memory_bytes.load(Ordering::Relaxed) {
            0 => None,
            v => Some(v),
        }
    }

    /// Last recorded compression ratio.
    #[must_use]
    pub fn last_compression_ratio(&self) -> Option<f32> {
        match self.compression_ratio_bits.load(Ordering::Relaxed) {
            0 => None,
            encoded => {
                let bits = u32::try_from(encoded & u64::from(u32::MAX)).ok()?;
                Some(f32::from_bits(bits))
            }
        }
    }
}

fn u64_to_f64(value: u64) -> f64 {
    const U32_RADIX: f64 = 4_294_967_296.0;
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high).mul_add(U32_RADIX, f64::from(low))
}

fn usize_to_f64(value: usize) -> f64 {
    u64_to_f64(u64::try_from(value).unwrap_or(u64::MAX))
}

impl VectorMetrics for SimpleMetrics {
    fn record_search_latency(&self, duration: Duration) {
        self.search_latencies.record(duration);
    }

    fn record_index_build_latency(&self, duration: Duration) {
        self.index_build_latencies.record(duration);
    }

    fn record_upsert_latency(&self, duration: Duration) {
        self.upsert_latencies.record(duration);
    }

    fn record_index_size(&self, size: usize) {
        let mut sizes = self.index_sizes.lock();
        if sizes.len() >= LATENCY_RESERVOIR_CAPACITY {
            sizes.remove(0);
        }
        sizes.push(size);
    }

    fn increment_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    fn increment_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_search_result_count(&self, count: usize) {
        let mut counts = self.search_result_counts.lock();
        if counts.len() >= LATENCY_RESERVOIR_CAPACITY {
            counts.remove(0);
        }
        counts.push(count);
    }

    fn record_vector_dimension(&self, dimension: usize) {
        self.last_dimension.store(dimension, Ordering::Relaxed);
    }

    fn record_error(&self, error_type: &str) {
        *self
            .errors
            .lock()
            .entry(error_type.to_string())
            .or_insert(0) += 1;
    }

    fn record_batch_size(&self, size: usize) {
        let _ = self
            .total_batches
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                Some(total.saturating_add(1))
            });
        let size = u64::try_from(size).unwrap_or(u64::MAX);
        let _ =
            self.total_batched_items
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                    Some(total.saturating_add(size))
                });
    }

    fn record_memory_usage(&self, bytes: usize) {
        self.last_memory_bytes.store(bytes, Ordering::Relaxed);
    }

    fn record_compression_ratio(&self, ratio: f32) {
        // The high bit marks presence, while the low 32 bits preserve the
        // exact IEEE-754 value without a lossy numeric cast.
        let encoded = (1_u64 << 32) | u64::from(ratio.max(0.0).to_bits());
        self.compression_ratio_bits
            .store(encoded, Ordering::Relaxed);
    }
}

/// Type alias for shared metrics
pub type SharedMetrics = Arc<dyn VectorMetrics>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_metrics() {
        let metrics = NoOpMetrics;
        metrics.record_search_latency(Duration::from_millis(10));
        metrics.increment_cache_hit();
        // Should not panic
    }

    #[test]
    fn test_simple_metrics() {
        let metrics = SimpleMetrics::new();
        metrics.record_search_latency(Duration::from_millis(10));
        metrics.record_search_latency(Duration::from_millis(20));
        metrics.increment_cache_hit();
        metrics.increment_cache_miss();
        metrics.increment_cache_miss();

        assert_eq!(
            metrics.average_search_latency(),
            Some(Duration::from_millis(15))
        );
        assert!((metrics.cache_hit_rate() - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_error_tracking() {
        let metrics = SimpleMetrics::new();
        metrics.record_error("InvalidVectorLength");
        metrics.record_error("InvalidVectorLength");
        metrics.record_error("StorageError");

        let errors = metrics.error_counts();
        assert_eq!(errors.get("InvalidVectorLength"), Some(&2));
        assert_eq!(errors.get("StorageError"), Some(&1));
    }

    #[test]
    fn test_latency_reservoir_bounded() {
        let metrics = SimpleMetrics::new();
        for i in 0..(LATENCY_RESERVOIR_CAPACITY + 128) {
            metrics
                .record_search_latency(Duration::from_micros(u64::try_from(i).unwrap_or(u64::MAX)));
        }
        assert_eq!(
            metrics.search_latency_sample_count(),
            LATENCY_RESERVOIR_CAPACITY,
            "reservoir must not exceed capacity"
        );
    }

    #[test]
    fn test_batch_and_dimension_and_memory_tracking() {
        let metrics = SimpleMetrics::new();
        metrics.record_batch_size(10);
        metrics.record_batch_size(20);
        metrics.record_vector_dimension(384);
        metrics.record_memory_usage(1024);
        metrics.record_compression_ratio(0.25);

        assert_eq!(metrics.total_batch_operations(), 2);
        assert_eq!(metrics.total_batched_items(), 30);
        assert_eq!(metrics.average_batch_size(), Some(15.0));
        assert_eq!(metrics.last_vector_dimension(), Some(384));
        assert_eq!(metrics.last_memory_usage_bytes(), Some(1024));
        let ratio = metrics.last_compression_ratio().expect("recorded");
        assert!((ratio - 0.25).abs() < 1e-3);
    }
}
