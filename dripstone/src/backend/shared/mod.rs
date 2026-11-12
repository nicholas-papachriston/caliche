//! Shared utilities and traits for tiered storage backends
//!
//! This module provides common functionality used by TieredChainBackend
//! and ColdForgeBackend, including:
//! - Access tracking and metrics
//! - Size estimation utilities
//! - Eviction policies
//! - Cache interfaces

pub mod cache;

use crate::embedding::EmbeddingRecord;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use time::OffsetDateTime;

pub use cache::Cache;

/// A spawned background task that stops when this guard drops.
///
/// Dropping a `JoinHandle` detaches its task, so a periodic task that holds
/// its own clones of a backend's stores keeps writing after the owner is
/// gone. The owner holds this guard instead; dropping it aborts the task at
/// its next await point.
#[derive(Debug)]
#[must_use = "dropping the guard stops the task"]
pub struct BackgroundTask(tokio::task::AbortHandle);

impl BackgroundTask {
    /// Spawn `task` on the current Tokio runtime.
    ///
    /// # Panics
    ///
    /// Panics when called outside a Tokio runtime, as `tokio::spawn` does.
    pub fn spawn<F>(task: F) -> Self
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        Self(tokio::spawn(task).abort_handle())
    }
}

impl Drop for BackgroundTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Access metrics for tracking vector usage patterns
#[derive(Debug, Clone)]
pub struct AccessMetrics {
    /// Number of times this vector has been accessed
    pub access_count: usize,
    /// Last time this vector was accessed
    pub last_accessed: OffsetDateTime,
    /// Estimated size in bytes
    pub size_bytes: usize,
}

impl AccessMetrics {
    /// Create new access metrics
    #[must_use]
    pub fn new(size_bytes: usize) -> Self {
        Self {
            access_count: 1,
            last_accessed: OffsetDateTime::now_utc(),
            size_bytes,
        }
    }

    /// Record a new access
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = OffsetDateTime::now_utc();
    }
}

/// Shared access tracker for monitoring vector usage
pub struct AccessTracker {
    /// Metrics per vector (keyed by composite key: namespace:id)
    metrics: Arc<RwLock<HashMap<String, AccessMetrics>>>,
}

impl AccessTracker {
    /// Create a new access tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record an access for a vector
    pub fn record_access(&self, key: String, size_bytes: usize) {
        let mut metrics = self.metrics.write();
        metrics
            .entry(key)
            .and_modify(AccessMetrics::record_access)
            .or_insert_with(|| AccessMetrics::new(size_bytes));
    }

    /// Get access metrics for a vector
    #[must_use]
    pub fn get_metrics(&self, key: &str) -> Option<AccessMetrics> {
        let metrics = self.metrics.read();
        metrics.get(key).cloned()
    }

    /// Check if a vector should be promoted based on access count
    #[must_use]
    pub fn should_promote(&self, key: &str, threshold: usize) -> bool {
        let metrics = self.metrics.read();
        metrics
            .get(key)
            .is_some_and(|m| m.access_count >= threshold)
    }

    /// Remove metrics for a vector
    #[must_use]
    pub fn remove(&self, key: &str) -> Option<AccessMetrics> {
        let mut metrics = self.metrics.write();
        metrics.remove(key)
    }

    /// Clear metrics for a namespace
    pub fn clear_namespace(&self, namespace: &str) {
        let mut metrics = self.metrics.write();
        let prefix = format!("{namespace}:");
        metrics.retain(|key, _| !key.starts_with(&prefix));
    }

    /// Get all metrics (for eviction decisions)
    #[must_use]
    pub fn get_all_metrics(&self) -> Vec<(String, AccessMetrics)> {
        let metrics = self.metrics.read();
        metrics
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate the size of an embedding record in bytes
#[must_use]
pub fn estimate_record_size(record: &EmbeddingRecord) -> usize {
    let base_size = std::mem::size_of::<EmbeddingRecord>();
    let vector_size = record.vector.len() * std::mem::size_of::<f32>();
    let id_size = record.id.len();
    let namespace_size = record.namespace.as_ref().map_or(0, |ns| ns.len());
    let metadata_size = record.metadata.as_ref().map_or(0, |m| m.to_string().len());

    base_size + vector_size + id_size + namespace_size + metadata_size + 100 // overhead
}

/// Build a composite key from namespace and ID
#[must_use]
pub fn build_composite_key(namespace: Option<&str>, id: &str) -> String {
    namespace.map_or_else(|| format!("default:{id}"), |ns| format!("{ns}:{id}"))
}

/// Parse a composite key into namespace and ID
#[must_use]
pub fn parse_composite_key(key: &str) -> (Option<&str>, &str) {
    if let Some((ns, id)) = key.split_once(':') {
        let namespace = if ns == "default" { None } else { Some(ns) };
        (namespace, id)
    } else {
        (None, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_tracker() {
        let tracker = AccessTracker::new();
        let key = "test:vec1".to_string();

        tracker.record_access(key.clone(), 1024);
        assert_eq!(
            tracker
                .get_metrics(&key)
                .expect("Expected metrics to exist")
                .access_count,
            1
        );

        tracker.record_access(key.clone(), 1024);
        assert_eq!(
            tracker
                .get_metrics(&key)
                .expect("Expected metrics to exist")
                .access_count,
            2
        );

        assert!(tracker.should_promote(&key, 2));
        assert!(!tracker.should_promote(&key, 3));
    }

    #[test]
    fn test_composite_key() {
        assert_eq!(build_composite_key(Some("ns1"), "id1"), "ns1:id1");
        assert_eq!(build_composite_key(None, "id1"), "default:id1");

        assert_eq!(parse_composite_key("ns1:id1"), (Some("ns1"), "id1"));
        assert_eq!(parse_composite_key("default:id1"), (None, "id1"));
    }
}
