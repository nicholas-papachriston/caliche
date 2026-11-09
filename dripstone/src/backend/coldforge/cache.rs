//! Caching layer for `ColdForge` backend
//!
//! Provides memory-based caching with LRU eviction for hot data.

use crate::backend::shared::cache::{Cache, CacheStats};
use crate::embedding::EmbeddingRecord;
use crate::utils::numeric::u64_to_f64;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::Arc;
use time::OffsetDateTime;

/// Cache entry metadata
///
/// # Memory Optimization
///
/// Records are stored as `Arc<EmbeddingRecord>` for cheap cloning when
/// returning cached values. This avoids copying ~6KB vectors per cache hit.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Record wrapped in Arc for cheap cloning
    record: Arc<EmbeddingRecord>,
    last_accessed: OffsetDateTime,
    size_bytes: usize,
}

/// LRU cache for vector records
///
/// This type is Send because all its fields are Send:
/// - `DashMap` is Send
/// - `Arc<Mutex<T>>` is Send if T is Send
/// - `parking_lot::Mutex` itself is Send (only guards aren't, but we don't hold them across await)
pub struct ColdForgeCache {
    /// Max cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: Arc<Mutex<usize>>,
    /// Cache entries by ID
    entries: DashMap<String, CacheEntry>,
    /// LRU queue (most recent at back)
    lru_queue: Arc<RwLock<VecDeque<String>>>,
    /// Access count for statistics
    hits: Arc<Mutex<u64>>,
    /// Miss count for statistics
    misses: Arc<Mutex<u64>>,
}

impl ColdForgeCache {
    /// Create a new cache with the given maximum size
    #[must_use]
    pub fn new(max_size_bytes: usize) -> Self {
        Self {
            max_size_bytes,
            current_size_bytes: Arc::new(Mutex::new(0)),
            entries: DashMap::new(),
            lru_queue: Arc::new(RwLock::new(VecDeque::new())),
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Estimate size of a record in bytes (uses shared utility)
    fn estimate_size(record: &EmbeddingRecord) -> usize {
        crate::backend::shared::estimate_record_size(record)
    }

    /// Evict entries to make room for new data
    fn evict_to_fit(&self, needed_bytes: usize) {
        let mut current_size = self.current_size_bytes.lock();

        while *current_size + needed_bytes > self.max_size_bytes {
            let id_to_evict = {
                let mut queue = self.lru_queue.write();
                queue.pop_front()
            };

            if let Some(id) = id_to_evict {
                if let Some((_, entry)) = self.entries.remove(&id) {
                    *current_size = current_size.saturating_sub(entry.size_bytes);
                    tracing::trace!("Evicted cache entry: {}", id);
                } else {
                    break; // No more entries to evict
                }
            } else {
                break; // Queue is empty
            }
        }
        drop(current_size);
    }

    /// Update LRU position for an ID
    fn touch(&self, id: &str) {
        let mut queue = self.lru_queue.write();

        // Remove old position
        queue.retain(|cached_id| cached_id != id);

        // Add to back (most recent)
        queue.push_back(id.to_string());
    }

    /// Put a record in the cache (takes ownership and wraps in Arc)
    pub fn put(&self, record: EmbeddingRecord) {
        // Build composite key including namespace for proper isolation
        let key =
            crate::backend::shared::build_composite_key(record.namespace.as_deref(), &record.id);
        let size = Self::estimate_size(&record);

        // Wrap record in Arc for efficient storage and retrieval
        let record_arc = Arc::new(record);

        // Check if already in cache
        if let Some(mut entry) = self.entries.get_mut(&key) {
            let old_size = entry.size_bytes;
            entry.record = record_arc;
            entry.last_accessed = OffsetDateTime::now_utc();
            entry.size_bytes = size;

            // Update size
            let mut current_size = self.current_size_bytes.lock();
            *current_size = current_size.saturating_sub(old_size).saturating_add(size);

            // Release locks before taking the LRU lock. Eviction acquires the
            // size lock before touching the entry map, so retaining either
            // guard here would invert the lock order.
            drop(current_size);
            drop(entry);
            self.touch(&key);

            return;
        }

        // Evict if necessary
        self.evict_to_fit(size);

        // Add new entry with Arc-wrapped record
        let entry = CacheEntry {
            record: record_arc,
            last_accessed: OffsetDateTime::now_utc(),
            size_bytes: size,
        };

        self.entries.insert(key.clone(), entry);

        // Update size
        *self.current_size_bytes.lock() += size;

        // Add to LRU queue
        self.lru_queue.write().push_back(key);
    }

    /// Get a record from the cache
    ///
    /// Returns a clone of the `EmbeddingRecord`. Since the record's fields use Arc
    /// internally (id, namespace, vector), this clone is cheap.
    #[must_use]
    pub fn get(&self, id: &str, namespace: Option<&str>) -> Option<EmbeddingRecord> {
        // Build composite key including namespace
        let key = crate::backend::shared::build_composite_key(namespace, id);

        if let Some(mut entry) = self.entries.get_mut(&key) {
            entry.last_accessed = OffsetDateTime::now_utc();
            *self.hits.lock() += 1;

            // Dereference Arc and clone - now cheap because EmbeddingRecord
            // uses Arc for its vector, id, and namespace fields
            let record = (*entry.record).clone();

            // Update LRU
            drop(entry);
            self.touch(&key);

            Some(record)
        } else {
            *self.misses.lock() += 1;
            None
        }
    }

    /// Remove one namespace's copy of `id`.
    ///
    /// The same chunk id may exist in more than one namespace. Deleting in
    /// one namespace must leave the others cached.
    pub fn remove_in_namespace(&self, id: &str, namespace: Option<&str>) {
        let key = crate::backend::shared::build_composite_key(namespace, id);
        self.remove_key(&key);
    }

    fn remove_key(&self, key: &str) {
        if let Some((_, entry)) = self.entries.remove(key) {
            let mut current_size = self.current_size_bytes.lock();
            *current_size = current_size.saturating_sub(entry.size_bytes);
            drop(current_size);
            let mut queue = self.lru_queue.write();
            queue.retain(|cached_id| cached_id != key);
        }
    }

    /// Remove a record from the cache in every namespace.
    pub fn remove(&self, id: &str) {
        // Try to remove with default namespace first, then try with explicit namespace
        // Since we don't know the namespace, we need to search for matching IDs
        let mut to_remove = Vec::new();
        {
            let queue = self.lru_queue.read();
            for cached_key in queue.iter() {
                let (_, cached_id) = crate::backend::shared::parse_composite_key(cached_key);
                if cached_id == id {
                    to_remove.push(cached_key.clone());
                }
            }
        }

        for key in to_remove {
            self.remove_key(&key);
        }
    }

    /// List all records in a namespace
    #[must_use]
    pub fn list_namespace(&self, namespace: Option<&str>) -> Vec<EmbeddingRecord> {
        self.entries
            .iter()
            .filter_map(|entry| {
                let ns_match = match (namespace, entry.record.namespace.as_deref()) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };

                if ns_match {
                    // Dereference Arc and clone - cheap due to internal Arc fields
                    Some((*entry.record).clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Drop every cached envelope. Does not touch object storage.
    ///
    /// Release the size lock before taking the LRU lock. Eviction holds the
    /// size lock and then the LRU lock, so the opposite order deadlocks.
    pub fn clear_all(&self) {
        self.entries.clear();
        *self.current_size_bytes.lock() = 0;
        self.lru_queue.write().clear();
    }

    /// Clear all records in a namespace
    pub fn clear_namespace(&self, namespace: Option<&str>) {
        let ids_to_remove: Vec<String> = self
            .entries
            .iter()
            .filter_map(|entry| {
                let ns_match = match (namespace, entry.record.namespace.as_deref()) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                };

                if ns_match {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();

        for id in ids_to_remove {
            self.remove(&id);
        }
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let hits = *self.hits.lock();
        let misses = *self.misses.lock();
        let total = hits.saturating_add(misses);
        let hit_rate = if total > 0 {
            u64_to_f64(hits) / u64_to_f64(total)
        } else {
            0.0
        };

        CacheStats {
            current_size_bytes: *self.current_size_bytes.lock(),
            max_size_bytes: self.max_size_bytes,
            entry_count: self.entries.len(),
            hits,
            misses,
            hit_rate,
        }
    }
}

// Implement the shared Cache trait
impl Cache for ColdForgeCache {
    fn put(&self, record: EmbeddingRecord) {
        self.put(record);
    }

    fn get(&self, id: &str, namespace: Option<&str>) -> Option<EmbeddingRecord> {
        self.get(id, namespace)
    }

    fn remove(&self, id: &str) {
        self.remove(id);
    }

    fn clear_namespace(&self, namespace: Option<&str>) {
        self.clear_namespace(namespace);
    }

    fn stats(&self) -> CacheStats {
        self.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(id: &str, vector_size: usize) -> EmbeddingRecord {
        EmbeddingRecord::new(
            id,
            (0..vector_size)
                .map(|_| fastrand::f32())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn test_cache_put_get() {
        let cache = ColdForgeCache::new(1024 * 1024); // 1MB
        let record = create_test_record("test1", 128);

        cache.put(record);

        let retrieved = cache.get("test1", None);
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved
                .expect("Expected cache entry to exist")
                .id
                .as_ref(),
            "test1"
        );
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ColdForgeCache::new(1000); // Small cache

        // Add records until eviction happens
        for i in 0..10 {
            let record = create_test_record(&format!("test{i}"), 128);
            cache.put(record);
        }

        // First records should be evicted
        assert!(cache.get("test0", None).is_none());

        // Recent records should still be there
        assert!(cache.get("test9", None).is_some());
    }

    #[test]
    fn test_cache_namespace_isolation() {
        let cache = ColdForgeCache::new(1024 * 1024);

        let mut record1 = create_test_record("test1", 128);
        record1.namespace = Some("ns1".into());

        let mut record2 = create_test_record("test1", 128);
        record2.namespace = Some("ns2".into());

        cache.put(record1);
        cache.put(record2);

        assert!(cache.get("test1", Some("ns1")).is_some());
        assert!(cache.get("test1", Some("ns2")).is_some());
    }

    #[test]
    fn test_cache_stats() {
        let cache = ColdForgeCache::new(1024 * 1024);
        let record = create_test_record("test1", 128);

        cache.put(record);

        let _ = cache.get("test1", None); // Hit
        let _ = cache.get("test2", None); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() <= f64::EPSILON);
    }
}
