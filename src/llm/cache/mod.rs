mod codec;
pub mod conversion;
pub mod response;

// Explicit exports from conversion module
pub use conversion::{
    ConversionCache, hash_messages_for_cache, hash_messages_for_cache_skip_system,
    hash_tools_for_cache,
};

pub use response::{CachePolicy, CacheStats, CachedResponse, LlmCache};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// Multi-level cache with L1 (memory) and L2 (Redis) tiers
pub struct MultiLevelCache<K: Hash + Eq + Clone, V: Clone> {
    l1_cache: Arc<DashMap<K, CacheEntry<V>>>,
    l1_max_size: usize,
    default_ttl: Duration,
}

#[derive(Clone)]
struct CacheEntry<V> {
    value: V,
    expires_at: SystemTime,
    hit_count: u64,
    last_accessed: SystemTime,
}

impl<K: Hash + Eq + Clone, V: Clone> MultiLevelCache<K, V> {
    #[must_use]
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            l1_cache: Arc::new(DashMap::new()),
            l1_max_size: max_size,
            default_ttl: ttl,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        if let Some(mut entry) = self.l1_cache.get_mut(key) {
            if entry.expires_at > SystemTime::now() {
                entry.hit_count += 1;
                entry.last_accessed = SystemTime::now();
                return Some(entry.value.clone());
            }
            drop(entry);
            self.l1_cache.remove(key);
        }
        None
    }

    pub fn put(&self, key: K, value: V, ttl: Option<Duration>) {
        if self.l1_cache.len() >= self.l1_max_size {
            self.evict_lfu();
        }

        let expires_at = SystemTime::now() + ttl.unwrap_or(self.default_ttl);
        self.l1_cache.insert(
            key,
            CacheEntry {
                value,
                expires_at,
                hit_count: 0,
                last_accessed: SystemTime::now(),
            },
        );
    }

    pub fn remove(&self, key: &K) {
        self.l1_cache.remove(key);
    }

    fn evict_lfu(&self) {
        // Evict least frequently used entry (hit-count based).
        if let Some(min_entry) = self
            .l1_cache
            .iter()
            .min_by_key(|entry| entry.value().hit_count)
        {
            let key = min_entry.key().clone();
            drop(min_entry);
            self.l1_cache.remove(&key);
        }
    }

    pub fn clear(&self) {
        self.l1_cache.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.l1_cache.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.l1_cache.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub hit_rate: f64,
}
