//! Cache backend abstraction.
//!
//! This module provides a unified interface for caching operations across
//! different backends. Implement the `CacheBackend` trait to add custom
//! cache implementations like Redis, Memcached, or custom in-memory stores.
//!
//! # Examples
//!
//! ## Using In-Memory Cache
//!
//! ```rust,no_run
//! use caliche::traits::cache::{CacheBackend, InMemoryCacheBackend};
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let cache = InMemoryCacheBackend::new(1000, Duration::from_secs(300));
//! cache.set("key", b"value", Some(Duration::from_secs(60))).await?;
//! let value = cache.get("key").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Typed Cache
//!
//! ```rust,no_run
//! use caliche::traits::cache::{TypedCacheBackend, InMemoryCacheBackend};
//! use serde::{Deserialize, Serialize};
//! use std::time::Duration;
//!
//! #[derive(Serialize, Deserialize)]
//! struct MyData {
//!     value: String,
//! }
//!
//! # async fn example() -> anyhow::Result<()> {
//! let cache = InMemoryCacheBackend::new(1000, Duration::from_secs(300));
//! let data = MyData { value: "test".to_string() };
//! cache.set_typed("key", &data, Some(Duration::from_secs(60))).await?;
//! let retrieved: Option<MyData> = cache.get_typed("key").await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use async_trait::async_trait;
use serde::{Serialize, de::DeserializeOwned};
use std::sync::Arc;
use std::time::Duration;

/// Cache statistics for monitoring and debugging
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Number of evictions
    pub evictions: u64,
    /// Current cache size (number of entries)
    pub size: usize,
    /// Hit rate (hits / (hits + misses))
    pub hit_rate: f64,
    /// Backend name
    pub backend: String,
}

/// Trait for cache backends that can store and retrieve cached data
#[async_trait]
pub trait CacheBackend: Send + Sync + std::fmt::Debug {
    /// Get a value from cache
    ///
    /// # Arguments
    /// * `key` - Cache key
    ///
    /// # Returns
    /// * `Ok(Some(Vec<u8>))` - Cached value if found
    /// * `Ok(None)` - Key not found
    /// * `Err` - Error retrieving from cache
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Set a value in cache with optional TTL
    ///
    /// # Arguments
    /// * `key` - Cache key
    /// * `value` - Value to cache
    /// * `ttl` - Time to live (None means use default TTL)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully cached
    /// * `Err` - Error storing in cache
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()>;

    /// Delete a value from cache
    ///
    /// # Arguments
    /// * `key` - Cache key to delete
    ///
    /// # Returns
    /// * `Ok(())` - Successfully deleted (or key didn't exist)
    /// * `Err` - Error deleting from cache
    async fn delete(&self, key: &str) -> Result<()>;

    /// Check if key exists
    ///
    /// Default implementation calls `get()` and checks if result is Some.
    /// Override for more efficient implementations.
    async fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.get(key).await?.is_some())
    }

    /// Clear all cache entries (use with caution)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully cleared
    /// * `Err` - Error clearing cache
    async fn clear(&self) -> Result<()>;

    /// Get cache statistics
    ///
    /// # Returns
    /// * `Ok(CacheStats)` - Current cache statistics
    /// * `Err` - Error retrieving statistics
    async fn stats(&self) -> Result<CacheStats>;

    /// Get backend name for logging
    fn backend_name(&self) -> &str;
}

/// Helper trait for typed caching (auto-implemented for all CacheBackend implementations)
///
/// This trait provides convenience methods for caching serializable types
/// without manually handling serialization.
pub trait TypedCacheBackend: CacheBackend {
    /// Get a typed value from cache
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize (must implement `DeserializeOwned + Send`)
    ///
    /// # Returns
    /// * `Ok(Some(T))` - Cached value if found and deserializable
    /// * `Ok(None)` - Key not found
    /// * `Err` - Error retrieving or deserializing
    fn get_typed<T: DeserializeOwned + Send>(
        &self,
        key: &str,
    ) -> impl std::future::Future<Output = Result<Option<T>>> + Send {
        async move {
            if let Some(bytes) = self.get(key).await? {
                // Deserialize synchronously - this is safe because bytes is owned
                match serde_json::from_slice(&bytes) {
                    Ok(value) => Ok(Some(value)),
                    Err(e) => Err(anyhow::anyhow!("Failed to deserialize cached value: {}", e)),
                }
            } else {
                Ok(None)
            }
        }
    }

    /// Set a typed value in cache
    ///
    /// # Type Parameters
    /// * `T` - Type to serialize (must implement `Serialize + Send`)
    ///
    /// # Arguments
    /// * `key` - Cache key
    /// * `value` - Value to cache
    /// * `ttl` - Time to live (None means use default TTL)
    ///
    /// # Returns
    /// * `Ok(())` - Successfully cached
    /// * `Err` - Error serializing or storing
    fn set_typed<T: Serialize + Send>(
        &self,
        key: &str,
        value: &T,
        ttl: Option<Duration>,
    ) -> impl std::future::Future<Output = Result<()>> + Send {
        // Serialize synchronously before async operation to ensure Send
        let serialization_result = serde_json::to_vec(value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize value: {}", e));
        let key = key.to_string();
        async move {
            let bytes = serialization_result?;
            self.set(&key, &bytes, ttl).await
        }
    }
}

// Auto-implement for all CacheBackend implementations
impl<T: CacheBackend> TypedCacheBackend for T {}

// Implement CacheBackend for Arc-wrapped CacheBackend
#[async_trait]
impl CacheBackend for Arc<dyn CacheBackend> {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        (**self).get(key).await
    }

    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        (**self).set(key, value, ttl).await
    }

    async fn delete(&self, key: &str) -> Result<()> {
        (**self).delete(key).await
    }

    async fn clear(&self) -> Result<()> {
        (**self).clear().await
    }

    async fn stats(&self) -> Result<CacheStats> {
        (**self).stats().await
    }

    fn backend_name(&self) -> &str {
        (**self).backend_name()
    }
}

/// In-memory cache backend using DashMap
#[derive(Debug)]
pub struct InMemoryCacheBackend {
    cache: dashmap::DashMap<String, CacheEntry>,
    max_size: usize,
    default_ttl: Duration,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    evictions: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: Vec<u8>,
    expires_at: std::time::SystemTime,
    hit_count: u64,
}

impl InMemoryCacheBackend {
    /// Create a new in-memory cache backend
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of entries before eviction
    /// * `default_ttl` - Default time-to-live for entries
    pub fn new(max_size: usize, default_ttl: Duration) -> Self {
        Self {
            cache: dashmap::DashMap::new(),
            max_size,
            default_ttl,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            evictions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn evict_lru(&self) {
        if let Some(min_entry) = self
            .cache
            .iter()
            .min_by_key(|entry| entry.value().hit_count)
        {
            let key = min_entry.key().clone();
            drop(min_entry);
            self.cache.remove(&key);
            self.evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn cleanup_expired(&self) {
        let now = std::time::SystemTime::now();
        let mut to_remove = Vec::new();

        for entry in self.cache.iter() {
            if entry.value().expires_at < now {
                to_remove.push(entry.key().clone());
            }
        }

        for key in to_remove {
            self.cache.remove(&key);
        }
    }
}

#[async_trait]
impl CacheBackend for InMemoryCacheBackend {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Periodic cleanup of expired entries
        if !self.cache.is_empty() && self.cache.len().is_multiple_of(100) {
            self.cleanup_expired();
        }

        if let Some(mut entry) = self.cache.get_mut(key) {
            let now = std::time::SystemTime::now();
            if entry.expires_at > now {
                entry.hit_count += 1;
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Some(entry.value.clone()));
            } else {
                // Expired, remove it
                drop(entry);
                self.cache.remove(key);
            }
        }

        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(None)
    }

    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        // Evict if at capacity
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        let ttl = ttl.unwrap_or(self.default_ttl);
        let expires_at = std::time::SystemTime::now() + ttl;

        self.cache.insert(
            key.to_string(),
            CacheEntry {
                value: value.to_vec(),
                expires_at,
                hit_count: 0,
            },
        );

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.cache.remove(key);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.cache.clear();
        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        let hits = self.hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.misses.load(std::sync::atomic::Ordering::Relaxed);
        let evictions = self.evictions.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };

        Ok(CacheStats {
            hits,
            misses,
            evictions,
            size: self.cache.len(),
            hit_rate,
            backend: "in-memory".to_string(),
        })
    }

    fn backend_name(&self) -> &str {
        "in-memory"
    }
}

/// No-op cache backend that doesn't cache anything (useful for testing)
#[derive(Debug, Clone)]
pub struct NoOpCacheBackend;

impl NoOpCacheBackend {
    /// Create a new no-op cache backend
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpCacheBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CacheBackend for NoOpCacheBackend {
    async fn get(&self, _key: &str) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    async fn set(&self, _key: &str, _value: &[u8], _ttl: Option<Duration>) -> Result<()> {
        Ok(())
    }

    async fn delete(&self, _key: &str) -> Result<()> {
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats> {
        Ok(CacheStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            size: 0,
            hit_rate: 0.0,
            backend: "no-op".to_string(),
        })
    }

    fn backend_name(&self) -> &str {
        "no-op"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_cache() {
        let cache = InMemoryCacheBackend::new(100, Duration::from_secs(60));

        // Set and get
        cache
            .set("key1", b"value1", None)
            .await
            .expect("Should set cache value");
        let value = cache.get("key1").await.expect("Should get cache value");
        assert_eq!(value, Some(b"value1".to_vec()));

        // Delete
        cache
            .delete("key1")
            .await
            .expect("Should delete cache value");
        let value = cache
            .get("key1")
            .await
            .expect("Should get cache value after deletion");
        assert_eq!(value, None);
    }

    #[tokio::test]
    async fn test_typed_cache() {
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        struct TestData {
            value: String,
        }

        let cache = InMemoryCacheBackend::new(100, Duration::from_secs(60));
        let data = TestData {
            value: "test".to_string(),
        };

        cache
            .set_typed("key", &data, None)
            .await
            .expect("Should set typed cache value");
        let retrieved: Option<TestData> = cache
            .get_typed("key")
            .await
            .expect("Should get typed cache value");
        assert_eq!(retrieved, Some(data));
    }

    #[tokio::test]
    async fn test_no_op_cache() {
        let cache = NoOpCacheBackend::new();

        cache
            .set("key", b"value", None)
            .await
            .expect("Should set cache value (no-op)");
        let value = cache
            .get("key")
            .await
            .expect("Should get cache value (no-op)");
        assert_eq!(value, None);
    }
}
