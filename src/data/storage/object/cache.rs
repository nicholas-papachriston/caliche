//! Caching layer for storage backends
//!
//! Provides configurable disk caching for storage operations to reduce
//! network calls and improve performance.

use crate::data::storage::object::backend::StorageBackend;
use async_trait::async_trait;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::warn;

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,
    /// Time-to-live in seconds (None = no expiration)
    pub ttl_seconds: Option<u64>,
    /// Maximum cache size in MB (None = unlimited)
    pub max_size_mb: Option<u64>,
    /// Maximum number of cache entries in memory index (None = unlimited, default: 10000)
    /// Prevents unbounded `HashMap` growth even with many small files
    pub max_entries: Option<usize>,
    /// Directory for cache files
    pub cache_dir: PathBuf,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_seconds: Some(3600),  // 1 hour default
            max_size_mb: Some(100),   // 100 MB default
            max_entries: Some(10000), // 10k entries default to prevent unbounded HashMap growth
            cache_dir: std::env::temp_dir().join("caliche_cache"),
        }
    }
}

impl CacheConfig {
    /// Create a new cache config with caching disabled
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create a new cache config with caching enabled
    #[must_use]
    pub fn enabled() -> Self {
        Self::default()
    }

    /// Set TTL in seconds
    #[must_use]
    pub const fn with_ttl(mut self, ttl_seconds: u64) -> Self {
        self.ttl_seconds = Some(ttl_seconds);
        self
    }

    /// Set maximum cache size in MB
    #[must_use]
    pub const fn with_max_size(mut self, max_size_mb: u64) -> Self {
        self.max_size_mb = Some(max_size_mb);
        self
    }

    /// Set cache directory
    #[must_use]
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache_dir = cache_dir;
        self
    }
}

/// Cache entry metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    file_path: PathBuf,
    size: u64,
    created_at: SystemTime,
}

/// Age-ordered secondary index. Ordered by `(created_at, key)` so identical
/// timestamps don't collide and we can pop oldest-first in O(log n).
type AgeIndex = BTreeSet<(SystemTime, String)>;

/// In-memory cache index. Keeps the primary `HashMap<key, entry>` for O(1)
/// lookups and a secondary `BTreeSet<(SystemTime, key)>` so oldest-first
/// eviction is O(log n) per step instead of the previous O(n) scan.
struct CacheIndex {
    entries: HashMap<String, CacheEntry>,
    by_age: AgeIndex,
    total_size: u64,
}

impl CacheIndex {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            by_age: AgeIndex::new(),
            total_size: 0,
        }
    }

    /// Insert or replace an entry. When `max_entries` is set, over-limit
    /// entries are evicted by oldest-first order. The evicted entries are
    /// returned to the caller so their backing files can be deleted from
    /// disk (see `CachedStorageBackend::write_to_cache`).
    fn add_entry(&mut self, entry: CacheEntry, max_entries: Option<usize>) -> Vec<CacheEntry> {
        let mut evicted = Vec::new();

        if let Some(old_entry) = self.entries.remove(&entry.key) {
            self.by_age
                .remove(&(old_entry.created_at, old_entry.key.clone()));
            self.total_size = self.total_size.saturating_sub(old_entry.size);
            evicted.push(old_entry);
        }

        self.total_size += entry.size;
        self.by_age.insert((entry.created_at, entry.key.clone()));
        self.entries.insert(entry.key.clone(), entry);

        if let Some(max) = max_entries {
            while self.entries.len() > max {
                let Some((oldest_time, oldest_key)) = self.by_age.iter().next().cloned() else {
                    break;
                };
                self.by_age.remove(&(oldest_time, oldest_key.clone()));
                if let Some(removed) = self.entries.remove(&oldest_key) {
                    self.total_size = self.total_size.saturating_sub(removed.size);
                    evicted.push(removed);
                }
            }
        }
        evicted
    }

    fn remove_entry(&mut self, key: &str) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(key) {
            self.by_age.remove(&(entry.created_at, entry.key.clone()));
            self.total_size = self.total_size.saturating_sub(entry.size);
            Some(entry)
        } else {
            None
        }
    }

    fn get_entry(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    fn cleanup_expired(&mut self, ttl: Duration) -> Vec<CacheEntry> {
        let now = SystemTime::now();
        let mut expired = Vec::new();

        // Walk the age index in ascending order; stop at the first entry
        // that hasn't expired yet since everything after is younger.
        while let Some((created_at, key)) = self.by_age.iter().next().cloned() {
            match now.duration_since(created_at) {
                Ok(age) if age > ttl => {
                    self.by_age.remove(&(created_at, key.clone()));
                    if let Some(removed) = self.entries.remove(&key) {
                        self.total_size = self.total_size.saturating_sub(removed.size);
                        expired.push(removed);
                    }
                }
                _ => break,
            }
        }

        expired
    }

    fn evict_oldest(&mut self, target_size: u64) -> Vec<CacheEntry> {
        if self.total_size <= target_size {
            return Vec::new();
        }

        let mut evicted = Vec::new();
        while self.total_size > target_size {
            let Some((created_at, key)) = self.by_age.iter().next().cloned() else {
                break;
            };
            self.by_age.remove(&(created_at, key.clone()));
            if let Some(removed) = self.entries.remove(&key) {
                self.total_size = self.total_size.saturating_sub(removed.size);
                evicted.push(removed);
            }
        }
        evicted
    }
}

/// Cached storage backend wrapper
pub struct CachedStorageBackend {
    inner: Arc<dyn StorageBackend>,
    config: CacheConfig,
    index: Arc<RwLock<CacheIndex>>,
}

impl CachedStorageBackend {
    /// Create a new cached storage backend
    pub fn new(inner: Arc<dyn StorageBackend>, config: CacheConfig) -> Self {
        // Ensure cache directory exists if caching is enabled
        if config.enabled
            && let Err(error) = std::fs::create_dir_all(&config.cache_dir)
        {
            warn!(
                path = %config.cache_dir.display(),
                error = %error,
                "failed to create cache directory"
            );
        }

        Self {
            inner,
            config,
            index: Arc::new(RwLock::new(CacheIndex::new())),
        }
    }

    /// Get cache file path for a key
    fn cache_path(&self, key: &str) -> PathBuf {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        self.config
            .cache_dir
            .join(format!("{:016x}.cache", hasher.finish()))
    }

    /// Check whether a cached entry is TTL-valid without touching its file.
    async fn is_cache_fresh(&self, key: &str) -> bool {
        if !self.config.enabled {
            return false;
        }
        let index = self.index.read().await;
        let Some(entry) = index.get_entry(key) else {
            return false;
        };
        let is_fresh = self.config.ttl_seconds.map_or(true, |ttl| {
            SystemTime::now()
                .duration_since(entry.created_at)
                .map_or(false, |age| age <= Duration::from_secs(ttl))
        });
        drop(index);
        is_fresh
    }

    /// Read from cache if available and valid. TTL-expired entries are
    /// invalidated (both the index entry and the backing file) as part of
    /// the read so the next call falls through to the inner backend rather
    /// than repeatedly observing the stale state.
    async fn read_from_cache(&self, key: &str) -> Option<String> {
        if !self.config.enabled {
            return None;
        }

        // Capture the file path + expiry under a short read-lock, then drop
        // it before taking the write-lock to invalidate on expiry.
        let (file_path, is_expired) = {
            let index = self.index.read().await;
            let entry = index.get_entry(key)?;
            let expired = self.config.ttl_seconds.map_or(false, |ttl| {
                SystemTime::now()
                    .duration_since(entry.created_at)
                    .map(|age| age > Duration::from_secs(ttl))
                    .unwrap_or(false)
            });
            let file_path = entry.file_path.clone();
            drop(index);
            (file_path, expired)
        };

        if is_expired {
            self.invalidate_cache(key).await;
            return None;
        }

        match tokio::fs::read_to_string(&file_path).await {
            Ok(content) => Some(content),
            Err(error) => {
                warn!(
                    path = %file_path.display(),
                    error = %error,
                    "failed to read cache file"
                );
                None
            }
        }
    }

    /// Write to cache
    async fn write_to_cache(&self, key: &str, content: &str) -> Result<(), anyhow::Error> {
        if !self.config.enabled {
            return Ok(());
        }

        let cache_path = self.cache_path(key);

        // Create parent directory if needed
        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        // Write content to cache file
        tokio::fs::write(&cache_path, content).await?;

        // Update index
        let size = content.len() as u64;
        let entry = CacheEntry {
            key: key.to_string(),
            file_path: cache_path,
            size,
            created_at: SystemTime::now(),
        };

        let mut index = self.index.write().await;

        // Check cache size limits
        if let Some(max_size_mb) = self.config.max_size_mb {
            let max_size_bytes = max_size_mb * 1024 * 1024;
            if index.total_size + size > max_size_bytes {
                // Evict oldest entries
                let evicted = index.evict_oldest(max_size_bytes.saturating_sub(size));
                for evicted_entry in evicted {
                    if let Err(error) = tokio::fs::remove_file(&evicted_entry.file_path).await {
                        warn!(
                            path = %evicted_entry.file_path.display(),
                            error = %error,
                            "failed to remove evicted cache file"
                        );
                    }
                }
            }
        }

        let evicted_by_count = index.add_entry(entry, self.config.max_entries);
        let entry_count = index.entries.len();
        drop(index);

        for evicted_entry in evicted_by_count {
            if let Err(error) = tokio::fs::remove_file(&evicted_entry.file_path).await {
                warn!(
                    path = %evicted_entry.file_path.display(),
                    error = %error,
                    "failed to remove cache file during max_entries eviction"
                );
            }
        }

        // Periodically evict TTL-expired entries (every 50 writes)
        if entry_count % 50 == 0 {
            self.cleanup_ttl_expired().await;
        }

        Ok(())
    }

    /// Invalidate cache entry
    async fn invalidate_cache(&self, key: &str) {
        if !self.config.enabled {
            return;
        }

        let mut index = self.index.write().await;
        if let Some(entry) = index.remove_entry(key)
            && let Err(error) = tokio::fs::remove_file(&entry.file_path).await
        {
            warn!(
                path = %entry.file_path.display(),
                error = %error,
                "failed to remove cache file during invalidation"
            );
        }
    }

    /// Cleanup expired cache entries based on TTL configuration.
    async fn cleanup_ttl_expired(&self) {
        if !self.config.enabled {
            return;
        }

        if let Some(ttl) = self.config.ttl_seconds {
            let mut index = self.index.write().await;
            let expired = index.cleanup_expired(Duration::from_secs(ttl));
            drop(index);
            for entry in expired {
                if let Err(error) = tokio::fs::remove_file(&entry.file_path).await {
                    warn!(
                        path = %entry.file_path.display(),
                        error = %error,
                        "failed to remove expired cache file"
                    );
                }
            }
        }
    }
}

#[async_trait]
impl StorageBackend for CachedStorageBackend {
    async fn read(&self, key: &str) -> Result<String, anyhow::Error> {
        // Try cache first
        if let Some(cached) = self.read_from_cache(key).await {
            return Ok(cached);
        }

        // Read from inner backend
        let content = self.inner.read(key).await?;

        // Write to cache
        if let Err(error) = self.write_to_cache(key, &content).await {
            warn!(key, error = %error, "failed to write cache entry after backend read");
        }

        Ok(content)
    }

    async fn write(&self, key: &str, content: &str) -> Result<(), anyhow::Error> {
        // Write-through: write to backend first
        self.inner.write(key, content).await?;

        // Then update cache
        if let Err(error) = self.write_to_cache(key, content).await {
            warn!(key, error = %error, "failed to write cache entry after backend write");
        }

        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>, anyhow::Error> {
        // List always goes to backend (cache is for individual reads)
        self.inner.list(prefix).await
    }

    async fn delete(&self, key: &str) -> Result<(), anyhow::Error> {
        // Delete from backend
        self.inner.delete(key).await?;

        // Invalidate cache
        self.invalidate_cache(key).await;

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool, anyhow::Error> {
        // Trust the cache only when the entry is still within TTL. A stale
        // entry must not short-circuit the backend round-trip: the actual
        // object may have been deleted or replaced upstream.
        if self.is_cache_fresh(key).await {
            return Ok(true);
        }

        // Check backend
        self.inner.exists(key).await
    }
}
