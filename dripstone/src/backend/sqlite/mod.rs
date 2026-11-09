//! `SQLite` vector storage backend with configurable memory management.
//!
//! This backend provides durable vector storage using `SQLite` with flexible memory strategies:
//! - **`LoadAll`**: Load all vectors into memory on startup (best for small datasets)
//! - **`LazyLRU`**: Cache only accessed chunks with LRU eviction (balanced approach)
//! - **`QueryOnly`**: Zero memory overhead, always query `SQLite` (for large datasets)
//! - **`WriteThrough`**: Write to both cache and `SQLite`, read from cache (best for workspaces)
//!
//! # Examples
//!
//! ```text
//! use dripstone::backend::sqlite::{SqliteBackend, SqliteBackendConfig, MemoryStrategy};
//! use std::path::PathBuf;
//!
//! // Write-through strategy for workspace embeddings
//! let config = SqliteBackendConfig {
//!     db_path: PathBuf::from("embeddings.db"),
//!     memory_strategy: MemoryStrategy::WriteThrough {
//!         max_cache_size: 10_000,
//!         ttl_secs: None,
//!     },
//!     max_connections: 5,
//!     table_name: "vectors".to_string(),
//!     ensure_schema: true,
//! };
//!
//! let backend = SqliteBackend::new(config).await?;
//! ```

use async_trait::async_trait;
use dashmap::DashMap;
use futures::stream::BoxStream;
use sqlx::Row;
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqlitePoolOptions, SqliteSynchronous,
};
use sqlx::types::chrono::Utc;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info};

use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::traits::Backend;

/// Configuration for the `SQLite` vector backend.
#[derive(Debug, Clone)]
pub struct SqliteBackendConfig {
    /// Path to the `SQLite` database file
    pub db_path: PathBuf,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Maximum number of `SQLite` connections
    pub max_connections: u32,
    /// Table name for storing vectors
    pub table_name: String,
    /// Whether to create the schema if it doesn't exist
    pub ensure_schema: bool,
}

impl Default for SqliteBackendConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("vectors.db"),
            memory_strategy: MemoryStrategy::default(),
            max_connections: 5,
            table_name: "vectors".to_string(),
            ensure_schema: true,
        }
    }
}

/// Memory management strategy for the `SQLite` backend.
///
/// Different strategies offer trade-offs between memory usage, search speed, and complexity.
#[derive(Debug, Clone)]
pub enum MemoryStrategy {
    /// Load all vectors into memory on startup.
    ///
    /// **Pros**: Fastest search performance, simplest implementation
    /// **Cons**: High memory usage, potential OOM for large datasets
    /// **Use case**: Small datasets (< 10K chunks), development/testing
    LoadAll,

    /// Lazy load vectors with LRU cache eviction.
    ///
    /// **Pros**: Bounded memory usage, good for datasets with locality
    /// **Cons**: Cache misses require `SQLite` queries, eviction overhead
    /// **Use case**: Medium datasets with access patterns (hot/cold data)
    LazyLRU {
        /// Maximum number of chunks to cache
        max_cache_size: usize,
        /// Optional TTL for cache entries (in seconds)
        ttl_secs: Option<u64>,
    },

    /// No caching, always query `SQLite`.
    ///
    /// **Pros**: Minimal memory overhead, predictable performance
    /// **Cons**: Slowest search, high I/O
    /// **Use case**: Very large datasets, infrequent queries, memory-constrained environments
    QueryOnly,

    /// Write-through cache with LRU eviction.
    ///
    /// **Pros**: Fast reads for cached data, durable writes, balanced memory usage
    /// **Cons**: Slightly slower writes (dual write), cache management overhead
    /// **Use case**: Workspace embeddings, frequently accessed data with persistence needs
    WriteThrough {
        /// Maximum number of chunks to cache
        max_cache_size: usize,
        /// Optional TTL for cache entries (in seconds)
        ttl_secs: Option<u64>,
    },
}

impl Default for MemoryStrategy {
    fn default() -> Self {
        Self::WriteThrough {
            max_cache_size: 10_000,
            ttl_secs: None,
        }
    }
}

/// Cached chunk entry with LRU tracking.
#[derive(Debug)]
struct CachedChunk {
    /// The chunk data (wrapped in Arc for cheap cloning)
    chunk: Arc<Chunk>,
    /// Last access time for LRU eviction
    last_accessed: Instant,
    /// Access count for statistics
    access_count: AtomicU64,
    /// Optional expiration time
    expires_at: Option<Instant>,
}

impl CachedChunk {
    fn new(chunk: Chunk, ttl_secs: Option<u64>) -> Self {
        let expires_at = ttl_secs.map(|secs| Instant::now() + Duration::from_secs(secs));
        Self {
            chunk: Arc::new(chunk),
            last_accessed: Instant::now(),
            access_count: AtomicU64::new(1),
            expires_at,
        }
    }

    fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|exp| Instant::now() >= exp)
    }

    fn touch(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// `SQLite` vector storage backend.
#[derive(Debug)]
pub struct SqliteBackend {
    /// `SQLite` connection pool
    pool: SqlitePool,
    /// In-memory cache (composite key: "namespace:id")
    cache: Arc<DashMap<String, CachedChunk>>,
    /// Configuration
    config: SqliteBackendConfig,
}

impl SqliteBackend {
    /// Create a new `SQLite` backend with the given configuration.
    ///
    /// This will:
    /// 1. Create the database file if it doesn't exist
    /// 2. Initialize the schema if `ensure_schema` is true
    /// 3. Preload data if using `LoadAll` strategy
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(config: SqliteBackendConfig) -> Result<Self, DripstoneError> {
        // Ensure parent directory exists
        if let Some(parent) = config.db_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| DripstoneError::storage(format!("Failed to create directory: {e}")))?;
        }

        // Create connection options
        let connect_opts =
            SqliteConnectOptions::from_str(&format!("sqlite://{}", config.db_path.display()))
                .map_err(|e| DripstoneError::storage(format!("Invalid database path: {e}")))?
                .create_if_missing(true)
                .journal_mode(SqliteJournalMode::Wal)
                .busy_timeout(Duration::from_secs(5))
                .synchronous(SqliteSynchronous::Normal);

        // One SQLite connection is one thread. Idle slots close so a quiet
        // store does not keep extra workers. WAL lets a reader proceed while
        // a writer commits.
        let pool = SqlitePoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(0)
            .idle_timeout(Duration::from_secs(30))
            .connect_with(connect_opts)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to connect to SQLite: {e}")))?;

        let backend = Self {
            pool,
            cache: Arc::new(DashMap::new()),
            config,
        };

        // Initialize schema if requested
        if backend.config.ensure_schema {
            backend.ensure_schema().await?;
        }

        // Preload data if using LoadAll strategy
        if matches!(backend.config.memory_strategy, MemoryStrategy::LoadAll) {
            backend.preload_all().await?;
        }

        // Start background cleanup for TTL-based strategies
        if backend.has_ttl() {
            backend.start_cleanup_task();
        }

        Ok(backend)
    }

    /// Check if the current strategy uses TTL.
    const fn has_ttl(&self) -> bool {
        matches!(
            &self.config.memory_strategy,
            MemoryStrategy::LazyLRU {
                ttl_secs: Some(_),
                ..
            } | MemoryStrategy::WriteThrough {
                ttl_secs: Some(_),
                ..
            }
        )
    }

    /// Get the TTL in seconds from the current strategy.
    const fn get_ttl_secs(&self) -> Option<u64> {
        match &self.config.memory_strategy {
            MemoryStrategy::LazyLRU { ttl_secs, .. }
            | MemoryStrategy::WriteThrough { ttl_secs, .. } => *ttl_secs,
            _ => None,
        }
    }

    /// Get the max cache size from the current strategy.
    const fn get_max_cache_size(&self) -> Option<usize> {
        match &self.config.memory_strategy {
            MemoryStrategy::LazyLRU { max_cache_size, .. }
            | MemoryStrategy::WriteThrough { max_cache_size, .. } => Some(*max_cache_size),
            MemoryStrategy::LoadAll => None, // Unlimited for LoadAll
            MemoryStrategy::QueryOnly => Some(0), // No cache
        }
    }

    /// Ensure the database schema exists.
    async fn ensure_schema(&self) -> Result<(), DripstoneError> {
        let table_name = &self.config.table_name;
        let create_table = format!(
            r"
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                chunk_data BLOB NOT NULL,
                created_at INTEGER NOT NULL,
                namespace TEXT NOT NULL DEFAULT 'default'
            )
            "
        );

        sqlx::query(&create_table)
            .execute(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to create schema: {e}")))?;

        // Create index on namespace for faster queries
        let create_index = format!(
            r"
            CREATE INDEX IF NOT EXISTS idx_{table_name}_namespace ON {table_name}(namespace)
            "
        );

        sqlx::query(&create_index)
            .execute(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to create index: {e}")))?;

        Ok(())
    }

    /// Preload all vectors from `SQLite` into memory (`LoadAll` strategy).
    async fn preload_all(&self) -> Result<(), DripstoneError> {
        debug!("Preloading all vectors from SQLite...");

        let table_name = &self.config.table_name;
        let query = format!("SELECT chunk_data FROM {table_name}");

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to preload vectors: {e}")))?;

        for row in rows {
            let chunk_blob: Vec<u8> = row
                .try_get("chunk_data")
                .map_err(|e| DripstoneError::storage(format!("Failed to get chunk_data: {e}")))?;

            let chunk = Chunk::decode_from_sqlite_blob(&chunk_blob)?;

            let key = Self::composite_key(&chunk.id, chunk.namespace.as_deref());
            self.cache
                .insert(key, CachedChunk::new(chunk, self.get_ttl_secs()));
        }

        info!(
            "Preloaded {} vectors from SQLite into memory",
            self.cache.len()
        );
        Ok(())
    }

    /// Get composite key for cache storage.
    fn composite_key(id: &str, namespace: Option<&str>) -> String {
        let ns = namespace.unwrap_or("default");
        format!("{ns}:{id}")
    }

    /// Extract namespace from composite key.
    fn extract_namespace(key: &str) -> &str {
        key.split(':').next().unwrap_or("default")
    }

    /// Start background task to clean up expired cache entries.
    fn start_cleanup_task(&self) {
        let cache = Arc::clone(&self.cache);
        let cleanup_interval = Duration::from_secs(60);

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(cleanup_interval).await;
                cache.retain(|_, v| !v.is_expired());
            }
        });
    }

    /// Evict LRU entry if cache is at capacity.
    fn evict_lru_if_needed(&self) {
        if let Some(max_size) = self.get_max_cache_size()
            && self.cache.len() >= max_size
        {
            // Find LRU entry (oldest last_accessed)
            let lru_key = self
                .cache
                .iter()
                .min_by_key(|entry| entry.value().last_accessed)
                .map(|entry| entry.key().clone());

            if let Some(key) = lru_key {
                self.cache.remove(&key);
                debug!("Evicted LRU cache entry: {}", key);
            }
        }
    }

    /// Touch a cache entry to update LRU tracking.
    fn touch_cache_entry(&self, key: &str) {
        if let Some(mut entry) = self.cache.get_mut(key) {
            entry.value_mut().touch();
            entry.value_mut().last_accessed = Instant::now();
        }
    }

    /// Add chunk to cache based on current strategy.
    fn add_to_cache(&self, chunk: Chunk) {
        match &self.config.memory_strategy {
            MemoryStrategy::QueryOnly => {
                // No caching
            }
            MemoryStrategy::LoadAll
            | MemoryStrategy::LazyLRU { .. }
            | MemoryStrategy::WriteThrough { .. } => {
                self.evict_lru_if_needed();
                let key = Self::composite_key(&chunk.id, chunk.namespace.as_deref());
                self.cache
                    .insert(key, CachedChunk::new(chunk, self.get_ttl_secs()));
            }
        }
    }

    /// Get chunk from cache.
    fn get_from_cache(&self, id: &str, namespace: Option<&str>) -> Option<Chunk> {
        let key = Self::composite_key(id, namespace);
        // Release the DashMap read guard before `touch_cache_entry` takes the
        // write guard on the same shard; holding both deadlocks.
        let cached = match self.cache.get(&key) {
            Some(entry) if entry.is_expired() => {
                drop(entry);
                self.cache.remove(&key);
                return None;
            }
            Some(entry) => (*entry.chunk).clone(),
            None => return None,
        };
        self.touch_cache_entry(&key);
        Some(cached)
    }

    /// Write chunk to `SQLite`.
    async fn write_to_db(&self, chunk: &Chunk) -> Result<(), DripstoneError> {
        let chunk_blob = chunk.encode_for_sqlite_blob()?;

        let created_at = Utc::now().timestamp();
        let namespace = chunk.namespace.as_deref().unwrap_or("default");
        let table_name = &self.config.table_name;

        let query = format!(
            r"
            INSERT INTO {table_name} (id, chunk_data, created_at, namespace)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                chunk_data = excluded.chunk_data,
                created_at = excluded.created_at,
                namespace = excluded.namespace
            "
        );

        sqlx::query(&query)
            .bind(&chunk.id)
            .bind(&chunk_blob)
            .bind(created_at)
            .bind(namespace)
            .execute(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to write chunk: {e}")))?;

        Ok(())
    }

    /// Read chunk from `SQLite`.
    async fn read_from_db(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let table_name = &self.config.table_name;

        let query = format!("SELECT chunk_data FROM {table_name} WHERE id = ? AND namespace = ?");

        let row = sqlx::query(&query)
            .bind(id)
            .bind(ns)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to read chunk: {e}")))?;

        if let Some(row) = row {
            let chunk_blob: Vec<u8> = row
                .try_get("chunk_data")
                .map_err(|e| DripstoneError::storage(format!("Failed to get chunk_data: {e}")))?;

            let chunk = Chunk::decode_from_sqlite_blob(&chunk_blob)?;

            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }
}

#[async_trait]
impl Backend for SqliteBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: None,
            max_batch: None,
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        for chunk in chunks {
            // Always write to SQLite
            self.write_to_db(chunk).await?;

            // Update cache based on strategy
            match &self.config.memory_strategy {
                MemoryStrategy::LoadAll | MemoryStrategy::WriteThrough { .. } => {
                    self.add_to_cache(chunk.clone());
                }
                MemoryStrategy::LazyLRU { .. } => {
                    // Opportunistically add if there's space
                    if let Some(max_size) = self.get_max_cache_size()
                        && self.cache.len() < max_size
                    {
                        self.add_to_cache(chunk.clone());
                    }
                }
                MemoryStrategy::QueryOnly => {
                    // No caching
                }
            }
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        // Try cache first for strategies that use it
        if !matches!(self.config.memory_strategy, MemoryStrategy::QueryOnly)
            && let Some(chunk) = self.get_from_cache(id, namespace)
        {
            return Ok(Some(chunk));
        }

        // Cache miss or QueryOnly - read from SQLite
        let chunk = self.read_from_db(id, namespace).await?;

        // Add to cache for lazy strategies
        if let Some(ref chunk) = chunk
            && matches!(self.config.memory_strategy, MemoryStrategy::LazyLRU { .. })
        {
            self.add_to_cache(chunk.clone());
        }

        Ok(chunk)
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        // Remove from cache
        let key = Self::composite_key(id, namespace);
        self.cache.remove(&key);

        // Delete from SQLite
        let ns = namespace.unwrap_or("default");
        let table_name = &self.config.table_name;
        let query = format!("DELETE FROM {table_name} WHERE id = ? AND namespace = ?");

        sqlx::query(&query)
            .bind(id)
            .bind(ns)
            .execute(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to delete chunk: {e}")))?;

        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let prefix = format!("{doc_id}#");

        // Remove from cache
        let keys_to_remove: Vec<String> = self
            .cache
            .iter()
            .filter(|entry| {
                let entry_ns = Self::extract_namespace(entry.key());
                entry_ns == ns && (entry.chunk.id.starts_with(&prefix) || entry.chunk.id == doc_id)
            })
            .map(|entry| entry.key().clone())
            .collect();

        for key in &keys_to_remove {
            self.cache.remove(key);
        }

        // Delete from SQLite
        let table_name = &self.config.table_name;
        let query =
            format!("DELETE FROM {table_name} WHERE namespace = ? AND (id LIKE ? OR id = ?)");

        let prefix_pattern = format!("{prefix}%");
        let result = sqlx::query(&query)
            .bind(ns)
            .bind(&prefix_pattern)
            .bind(doc_id)
            .execute(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to delete document: {e}")))?;

        usize::try_from(result.rows_affected())
            .map_err(|_| DripstoneError::storage("deleted row count exceeds addressable size"))
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        match &self.config.memory_strategy {
            MemoryStrategy::LoadAll => {
                // All data in cache, search in memory
                self.search_in_cache(query)
            }
            MemoryStrategy::LazyLRU { .. } | MemoryStrategy::WriteThrough { .. } => {
                // Hybrid: search in cache, then load missing from DB if needed
                self.hybrid_search(query).await
            }
            MemoryStrategy::QueryOnly => {
                // Always query SQLite
                self.search_in_db(query).await
            }
        }
    }

    fn search_stream<'a>(
        &'a self,
        _query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        // Streaming not supported for SQLite backend
        Box::pin(futures::stream::empty())
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let table_name = &self.config.table_name;
        let query = format!("SELECT DISTINCT namespace FROM {table_name}");

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to get namespaces: {e}")))?;

        let mut namespaces = Vec::new();
        for row in rows {
            let ns: String = row
                .try_get("namespace")
                .map_err(|e| DripstoneError::storage(format!("Failed to get namespace: {e}")))?;
            namespaces.push(ns);
        }

        namespaces.sort();
        namespaces.dedup();
        Ok(namespaces)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        // Remove from cache
        let prefix = format!("{namespace}:");
        self.cache.retain(|key, _| !key.starts_with(&prefix));

        // Delete from SQLite
        let table_name = &self.config.table_name;
        let query = format!("DELETE FROM {table_name} WHERE namespace = ?");

        sqlx::query(&query)
            .bind(namespace)
            .execute(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to delete namespace: {e}")))?;

        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let table_name = &self.config.table_name;
        let (query, ns) = namespace.map_or_else(
            || (format!("SELECT COUNT(*) as count FROM {table_name}"), None),
            |namespace| {
                (
                    format!("SELECT COUNT(*) as count FROM {table_name} WHERE namespace = ?"),
                    Some(namespace),
                )
            },
        );

        let mut q = sqlx::query(&query);
        if let Some(ns) = ns {
            q = q.bind(ns);
        }

        let row = q
            .fetch_one(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to count chunks: {e}")))?;

        let count: i64 = row
            .try_get("count")
            .map_err(|e| DripstoneError::storage(format!("Failed to get count: {e}")))?;

        usize::try_from(count)
            .map_err(|_| DripstoneError::storage("SQLite count is negative or out of range"))
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let table_name = &self.config.table_name;
        let base_query = namespace.map_or_else(
            || format!("SELECT chunk_data FROM {table_name}"),
            |_| format!("SELECT chunk_data FROM {table_name} WHERE namespace = ?"),
        );

        let query = if let Some(lim) = limit {
            format!("{base_query} LIMIT {lim}")
        } else {
            base_query
        };

        let mut q = sqlx::query(&query);
        if let Some(ns) = namespace {
            q = q.bind(ns);
        }

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to list chunks: {e}")))?;

        let mut chunks = Vec::new();
        for row in rows {
            let chunk_blob: Vec<u8> = row
                .try_get("chunk_data")
                .map_err(|e| DripstoneError::storage(format!("Failed to get chunk_data: {e}")))?;

            let chunk = Chunk::decode_from_sqlite_blob(&chunk_blob)?;

            chunks.push(chunk);
        }

        Ok(chunks)
    }
}

/// Whether `chunk` satisfies `query.filter` (if any).
fn chunk_matches_query_filter(query: &SearchQuery, chunk: &Chunk) -> bool {
    query
        .filter
        .as_ref()
        .is_none_or(|f| f.matches(&chunk.metadata))
}

// Helper methods for search implementations
impl SqliteBackend {
    /// Search only in the cache (for `LoadAll` strategy).
    fn search_in_cache(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let mut results: Vec<SearchResult> = self
            .cache
            .iter()
            .filter(|entry| {
                // Skip expired entries
                if entry.value().is_expired() {
                    return false;
                }

                // Namespace filter
                if let Some(ref ns) = query.namespace {
                    let entry_ns = Self::extract_namespace(entry.key());
                    if entry_ns != ns {
                        return false;
                    }
                }

                // Metadata filter
                if !chunk_matches_query_filter(query, &entry.value().chunk) {
                    return false;
                }

                true
            })
            .map(|entry| -> Result<SearchResult, DripstoneError> {
                let chunk = &entry.value().chunk;
                let score = query.metric.compute(&query.vector, &chunk.vector)?;
                Ok(SearchResult::new(chunk.as_ref().clone(), score, 0))
            })
            .collect::<Result<Vec<_>, DripstoneError>>()?;

        // Sort by score
        if query.metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Take top k and update ranks
        results.truncate(query.limit);
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    /// Hybrid search: check cache, load from DB on miss (for LazyLRU/WriteThrough).
    async fn hybrid_search(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        // First, try searching in cache
        let cache_results = self.search_in_cache(query)?;

        // If we have enough results from cache, return them
        if cache_results.len() >= query.limit {
            return Ok(cache_results);
        }

        // Otherwise, fall back to full DB search
        self.search_in_db(query).await
    }

    /// Search by loading all chunks from `SQLite` and computing similarity (for `QueryOnly`).
    async fn search_in_db(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        // Load all chunks from SQLite (filtered by namespace if provided)
        let chunks = self.list(query.namespace.as_deref(), None).await?;

        // Compute similarities
        let mut results: Vec<SearchResult> = chunks
            .into_iter()
            .filter(|chunk| chunk_matches_query_filter(query, chunk))
            .map(|chunk| -> Result<SearchResult, DripstoneError> {
                let score = query.metric.compute(&query.vector, &chunk.vector)?;
                Ok(SearchResult::new(chunk, score, 0))
            })
            .collect::<Result<Vec<_>, DripstoneError>>()?;

        // Sort by score
        if query.metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Take top k and update ranks
        results.truncate(query.limit);
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }
}

#[cfg(test)]
mod metadata_filter_unit_tests {
    use super::chunk_matches_query_filter;
    use crate::core::{Chunk, Metric, SearchQuery};
    use crate::search::Filter;
    use serde_json::json;
    use std::sync::Arc;

    fn tiny_chunk(doc_id: &str, category: &str) -> Chunk {
        let v = vec![0.0_f32, 0.1, 0.2, 0.3];
        Chunk::new(
            doc_id,
            0,
            "x",
            v,
            0,
            1,
            1,
            json!({ "category": category }),
            None,
        )
    }

    #[test]
    fn no_filter_accepts_all() {
        let q = SearchQuery {
            vector: Arc::from([0.0_f32, 0.1, 0.2, 0.3]),
            limit: 10,
            metric: Metric::Cosine,
            namespace: None,
            filter: None,
            hybrid_weight: None,
            sparse_vector: None,
            include_content: true,
            include_vectors: false,
            query_multi_vectors: None,
            maxsim: None,
        };
        let c = tiny_chunk("a", "A");
        assert!(chunk_matches_query_filter(&q, &c));
    }

    #[test]
    fn equal_filter_on_metadata() {
        let filter = Filter::Equal {
            field: "category".to_string(),
            value: json!("keep"),
        };
        let q = SearchQuery {
            vector: Arc::from([0.0_f32, 0.1, 0.2, 0.3]),
            limit: 10,
            metric: Metric::Cosine,
            namespace: None,
            filter: Some(filter),
            hybrid_weight: None,
            sparse_vector: None,
            include_content: true,
            include_vectors: false,
            query_multi_vectors: None,
            maxsim: None,
        };
        assert!(chunk_matches_query_filter(&q, &tiny_chunk("a", "keep")));
        assert!(!chunk_matches_query_filter(&q, &tiny_chunk("b", "drop")));
    }

    #[test]
    fn and_filter_on_metadata() {
        let filter = Filter::And {
            left: Box::new(Filter::Equal {
                field: "category".to_string(),
                value: json!("A"),
            }),
            right: Box::new(Filter::Equal {
                field: "tier".to_string(),
                value: json!("pro"),
            }),
        };
        let q = SearchQuery {
            vector: Arc::from([0.0_f32, 0.1, 0.2, 0.3]),
            limit: 10,
            metric: Metric::Cosine,
            namespace: None,
            filter: Some(filter),
            hybrid_weight: None,
            sparse_vector: None,
            include_content: true,
            include_vectors: false,
            query_multi_vectors: None,
            maxsim: None,
        };
        let mut c = tiny_chunk("x", "A");
        c.metadata = json!({ "category": "A", "tier": "pro" });
        assert!(chunk_matches_query_filter(&q, &c));
        c.metadata = json!({ "category": "A", "tier": "free" });
        assert!(!chunk_matches_query_filter(&q, &c));
    }
}

#[cfg(test)]
mod pool_tests {
    use super::{MemoryStrategy, SqliteBackend, SqliteBackendConfig};

    #[tokio::test]
    async fn sqlite_backend_opens_in_wal_mode() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("pool.db");
        let backend = SqliteBackend::new(SqliteBackendConfig {
            db_path: path.clone(),
            memory_strategy: MemoryStrategy::QueryOnly,
            max_connections: 2,
            table_name: "vectors".to_string(),
            ensure_schema: true,
        })
        .await
        .expect("sqlite backend opens");
        let mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&backend.pool)
            .await
            .expect("journal mode");
        assert_eq!(mode.to_ascii_lowercase(), "wal");
        assert_eq!(backend.pool.options().get_max_connections(), 2);
        drop(backend);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }
}
