//! In-memory vector database backend with TTL support.
//!
//! This backend provides a fast, lock-free in-memory storage for chunks
//! with automatic TTL-based expiration and configurable capacity limits.
//!
//! # Features
//!
//! - **Lock-free concurrency**: Built on `DashMap` for high-performance concurrent access
//! - **TTL support**: Automatic expiration of entries after a configurable time
//! - **Capacity management**: Optional maximum entry count with LRU-style eviction
//! - **Namespace support**: Organize chunks into separate namespaces
//! - **Similarity search**: Cosine, dot product, and Euclidean distance metrics
//! - **Background cleanup**: Optional periodic cleanup of expired entries
//!
//! # Example
//!
//! ```text
//! use dripstone::{MemoryBackend, DripstoneBuilder};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Wire MemoryBackend into a Dripstone instance via the builder. The
//! // embedder is constructed from an environment-appropriate provider
//! // (see `examples/` for concrete `with_*_embedder` calls).
//! let vf = DripstoneBuilder::new()
//!     .backend(MemoryBackend::new())
//!     // .with_*_embedder(...)
//!     .build()?;
//!
//! vf.ingest("doc1", "Hello, world!").await?;
//! let _results = vf.search("greeting").limit(5).await?;
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use dashmap::DashMap;
use futures::stream::BoxStream;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::traits::Backend;

/// Configuration for the in-memory backend
#[derive(Debug, Clone)]
pub struct InMemoryConfig {
    /// Default TTL for entries, in seconds.
    ///
    /// `None` keeps an entry until it is deleted.
    pub default_ttl_secs: Option<u64>,
    /// Maximum number of entries before eviction
    pub max_entries: Option<usize>,
    /// Whether to automatically clean up expired entries
    pub auto_cleanup: bool,
}

impl Default for InMemoryConfig {
    fn default() -> Self {
        Self {
            default_ttl_secs: Some(3600),
            max_entries: Some(10000),
            auto_cleanup: true,
        }
    }
}

/// Storage entry for Chunk with TTL
///
/// # Memory Optimization
///
/// Chunks are stored as `Arc<Chunk>` to enable cheap cloning when
/// returning search results. This avoids copying ~6KB vectors per result.
#[derive(Debug, Clone)]
struct ChunkEntry {
    /// Chunk wrapped in Arc for cheap cloning
    chunk: Arc<Chunk>,
    created_at: Instant,
    ttl: Option<Duration>,
}

impl ChunkEntry {
    fn is_expired(&self) -> bool {
        self.ttl.is_some_and(|ttl| self.created_at.elapsed() >= ttl)
    }
}

/// Memory backend for `Dripstone`.
///
/// Implements the `Backend` trait for in-memory storage with TTL support.
#[derive(Debug)]
pub struct MemoryBackend {
    /// Storage indexed by composite key: "namespace:id"
    store: Arc<DashMap<String, ChunkEntry>>,
    config: InMemoryConfig,
}

impl MemoryBackend {
    /// Create a new memory backend with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(InMemoryConfig::default())
    }

    /// TTL passed to [`Self::with_config`], in seconds.
    ///
    /// `None` means entries stay until they are deleted.
    #[cfg(test)]
    pub(crate) fn default_ttl_secs(&self) -> Option<u64> {
        self.config.default_ttl_secs
    }

    /// Create a new memory backend with custom configuration.
    #[must_use]
    pub fn with_config(config: InMemoryConfig) -> Self {
        let backend = Self {
            store: Arc::new(DashMap::new()),
            config,
        };

        if backend.config.auto_cleanup {
            backend.start_cleanup_task();
        }

        backend
    }

    /// Insert chunks without going through the async backend trait.
    ///
    /// [`crate::Dripstone::open`] uses this before a runtime exists.
    #[cfg_attr(
        not(feature = "local"),
        allow(
            dead_code,
            reason = "Dripstone::open is the caller, and that constructor is behind the local feature"
        )
    )]
    pub(crate) fn insert_chunks(&self, chunks: &[Chunk]) {
        for chunk in chunks {
            self.store_chunk(chunk);
        }
    }

    fn store_chunk(&self, chunk: &Chunk) {
        let key = Self::composite_key(&chunk.id, chunk.namespace.as_deref());
        let entry = ChunkEntry {
            chunk: Arc::new(chunk.clone()),
            created_at: Instant::now(),
            ttl: self.config.default_ttl_secs.map(Duration::from_secs),
        };
        self.store.insert(key, entry);
    }

    /// Get the composite key for an entry.
    fn composite_key(id: &str, namespace: Option<&str>) -> String {
        let ns = namespace.unwrap_or("default");
        format!("{ns}:{id}")
    }

    /// Extract namespace from a composite key.
    fn extract_namespace(composite_key: &str) -> &str {
        composite_key.split(':').next().unwrap_or("default")
    }

    /// Start a background task to periodically clean up expired entries.
    ///
    /// Spawned on the current Tokio runtime via [`tokio::spawn`]. If there is
    /// no current runtime the cleanup task is skipped rather than panicking,
    /// so `MemoryBackend::with_config` remains usable in synchronous tests.
    fn start_cleanup_task(&self) {
        let store = Arc::clone(&self.store);
        let cleanup_interval = Duration::from_secs(60);

        // Only spawn if we're inside a Tokio runtime; otherwise silently skip.
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(cleanup_interval);
            // Skip the immediate first tick so the task actually waits.
            ticker.tick().await;
            loop {
                ticker.tick().await;
                store.retain(|_, v| !v.is_expired());
            }
        });
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Backend for MemoryBackend {
    fn capabilities(&self) -> Capabilities {
        let mut caps = Capabilities::memory();
        caps.native_maxsim = true;
        caps
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        for chunk in chunks {
            self.store_chunk(chunk);
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let key = Self::composite_key(id, namespace);
        Ok(self.store.get(&key).and_then(|entry| {
            if entry.value().is_expired() {
                None
            } else {
                // Dereference Arc and clone - still needed as API returns owned Chunk
                // But the internal Arc clone is cheap during filtering/processing
                Some((*entry.value().chunk).clone())
            }
        }))
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let key = Self::composite_key(id, namespace);
        self.store.remove(&key);
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let prefix = format!("{doc_id}#");
        let ns = namespace.unwrap_or("default");
        let full_prefix = format!("{ns}:{prefix}");

        let mut count = 0;
        self.store.retain(|key, _| {
            if key.starts_with(&full_prefix) || key.ends_with(&format!(":{doc_id}")) {
                count += 1;
                false
            } else {
                true
            }
        });
        Ok(count)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let mut candidate_chunks: Vec<Arc<Chunk>> = Vec::new();
        for entry in self.store.iter() {
            let stored = entry.value();
            if stored.is_expired() {
                continue;
            }

            if let Some(ref ns) = query.namespace {
                let entry_ns = Self::extract_namespace(entry.key());
                if entry_ns != ns {
                    continue;
                }
            }

            if let Some(ref filter) = query.filter
                && !filter.matches(&stored.chunk.metadata)
            {
                continue;
            }

            candidate_chunks.push(Arc::clone(&stored.chunk));
        }

        // Exact MaxSim path when every candidate carries multi-vectors and the
        // caller did not ask to fuse with dense (correctness / late-interaction).
        if query.query_multi_vectors.is_some()
            && query.maxsim.is_none_or(|o| !o.fuse_with_dense)
            && candidate_chunks.iter().all(|c| c.multi_vectors.is_some())
        {
            let owned: Vec<Chunk> = candidate_chunks
                .iter()
                .map(|c| c.as_ref().clone())
                .collect();
            return crate::search::exact_maxsim_search(&owned, query);
        }

        let prepared = query.metric.prepare(&query.vector)?;
        let mut candidate_vectors = Vec::with_capacity(candidate_chunks.len());
        for chunk in &candidate_chunks {
            candidate_vectors.push(chunk.vector.as_ref());
        }
        let scores = prepared.score_dense_batch(&candidate_vectors)?;

        let mut results = Vec::with_capacity(candidate_chunks.len());
        for (chunk, score) in candidate_chunks.into_iter().zip(scores) {
            results.push(SearchResult::new(chunk.as_ref().clone(), score, 0));
        }

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

        let keep = if query.query_multi_vectors.is_some() {
            crate::search::overfetch_limit(query)
        } else {
            query.limit
        };
        results.truncate(keep.max(1));
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        if query.query_multi_vectors.is_some() {
            return crate::search::rerank_with_maxsim(results, query);
        }

        Ok(results)
    }

    fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        use futures::StreamExt;

        let query = query.clone();
        Box::pin(
            futures::stream::once(async move { self.search(&query).await }).flat_map(|result| {
                match result {
                    Ok(results) => futures::stream::iter(results.into_iter().map(Ok)).boxed(),
                    Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
                }
            }),
        )
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let mut namespaces: Vec<String> = self
            .store
            .iter()
            .filter(|entry| !entry.value().is_expired())
            .map(|entry| Self::extract_namespace(entry.key()).to_string())
            .collect();

        namespaces.sort();
        namespaces.dedup();
        Ok(namespaces)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let prefix = format!("{namespace}:");
        self.store.retain(|key, _| !key.starts_with(&prefix));
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let count = self
            .store
            .iter()
            .filter(|entry| {
                if entry.value().is_expired() {
                    return false;
                }
                namespace.is_none_or(|ns| Self::extract_namespace(entry.key()) == ns)
            })
            .count();
        Ok(count)
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let mut chunks: Vec<Chunk> = self
            .store
            .iter()
            .filter(|entry| {
                if entry.value().is_expired() {
                    return false;
                }
                namespace.is_none_or(|ns| Self::extract_namespace(entry.key()) == ns)
            })
            .map(|entry| (*entry.value().chunk).clone())
            .collect();

        if let Some(limit) = limit {
            chunks.truncate(limit);
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_chunks_stores_without_a_runtime() {
        let backend = MemoryBackend::with_config(InMemoryConfig {
            default_ttl_secs: None,
            max_entries: None,
            auto_cleanup: false,
        });
        let chunk = Chunk::new(
            "alpha",
            0,
            "penguins",
            vec![1.0_f32],
            0,
            8,
            1,
            serde_json::json!({}),
            None,
        );
        backend.insert_chunks(std::slice::from_ref(&chunk));
        let stored = backend.store.iter().next().expect("stored");
        assert!(!stored.is_expired());
        assert_eq!(stored.chunk.doc_id, "alpha");
    }
}
