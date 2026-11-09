//! Single-WAL, multi-shard [`Backend`] (turbopuffer-inspired scale path).
//!
//! One logical namespace keeps a shared write-ahead log. Each shard applies
//! `hash(id) % N` of that log into its own index. Clients keep talking to a
//! single `Backend`; indexing and search fan out across shards.

#[cfg(all(feature = "coldforge-backend", feature = "memory-backend"))]
pub mod coldforge;

#[cfg(all(feature = "coldforge-backend", feature = "memory-backend"))]
pub use coldforge::{
    ColdForgeShardOptions, build_coldforge_wal_sharded, build_memory_coldforge_wal_sharded,
};

use async_trait::async_trait;
use futures::future::try_join_all;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::distributed::{NamespaceWalOp, ShardManager, SharedNamespaceWal, SharedWalConfig};
use crate::errors::DripstoneError;
use crate::storage::ObjectStore;
use crate::traits::Backend;

/// Builder for [`WalShardedBackend`].
pub struct WalShardedBackendBuilder {
    shard_count: usize,
    shard_factory: Option<Box<dyn FnMut(usize) -> Arc<dyn Backend> + Send>>,
    shards: Option<Vec<Arc<dyn Backend>>>,
    wal: Option<SharedNamespaceWal>,
}

impl Default for WalShardedBackendBuilder {
    fn default() -> Self {
        Self {
            shard_count: 4,
            shard_factory: None,
            shards: None,
            wal: None,
        }
    }
}

impl WalShardedBackendBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn shard_count(mut self, shard_count: usize) -> Self {
        self.shard_count = shard_count.max(1);
        self
    }

    /// Provide a factory that builds each shard backend (`0..shard_count`).
    #[must_use]
    pub fn shard_factory<F>(mut self, factory: F) -> Self
    where
        F: FnMut(usize) -> Arc<dyn Backend> + Send + 'static,
    {
        self.shard_factory = Some(Box::new(factory));
        self
    }

    /// Use pre-built shard backends (for async constructors like ColdForge).
    ///
    /// Sets `shard_count` from `shards.len()`. Prefer this when shard setup is
    /// async and cannot run inside the sync [`Self::shard_factory`] closure.
    #[must_use]
    pub fn shards(mut self, shards: Vec<Arc<dyn Backend>>) -> Self {
        self.shard_count = shards.len().max(1);
        self.shards = Some(shards);
        self
    }

    #[must_use]
    pub fn wal(mut self, wal: SharedNamespaceWal) -> Self {
        self.wal = Some(wal);
        self
    }

    /// Convenience: durable WAL on an object store under `key_prefix`.
    #[must_use]
    pub fn object_store_wal(
        mut self,
        store: Arc<dyn ObjectStore>,
        bucket: impl Into<String>,
        key_prefix: impl Into<String>,
    ) -> Self {
        self.wal = Some(SharedNamespaceWal::with_object_store(
            store,
            SharedWalConfig {
                bucket: bucket.into(),
                key_prefix: key_prefix.into(),
            },
        ));
        self
    }

    /// Build the sharded backend.
    ///
    /// # Errors
    ///
    /// Returns when neither [`Self::shards`] nor [`Self::shard_factory`] was set,
    /// or when a prebuilt shard list is empty.
    pub fn build(mut self) -> Result<WalShardedBackend, DripstoneError> {
        let shards = if let Some(shards) = self.shards.take() {
            if shards.is_empty() {
                return Err(DripstoneError::pipeline(
                    "WalShardedBackendBuilder::shards requires at least one shard",
                ));
            }
            shards
        } else if let Some(mut factory) = self.shard_factory.take() {
            let shard_count = self.shard_count.max(1);
            let mut shards = Vec::with_capacity(shard_count);
            for i in 0..shard_count {
                shards.push(factory(i));
            }
            shards
        } else {
            return Err(DripstoneError::pipeline(
                "WalShardedBackendBuilder requires shards(...) or shard_factory(...)",
            ));
        };
        let shard_count = shards.len();
        Ok(WalShardedBackend {
            shards,
            shard_manager: ShardManager::new(shard_count),
            wal: Arc::new(self.wal.unwrap_or_else(SharedNamespaceWal::in_memory)),
        })
    }
}

/// Namespace facade: shared WAL + N independent shard backends.
pub struct WalShardedBackend {
    shards: Vec<Arc<dyn Backend>>,
    shard_manager: ShardManager,
    wal: Arc<SharedNamespaceWal>,
}

impl std::fmt::Debug for WalShardedBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalShardedBackend")
            .field("shard_count", &self.shards.len())
            .field("wal_entries", &self.wal.len())
            .finish()
    }
}

impl WalShardedBackend {
    /// Construct from an already-built shard list and WAL.
    ///
    /// # Errors
    ///
    /// Returns when `shards` is empty.
    pub fn from_shards(
        shards: Vec<Arc<dyn Backend>>,
        wal: SharedNamespaceWal,
    ) -> Result<Self, DripstoneError> {
        Self::builder().shards(shards).wal(wal).build()
    }

    #[must_use]
    pub fn builder() -> WalShardedBackendBuilder {
        WalShardedBackendBuilder::new()
    }

    /// Number of shards in this namespace.
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Shared WAL handle (for inspection / replay tooling).
    #[must_use]
    pub fn wal(&self) -> &Arc<SharedNamespaceWal> {
        &self.wal
    }

    /// Owning shard index for `id` (`hash(chunk.id) % N`).
    ///
    /// Routing stays on the chunk id. A cluster index is not a shard key, and a
    /// membership move must not change which shard holds the chunk.
    #[must_use]
    pub fn shard_index_for_id(&self, id: &str) -> Option<usize> {
        self.shard_manager.shard_index_for_key(id)
    }

    /// Count of vectors currently resident on each shard (for balance checks).
    ///
    /// # Errors
    ///
    /// Returns when any shard `count` fails.
    pub async fn shard_counts(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<usize>, DripstoneError> {
        let futs = self.shards.iter().map(|s| s.count(namespace));
        try_join_all(futs).await
    }

    fn shard_for_id(&self, id: &str) -> Result<&Arc<dyn Backend>, DripstoneError> {
        let idx = self
            .shard_manager
            .shard_index_for_key(id)
            .ok_or_else(|| DripstoneError::pipeline("WalShardedBackend has zero shards"))?;
        self.shards
            .get(idx)
            .ok_or_else(|| DripstoneError::pipeline(format!("shard index {idx} out of range")))
    }

    async fn apply_op(&self, op: &NamespaceWalOp) -> Result<(), DripstoneError> {
        match op {
            NamespaceWalOp::Upsert(chunk) => {
                let shard = self.shard_for_id(&chunk.id)?;
                shard.upsert(std::slice::from_ref(chunk)).await
            }
            NamespaceWalOp::Delete { id, namespace } => {
                let shard = self.shard_for_id(id)?;
                shard.delete(id, namespace.as_deref()).await
            }
            NamespaceWalOp::Clear { namespace } => {
                let ns = namespace.as_deref();
                if let Some(name) = ns {
                    let futs = self.shards.iter().map(|s| s.delete_namespace(name));
                    try_join_all(futs).await?;
                } else {
                    // Clear default namespace: delete every listed chunk id.
                    for shard in &self.shards {
                        let chunks = shard.list(None, None).await?;
                        for chunk in chunks {
                            shard.delete(&chunk.id, None).await?;
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Rebuild shard indexes by replaying the in-memory WAL from scratch.
    ///
    /// # Errors
    ///
    /// Returns when clearing shards or applying WAL ops fails.
    pub async fn replay_wal(&self) -> Result<usize, DripstoneError> {
        for shard in &self.shards {
            for ns in shard.namespaces().await? {
                shard.delete_namespace(&ns).await?;
            }
            let default_chunks = shard.list(None, None).await?;
            for chunk in default_chunks {
                shard.delete(&chunk.id, None).await?;
            }
        }
        let entries = self.wal.entries();
        for entry in &entries {
            self.apply_op(&entry.operation).await?;
        }
        Ok(entries.len())
    }

    /// Load durable WAL segments (if configured) then replay into shards.
    ///
    /// # Errors
    ///
    /// Returns when object-store load or replay fails.
    pub async fn recover(&self) -> Result<usize, DripstoneError> {
        self.wal.load_from_store().await?;
        self.replay_wal().await
    }

    fn merge_search_results(
        shards: Vec<Vec<SearchResult>>,
        limit: usize,
        higher_is_better: bool,
    ) -> Vec<SearchResult> {
        let mut best: HashMap<String, SearchResult> = HashMap::new();
        for results in shards {
            for result in results {
                match best.entry(result.chunk.id.clone()) {
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(result);
                    }
                    std::collections::hash_map::Entry::Occupied(mut slot) => {
                        let replace = if higher_is_better {
                            result.score > slot.get().score
                        } else {
                            result.score < slot.get().score
                        };
                        if replace {
                            let _ = slot.insert(result);
                        }
                    }
                }
            }
        }
        let mut merged: Vec<SearchResult> = best.into_values().collect();
        if higher_is_better {
            merged.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            merged.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        merged.truncate(limit.max(1));
        for (rank, result) in merged.iter_mut().enumerate() {
            result.rank = rank;
        }
        merged
    }
}

#[async_trait]
impl Backend for WalShardedBackend {
    fn capabilities(&self) -> Capabilities {
        let mut caps = self
            .shards
            .first()
            .map(|s| s.capabilities())
            .unwrap_or_else(Capabilities::memory);
        // Sharding is transparent; advertise the union conservatively.
        caps.supports_namespaces = caps.supports_namespaces
            || self
                .shards
                .iter()
                .any(|s| s.capabilities().supports_namespaces);
        caps
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        // Group by shard after WAL append so the log remains the source of truth.
        for chunk in chunks {
            let op = NamespaceWalOp::Upsert(chunk.clone());
            self.wal.append(op.clone()).await?;
            self.apply_op(&op).await?;
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.shard_for_id(id)?.get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let op = NamespaceWalOp::Delete {
            id: id.to_string(),
            namespace: namespace.map(str::to_string),
        };
        self.wal.append(op.clone()).await?;
        self.apply_op(&op).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // WAL each chunk delete so recovery stays consistent across shards.
        let chunks = self.list(namespace, None).await?;
        let mut deleted = 0_usize;
        for chunk in chunks {
            if chunk.doc_id == doc_id {
                self.delete(&chunk.id, namespace).await?;
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        // For MaxSim, each shard must return an overfetched ranked list so the
        // global merge can recover the true top-`limit` across shards.
        let (shard_query, merge_limit, higher_is_better) = if query.query_multi_vectors.is_some() {
            let mut q = query.clone();
            q.limit = crate::search::overfetch_limit(query);
            // MaxSim scores are always higher-is-better after rerank.
            (q, query.limit.max(1), true)
        } else {
            (
                query.clone(),
                query.limit.max(1),
                query.metric.higher_is_better(),
            )
        };
        let futs = self.shards.iter().map(|s| s.search(&shard_query));
        let shard_results = try_join_all(futs).await?;
        Ok(Self::merge_search_results(
            shard_results,
            merge_limit,
            higher_is_better,
        ))
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let futs = self.shards.iter().map(|s| s.namespaces());
        let parts = try_join_all(futs).await?;
        let mut set = HashSet::new();
        for part in parts {
            for ns in part {
                set.insert(ns);
            }
        }
        let mut out: Vec<String> = set.into_iter().collect();
        out.sort();
        Ok(out)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let op = NamespaceWalOp::Clear {
            namespace: Some(namespace.to_string()),
        };
        self.wal.append(op.clone()).await?;
        self.apply_op(&op).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let futs = self.shards.iter().map(|s| s.count(namespace));
        let counts = try_join_all(futs).await?;
        Ok(counts.into_iter().sum())
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let futs = self.shards.iter().map(|s| s.list(namespace, None));
        let parts = try_join_all(futs).await?;
        let mut out = Vec::new();
        for part in parts {
            out.extend(part);
        }
        if let Some(limit) = limit {
            out.truncate(limit);
        }
        Ok(out)
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        let futs = self.shards.iter().map(|s| s.flush());
        try_join_all(futs).await?;
        Ok(())
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        let futs = self.shards.iter().map(|s| s.close());
        try_join_all(futs).await?;
        Ok(())
    }
}
