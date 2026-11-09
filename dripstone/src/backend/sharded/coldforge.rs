//! Compose [`ColdForgeBackend`] shards under a shared namespace WAL.

use std::sync::Arc;

use crate::backend::coldforge::{ColdForgeBackend, ColdForgeConfig};
use crate::backend::memory::{InMemoryConfig, MemoryBackend};
use crate::distributed::{SharedNamespaceWal, SharedWalConfig};
use crate::errors::DripstoneError;
use crate::storage::{MemoryObjectStore, ObjectStore};
use crate::traits::Backend;

use super::WalShardedBackend;

/// Options for [`build_coldforge_wal_sharded`].
#[derive(Debug, Clone)]
pub struct ColdForgeShardOptions {
    pub shard_count: usize,
    pub vector_dimension: usize,
    pub num_centroids: usize,
    pub bucket: String,
    /// Prefix for the shared namespace WAL (must not overlap shard prefixes).
    pub shared_wal_prefix: String,
    /// When true, each shard gets its own [`MemoryBackend`] hot tier with write-through.
    pub memory_hot_tier: bool,
}

impl Default for ColdForgeShardOptions {
    fn default() -> Self {
        Self {
            shard_count: 4,
            vector_dimension: 32,
            num_centroids: 16,
            bucket: "dripstone".to_string(),
            shared_wal_prefix: "namespaces/default/wal/".to_string(),
            memory_hot_tier: true,
        }
    }
}

fn unlimited_memory_hot() -> Arc<MemoryBackend> {
    Arc::new(MemoryBackend::with_config(InMemoryConfig {
        default_ttl_secs: Some(86_400),
        max_entries: None,
        auto_cleanup: false,
    }))
}

/// Build a [`WalShardedBackend`] whose shards are ColdForge instances sharing one
/// object store, each with an isolated `key_prefix`, optional Memory hot tier,
/// and a shared namespace WAL for atomic write order.
///
/// # Errors
///
/// Returns when shard construction or WAL wiring fails.
pub fn build_coldforge_wal_sharded(
    store: Arc<dyn ObjectStore>,
    options: ColdForgeShardOptions,
) -> Result<WalShardedBackend, DripstoneError> {
    let shard_count = options.shard_count.max(1);
    let mut shards: Vec<Arc<dyn Backend>> = Vec::with_capacity(shard_count);
    for i in 0..shard_count {
        let cfg = ColdForgeConfig {
            bucket: options.bucket.clone(),
            key_prefix: format!("shards/{i}/"),
            cache_size_bytes: 32 * 1024 * 1024,
            num_centroids: options.num_centroids.max(1),
            wal_batch_interval_ms: 50,
            wal_max_batch_size: 64,
            enable_compression: false,
            vector_dimension: options.vector_dimension,
            hot_tier: None,
            ..ColdForgeConfig::default()
        };
        let shard: Arc<dyn Backend> = if options.memory_hot_tier {
            Arc::new(ColdForgeBackend::with_object_store_and_hot_tier(
                Arc::clone(&store),
                cfg,
                unlimited_memory_hot(),
            ))
        } else {
            Arc::new(ColdForgeBackend::with_object_store(Arc::clone(&store), cfg))
        };
        shards.push(shard);
    }

    let wal = SharedNamespaceWal::with_object_store(
        store,
        SharedWalConfig {
            bucket: options.bucket,
            key_prefix: options.shared_wal_prefix,
        },
    );
    WalShardedBackend::from_shards(shards, wal)
}

/// Convenience: shared [`MemoryObjectStore`] + ColdForge shards + Memory hot tiers.
///
/// # Errors
///
/// Returns when construction fails.
pub fn build_memory_coldforge_wal_sharded(
    options: ColdForgeShardOptions,
) -> Result<WalShardedBackend, DripstoneError> {
    build_coldforge_wal_sharded(MemoryObjectStore::shared(), options)
}
