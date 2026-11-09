//! `ColdForge`: S3-backed vector storage with `SPFresh` indexing
//!
//! `ColdForge` combines object storage with intelligent caching for cost-effective,
//! scalable vector search.
//!
//! ## Architecture
//!
//! - **Cold Storage**: Chunk envelopes in per-id objects (`bodies/{namespace}/{chunk_id}`),
//!   plus a write-ahead log of the same records
//! - **Hot Cache**: LRU of envelopes. Eviction does not remove the body store
//! - **`SPFresh` Index**: Centroid ANN. `records` keeps the pooled vector and
//!   `multi_vectors`. Search takes the hit id and hydrates the chunk-id body
//! - **Write-Ahead Log**: Batched segments on the object store. `upsert` is durable
//!   only after `flush`, a full batch, or the background interval. `WalCompactor`
//!   deletes a prefix only after [`ColdForgeBackend::compact_wal`] saves a
//!   membership snapshot and every dropped envelope matches a chunk-id body.
//!   Shard routing stays `hash(chunk.id)`
//!
//! Live timings are [`live`]: one scene, backends `minio`, `s3`, and `s3-redis`.
//! MinIO is the local floor, not an AWS network measurement. The scene uses
//! [`DEFAULT_WAL_REFRESH_INTERVAL_MS`]. `wal_refresh_interval_ms = 0` lists the
//! WAL on every refreshing read.
//!
//! Replaying a WAL segment installs envelopes in memory. It does not PUT chunk
//! bodies again. Upsert writes those objects with bounded concurrency
//! ([`DEFAULT_BODY_PUT_CONCURRENCY`]) before the WAL append. Delete removes them.

pub mod cache;
pub mod live;
mod live_corpus;

mod backend;
mod body;
mod index;

use crate::errors::DripstoneError;
use crate::index::spfresh::CentroidIndex;
use crate::traits::Backend;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};

use crate::storage::{ObjectStore, ObjectStoreConfig, create_object_store};

use crate::backend::s3::wal::WalWriter;
use cache::ColdForgeCache;

pub use index::CacheWarmStats;

/// Minimum gap between shared-WAL listings.
///
/// `0` lists object storage on every refreshing read, including `get`, and that
/// list runs before a hot tier. The first read still replays unseen segments.
pub const DEFAULT_WAL_REFRESH_INTERVAL_MS: u64 = 1_000;

/// Concurrent chunk-body PUTs on the upsert path.
///
/// Hydrate already issues this many GETs. A serial PUT was the slow part of
/// large soaks (tens of milliseconds per object on EC2).
pub const DEFAULT_BODY_PUT_CONCURRENCY: usize = 32;

/// Configuration for `ColdForge` backend
#[derive(Clone)]
pub struct ColdForgeConfig {
    /// S3 bucket name
    pub bucket: String,
    /// S3 key prefix
    pub key_prefix: String,
    /// AWS region
    pub region: String,
    /// Cache size in bytes (used if no `hot_tier` is provided)
    pub cache_size_bytes: usize,
    /// Number of centroids for `SPFresh` indexing
    pub num_centroids: usize,
    /// WAL batch interval in milliseconds.
    ///
    /// `upsert` returns before the segment is durable unless this is `0`
    /// (flush that batch before returning) or the caller awaits [`ColdForgeBackend::flush`].
    /// The background task waits one full interval before its first flush.
    /// A failed flush keeps the batch queued and surfaces the error.
    pub wal_batch_interval_ms: u64,
    /// Max operations per WAL batch before flush (larger batches reduce S3 PUT count).
    pub wal_max_batch_size: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Optional hot tier backend (Redis, memory, and so on).
    ///
    /// The hot tier is a `get` cache. Search reads the in-process `SPFresh` index.
    /// When this is `Some` and [`Self::refresh_shared_wal`] is `None`, reads
    /// reload flushed WAL segments so other processes sharing the bucket are visible.
    /// A hot-tier hit is returned only when the namespace index still contains the id.
    /// Note: This field cannot be Debug because trait objects don't implement Debug
    #[allow(missing_debug_implementations)]
    pub hot_tier: Option<Arc<dyn Backend + Send + Sync>>,
    /// Shared-WAL refresh policy.
    ///
    /// `None` refreshes on read iff [`Self::hot_tier`] is set. `Some(true)` always
    /// refreshes. `Some(false)` is single-writer: the index loads once and ignores
    /// later segments until [`ColdForgeBackend::refresh_shared_wal`].
    pub refresh_shared_wal: Option<bool>,
    /// Minimum gap between shared-WAL listings.
    ///
    /// `0` lists object storage on every refreshing read, including `get`.
    /// That list runs before the Redis hot tier, so a hot-tier get cannot beat
    /// S3 while this stays `0`. The default is 1000 ms. The first read still
    /// replays unseen segments.
    pub wal_refresh_interval_ms: u64,
    /// AWS access key ID (optional, uses default credentials if not set)
    pub access_key_id: Option<String>,
    /// AWS secret access key (optional, uses default credentials if not set)
    pub secret_access_key: Option<String>,
    /// AWS session token for temporary credentials (optional)
    pub session_token: Option<String>,
    /// Object store configuration (optional, uses `from_env()` if not provided)
    /// Use this for S3-compatible services or programmatic configuration
    pub object_store_config: Option<crate::storage::ObjectStoreConfig>,
    /// How many chunk-body PUTs may be in flight during one upsert batch.
    ///
    /// `1` is serial. The default matches hydrate parallelism. WAL appends stay
    /// in input order after the PUTs finish, and replay still does not PUT.
    pub body_put_concurrency: usize,
}

impl std::fmt::Debug for ColdForgeConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdForgeConfig")
            .field("bucket", &self.bucket)
            .field("key_prefix", &self.key_prefix)
            .field("region", &self.region)
            .field("cache_size_bytes", &self.cache_size_bytes)
            .field("num_centroids", &self.num_centroids)
            .field("wal_batch_interval_ms", &self.wal_batch_interval_ms)
            .field("wal_max_batch_size", &self.wal_max_batch_size)
            .field("enable_compression", &self.enable_compression)
            .field("refresh_shared_wal", &self.refresh_shared_wal)
            .field("wal_refresh_interval_ms", &self.wal_refresh_interval_ms)
            .field("vector_dimension", &self.vector_dimension)
            .field(
                "hot_tier",
                &self.hot_tier.as_ref().map(|_| "<EmbeddingBackend>"),
            )
            .field(
                "access_key_id",
                &self.access_key_id.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "secret_access_key",
                &self.secret_access_key.as_ref().map(|_| "<redacted>"),
            )
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .field("body_put_concurrency", &self.body_put_concurrency)
            .finish_non_exhaustive()
    }
}

impl Default for ColdForgeConfig {
    fn default() -> Self {
        Self {
            bucket: String::new(),
            key_prefix: "coldforge/".to_string(),
            region: "us-east-1".to_string(),
            cache_size_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            num_centroids: 256,
            wal_batch_interval_ms: 1000,
            wal_max_batch_size: 256,
            enable_compression: true,
            vector_dimension: 1536,
            hot_tier: None,
            refresh_shared_wal: None,
            wal_refresh_interval_ms: DEFAULT_WAL_REFRESH_INTERVAL_MS,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            object_store_config: None,
            body_put_concurrency: DEFAULT_BODY_PUT_CONCURRENCY,
        }
    }
}

/// `ColdForge` backend implementation
pub struct ColdForgeBackend {
    /// Object store
    store: Arc<dyn ObjectStore>,
    /// Configuration
    config: ColdForgeConfig,
    /// `SPFresh` index per namespace
    indexes: Arc<RwLock<HashMap<String, Arc<RwLock<CentroidIndex>>>>>,
    /// WAL writer per namespace
    wal_writers: Arc<RwLock<HashMap<String, Arc<WalWriter>>>>,
    /// Hot tier backend (Redis/Postgres/etc.) for distributed deployments
    hot_tier: Option<Arc<dyn Backend + Send + Sync>>,
    /// In-memory cache (used if no `hot_tier` is provided, or as secondary cache)
    cache: Arc<ColdForgeCache>,
    /// Chunk envelopes keyed by chunk id, independent of cluster membership.
    bodies: body::ChunkBodyStore,
    /// Per-namespace snapshot and WAL high-water mark.
    durability: Arc<parking_lot::RwLock<HashMap<String, NamespaceDurability>>>,
    /// Serializes snapshot load and WAL replay for a backend.
    wal_gate: Arc<Mutex<()>>,
}

#[derive(Default)]
struct NamespaceDurability {
    snapshot_loaded: bool,
    initial_replay_done: bool,
    applied_wal_keys: HashSet<String>,
    last_refresh: Option<Instant>,
}

impl ColdForgeBackend {
    /// Build ColdForge on top of an already-constructed [`ObjectStore`].
    ///
    /// Prefer this for tests and multi-shard layouts that share a
    /// [`crate::storage::MemoryObjectStore`] (or one S3 store with distinct
    /// `key_prefix` values per shard).
    #[must_use]
    pub fn with_object_store(store: Arc<dyn ObjectStore>, config: ColdForgeConfig) -> Self {
        let cache = Arc::new(ColdForgeCache::new(config.cache_size_bytes));
        let hot_tier = config.hot_tier.clone();
        let bodies = body::ChunkBodyStore::new(
            Arc::clone(&store),
            config.bucket.clone(),
            config.key_prefix.clone(),
        );
        Self {
            store,
            config,
            indexes: Arc::new(RwLock::new(HashMap::new())),
            wal_writers: Arc::new(RwLock::new(HashMap::new())),
            hot_tier,
            cache,
            bodies,
            durability: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            wal_gate: Arc::new(Mutex::new(())),
        }
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(config: ColdForgeConfig) -> Result<Self, DripstoneError> {
        let store_config = config.object_store_config.clone().unwrap_or_else(|| {
            if let (Some(access_key), Some(secret_key)) = (
                config.access_key_id.as_ref(),
                config.secret_access_key.as_ref(),
            ) {
                ObjectStoreConfig::aws_s3(
                    Some(config.region.clone()),
                    Some(access_key.clone()),
                    Some(secret_key.clone()),
                    config.session_token.clone(),
                )
            } else {
                ObjectStoreConfig::from_env().unwrap_or_default()
            }
        });
        let store = create_object_store(store_config).await?;
        Ok(Self::with_object_store(store, config))
    }

    /// Create a new `ColdForge` backend with a hot tier backend
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_hot_tier<H: Backend + Send + Sync + 'static>(
        mut config: ColdForgeConfig,
        hot_tier: Arc<H>,
    ) -> Result<Self, DripstoneError> {
        config.hot_tier = Some(hot_tier as Arc<dyn Backend + Send + Sync>);
        Self::new(config).await
    }

    /// Like [`Self::with_object_store`] but also attaches a hot tier backend.
    #[must_use]
    pub fn with_object_store_and_hot_tier(
        store: Arc<dyn ObjectStore>,
        mut config: ColdForgeConfig,
        hot_tier: Arc<dyn Backend + Send + Sync>,
    ) -> Self {
        config.hot_tier = Some(hot_tier);
        Self::with_object_store(store, config)
    }
}

#[cfg(test)]
mod hydrate_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> ColdForgeConfig {
        ColdForgeConfig {
            bucket: "test-bucket".to_string(),
            key_prefix: "test/".to_string(),
            region: "us-east-1".to_string(),
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            cache_size_bytes: 100 * 1024 * 1024,
            num_centroids: 10,
            wal_batch_interval_ms: 1000,
            wal_max_batch_size: 100,
            enable_compression: true,
            vector_dimension: 128,
            hot_tier: None,
            refresh_shared_wal: None,
            wal_refresh_interval_ms: 0,
            object_store_config: None,
            body_put_concurrency: DEFAULT_BODY_PUT_CONCURRENCY,
        }
    }

    #[tokio::test]
    async fn with_object_store_builds_without_network() {
        let store = crate::storage::MemoryObjectStore::shared();
        let backend = ColdForgeBackend::with_object_store(store, create_test_config());
        assert_eq!(backend.capabilities().max_dimension, Some(128));
    }
}
