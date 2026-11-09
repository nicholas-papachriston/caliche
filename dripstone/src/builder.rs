//! Backend-only builders (secondary API).
//!
//! Prefer [`crate::Dripstone`] when you need ingest and search with an embedder.
//! Use this module only when you already have embeddings and want a fluent
//! backend constructor (`VectorStoreBuilder`, `IndexedBuilder`).
//!
use crate::traits::Backend;
use std::sync::Arc;

// Backend imports
#[cfg(feature = "memory-backend")]
use crate::backend::memory::{InMemoryConfig, MemoryBackend};

#[cfg(feature = "redis-backend")]
use crate::backend::redis::RedisBackend;

#[cfg(feature = "postgres-backend")]
use crate::backend::postgres::{PostgresBackend, PostgresBackendOptions};

#[cfg(feature = "milvus-backend")]
use crate::backend::milvus::{MilvusBackend, MilvusBackendOptions};
#[cfg(feature = "qdrant-backend")]
use crate::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};
#[cfg(feature = "weaviate-backend")]
use crate::backend::weaviate::{WeaviateBackend, WeaviateBackendOptions};
#[cfg(feature = "qdrant-backend")]
use crate::embedding::SimilarityMetric;

use crate::backend::full_text_indexed::{FullTextIndexedBackend, FullTextIndexedBackendConfig};
#[cfg(any(
    feature = "memory-backend",
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-grpc-backend",
    feature = "qdrant-backend",
    feature = "pinecone-backend",
    feature = "chroma-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend",
    feature = "s3-backend",
    feature = "coldforge-backend",
    feature = "mmap-backend"
))]
use crate::backend::tiered::TieredChainBuilder;
use crate::errors::DripstoneError;
use crate::fts::{FullTextIndexConfig, InMemoryFullTextIndex};
use crate::search::SimpleTokenizer;

/// Builder for creating vector store backends with various storage implementations.
///
/// This builder provides a unified interface for configuring and creating
/// `Backend` trait implementations regardless of the underlying storage backend.
#[must_use]
pub struct VectorStoreBuilder;

impl VectorStoreBuilder {
    /// Create a new vector store builder.
    pub const fn new() -> Self {
        Self
    }

    /// Configure an in-memory backend.
    ///
    /// # Example
    ///
    /// ```rust
    /// use dripstone::VectorStoreBuilder;
    ///
    /// let client = VectorStoreBuilder::new()
    ///     .in_memory()
    ///     .with_ttl(3600)
    ///     .build();
    /// ```
    #[cfg(feature = "memory-backend")]
    pub fn in_memory(self) -> InMemoryBuilder {
        InMemoryBuilder {
            config: InMemoryConfig::default(),
        }
    }

    /// Configure a Redis backend.
    ///
    /// # Arguments
    ///
    /// * `connection_string` - Redis connection URL (e.g., <redis://localhost:6379>)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dripstone::VectorStoreBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = VectorStoreBuilder::new()
    ///     .redis("redis://localhost:6379")
    ///     .with_prefix("vectors")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "redis-backend")]
    pub fn redis(self, connection_string: impl Into<String>) -> RedisBuilder {
        RedisBuilder {
            connection_string: connection_string.into(),
            prefix: None,
            pool_size: None,
        }
    }

    /// Configure a `PostgreSQL` backend.
    ///
    /// # Arguments
    ///
    /// * `connection_string` - `PostgreSQL` connection URL
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dripstone::VectorStoreBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = VectorStoreBuilder::new()
    ///     .postgres("postgresql://localhost/mydb")
    ///     .with_table("embeddings")
    ///     .with_dimension(768)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "postgres-backend")]
    pub fn postgres(self, connection_string: impl Into<String>) -> PostgresBuilder {
        PostgresBuilder {
            connection_string: connection_string.into(),
            table_name: None,
            dimension: None,
        }
    }

    /// Configure a Qdrant backend.
    ///
    /// # Arguments
    ///
    /// * `url` - Qdrant server URL
    /// * `collection` - Collection name
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dripstone::VectorStoreBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = VectorStoreBuilder::new()
    ///     .qdrant("http://localhost:6334", "my_collection")
    ///     .with_dimension(768)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "qdrant-backend")]
    pub fn qdrant(self, url: impl Into<String>, collection: impl Into<String>) -> QdrantBuilder {
        QdrantBuilder {
            url: url.into(),
            collection: collection.into(),
            dimension: None,
            api_key: None,
        }
    }

    /// Configure a Weaviate backend.
    ///
    /// # Arguments
    ///
    /// * `url` - Weaviate server URL
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dripstone::VectorStoreBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = VectorStoreBuilder::new()
    ///     .weaviate("http://localhost:8081")
    ///     .with_dimension(768)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "weaviate-backend")]
    pub fn weaviate(self, url: impl Into<String>) -> WeaviateBuilder {
        WeaviateBuilder {
            url: url.into(),
            dimension: None,
            class_prefix: None,
            api_key: None,
        }
    }

    /// Configure a Milvus backend.
    ///
    /// # Arguments
    ///
    /// * `endpoint` - Milvus server endpoint URL
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use dripstone::VectorStoreBuilder;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = VectorStoreBuilder::new()
    ///     .milvus("http://localhost:19530")
    ///     .with_dimension(768)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "milvus-backend")]
    pub fn milvus(self, endpoint: impl Into<String>) -> MilvusBuilder {
        MilvusBuilder {
            endpoint: endpoint.into(),
            dimension: None,
            collection_prefix: None,
            index_type: None,
            metric_type: None,
            nlist: None,
        }
    }

    /// Wrap a pre-built backend with an ANN index for accelerated search.
    ///
    /// # Example
    ///
    /// ```text
    /// use dripstone::{VectorStoreBuilder, Backend};
    /// use dripstone::index::{HnswIndex, HnswConfig};
    /// use std::sync::Arc;
    ///
    /// let inner: Arc<dyn Backend> = VectorStoreBuilder::new()
    ///     .in_memory()
    ///     .build();
    ///
    /// let indexed: Arc<dyn Backend> = VectorStoreBuilder::new()
    ///     .indexed(inner)
    ///     .with_hnsw(HnswConfig::default(), 384)
    ///     .build();
    /// ```
    #[cfg(feature = "tokio")]
    pub fn indexed(self, inner: Arc<dyn Backend>) -> IndexedBuilder {
        IndexedBuilder {
            inner,
            index: None,
            config: crate::backend::indexed::IndexedBackendConfig::default(),
        }
    }

    /// Wrap a pre-built backend with a native BM25/MAXSCORE full-text index.
    pub fn full_text_indexed(self, inner: Arc<dyn Backend>) -> FullTextIndexedBuilder {
        FullTextIndexedBuilder {
            inner,
            config: FullTextIndexedBackendConfig::default(),
            tokenizer: SimpleTokenizer::new().with_english_stopwords(),
            block_size: 32,
        }
    }

    /// Configure an N-tier chain backend (cold -> hot). Add tiers in order with
    /// `.tier(backend, TierConfig::new("name"))`, then call `.build()`.
    #[cfg(any(
        feature = "memory-backend",
        feature = "redis-backend",
        feature = "postgres-backend",
        feature = "qdrant-grpc-backend",
        feature = "qdrant-backend",
        feature = "pinecone-backend",
        feature = "chroma-backend",
        feature = "weaviate-backend",
        feature = "milvus-backend",
        feature = "s3-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend"
    ))]
    #[must_use]
    pub fn tiered(self) -> TieredChainBuilder {
        TieredChainBuilder::new()
    }
}

impl Default for VectorStoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== In-Memory Builder ====================

/// Builder for in-memory vector store.
#[cfg(feature = "memory-backend")]
#[must_use]
pub struct InMemoryBuilder {
    config: InMemoryConfig,
}

#[cfg(feature = "memory-backend")]
impl InMemoryBuilder {
    /// Set the default TTL for entries (in seconds).
    ///
    /// Default is 3600 seconds (1 hour).
    pub const fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.config.default_ttl_secs = Some(ttl_secs);
        self
    }

    /// Set the maximum number of entries before eviction starts.
    ///
    /// Default is 10,000 entries.
    pub const fn with_max_entries(mut self, max: usize) -> Self {
        self.config.max_entries = Some(max);
        self
    }

    /// Disable the maximum entry limit (unlimited storage).
    pub const fn unlimited(mut self) -> Self {
        self.config.max_entries = None;
        self
    }

    /// Enable or disable automatic cleanup of expired entries.
    ///
    /// Default is enabled.
    pub const fn with_auto_cleanup(mut self, enabled: bool) -> Self {
        self.config.auto_cleanup = enabled;
        self
    }

    /// Build the in-memory backend.
    #[must_use]
    pub fn build(self) -> Arc<dyn Backend> {
        Arc::new(MemoryBackend::with_config(self.config))
    }
}

#[cfg(all(test, feature = "memory-backend"))]
mod in_memory_builder_tests {
    use super::*;

    #[test]
    fn with_ttl_stores_the_requested_seconds() {
        let builder = VectorStoreBuilder::new().in_memory().with_ttl(15);
        assert_eq!(builder.config.default_ttl_secs, Some(15));
    }
}

// ==================== Redis Builder ====================

#[cfg(feature = "redis-backend")]
/// Builder for Redis vector store.
#[must_use]
pub struct RedisBuilder {
    connection_string: String,
    prefix: Option<String>,
    pool_size: Option<usize>,
}

#[cfg(feature = "redis-backend")]
impl RedisBuilder {
    /// Set the key prefix for all entries.
    ///
    /// Default is no prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set the connection pool size.
    ///
    /// Default is determined by the Redis client.
    pub const fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = Some(size);
        self
    }

    /// Build the Redis backend.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if the Redis backend cannot connect.
    pub async fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        let prefix = self.prefix.unwrap_or_else(|| "embeddings".to_string());
        let backend = RedisBackend::with_prefix(&self.connection_string, &prefix).await?;
        Ok(Arc::new(backend))
    }
}

// ==================== PostgreSQL Builder ====================

#[cfg(feature = "postgres-backend")]
/// Builder for `PostgreSQL` vector store.
#[must_use]
pub struct PostgresBuilder {
    connection_string: String,
    table_name: Option<String>,
    dimension: Option<usize>,
}

#[cfg(feature = "postgres-backend")]
impl PostgresBuilder {
    /// Set the table name for storing embeddings.
    ///
    /// Default is "embeddings".
    pub fn with_table(mut self, table: impl Into<String>) -> Self {
        self.table_name = Some(table.into());
        self
    }

    /// Set the vector dimension.
    ///
    /// Required for creating the table schema if it doesn't exist.
    pub const fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Build the `PostgreSQL` backend.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if the dimension is missing or the backend
    /// cannot connect or initialize its schema.
    pub async fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        let dimension = self
            .dimension
            .ok_or_else(|| DripstoneError::Misconfiguration("dimension is required".to_string()))?;
        let table = self.table_name.unwrap_or_else(|| "embeddings".to_string());
        let vector_dimension = i32::try_from(dimension)
            .map_err(|_| DripstoneError::configuration("dimension exceeds i32 range"))?;
        let backend = PostgresBackend::with_options(
            &self.connection_string,
            PostgresBackendOptions {
                table_name: table,
                ensure_schema: true,
                max_connections: 10,
                vector_dimension: Some(vector_dimension),
                hnsw_index: None,
            },
        )
        .await?;
        Ok(Arc::new(backend))
    }
}

// ==================== Qdrant Builder ====================

#[cfg(feature = "qdrant-backend")]
/// Builder for Qdrant vector store.
#[must_use]
pub struct QdrantBuilder {
    url: String,
    collection: String,
    dimension: Option<usize>,
    api_key: Option<String>,
}

#[cfg(feature = "qdrant-backend")]
impl QdrantBuilder {
    /// Set the vector dimension.
    ///
    /// Required for creating the collection if it doesn't exist.
    pub const fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set the API key for authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Build the Qdrant backend.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if the dimension is missing or the backend
    /// cannot connect.
    pub async fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        let dimension = self
            .dimension
            .ok_or_else(|| DripstoneError::Misconfiguration("dimension is required".to_string()))?;
        let backend = QdrantHttpBackend::with_options(QdrantBackendOptions {
            retry_config: None,
            base_url: self.url,
            collection_prefix: Some(self.collection),
            vector_dimension: dimension,
            distance: SimilarityMetric::Cosine,
            timeout: None,
            api_key: self.api_key,
            batch_size: Some(100),
        })
        .await?;
        Ok(Arc::new(backend))
    }
}

// ==================== Weaviate Builder ====================

#[cfg(feature = "weaviate-backend")]
/// Builder for Weaviate vector store.
#[must_use]
pub struct WeaviateBuilder {
    url: String,
    dimension: Option<usize>,
    class_prefix: Option<String>,
    api_key: Option<String>,
}

#[cfg(feature = "weaviate-backend")]
impl WeaviateBuilder {
    /// Set the vector dimension.
    ///
    /// Required for creating the class schema if it doesn't exist.
    pub const fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set the class prefix for all classes.
    ///
    /// Default is "Dripstone_".
    pub fn with_class_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.class_prefix = Some(prefix.into());
        self
    }

    /// Set the API key for authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Build the Weaviate backend.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if the dimension is missing or the backend
    /// cannot connect.
    pub async fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        let dimension = self
            .dimension
            .ok_or_else(|| DripstoneError::Misconfiguration("dimension is required".to_string()))?;
        let backend = WeaviateBackend::with_options(WeaviateBackendOptions {
            base_url: self.url,
            class_prefix: self.class_prefix,
            vector_dimension: dimension,
            api_key: self.api_key,
            timeout: None,
            batch_size: None,
            retry_config: None,
        })
        .await?;
        Ok(Arc::new(backend))
    }
}

// ==================== Milvus Builder ====================

#[cfg(feature = "milvus-backend")]
/// Builder for Milvus vector store.
#[must_use]
pub struct MilvusBuilder {
    endpoint: String,
    dimension: Option<usize>,
    collection_prefix: Option<String>,
    index_type: Option<String>,
    metric_type: Option<String>,
    nlist: Option<u32>,
}

#[cfg(feature = "milvus-backend")]
impl MilvusBuilder {
    /// Set the vector dimension.
    ///
    /// Required for creating the collection schema if it doesn't exist.
    pub const fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Set the collection prefix for all collections.
    ///
    /// Default is "dripstone_".
    pub fn with_collection_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.collection_prefix = Some(prefix.into());
        self
    }

    /// Set the index type (e.g., `IVF_FLAT`, `HNSW`).
    ///
    /// Default is `IVF_FLAT`.
    pub fn with_index_type(mut self, index_type: impl Into<String>) -> Self {
        self.index_type = Some(index_type.into());
        self
    }

    /// Set the metric type (e.g., "COSINE", "L2", "IP").
    ///
    /// Default is "COSINE".
    pub fn with_metric_type(mut self, metric_type: impl Into<String>) -> Self {
        self.metric_type = Some(metric_type.into());
        self
    }

    /// Set the nlist parameter for IVF index.
    ///
    /// Default is 128.
    pub const fn with_nlist(mut self, nlist: u32) -> Self {
        self.nlist = Some(nlist);
        self
    }

    /// Build the Milvus backend.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if the dimension is missing or the backend
    /// cannot connect.
    pub async fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        let dimension = self
            .dimension
            .ok_or_else(|| DripstoneError::Misconfiguration("dimension is required".to_string()))?;
        let backend = MilvusBackend::with_options(MilvusBackendOptions {
            endpoint: self.endpoint,
            collection_prefix: self.collection_prefix,
            vector_dimension: dimension,
            index_type: self.index_type,
            metric_type: self.metric_type,
            nlist: self.nlist,
            timeout: None,
            batch_size: None,
            retry_config: None,
            flush_policy: None,
        })
        .await?;
        Ok(Arc::new(backend))
    }
}

// ==================== Indexed (ANN) Builder ====================

/// Builder for wrapping an existing backend with ANN-accelerated search.
#[cfg(feature = "tokio")]
#[must_use]
pub struct IndexedBuilder {
    inner: Arc<dyn Backend>,
    index: Option<Box<dyn crate::index::VectorIndex>>,
    config: crate::backend::indexed::IndexedBackendConfig,
}

/// Builder for wrapping an existing backend with a native full-text index.
#[must_use]
pub struct FullTextIndexedBuilder {
    inner: Arc<dyn Backend>,
    config: FullTextIndexedBackendConfig,
    tokenizer: SimpleTokenizer,
    block_size: usize,
}

impl FullTextIndexedBuilder {
    /// Sets the number of mutations that trigger an automatic index rebuild.
    pub const fn with_auto_rebuild_threshold(mut self, threshold: usize) -> Self {
        self.config.auto_rebuild_threshold = threshold;
        self
    }

    /// Sets the optional path used to persist full-text index snapshots.
    pub fn with_snapshot_path(mut self, snapshot_path: impl Into<String>) -> Self {
        self.config.snapshot_path = Some(snapshot_path.into());
        self
    }

    /// Enables rebuilding a dirty index immediately before search.
    pub const fn with_rebuild_on_search(mut self, rebuild_on_search: bool) -> Self {
        self.config.rebuild_on_search = rebuild_on_search;
        self
    }

    /// Replaces the tokenizer used by the full-text index.
    pub fn with_tokenizer(mut self, tokenizer: SimpleTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Sets the `MAXSCORE` posting block size. Values below one become one.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size.max(1);
        self
    }

    /// Builds the full-text indexed backend.
    #[must_use]
    pub fn build(self) -> Arc<dyn Backend> {
        let index = Arc::new(InMemoryFullTextIndex::with_config(FullTextIndexConfig {
            tokenizer: self.tokenizer,
            block_size: self.block_size,
        }));
        Arc::new(FullTextIndexedBackend::new(self.inner, index, self.config))
    }
}

#[cfg(feature = "tokio")]
impl IndexedBuilder {
    /// Set the HNSW index with the given configuration and vector dimension.
    #[cfg(feature = "hnsw")]
    pub fn with_hnsw(mut self, hnsw_config: crate::index::HnswConfig, dimension: usize) -> Self {
        self.index = Some(Box::new(crate::index::HnswIndex::new(
            hnsw_config,
            dimension,
        )));
        self
    }

    /// Set the `SPFresh` index with the given configuration and vector dimension.
    pub fn with_spfresh(
        mut self,
        spfresh_config: crate::index::spfresh::SPFreshConfig,
        dimension: usize,
    ) -> Self {
        self.index = Some(Box::new(crate::index::spfresh::CentroidIndex::new(
            spfresh_config,
            dimension,
        )));
        self
    }

    /// Provide a pre-built [`VectorIndex`](crate::index::VectorIndex) implementation.
    pub fn with_index(mut self, index: Box<dyn crate::index::VectorIndex>) -> Self {
        self.index = Some(index);
        self
    }

    /// Set the number of upserts before the index is fully rebuilt.
    /// Default is 1000. Set to 0 to disable periodic rebuilds.
    pub const fn with_auto_rebuild_threshold(mut self, threshold: usize) -> Self {
        self.config.auto_rebuild_threshold = threshold;
        self
    }

    /// When the index is empty, fall back to the inner backend's brute-force search.
    /// Default is `true`.
    pub const fn with_fallback(mut self, fallback: bool) -> Self {
        self.config.fallback_to_brute_force = fallback;
        self
    }

    /// Set the over-fetch multiplier for ANN candidate retrieval.
    /// Default is 2 (retrieve `limit * 2` candidates).
    pub const fn with_overfetch_factor(mut self, factor: usize) -> Self {
        self.config.overfetch_factor = factor;
        self
    }

    /// Commit vectors to the inner backend first and queue ANN inserts separately.
    /// Search merges index hits with a brute-force tail scan by default.
    pub const fn with_deferred_index_updates(mut self) -> Self {
        self.config.index_update_mode = crate::backend::indexed::IndexUpdateMode::Deferred;
        self
    }

    /// When deferred indexing is enabled, scan unindexed vectors on each search
    /// so recent upserts stay visible. Default is `true`.
    pub const fn with_scan_unindexed_tail(mut self, scan: bool) -> Self {
        self.config.scan_unindexed_tail = scan;
        self
    }

    /// When deferred indexing is enabled, flush the pending queue after this many
    /// queued records. Default is 256. Set to `0` to disable automatic flush.
    pub const fn with_deferred_flush_threshold(mut self, threshold: usize) -> Self {
        self.config.deferred_flush_threshold = threshold;
        self
    }

    /// When deferred indexing is enabled, flush the pending queue at search time
    /// when pending count reaches this threshold. Default is `0` (disabled).
    pub const fn with_deferred_search_flush_threshold(mut self, threshold: usize) -> Self {
        self.config.deferred_search_flush_threshold = threshold;
        self
    }

    /// When deferred indexing is enabled, cap tail scan to the last N pending records.
    /// Default is `0` (scan the full pending queue).
    pub const fn with_max_tail_scan_records(mut self, max_records: usize) -> Self {
        self.config.max_tail_scan_records = max_records;
        self
    }

    /// Build the indexed backend, returning `Arc<dyn Backend>`.
    ///
    /// # Errors
    ///
    /// Returns `DripstoneError::Misconfiguration` if no index was configured.
    pub fn build(self) -> Result<Arc<dyn Backend>, DripstoneError> {
        let index = self.index.ok_or_else(|| {
            DripstoneError::Misconfiguration(
                "IndexedBuilder: no index configured -- call .with_hnsw(), .with_spfresh(), or .with_index()".to_string(),
            )
        })?;
        Ok(Arc::new(crate::backend::indexed::IndexedBackend::new(
            self.inner,
            index,
            self.config,
        )))
    }
}
