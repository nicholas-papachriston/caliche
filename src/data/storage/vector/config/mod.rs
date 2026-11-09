//! Configuration for vector store backends.
//!
//! The public surface is the [`VectorBackendConfig`] enum plus the
//! [`VectorBackendConfigWrapper`] helper that attaches metrics preferences.
//! The implementation is split across a few submodules to keep each file
//! manageable:
//!
//! * [`env`]  — environment-variable-driven construction
//!   ([`VectorBackendConfig::from_env`]) and shared AWS credential plumbing.
//! * [`build`] — the `build_backend` dispatcher that turns a config into a
//!   concrete `Dripstone` backend.
//! * [`tests`] — behavioural unit tests.

mod build;
mod env;
mod json;
#[cfg(test)]
mod tests;

pub use json::{
    tier_config_from_json_value, tiered_tiers_from_json_str, tiered_tiers_from_json_value,
};

use anyhow::Result;
use dripstone::Backend;
use std::sync::Arc;

/// Configuration wrapper that includes backend config and metrics settings
#[derive(Debug, Clone)]
pub struct VectorBackendConfigWrapper {
    /// The backend configuration
    pub backend: VectorBackendConfig,
    /// Enable metrics collection (applies to all backends)
    pub metrics_enabled: bool,
}

/// Configuration for different vector store backends.
#[derive(Debug, Clone)]
pub enum VectorBackendConfig {
    /// In-memory backend with TTL support (default)
    Memory {
        /// Default TTL in seconds (default: 86400 = 24 hours)
        ttl_secs: u64,
        /// Maximum number of entries (default: 100,000)
        max_entries: Option<usize>,
    },

    /// Redis backend
    #[cfg(feature = "vector-store-redis")]
    Redis {
        /// Redis connection URL (e.g., "<redis://127.0.0.1>/")
        url: String,
        /// Key prefix for embeddings
        prefix: String,
    },

    /// Valkey backend (Redis-compatible)
    #[cfg(feature = "vector-store-redis")]
    Valkey {
        /// Valkey connection URL (e.g., "<redis://127.0.0.1>/")
        url: String,
        /// Key prefix for embeddings
        prefix: String,
    },

    /// `PostgreSQL` + pgvector backend
    #[cfg(feature = "vector-store-postgres")]
    Postgres {
        /// `PostgreSQL` connection string
        connection_string: String,
        /// Table name for embeddings
        table_name: String,
        /// Vector dimension (if known)
        vector_dimension: Option<usize>,
        /// Maximum number of connections in the pool
        max_connections: u32,
        /// HNSW index configuration (optional)
        hnsw_config: Option<crate::data::storage::vector::HnswConfig>,
    },

    /// Qdrant gRPC backend
    #[cfg(feature = "vector-store-qdrant-grpc")]
    QdrantGrpc {
        /// Qdrant gRPC URL (e.g., "<http://localhost:6334>")
        grpc_url: String,
        /// Collection name prefix
        collection_prefix: Option<String>,
        /// Vector dimension
        vector_dimension: usize,
        /// API key (optional)
        api_key: Option<String>,
        /// Batch size for operations
        batch_size: Option<usize>,
    },

    /// Qdrant HTTP backend
    #[cfg(feature = "vector-store-qdrant")]
    Qdrant {
        /// Qdrant HTTP URL (e.g., "<http://localhost:6333>")
        url: String,
        /// Collection name prefix
        collection_prefix: Option<String>,
        /// Vector dimension
        vector_dimension: usize,
        /// API key (optional)
        api_key: Option<String>,
        /// Batch size for operations
        batch_size: Option<usize>,
        /// Request timeout
        timeout: Option<std::time::Duration>,
    },

    /// Pinecone backend
    #[cfg(feature = "vector-store-pinecone")]
    Pinecone {
        /// Pinecone API key
        api_key: String,
        /// Pinecone index host URL
        base_url: String,
        /// Vector dimension
        dimension: usize,
    },

    /// Chroma backend
    #[cfg(feature = "vector-store-chroma")]
    Chroma {
        /// Chroma server URL
        url: String,
        /// Collection name
        collection_name: String,
        /// Vector dimension
        dimension: usize,
    },

    /// Weaviate backend
    #[cfg(feature = "vector-store-weaviate")]
    Weaviate {
        /// Weaviate server URL
        url: String,
        /// Class name
        class_name: String,
        /// Vector dimension
        dimension: usize,
        /// API key (optional)
        api_key: Option<String>,
    },

    /// Milvus backend
    #[cfg(feature = "vector-store-milvus")]
    Milvus {
        /// Milvus server URL
        url: String,
        /// Collection name
        collection_name: String,
        /// Vector dimension
        dimension: usize,
    },

    /// S3-based vector storage
    #[cfg(feature = "vector-store-s3-backend")]
    S3 {
        /// S3 bucket name
        bucket: String,
        /// Key prefix for vectors
        prefix: String,
        /// AWS region
        region: Option<String>,
        /// Vector dimension
        dimension: usize,
    },

    /// `ColdForge` (tiered S3 storage)
    #[cfg(feature = "vector-store-coldforge")]
    ColdForge {
        /// S3 bucket name for cold storage
        bucket: String,
        /// Key prefix
        prefix: String,
        /// AWS region
        region: Option<String>,
        /// Vector dimension
        dimension: usize,
        /// Enable compression
        compression: bool,
    },

    /// DynamoDB native vector search (`SearchVectors`)
    #[cfg(feature = "vector-store-dynamodb")]
    DynamoDb {
        /// DynamoDB table name
        table_name: String,
        /// Vector index name on the table
        index_name: String,
        /// AWS region
        region: Option<String>,
        /// Optional custom endpoint (LocalStack)
        endpoint_url: Option<String>,
        /// Vector dimension (must match the index)
        dimension: usize,
        /// Create table + vector index when missing
        ensure_table: bool,
    },

    /// `Dripstone` memory-mapped backend for read-mostly serving or local snapshots.
    #[cfg(feature = "vector-store-mmap")]
    Mmap {
        /// Path to the mmap snapshot file.
        file_path: String,
        /// Vector dimension expected by the snapshot.
        dimension: usize,
        /// Whether the mmap backend should allow writes and snapshot flushing.
        read_write: bool,
        /// Auto-flush interval for read-write mode. `0` disables auto-flush.
        flush_every_ops: usize,
        /// Optional vector encoding for snapshots written by the mmap backend.
        vector_encoding: Option<dripstone::MmapVectorEncoding>,
    },

    /// Decorate any backend with a `Dripstone` ANN index.
    #[cfg(feature = "vector-memory")]
    Indexed {
        /// Inner backend to wrap with ANN acceleration.
        inner: Box<Self>,
        /// Vector dimension used by the ANN index.
        dimension: usize,
        /// Optional HNSW configuration for the index.
        hnsw_config: Option<crate::data::storage::vector::HnswConfig>,
        /// Optional indexed backend tuning knobs.
        config: Option<dripstone::IndexedBackendConfig>,
    },

    /// Decorate any backend with `Dripstone` lexical full-text indexing.
    #[cfg(feature = "vector-memory")]
    FullTextIndexed {
        /// Inner backend to wrap with full-text indexing.
        inner: Box<Self>,
        /// Optional full-text indexed backend tuning knobs.
        config: Option<dripstone::FullTextIndexedBackendConfig>,
    },

    /// N-tier chain (cold -> hot). Each tier is a backend config + `TierConfig`.
    /// At build time, [`VectorBackendConfig::build_backend`] builds each tier's backend and
    /// assembles them with [`dripstone::TieredChainBuilder`].
    #[cfg(feature = "vector-memory")]
    Tiered {
        /// Tiers in order: index 0 = coldest (source of truth), last = hottest.
        tiers: Vec<(Self, dripstone::TierConfig)>,
        /// How writes are distributed; None = default (`WriteToSourceAndHot`).
        write_strategy: Option<dripstone::WriteStrategy>,
        /// How `search()` fans out across tiers; None = default (`SearchAllTiers`).
        search_strategy: Option<dripstone::SearchStrategy>,
        /// Enable periodic sync from hotter tiers down to colder.
        background_sync: bool,
        /// Sync interval in seconds.
        sync_interval_secs: u64,
    },
}

impl VectorBackendConfig {
    /// Create a default in-memory configuration
    #[must_use]
    pub const fn memory() -> Self {
        Self::Memory {
            ttl_secs: 86400, // 24 hours
            max_entries: Some(100_000),
        }
    }

    /// Create a memory configuration with custom TTL and max entries
    #[must_use]
    pub const fn memory_with_config(ttl_secs: u64, max_entries: Option<usize>) -> Self {
        Self::Memory {
            ttl_secs,
            max_entries,
        }
    }

    #[cfg(feature = "vector-store-mmap")]
    /// Create a memory-mapped backend configuration.
    pub fn mmap(file_path: impl Into<String>, dimension: usize) -> Self {
        Self::Mmap {
            file_path: file_path.into(),
            dimension,
            read_write: false,
            flush_every_ops: 512,
            vector_encoding: None,
        }
    }

    /// Create a wrapper with metrics enabled
    #[must_use]
    pub const fn with_metrics(self, enabled: bool) -> VectorBackendConfigWrapper {
        VectorBackendConfigWrapper {
            backend: self,
            metrics_enabled: enabled,
        }
    }

    #[cfg(feature = "vector-store-redis")]
    /// Create a Redis backend configuration
    pub fn redis(url: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self::Redis {
            url: url.into(),
            prefix: prefix.into(),
        }
    }

    #[cfg(feature = "vector-store-redis")]
    /// Create a Valkey backend configuration
    pub fn valkey(url: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self::Valkey {
            url: url.into(),
            prefix: prefix.into(),
        }
    }

    #[cfg(feature = "vector-store-postgres")]
    /// Create a `PostgreSQL` backend configuration
    pub fn postgres(
        connection_string: impl Into<String>,
        table_name: impl Into<String>,
        vector_dimension: Option<usize>,
        max_connections: u32,
    ) -> Self {
        Self::Postgres {
            connection_string: connection_string.into(),
            table_name: table_name.into(),
            vector_dimension,
            max_connections,
            hnsw_config: None,
        }
    }

    #[cfg(feature = "vector-store-qdrant-grpc")]
    /// Create a Qdrant gRPC backend configuration
    pub fn qdrant_grpc(
        grpc_url: impl Into<String>,
        vector_dimension: usize,
        collection_prefix: Option<String>,
        api_key: Option<String>,
        batch_size: Option<usize>,
    ) -> Self {
        Self::QdrantGrpc {
            grpc_url: grpc_url.into(),
            collection_prefix,
            vector_dimension,
            api_key,
            batch_size,
        }
    }

    #[cfg(feature = "vector-store-qdrant")]
    /// Create a Qdrant HTTP backend configuration
    pub fn qdrant_http(
        url: impl Into<String>,
        vector_dimension: usize,
        collection_prefix: Option<String>,
        api_key: Option<String>,
        batch_size: Option<usize>,
        timeout: Option<std::time::Duration>,
    ) -> Self {
        Self::Qdrant {
            url: url.into(),
            collection_prefix,
            vector_dimension,
            api_key,
            batch_size,
            timeout,
        }
    }

    #[cfg(feature = "vector-store-pinecone")]
    /// Create a Pinecone backend configuration
    pub fn pinecone(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self::Pinecone {
            api_key: api_key.into(),
            base_url: base_url.into(),
            dimension,
        }
    }

    #[cfg(feature = "vector-store-chroma")]
    /// Create a Chroma backend configuration
    pub fn chroma(
        url: impl Into<String>,
        collection_name: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self::Chroma {
            url: url.into(),
            collection_name: collection_name.into(),
            dimension,
        }
    }

    #[cfg(feature = "vector-store-weaviate")]
    /// Create a Weaviate backend configuration
    pub fn weaviate(
        url: impl Into<String>,
        class_name: impl Into<String>,
        dimension: usize,
        api_key: Option<String>,
    ) -> Self {
        Self::Weaviate {
            url: url.into(),
            class_name: class_name.into(),
            dimension,
            api_key,
        }
    }

    #[cfg(feature = "vector-store-milvus")]
    /// Create a Milvus backend configuration
    pub fn milvus(
        url: impl Into<String>,
        collection_name: impl Into<String>,
        dimension: usize,
    ) -> Self {
        Self::Milvus {
            url: url.into(),
            collection_name: collection_name.into(),
            dimension,
        }
    }

    #[cfg(feature = "vector-store-s3-backend")]
    /// Create an S3 backend configuration
    pub fn s3(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: Option<String>,
        dimension: usize,
    ) -> Self {
        Self::S3 {
            bucket: bucket.into(),
            prefix: prefix.into(),
            region,
            dimension,
        }
    }

    #[cfg(feature = "vector-store-coldforge")]
    /// Create a `ColdForge` (tiered S3) backend configuration
    pub fn coldforge(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        region: Option<String>,
        dimension: usize,
        compression: bool,
    ) -> Self {
        Self::ColdForge {
            bucket: bucket.into(),
            prefix: prefix.into(),
            region,
            dimension,
            compression,
        }
    }

    #[cfg(feature = "vector-store-dynamodb")]
    /// Create a DynamoDB native vector backend configuration
    pub fn dynamodb(
        table_name: impl Into<String>,
        index_name: impl Into<String>,
        region: Option<String>,
        dimension: usize,
        ensure_table: bool,
    ) -> Self {
        Self::DynamoDb {
            table_name: table_name.into(),
            index_name: index_name.into(),
            region,
            endpoint_url: None,
            dimension,
            ensure_table,
        }
    }
}

impl VectorBackendConfigWrapper {
    /// Build a backend from this configuration wrapper
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn build_backend(&self) -> Result<Arc<dyn Backend + Send + Sync>> {
        self.backend.build_backend().await
    }
}

impl Default for VectorBackendConfig {
    fn default() -> Self {
        Self::memory()
    }
}
