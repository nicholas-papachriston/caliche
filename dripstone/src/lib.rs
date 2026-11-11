//! # `Dripstone`
//!
//! Build powerful vector operations with Rust's performance and safety.
//!
//! `Dripstone` is a high-performance library for embedding storage, similarity search,
//! and RAG (Retrieval-Augmented Generation) workflows.
//!
//! ## Entry points
//!
//! Prefer [`Dripstone`] when ingest and search should share one embedder.
//! Use [`Backend`] directly when vectors already exist.
//! [`crate::builder::VectorStoreBuilder`] is a secondary helper for backend-only setup.
//!
//! ## Quick Start
//!
//! ```text
//! use dripstone::Dripstone;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), dripstone::DripstoneError> {
//!     let store = Dripstone::open("./store")?;
//!     store.add("alpha", "penguins swim in cold water").await?;
//!     let hits = store.search("penguins").limit(1).await?;
//!     println!("{}", hits[0].chunk.doc_id);
//!     Ok(())
//! }
//! ```
//!
//! `open` is enabled by the default `local` feature. It uses an in-process
//! store and a hashing embedder. `new` and the builder still take an explicit
//! backend and embedder and keep dense search.
//!
//! ## Features
//!
//! - **Pluggable backends**: Memory, Redis, `PostgreSQL`, Qdrant
//! - **Local embeddings**: ONNX Runtime, Candle
//! - **RAG features**: Chunking, hybrid search, reranking
//! - **Performance**: SIMD optimizations, streaming, batching
//!
//! ## Feature Flags
//!
//! - `core` and `local` (default): core types, plus an in-process store and hashing embedder
//! - `backends`: Remote and service storage backends (not in the default set)
//! - `embeddings`: Local embedding generation
//! - `rag`: Hybrid search and reranking

// Lint policy is declared in this crate's Cargo.toml so the crate can build
// as a standalone library. Intentional mmap/SIMD unsafe stays crate-allowed;
// panic/expect/unreachable exceptions use `#[expect(..., reason = "...")]`
// at the call site.
#![allow(
    unsafe_code,
    reason = "mmap backends and SIMD paths require carefully scoped unsafe"
)]

#[cfg(any(feature = "s3-backend", feature = "dynamodb-backend"))]
pub mod aws;
pub mod config;
pub mod core;
pub mod dripstone;
pub mod fts;
#[cfg(feature = "local")]
pub mod local;
pub mod traits;

// Chunk hash store for incremental indexing
pub mod chunk_hash_store;

// Hashline module (content-addressed tagging for RAG and line-tagged edits)
pub mod hashline;

// Re-export main types at crate root
pub use chunk_hash_store::{ChunkHashStore, MemoryChunkHashStore};
pub use config::{
    ChunkStrategy, ChunkingConfig, DripstoneConfig, HashlineConfig, IndexConfig, IndexStrategy,
    SearchConfig,
};
pub use core::{
    Capabilities, Chunk, Document, FullTextOptions, FullTextQuery, Metric, PreChunk, SearchMode,
    SearchQuery, SearchResult,
};
pub use dripstone::{Dripstone, DripstoneBuilder, DynDripstone, IngestChunksBuilder, IngestResult};
pub use fts::{
    DocId, FullTextCodecVersion, FullTextHit, FullTextIndex, FullTextIndexConfig,
    InMemoryFullTextIndex, LexiconEntry, Posting, PostingBlock, TermId, TextDocumentMeta,
};
pub use hashline::{HashlineOptions, compute_content_hash, format_content_with_hashlines};
#[cfg(feature = "local")]
pub use local::{LocalEmbedder, OpenBackend};
pub use traits::{Backend, DynBackend, DynEmbedder, Embedder};

// Re-export new backend implementations
#[cfg(feature = "chroma-backend")]
pub use backend::chroma::ChromaBackend;
#[cfg(feature = "coldforge-backend")]
pub use backend::coldforge::ColdForgeBackend;
#[cfg(feature = "dynamodb-backend")]
pub use backend::dynamodb::{DynamoDbVectorBackend, DynamoDbVectorConfig};
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
    feature = "dynamodb-backend",
    feature = "coldforge-backend",
    feature = "mmap-backend",
    feature = "sqlite-backend"
))]
pub use backend::full_text_indexed::{FullTextIndexedBackend, FullTextIndexedBackendConfig};
#[cfg(all(
    feature = "tokio",
    any(
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
        feature = "dynamodb-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend"
    )
))]
pub use backend::hot_swappable_indexed::HotSwappableIndexedBackend;
#[cfg(all(
    feature = "tokio",
    any(
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
        feature = "dynamodb-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend"
    )
))]
pub use backend::indexed::{IndexUpdateMode, IndexedBackend, IndexedBackendConfig, ReindexError};
#[cfg(all(
    feature = "tokio",
    any(
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
        feature = "dynamodb-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend",
        feature = "sqlite-backend"
    )
))]
pub use backend::late_interaction::{
    LateInteractionIndexedBackend, LateInteractionIndexedBackendConfig,
};
#[cfg(feature = "memory-backend")]
pub use backend::memory::MemoryBackend;
#[cfg(feature = "milvus-backend")]
pub use backend::milvus::MilvusBackend;
#[cfg(feature = "mmap-backend")]
pub use backend::mmap::{
    MmapBackend, MmapBackendConfig, MmapVectorEncoding, write_dripstone_mmap,
    write_dripstone_mmap_v1, write_dripstone_mmap_with_encoding,
};
#[cfg(feature = "pinecone-backend")]
pub use backend::pinecone::PineconeBackend;
#[cfg(feature = "postgres-backend")]
pub use backend::postgres::PostgresBackend;
#[cfg(feature = "qdrant-grpc-backend")]
pub use backend::qdrant::QdrantBackend;
#[cfg(feature = "qdrant-backend")]
pub use backend::qdrant::QdrantHttpBackend;
#[cfg(feature = "redis-backend")]
pub use backend::redis::RedisBackend;
#[cfg(feature = "s3-backend")]
pub use backend::s3::S3Backend;
#[cfg(all(
    feature = "tokio",
    feature = "coldforge-backend",
    feature = "memory-backend"
))]
pub use backend::sharded::{
    ColdForgeShardOptions, build_coldforge_wal_sharded, build_memory_coldforge_wal_sharded,
};
#[cfg(all(
    feature = "tokio",
    any(
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
        feature = "dynamodb-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend",
        feature = "sqlite-backend"
    )
))]
pub use backend::sharded::{WalShardedBackend, WalShardedBackendBuilder};
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
    feature = "dynamodb-backend",
    feature = "coldforge-backend",
    feature = "mmap-backend"
))]
pub use backend::tiered::{
    SearchStrategy, TierConfig, TieredChainBackend, TieredChainBuilder, TieredChainConfig,
    WriteStrategy,
};
#[cfg(feature = "weaviate-backend")]
pub use backend::weaviate::WeaviateBackend;
#[cfg(all(
    feature = "tokio",
    any(
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
        feature = "dynamodb-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend"
    )
))]
#[doc(inline)]
/// Secondary backend-only helper. Prefer [`Dripstone`] for ingest+search.
pub use builder::IndexedBuilder;
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
    feature = "dynamodb-backend",
    feature = "coldforge-backend",
    feature = "mmap-backend"
))]
#[doc(inline)]
/// Secondary backend-only helper. Prefer [`Dripstone`] for ingest+search.
pub use builder::VectorStoreBuilder;

// =============================================================================
// EXISTING MODULES
// =============================================================================

// Backend module requires at least one backend feature
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
    feature = "dynamodb-backend",
    feature = "coldforge-backend",
    feature = "mmap-backend",
    feature = "sqlite-backend"
))]
pub mod backend;
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
    feature = "dynamodb-backend",
    feature = "coldforge-backend",
    feature = "mmap-backend"
))]
pub mod builder;
#[cfg(feature = "tokio")]
pub mod distributed;
// Shared embedding types and search option builders.
#[cfg(feature = "corpus")]
pub mod corpus;
pub mod embedding;
pub mod errors;
#[cfg(any(
    feature = "embedding-candle",
    feature = "embedding-ort",
    feature = "embedding-mlx",
    feature = "local-candle",
    feature = "local-ort",
    feature = "local-mlx"
))]
pub mod generation;
#[cfg(feature = "tokio")]
pub mod graph;
#[cfg(feature = "tokio")]
pub mod index;
#[cfg(any(
    feature = "inference-candle",
    feature = "inference-ort",
    feature = "inference-mlx",
    feature = "inference-llamacpp"
))]
pub mod inference;
#[cfg(any(
    feature = "embedding-candle",
    feature = "embedding-ort",
    feature = "embedding-mlx",
    feature = "inference-candle",
    feature = "inference-ort",
    feature = "inference-mlx",
    feature = "model-download",
    feature = "local-candle",
    feature = "local-ort",
    feature = "local-mlx"
))]
pub mod loader;

pub mod metrics;
#[cfg(feature = "tokio")]
pub mod multi_tenant;
#[cfg(any(
    feature = "embedding-candle",
    feature = "embedding-ort",
    feature = "embedding-mlx",
    feature = "video-processing"
))]
pub mod multimodal;
pub mod pipeline;
pub mod prelude;
pub mod quantization;
pub mod rag;
pub mod runtime;
pub mod search;
pub mod simd;
pub mod sparse;
pub mod storage;
pub mod utils;

// Re-exports
#[cfg(feature = "tokio")]
pub use distributed::{
    ConsistentHashRing, NamespaceWalEntry, NamespaceWalOp, ReplicationManager, ShardManager,
    SharedNamespaceWal, SharedWalConfig,
};
pub use errors::DripstoneError;
#[cfg(feature = "tokio")]
pub use graph::{
    Edge, GraphBackend, GraphError, GraphSchema, GraphValue, Neighbor, Node, Path,
    QueryCapabilities, QueryParams, QueryResult, QueryableGraphBackend, Record,
};
#[cfg(feature = "tokio")]
pub use index::spfresh::{CentroidIndex, SPFreshConfig};
#[cfg(feature = "hnsw")]
pub use index::{HnswConfig, HnswIndex};
#[cfg(feature = "tokio")]
pub use index::{PlaidConfig, PlaidIndex, VectorIndex};
pub use metrics::{NoOpMetrics, SharedMetrics, SimpleMetrics, VectorMetrics};
#[cfg(feature = "tokio")]
pub use multi_tenant::{MultiTenantManager, TenantQuota, tenant_namespace};
pub use pipeline::{DiversityConfig, SearchPipeline, SearchPipelineBuilder};
#[cfg(all(
    feature = "tokio",
    any(
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
        feature = "dynamodb-backend",
        feature = "coldforge-backend",
        feature = "mmap-backend"
    )
))]
pub use quantization::ProductQuantizer;
#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "model-download"
))]
pub use runtime::{
    CapabilityProbe, DefaultCapabilityProbe, FitOptions, FitStatus, HostCapabilities, ModelFit,
    ModelResourceHint,
};
pub use runtime::{ResolvedRuntime, RuntimeConfig};
pub use search::{
    DEFAULT_MAX_TOKEN_VECTORS, Filter, FilterBuilder, MaxSimOptions, MultiVectors,
    exact_maxsim_search, maxsim_score, multi_vectors_from_owned, overfetch_limit,
    pooled_vector_from_multi, rerank_with_maxsim, tokens_for_late_interaction,
};
pub use sparse::SparseVector;
pub use storage::MemoryObjectStore;
pub use traits::search_with_maxsim_fallback;

#[cfg(feature = "local-candle")]
pub use generation::CandleEmbedder;
#[cfg(feature = "local-mlx")]
pub use generation::MlxEmbedder;
#[cfg(feature = "local-ort")]
pub use generation::OrtEmbedder;
#[cfg(any(feature = "local-candle", feature = "local-ort", feature = "local-mlx"))]
pub use generation::{EmbeddingConfig, PoolingStrategy, VectorGenerator};

#[cfg(feature = "vision-candle")]
pub use multimodal::CandleVisionEmbedder;
#[cfg(feature = "vision-ort")]
pub use multimodal::OrtVisionEmbedder;
#[cfg(any(
    feature = "embedding-candle",
    feature = "embedding-ort",
    feature = "embedding-mlx",
    feature = "video-processing"
))]
pub use multimodal::{ImageInput, ImagePreprocessor, MediaGenerator, VideoInput, VisionConfig};
