//! Vector storage backends implementing [`crate::Backend`].
//!
//! Layout convention:
//! - Small backends may live in a single `mod.rs`.
//! - Larger backends (`qdrant`, `weaviate`, `dynamodb`, `mmap`, …) use a
//!   directory with focused files (`backend.rs`, `client.rs`, `convert.rs`, …).
//! - New backends over ~400 lines should follow the multi-file layout.
//!
//! Related modules outside this tree:
//! - [`crate::storage`]: object stores (not ANN backends)
//! - [`crate::index`]: in-process ANN structures used by decorators
//!
pub mod chunk_envelope;

#[cfg(feature = "redis-backend")]
pub mod redis;

#[cfg(feature = "postgres-backend")]
pub mod postgres;

#[cfg(any(feature = "qdrant-backend", feature = "qdrant-grpc-backend"))]
pub mod qdrant;

#[cfg(feature = "chroma-backend")]
pub mod chroma;

#[cfg(feature = "pinecone-backend")]
pub mod pinecone;

#[cfg(feature = "weaviate-backend")]
pub mod weaviate;

#[cfg(feature = "milvus-backend")]
pub mod milvus;

#[cfg(feature = "memory-backend")]
pub mod memory;

#[cfg(feature = "mmap-backend")]
pub mod mmap;

#[cfg(feature = "sqlite-backend")]
pub mod sqlite;

#[cfg(feature = "s3-backend")]
pub mod s3;

#[cfg(feature = "dynamodb-backend")]
pub mod dynamodb;

#[cfg(feature = "coldforge-backend")]
pub mod coldforge;

/// ANN-accelerated backend decorator (wraps any Backend with a VectorIndex)
#[cfg(feature = "tokio")]
pub mod indexed;

/// Hot-swappable ANN backend for read-available index cutovers.
#[cfg(feature = "tokio")]
pub mod hot_swappable_indexed;

/// Full-text-accelerated backend decorator (wraps any Backend with a BM25/MAXSCORE index)
pub mod full_text_indexed;

/// Late-interaction (PLAID/MaxSim) backend decorator
#[cfg(feature = "tokio")]
pub mod late_interaction;

/// Shared utilities for tiered storage backends
pub mod shared;

/// N-tier chain backend (ordered cold -> hot)
pub mod tiered;

/// Single-WAL multi-shard namespace backend (scale-out path)
#[cfg(feature = "tokio")]
pub mod sharded;

#[cfg(any(
    feature = "qdrant-backend",
    feature = "chroma-backend",
    feature = "pinecone-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend"
))]
pub mod http;

#[cfg(any(
    feature = "qdrant-backend",
    feature = "chroma-backend",
    feature = "pinecone-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend"
))]
pub mod retry;

/// Compute optimal batch size for vector operations based on dimension.
///
/// Returns a batch size that targets approximately 10-20MB per batch,
/// accounting for vector data (dimension * 4 bytes per f32) plus overhead.
///
/// # Arguments
/// * `dimension` - The dimension of the vectors
///
/// # Returns
/// Optimal batch size (minimum 1, maximum 10,000)
#[must_use]
pub fn compute_batch_size(dimension: usize) -> usize {
    // Target ~10-20MB per batch
    // Each vector: dimension * 4 bytes (f32) + ~500 bytes overhead
    let bytes_per_vector = (dimension * 4) + 500;
    let target_batch_memory_bytes = 15_000_000; // ~15MB

    let batch_size = target_batch_memory_bytes / bytes_per_vector;

    // Clamp between reasonable bounds
    batch_size.clamp(1, 10_000)
}
