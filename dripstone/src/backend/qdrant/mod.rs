mod helpers;

#[cfg(feature = "qdrant-grpc-backend")]
mod grpc_client;
#[cfg(feature = "qdrant-grpc-backend")]
pub use grpc_client::{QdrantBackend, QdrantGrpcBackend, QdrantGrpcBackendOptions};

#[cfg(feature = "qdrant-backend")]
mod backend;
#[cfg(feature = "qdrant-backend")]
mod client;
#[cfg(feature = "qdrant-backend")]
mod convert;

#[cfg(feature = "qdrant-backend")]
pub use backend::QdrantHttpBackend;
#[cfg(feature = "qdrant-backend")]
pub use client::{QdrantBackendOptions, QdrantEmbeddingBackend};
