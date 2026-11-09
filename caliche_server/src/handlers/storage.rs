use axum::response::Json;
use caliche::capabilities::vector_backends;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct StorageOption {
    pub name: String,
    pub display_name: String,
    pub storage_type: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct StorageOptionsResponse {
    pub vector_storage: Vec<StorageOption>,
    // Future: other storage types will be added here
    // pub memory_storage: Vec<StorageOption>,
    // pub document_storage: Vec<StorageOption>,
}

/// GET /api/storage/options - List available storage options
/// Focuses on vector storage first, but designed to be extensible for other storage types
pub async fn storage_options_handler() -> Json<StorageOptionsResponse> {
    let vector_storage: Vec<StorageOption> = vector_backends::available_vector_backends()
        .iter()
        .map(|backend| {
            let (display_name, description) = match *backend {
                "memory" => (
                    "Memory",
                    Some(
                        "In-memory vector storage with TTL support. Fast but ephemeral."
                            .to_string(),
                    ),
                ),
                "redis" => (
                    "Redis",
                    Some("Redis-based vector storage. Persistent and scalable.".to_string()),
                ),
                "valkey" => (
                    "Valkey",
                    Some(
                        "Valkey-based vector storage (Redis-compatible). Persistent and scalable."
                            .to_string(),
                    ),
                ),
                "postgres" | "postgresql" => (
                    "PostgreSQL (pgvector)",
                    Some(
                        "PostgreSQL with pgvector extension. SQL-native vector storage."
                            .to_string(),
                    ),
                ),
                "qdrant" => (
                    "Qdrant (HTTP)",
                    Some(
                        "Qdrant vector database via HTTP API. High-performance vector search."
                            .to_string(),
                    ),
                ),
                "qdrant-grpc" | "qdrant_grpc" => (
                    "Qdrant (gRPC)",
                    Some(
                        "Qdrant vector database via gRPC. Optimized for high-throughput."
                            .to_string(),
                    ),
                ),
                "pinecone" => (
                    "Pinecone",
                    Some(
                        "Managed Pinecone vector database. Cloud-native vector storage."
                            .to_string(),
                    ),
                ),
                "chroma" => (
                    "Chroma",
                    Some("Chroma vector database. Open-source embedding database.".to_string()),
                ),
                "milvus" => (
                    "Milvus",
                    Some("Milvus vector database. Open-source vector database.".to_string()),
                ),
                "weaviate" => (
                    "Weaviate",
                    Some("Weaviate vector database. Open-source vector database.".to_string()),
                ),
                "s3" => (
                    "S3",
                    Some(
                        "S3-backed vector storage. Cost-effective for large-scale storage."
                            .to_string(),
                    ),
                ),
                "coldforge" => (
                    "ColdForge",
                    Some(
                        "S3-backed with SPFresh indexing. Intelligent caching for cold storage."
                            .to_string(),
                    ),
                ),
                _ => (*backend, None),
            };

            StorageOption {
                name: backend.to_string(),
                display_name: display_name.to_string(),
                storage_type: "vector".to_string(),
                description,
            }
        })
        .collect();

    Json(StorageOptionsResponse { vector_storage })
}
