//! Environment-variable based construction of [`VectorBackendConfig`] plus
//! the small `EnvAwsCredentials` helper used by the S3-family backends.

use super::VectorBackendConfig;
use crate::capabilities::vector_backends;
use anyhow::{Result, anyhow};

/// Optional AWS credentials resolved from the environment. We intentionally
/// keep this minimal (access key, secret, session) so that everything falls
/// back to the default AWS credential chain (IAM role, SSO, profile, etc.)
/// when the variables are not set, mirroring how the SDK behaves.
#[cfg(any(
    feature = "vector-store-s3-backend",
    feature = "vector-store-coldforge",
    feature = "vector-store-dynamodb",
    feature = "s3"
))]
#[derive(Debug, Default, Clone)]
pub struct EnvAwsCredentials {
    pub(crate) access_key_id: Option<String>,
    pub(crate) secret_access_key: Option<String>,
    pub(crate) session_token: Option<String>,
}

#[cfg(any(
    feature = "vector-store-s3-backend",
    feature = "vector-store-coldforge",
    feature = "vector-store-dynamodb",
    feature = "s3"
))]
impl EnvAwsCredentials {
    pub(crate) fn from_env() -> Self {
        let trimmed = |var: &str| {
            caliche_contracts::credential_env::var(var)
                .ok()
                .and_then(|v| {
                    let v = v.trim().to_string();
                    if v.is_empty() { None } else { Some(v) }
                })
        };
        Self {
            access_key_id: trimmed("AWS_ACCESS_KEY_ID"),
            secret_access_key: trimmed("AWS_SECRET_ACCESS_KEY"),
            session_token: trimmed("AWS_SESSION_TOKEN"),
        }
    }
}

impl VectorBackendConfig {
    /// Read vector store configuration from environment variables.
    ///
    /// Uses `VECTOR_BACKEND` to determine which backend to use.
    /// Falls back to in-memory if not specified.
    ///
    /// # Environment Variables
    ///
    /// ## General
    /// - `VECTOR_BACKEND`: Backend type (memory, redis, valkey, postgres, qdrant, qdrant-grpc, qdrant-http, pinecone, chroma, weaviate, milvus, s3, coldforge)
    /// - `VECTOR_BACKEND_DIMENSION`: Vector dimension (default: 1536)
    /// - `CALICHE_DEFERRED_ANN_INDEX`: When `true`, use deferred HNSW indexing on
    ///   workspace disk stores and default `Indexed` backend configs (opt-in)
    /// - `CALICHE_DEFERRED_SEARCH_FLUSH_THRESHOLD`: Flush pending ANN queue at search
    ///   when pending count reaches this value (requires deferred ANN index)
    /// - `CALICHE_DEFERRED_MAX_TAIL_SCAN`: Cap deferred tail scan to the last N pending
    ///   records (requires deferred ANN index)
    ///
    /// ## Redis
    /// - `REDIS_URL`: Redis connection URL (default: "<redis://127.0.0.1>/")
    /// - `VECTOR_BACKEND_REDIS_PREFIX`: Key prefix (default: "embeddings")
    ///
    /// ## Valkey
    /// - `VALKEY_URL`: Valkey connection URL (default: "<redis://127.0.0.1>/")
    /// - `VECTOR_BACKEND_VALKEY_PREFIX`: Key prefix (default: "embeddings")
    ///
    /// ## `PostgreSQL`
    /// - `POSTGRES_URL` or `VECTOR_BACKEND_POSTGRES_URL`: `PostgreSQL` connection string
    /// - `VECTOR_BACKEND_POSTGRES_TABLE`: Table name (default: "embeddings")
    /// - `VECTOR_BACKEND_POSTGRES_DIMENSION`: Vector dimension (optional)
    /// - `VECTOR_BACKEND_POSTGRES_MAX_CONNECTIONS`: Max connections (default: 10)
    ///
    /// ## Qdrant gRPC
    /// - `QDRANT_GRPC_URL`: Qdrant gRPC URL (default: "<http://localhost:6334>")
    /// - `QDRANT_API_KEY`: API key (optional)
    /// - `VECTOR_BACKEND_QDRANT_COLLECTION_PREFIX`: Collection prefix (optional)
    /// - `VECTOR_BACKEND_BATCH_SIZE`: Batch size (optional)
    ///
    /// ## Qdrant HTTP
    /// - `QDRANT_URL`: Qdrant HTTP URL (default: "<http://localhost:6333>")
    /// - `QDRANT_API_KEY`: API key (optional)
    /// - `VECTOR_BACKEND_QDRANT_COLLECTION_PREFIX`: Collection prefix (optional)
    /// - `VECTOR_BACKEND_BATCH_SIZE`: Batch size (optional)
    /// - `VECTOR_BACKEND_TIMEOUT_SECS`: Request timeout in seconds (optional)
    ///
    /// ## Pinecone
    /// - `PINECONE_API_KEY`: Pinecone API key (required)
    /// - `PINECONE_INDEX_HOST`: Pinecone index host URL (required)
    ///
    /// ## Chroma
    /// - `CHROMA_URL`: Chroma server URL (default: "<http://localhost:8000>")
    /// - `CHROMA_COLLECTION`: Collection name (default: "embeddings")
    ///
    /// ## Weaviate
    /// - `WEAVIATE_URL`: Weaviate server URL (default: "<http://localhost:8080>")
    /// - `WEAVIATE_CLASS`: Class name (default: "Embedding")
    /// - `WEAVIATE_API_KEY`: API key (optional)
    ///
    /// ## Milvus
    /// - `MILVUS_URL`: Milvus server URL (default: "<http://localhost:19530>")
    /// - `MILVUS_COLLECTION`: Collection name (default: "embeddings")
    ///
    /// ## S3
    /// - `S3_BUCKET`: S3 bucket name (required)
    /// - `S3_PREFIX`: Key prefix (default: "vectors/")
    /// - `AWS_REGION`: AWS region (optional)
    ///
    /// ## `ColdForge`
    /// - `COLDFORGE_BUCKET`: S3 bucket name for cold storage (required)
    /// - `COLDFORGE_PREFIX`: Key prefix (default: "cold/")
    /// - `AWS_REGION`: AWS region (optional)
    /// - `COLDFORGE_COMPRESSION`: Enable compression (default: true)
    ///
    /// ## DynamoDB
    /// - `DRIPSTONE_DYNAMODB_TABLE` or `DYNAMODB_TABLE`: table name (required)
    /// - `DRIPSTONE_DYNAMODB_INDEX` or `DYNAMODB_INDEX`: vector index (default `embeddings`)
    /// - `AWS_REGION` / `DRIPSTONE_DYNAMODB_REGION`: region (optional)
    /// - `DRIPSTONE_DYNAMODB_ENDPOINT`: optional custom endpoint
    /// - `DRIPSTONE_DYNAMODB_ENSURE`: create table/index when missing (`1`/`true`)
    /// - `VECTOR_BACKEND_DIMENSION`: vector dimension (default 1536)
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub fn from_env() -> Result<Self> {
        let backend = caliche_contracts::credential_env::var("VECTOR_BACKEND")
            .unwrap_or_else(|_| "memory".to_string())
            .to_lowercase();

        match backend.as_str() {
            "memory" | "redis" | "valkey" => Self::from_env_local(&backend),
            "postgres" | "postgresql" | "qdrant" | "qdrant-grpc" | "qdrant_grpc"
            | "qdrant-http" | "qdrant_http" => Self::from_env_database(&backend),
            "pinecone" | "chroma" => Self::from_env_service(&backend),
            "weaviate" | "milvus" => Self::from_env_service_extended(&backend),
            "s3" | "coldforge" | "dynamodb" | "dynamodb-vector" | "dynamodb_vector" => {
                Self::from_env_storage(&backend)
            }
            other => {
                if !vector_backends::is_vector_backend_available(other) {
                    return Err(vector_backends::vector_backend_not_available_error(other));
                }
                Err(anyhow!(
                    "Unknown vector store backend '{}'. Available: {}",
                    other,
                    vector_backends::available_vector_backends().join(", ")
                ))
            }
        }
    }

    fn from_env_local(backend: &str) -> Result<Self> {
        match backend {
            "memory" => Ok(Self::Memory {
                ttl_secs: caliche_contracts::credential_env::var("VECTOR_BACKEND_MEMORY_TTL")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(86400),
                max_entries: caliche_contracts::credential_env::var(
                    "VECTOR_BACKEND_MEMORY_MAX_ENTRIES",
                )
                .ok()
                .and_then(|s| s.parse().ok()),
            }),
            "redis" => {
                if !vector_backends::is_vector_backend_available("redis") {
                    return Err(vector_backends::vector_backend_not_available_error("redis"));
                }
                #[cfg(feature = "vector-store-redis")]
                {
                    Ok(Self::Redis {
                        url: caliche_contracts::credential_env::var("REDIS_URL")
                            .unwrap_or_else(|_| "redis://127.0.0.1/".to_string()),
                        prefix: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_REDIS_PREFIX",
                        )
                        .unwrap_or_else(|_| "embeddings".to_string()),
                    })
                }
                #[cfg(not(feature = "vector-store-redis"))]
                {
                    Err(vector_backends::vector_backend_not_available_error("redis"))
                }
            }
            "valkey" => {
                if !vector_backends::is_vector_backend_available("valkey") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "valkey",
                    ));
                }
                #[cfg(feature = "vector-store-redis")]
                {
                    Ok(Self::Valkey {
                        url: caliche_contracts::credential_env::var("VALKEY_URL")
                            .unwrap_or_else(|_| "redis://127.0.0.1/".to_string()),
                        prefix: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_VALKEY_PREFIX",
                        )
                        .unwrap_or_else(|_| "embeddings".to_string()),
                    })
                }
                #[cfg(not(feature = "vector-store-redis"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "valkey",
                    ))
                }
            }
            other => Err(anyhow!("Unsupported vector backend group entry: {other}")),
        }
    }

    fn from_env_database(backend: &str) -> Result<Self> {
        match backend {
            "postgres" | "postgresql" => {
                if !vector_backends::is_vector_backend_available("postgres") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "postgres",
                    ));
                }
                #[cfg(feature = "vector-store-postgres")]
                {
                    let connection_string = caliche_contracts::credential_env::var("POSTGRES_URL")
                        .or_else(|_| caliche_contracts::credential_env::var("VECTOR_BACKEND_POSTGRES_URL"))
                        .map_err(|_| anyhow!("POSTGRES_URL or VECTOR_BACKEND_POSTGRES_URL required for Postgres backend"))?;

                    Ok(Self::Postgres {
                        connection_string,
                        table_name: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_POSTGRES_TABLE",
                        )
                        .unwrap_or_else(|_| "embeddings".to_string()),
                        vector_dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_POSTGRES_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok()),
                        max_connections: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_POSTGRES_MAX_CONNECTIONS",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(10),
                        hnsw_config: None,
                    })
                }
                #[cfg(not(feature = "vector-store-postgres"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "postgres",
                    ))
                }
            }
            "qdrant" | "qdrant-grpc" | "qdrant_grpc" => {
                if !vector_backends::is_vector_backend_available("qdrant-grpc") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "qdrant-grpc",
                    ));
                }
                #[cfg(feature = "vector-store-qdrant-grpc")]
                {
                    Ok(Self::QdrantGrpc {
                        grpc_url: caliche_contracts::credential_env::var("QDRANT_GRPC_URL")
                            .unwrap_or_else(|_| "http://localhost:6334".to_string()),
                        collection_prefix: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_QDRANT_COLLECTION_PREFIX",
                        )
                        .ok(),
                        vector_dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                        api_key: caliche_contracts::credential_env::var("QDRANT_API_KEY").ok(),
                        batch_size: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_BATCH_SIZE",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok()),
                    })
                }
                #[cfg(not(feature = "vector-store-qdrant-grpc"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "qdrant-grpc",
                    ))
                }
            }
            "qdrant-http" | "qdrant_http" => {
                if !vector_backends::is_vector_backend_available("qdrant") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "qdrant",
                    ));
                }
                #[cfg(feature = "vector-store-qdrant")]
                {
                    Ok(Self::Qdrant {
                        url: caliche_contracts::credential_env::var("QDRANT_URL")
                            .unwrap_or_else(|_| "http://localhost:6333".to_string()),
                        collection_prefix: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_QDRANT_COLLECTION_PREFIX",
                        )
                        .ok(),
                        vector_dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                        api_key: caliche_contracts::credential_env::var("QDRANT_API_KEY").ok(),
                        batch_size: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_BATCH_SIZE",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok()),
                        timeout: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_TIMEOUT_SECS",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .map(std::time::Duration::from_secs),
                    })
                }
                #[cfg(not(feature = "vector-store-qdrant"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "qdrant",
                    ))
                }
            }
            other => Err(anyhow!("Unsupported vector backend group entry: {other}")),
        }
    }

    fn from_env_service(backend: &str) -> Result<Self> {
        match backend {
            "pinecone" => {
                if !vector_backends::is_vector_backend_available("pinecone") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "pinecone",
                    ));
                }
                #[cfg(feature = "vector-store-pinecone")]
                {
                    let api_key = caliche_contracts::credential_env::var("PINECONE_API_KEY")
                        .map_err(|_| anyhow!("PINECONE_API_KEY required for Pinecone backend"))?;
                    let base_url = caliche_contracts::credential_env::var("PINECONE_INDEX_HOST")
                        .map_err(|_| {
                            anyhow!("PINECONE_INDEX_HOST required for Pinecone backend")
                        })?;
                    Ok(Self::Pinecone {
                        api_key,
                        base_url,
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                    })
                }
                #[cfg(not(feature = "vector-store-pinecone"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "pinecone",
                    ))
                }
            }
            "chroma" => {
                if !vector_backends::is_vector_backend_available("chroma") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "chroma",
                    ));
                }
                #[cfg(feature = "vector-store-chroma")]
                {
                    Ok(Self::Chroma {
                        url: caliche_contracts::credential_env::var("CHROMA_URL")
                            .unwrap_or_else(|_| "http://localhost:8000".to_string()),
                        collection_name: caliche_contracts::credential_env::var(
                            "CHROMA_COLLECTION",
                        )
                        .unwrap_or_else(|_| "embeddings".to_string()),
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                    })
                }
                #[cfg(not(feature = "vector-store-chroma"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "chroma",
                    ))
                }
            }
            other => Err(anyhow!("Unsupported vector backend group entry: {other}")),
        }
    }

    fn from_env_service_extended(backend: &str) -> Result<Self> {
        match backend {
            "weaviate" => {
                if !vector_backends::is_vector_backend_available("weaviate") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "weaviate",
                    ));
                }
                #[cfg(feature = "vector-store-weaviate")]
                {
                    Ok(Self::Weaviate {
                        url: caliche_contracts::credential_env::var("WEAVIATE_URL")
                            .unwrap_or_else(|_| "http://localhost:8080".to_string()),
                        class_name: caliche_contracts::credential_env::var("WEAVIATE_CLASS")
                            .unwrap_or_else(|_| "Embedding".to_string()),
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                        api_key: caliche_contracts::credential_env::var("WEAVIATE_API_KEY").ok(),
                    })
                }
                #[cfg(not(feature = "vector-store-weaviate"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "weaviate",
                    ))
                }
            }
            "milvus" => {
                if !vector_backends::is_vector_backend_available("milvus") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "milvus",
                    ));
                }
                #[cfg(feature = "vector-store-milvus")]
                {
                    Ok(Self::Milvus {
                        url: caliche_contracts::credential_env::var("MILVUS_URL")
                            .unwrap_or_else(|_| "http://localhost:19530".to_string()),
                        collection_name: caliche_contracts::credential_env::var(
                            "MILVUS_COLLECTION",
                        )
                        .unwrap_or_else(|_| "embeddings".to_string()),
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                    })
                }
                #[cfg(not(feature = "vector-store-milvus"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "milvus",
                    ))
                }
            }
            other => Err(anyhow!("Unsupported vector backend group entry: {other}")),
        }
    }

    fn from_env_storage(backend: &str) -> Result<Self> {
        match backend {
            "s3" => {
                if !vector_backends::is_vector_backend_available("s3") {
                    return Err(vector_backends::vector_backend_not_available_error("s3"));
                }
                #[cfg(feature = "vector-store-s3-backend")]
                {
                    let bucket = caliche_contracts::credential_env::var("S3_BUCKET")
                        .map_err(|_| anyhow!("S3_BUCKET required for S3 backend"))?;
                    Ok(Self::S3 {
                        bucket,
                        prefix: caliche_contracts::credential_env::var("S3_PREFIX")
                            .unwrap_or_else(|_| "vectors/".to_string()),
                        region: caliche_contracts::credential_env::var("AWS_REGION").ok(),
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                    })
                }
                #[cfg(not(feature = "vector-store-s3-backend"))]
                {
                    Err(vector_backends::vector_backend_not_available_error("s3"))
                }
            }
            "coldforge" => {
                if !vector_backends::is_vector_backend_available("coldforge") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "coldforge",
                    ));
                }
                #[cfg(feature = "vector-store-coldforge")]
                {
                    let bucket = caliche_contracts::credential_env::var("COLDFORGE_BUCKET")
                        .map_err(|_| anyhow!("COLDFORGE_BUCKET required for ColdForge backend"))?;
                    Ok(Self::ColdForge {
                        bucket,
                        prefix: caliche_contracts::credential_env::var("COLDFORGE_PREFIX")
                            .unwrap_or_else(|_| "cold/".to_string()),
                        region: caliche_contracts::credential_env::var("AWS_REGION").ok(),
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                        compression: caliche_contracts::credential_env::var(
                            "COLDFORGE_COMPRESSION",
                        )
                        .map(|v| v == "true" || v == "1")
                        .unwrap_or(true),
                    })
                }
                #[cfg(not(feature = "vector-store-coldforge"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "coldforge",
                    ))
                }
            }
            "dynamodb" | "dynamodb-vector" | "dynamodb_vector" => {
                if !vector_backends::is_vector_backend_available("dynamodb") {
                    return Err(vector_backends::vector_backend_not_available_error(
                        "dynamodb",
                    ));
                }
                #[cfg(feature = "vector-store-dynamodb")]
                {
                    let table_name = caliche_contracts::credential_env::var("DRIPSTONE_DYNAMODB_TABLE")
                        .or_else(|_| caliche_contracts::credential_env::var("DYNAMODB_TABLE"))
                        .map_err(|_| {
                            anyhow!(
                                "DRIPSTONE_DYNAMODB_TABLE (or DYNAMODB_TABLE) required for DynamoDB backend"
                            )
                        })?;
                    let index_name =
                        caliche_contracts::credential_env::var("DRIPSTONE_DYNAMODB_INDEX")
                            .or_else(|_| caliche_contracts::credential_env::var("DYNAMODB_INDEX"))
                            .unwrap_or_else(|_| "embeddings".to_string());
                    let region =
                        caliche_contracts::credential_env::var("DRIPSTONE_DYNAMODB_REGION")
                            .or_else(|_| caliche_contracts::credential_env::var("AWS_REGION"))
                            .ok();
                    let endpoint_url =
                        caliche_contracts::credential_env::var("DRIPSTONE_DYNAMODB_ENDPOINT").ok();
                    let ensure_table =
                        caliche_contracts::credential_env::var("DRIPSTONE_DYNAMODB_ENSURE")
                            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                            .unwrap_or(false);
                    Ok(Self::DynamoDb {
                        table_name,
                        index_name,
                        region,
                        endpoint_url,
                        dimension: caliche_contracts::credential_env::var(
                            "VECTOR_BACKEND_DIMENSION",
                        )
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1536),
                        ensure_table,
                    })
                }
                #[cfg(not(feature = "vector-store-dynamodb"))]
                {
                    Err(vector_backends::vector_backend_not_available_error(
                        "dynamodb",
                    ))
                }
            }
            other => Err(anyhow!("Unsupported vector backend group entry: {other}")),
        }
    }
}
