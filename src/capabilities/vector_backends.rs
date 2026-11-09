//! Vector backend capability detection and availability functions.

/// Check if a vector backend is available at compile time.
///
/// Returns `true` if the backend feature is enabled, `false` otherwise.
pub fn is_vector_backend_available(backend: &str) -> bool {
    match backend {
        "memory" => true, // Always available
        "redis" => cfg!(feature = "vector-store-redis"),
        "valkey" => cfg!(feature = "vector-store-redis"), // Uses same feature as Redis
        "postgres" | "postgresql" => cfg!(feature = "vector-store-postgres"),
        "qdrant-grpc" | "qdrant_grpc" => cfg!(feature = "vector-store-qdrant-grpc"),
        "qdrant" | "qdrant-http" | "qdrant_http" => cfg!(feature = "vector-store-qdrant"),
        "pinecone" => cfg!(feature = "vector-store-pinecone"),
        "chroma" => cfg!(feature = "vector-store-chroma"),
        "weaviate" => cfg!(feature = "vector-store-weaviate"),
        "milvus" => cfg!(feature = "vector-store-milvus"),
        "s3" => cfg!(feature = "vector-store-s3-backend"),
        "coldforge" => cfg!(feature = "vector-store-coldforge"),
        "dynamodb" | "dynamodb-vector" | "dynamodb_vector" => {
            cfg!(feature = "vector-store-dynamodb")
        }
        _ => false,
    }
}

/// Get a list of all available vector backend names as strings.
///
/// Returns a static slice of backend names that are compiled into the binary.
pub fn available_vector_backends() -> &'static [&'static str] {
    &[
        "memory",
        #[cfg(feature = "vector-store-redis")]
        "redis",
        #[cfg(feature = "vector-store-redis")]
        "valkey",
        #[cfg(feature = "vector-store-postgres")]
        "postgres",
        #[cfg(feature = "vector-store-qdrant-grpc")]
        "qdrant-grpc",
        #[cfg(feature = "vector-store-qdrant")]
        "qdrant",
        #[cfg(feature = "vector-store-pinecone")]
        "pinecone",
        #[cfg(feature = "vector-store-chroma")]
        "chroma",
        #[cfg(feature = "vector-store-weaviate")]
        "weaviate",
        #[cfg(feature = "vector-store-milvus")]
        "milvus",
        #[cfg(feature = "vector-store-s3-backend")]
        "s3",
        #[cfg(feature = "vector-store-coldforge")]
        "coldforge",
        #[cfg(feature = "vector-store-dynamodb")]
        "dynamodb",
    ]
}

/// Generate a helpful error message for an unavailable vector backend.
pub fn vector_backend_not_available_error(backend: &str) -> anyhow::Error {
    let available = available_vector_backends();
    anyhow::anyhow!(
        "Vector backend '{}' requires a feature flag to be enabled. Available backends: {}. \
         To enable this backend, add the corresponding feature flag to your Cargo.toml.",
        backend,
        available.join(", ")
    )
}

/// `VECTOR_BACKEND` is unset. Retrieval is not configured.
pub const VECTOR_RETRIEVAL_UNCONFIGURED: &str = "unconfigured";

/// `VECTOR_BACKEND` names a backend this binary can construct.
///
/// The feature for that name is compiled in. This is not a check that a
/// remote store is reachable or healthy.
pub const VECTOR_RETRIEVAL_CONFIGURED: &str = "configured";

/// `VECTOR_BACKEND` is set, but the name is unknown or the feature is off.
pub const VECTOR_RETRIEVAL_UNAVAILABLE: &str = "unavailable";

/// Process-level Dripstone retrieval status from `VECTOR_BACKEND`.
///
/// This is optional retrieval configuration for one process. It is not a
/// tenant boundary, an authorization check, or per-caller isolation.
/// Unset is [`VECTOR_RETRIEVAL_UNCONFIGURED`]. `VectorBackendConfig::from_env`
/// may still build an in-memory backend when a caller asks it to and the
/// variable is missing. That constructor fallback is not this status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorRetrievalStatus {
    /// `unconfigured`, `configured`, or `unavailable`.
    pub state: &'static str,
    /// Lowercased backend kind when `VECTOR_BACKEND` is non-empty.
    pub backend: Option<String>,
}

/// Names accepted by `VectorBackendConfig::from_env`, including aliases.
const KNOWN_VECTOR_BACKENDS: &[&str] = &[
    "memory",
    "redis",
    "valkey",
    "postgres",
    "postgresql",
    "qdrant",
    "qdrant-grpc",
    "qdrant_grpc",
    "qdrant-http",
    "qdrant_http",
    "pinecone",
    "chroma",
    "weaviate",
    "milvus",
    "s3",
    "coldforge",
    "dynamodb",
    "dynamodb-vector",
    "dynamodb_vector",
];

#[must_use]
pub fn is_known_vector_backend(backend: &str) -> bool {
    KNOWN_VECTOR_BACKENDS.contains(&backend)
}

/// Status for one raw `VECTOR_BACKEND` value. `None` and blank are unset.
#[must_use]
pub fn vector_retrieval_status_from_raw(value: Option<&str>) -> VectorRetrievalStatus {
    let Some(raw) = value.map(str::trim).filter(|name| !name.is_empty()) else {
        return VectorRetrievalStatus {
            state: VECTOR_RETRIEVAL_UNCONFIGURED,
            backend: None,
        };
    };
    let backend = raw.to_ascii_lowercase();
    let state = if is_known_vector_backend(&backend) && is_vector_backend_available(&backend) {
        VECTOR_RETRIEVAL_CONFIGURED
    } else {
        VECTOR_RETRIEVAL_UNAVAILABLE
    };
    VectorRetrievalStatus {
        state,
        backend: Some(backend),
    }
}

/// Read `VECTOR_BACKEND` for this process.
///
/// Does not treat a missing variable as `memory`.
#[must_use]
pub fn vector_retrieval_status() -> VectorRetrievalStatus {
    let value = std::env::var("VECTOR_BACKEND").ok();
    vector_retrieval_status_from_raw(value.as_deref())
}

#[cfg(test)]
mod tests {
    use super::{
        VECTOR_RETRIEVAL_CONFIGURED, VECTOR_RETRIEVAL_UNAVAILABLE, VECTOR_RETRIEVAL_UNCONFIGURED,
        vector_retrieval_status_from_raw,
    };

    #[test]
    fn unset_vector_backend_is_unconfigured() {
        let status = vector_retrieval_status_from_raw(None);
        assert_eq!(status.state, VECTOR_RETRIEVAL_UNCONFIGURED);
        assert_eq!(status.backend, None);
        let blank = vector_retrieval_status_from_raw(Some("  "));
        assert_eq!(blank.state, VECTOR_RETRIEVAL_UNCONFIGURED);
        assert_eq!(blank.backend, None);
    }

    #[test]
    fn explicit_memory_is_configured() {
        let status = vector_retrieval_status_from_raw(Some("Memory"));
        assert_eq!(status.state, VECTOR_RETRIEVAL_CONFIGURED);
        assert_eq!(status.backend.as_deref(), Some("memory"));
    }

    #[test]
    fn unknown_backend_is_unavailable() {
        let status = vector_retrieval_status_from_raw(Some("not-a-backend"));
        assert_eq!(status.state, VECTOR_RETRIEVAL_UNAVAILABLE);
        assert_eq!(status.backend.as_deref(), Some("not-a-backend"));
    }

    #[test]
    fn redis_state_follows_the_compiled_feature() {
        let status = vector_retrieval_status_from_raw(Some("redis"));
        #[cfg(feature = "vector-store-redis")]
        assert_eq!(status.state, VECTOR_RETRIEVAL_CONFIGURED);
        #[cfg(not(feature = "vector-store-redis"))]
        assert_eq!(status.state, VECTOR_RETRIEVAL_UNAVAILABLE);
        assert_eq!(status.backend.as_deref(), Some("redis"));
    }
}
