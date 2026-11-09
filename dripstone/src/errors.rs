use std::error::Error as StdError;

use thiserror::Error;

/// Crate-wide error for storage, pipeline, HTTP, and embedding failures.
#[derive(Debug, Error)]
pub enum DripstoneError {
    #[error("backend misconfiguration: {0}")]
    Misconfiguration(String),
    #[error("storage error: {message}")]
    Storage {
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    #[error("pipeline error: {message}")]
    Pipeline {
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("record not found: {0}")]
    NotFound(String),
    #[error("invalid vector length: expected {expected}, got {actual}")]
    InvalidVectorLength { expected: usize, actual: usize },
    #[error("HTTP {status:?}: {message}")]
    Http {
        status: Option<u16>,
        message: String,
        retryable: bool,
    },
}

impl DripstoneError {
    /// Create a configuration error.
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Misconfiguration(message.into())
    }

    pub fn storage(message: impl Into<String>) -> Self {
        Self::Storage {
            message: message.into(),
            source: None,
        }
    }

    pub fn storage_with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Storage {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn pipeline(message: impl Into<String>) -> Self {
        Self::Pipeline {
            message: message.into(),
            source: None,
        }
    }

    pub fn pipeline_with_source(
        message: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self::Pipeline {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create an HTTP error with a known status code.
    pub fn http(status: u16, message: impl Into<String>) -> Self {
        let retryable = matches!(status, 408 | 425 | 429) || (500..=599).contains(&status);
        Self::Http {
            status: Some(status),
            message: message.into(),
            retryable,
        }
    }

    /// Create an HTTP error without a status code (e.g., connection failure).
    pub fn http_network(message: impl Into<String>) -> Self {
        Self::Http {
            status: None,
            message: message.into(),
            retryable: true,
        }
    }

    /// Returns true if this error indicates "not found" (HTTP 404 or storage `NoSuchKey`).
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        match self {
            Self::NotFound(_)
            | Self::Http {
                status: Some(404), ..
            } => true,
            Self::Storage { message, .. } => storage_message_is_not_found(message),
            _ => false,
        }
    }

    /// Returns true if this error is retryable per HTTP semantics.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Http {
                retryable: true,
                ..
            }
        )
    }
}

#[cfg(feature = "redis-backend")]
impl From<redis::RedisError> for DripstoneError {
    fn from(value: redis::RedisError) -> Self {
        Self::storage_with_source("redis command failed", value)
    }
}

#[cfg(feature = "postgres-backend")]
impl From<sqlx::Error> for DripstoneError {
    fn from(value: sqlx::Error) -> Self {
        Self::storage_with_source("postgres command failed", value)
    }
}

#[cfg(any(
    feature = "qdrant-backend",
    feature = "chroma-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend",
    feature = "pinecone-backend"
))]
impl From<reqwest::Error> for DripstoneError {
    fn from(value: reqwest::Error) -> Self {
        Self::storage_with_source("http request failed", value)
    }
}

/// True when an S3 or HTTP client message is a missing object / missing collection.
///
/// Matching is case-insensitive. Qdrant and the shared HTTP client report a missing
/// collection as `permanent error: HTTP 404: ... Not found ...`, which does not
/// contain the lowercase substring `not found`. MinIO/Chainguard missing keys are
/// often `NoSuchKey` only in the error code, so callers should also return
/// [`DripstoneError::NotFound`] from the storage client.
fn storage_message_is_not_found(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("nosuchkey")
        || lower.contains("no such key")
        || lower.contains("not found")
        || lower.contains("notfound")
        || lower.contains("does not exist")
        || lower.contains("doesn't exist")
        || lower.contains("http 404")
        || lower.contains("status 404")
        || lower.contains("status: 404")
}

#[cfg(test)]
mod tests {
    use super::DripstoneError;

    #[test]
    fn qdrant_missing_collection_http_404_is_not_found() {
        let err = DripstoneError::storage(
            "permanent error: HTTP 404: {\"status\":{\"error\":\"Not found: Collection `embeddings_ns-cleanup` doesn't exist!\"}}",
        );
        assert!(err.is_not_found());
    }

    #[test]
    fn minio_nosuchkey_display_is_not_found() {
        let err = DripstoneError::storage(
            "Failed to get object from S3-compatible storage: NoSuchKey: The specified key does not exist.",
        );
        assert!(err.is_not_found());
    }

    #[test]
    fn not_found_variant_is_not_found() {
        let err = DripstoneError::NotFound("NoSuchKey: bucket/indexes/ns/index.bin".to_string());
        assert!(err.is_not_found());
        assert!(err.to_string().contains("NoSuchKey"));
    }

    #[test]
    fn unrelated_storage_error_is_not_missing() {
        let err = DripstoneError::storage("Failed to get object from S3: AccessDenied");
        assert!(!err.is_not_found());
    }

    #[test]
    fn sdk_service_error_display_is_not_a_missing_key() {
        // `SdkError`'s Display is "service error" even when the source is NoSuchKey.
        // Callers must map that to `NotFound` before this string is stored.
        let err = DripstoneError::storage(
            "Failed to get object from S3-compatible storage: service error",
        );
        assert!(!err.is_not_found());
    }
}
