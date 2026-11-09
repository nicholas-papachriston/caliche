use anyhow::Result;

#[cfg(feature = "s3")]
use crate::data::storage::object::{
    ObjectStore, ObjectStoreConfig, create_object_store, types::ObjectStoreError,
};
#[cfg(feature = "s3")]
use bytes::Bytes;
#[cfg(feature = "s3")]
use std::sync::Arc;
#[cfg(feature = "s3")]
use tokio::sync::OnceCell;

/// Global object store instance (for `default/from_env` usage).
///
/// Stores `Result<Arc<dyn ObjectStore>>` so initialization errors are propagated
/// instead of panicking inside `OnceCell::get_or_init`.
#[cfg(feature = "s3")]
static OBJECT_STORE: OnceCell<Result<Arc<dyn ObjectStore>, String>> = OnceCell::const_new();

/// Get or initialize the object store using environment variables.
///
/// This is a convenience function that uses `ObjectStoreConfig::from_env()`.
/// For programmatic configuration, use `get_object_store_with_config()` instead.
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn get_object_store() -> Result<Arc<dyn ObjectStore>> {
    let result = OBJECT_STORE
        .get_or_init(|| async {
            tracing::debug!("Initializing object store from environment");
            let config = ObjectStoreConfig::from_env().unwrap_or_else(|error| {
                tracing::warn!(
                    error = %error,
                    "failed to load object store config from env; using AWS defaults"
                );
                ObjectStoreConfig::aws_s3(None, None, None, None)
            });
            match create_object_store(config).await {
                Ok(store) => Ok(store),
                Err(e) => {
                    tracing::error!("Failed to create object store: {e}");
                    Err(format!("{e:#}"))
                }
            }
        })
        .await;

    match result {
        Ok(store) => Ok(Arc::clone(store)),
        Err(msg) => Err(anyhow::anyhow!("Object store initialization failed: {msg}")),
    }
}

/// Get or initialize an object store with explicit configuration
///
/// This function allows programmatic configuration of the object store
/// without relying on environment variables. Use this when you need to
/// configure S3-compatible services (like `RustFS`, `MinIO`) or provide
/// credentials programmatically.
///
/// # Example
/// ```no_run
/// use caliche::storage::{ObjectStoreConfig, create_object_store};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = ObjectStoreConfig::generic_s3(
///     "http://localhost:9010".to_string(),
///     "us-east-1".to_string(),
///     true, // path_style
///     Some("access-key".to_string()),
///     Some("secret-key".to_string()),
///     None, // session_token
/// );
/// let store = create_object_store(config).await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn get_object_store_with_config(
    config: ObjectStoreConfig,
) -> Result<Arc<dyn ObjectStore>> {
    tracing::debug!("Creating object store with explicit configuration");
    let store = create_object_store(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create object store: {e}"))?;
    Ok(store)
}

/// Parse S3 URI into bucket and key
/// Format: <s3://bucket-name/path/to/file>
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    if !uri.starts_with("s3://") {
        return Err(anyhow::anyhow!("Invalid S3 URI: must start with s3://"));
    }

    // SAFETY: the `starts_with("s3://")` guard above guarantees this succeeds.
    let path = &uri["s3://".len()..];
    let parts: Vec<&str> = path.splitn(2, '/').collect();

    if parts.is_empty() || parts[0].is_empty() {
        return Err(anyhow::anyhow!("Invalid S3 URI: missing bucket name"));
    }

    let bucket = parts[0].to_string();
    let key = if parts.len() > 1 {
        parts[1].to_string()
    } else {
        String::new()
    };

    if key.is_empty() {
        return Err(anyhow::anyhow!("Invalid S3 URI: missing object key"));
    }

    Ok((bucket, key))
}

/// Check if a path is an S3 URI
#[must_use]
pub fn is_s3_uri(path: &str) -> bool {
    path.starts_with("s3://")
}

/// Download a file from object storage to a temporary location
/// Returns the path to the temporary file
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn download_from_s3(uri: &str) -> Result<std::path::PathBuf> {
    let (bucket, key) = parse_s3_uri(uri)?;

    tracing::info!(
        "Downloading from object storage: bucket={}, key={}",
        bucket,
        key
    );

    let store = get_object_store().await?;
    let bytes = store.get_object(&bucket, &key).await?;

    // Create temporary file
    let temp_dir = std::env::temp_dir();
    let file_name = key.split('/').next_back().unwrap_or("download");

    let temp_path = temp_dir.join(format!(
        "s3_download_{}_{}",
        uuid::Uuid::now_v7(),
        file_name
    ));

    // Write to temporary file
    tokio::fs::write(&temp_path, bytes)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to write temporary file: {e}"))?;

    tracing::info!(
        "Downloaded S3 object to temporary file: {}",
        temp_path.display()
    );

    Ok(temp_path)
}

/// Upload a file to object storage
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn upload_to_s3(local_path: &std::path::Path, uri: &str) -> Result<()> {
    let (bucket, key) = parse_s3_uri(uri)?;

    tracing::info!(
        "Uploading to object storage: bucket={}, key={}",
        bucket,
        key
    );

    let store = get_object_store().await?;

    // Read the file
    let data = tokio::fs::read(local_path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read file: {e}"))?;

    // Determine content type from file extension
    let content_type = local_path
        .extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| match ext {
            "json" => Some("application/json"),
            "txt" => Some("text/plain"),
            "md" => Some("text/markdown"),
            "html" => Some("text/html"),
            "csv" => Some("text/csv"),
            _ => None,
        });

    // Upload to object storage
    store
        .put_object(&bucket, &key, Bytes::from(data), content_type)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to upload to object storage: {e}"))?;

    tracing::info!("Successfully uploaded to object storage: {}", uri);

    Ok(())
}

/// Check if an object exists in object storage
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn s3_object_exists(uri: &str) -> Result<bool> {
    let (bucket, key) = parse_s3_uri(uri)?;

    let store = get_object_store().await?;

    match store.head_object(&bucket, &key).await {
        Ok(_) => Ok(true),
        Err(e) => match ObjectStoreError::classify(&key, &e) {
            ObjectStoreError::NotFound { .. } => Ok(false),
            _ => Err(anyhow::anyhow!("Failed to check object: {e}")),
        },
    }
}

/// Get object metadata from object storage
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn get_s3_metadata(uri: &str) -> Result<S3ObjectMetadata> {
    let (bucket, key) = parse_s3_uri(uri)?;

    let store = get_object_store().await?;

    let metadata = store
        .head_object(&bucket, &key)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to get object metadata: {e}"))?;

    Ok(S3ObjectMetadata {
        size: metadata.size,
        content_type: metadata.content_type,
        last_modified: metadata.last_modified.map(|dt| {
            dt.duration_since(std::time::UNIX_EPOCH)
                .map_or_else(|_| "0".to_string(), |d| d.as_secs().to_string())
        }),
        etag: metadata.etag,
    })
}

#[derive(Debug, Clone)]
pub struct S3ObjectMetadata {
    pub size: u64,
    pub content_type: Option<String>,
    pub last_modified: Option<String>,
    pub etag: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_uri() {
        let (bucket, key) =
            parse_s3_uri("s3://my-bucket/path/to/file.txt").expect("valid S3 URI should parse");
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.txt");

        let (bucket, key) =
            parse_s3_uri("s3://bucket/file.txt").expect("valid S3 URI should parse");
        assert_eq!(bucket, "bucket");
        assert_eq!(key, "file.txt");

        assert!(parse_s3_uri("https://example.com/file.txt").is_err());
        assert!(parse_s3_uri("s3://bucket").is_err());
        assert!(parse_s3_uri("s3://").is_err());
    }

    #[test]
    fn test_is_s3_uri() {
        assert!(is_s3_uri("s3://bucket/key"));
        assert!(!is_s3_uri("/local/path"));
        assert!(!is_s3_uri("https://example.com"));
    }
}
