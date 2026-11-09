//! Object storage module - S3-compatible storage backends

pub mod backend;
pub mod cache;
pub mod config;
pub mod s3_client;
#[cfg(feature = "s3")]
pub mod s3_store;
mod types;

#[cfg(feature = "s3")]
pub use backend::ObjectStoreBackend;
pub use backend::{FilesystemBackend, StorageBackend};
pub use cache::{CacheConfig, CachedStorageBackend};
pub use config::{ObjectStoreConfig, ObjectStoreProvider, StorageBackendType, StorageConfig};
pub use s3_client::{S3ObjectMetadata, is_s3_uri, parse_s3_uri};
#[cfg(feature = "s3")]
pub use s3_client::{download_from_s3, get_object_store, upload_to_s3};
#[cfg(feature = "s3")]
pub use s3_store::{S3Store, S3StoreBuilder};
pub use types::{ObjectInfo, ObjectMetadata, ObjectStore};

/// Create an object store from configuration.
///
/// All three S3-compatible providers (`AwsS3`, `CloudflareR2`, `GenericS3`)
/// now route through the unified [`S3Store`] builder.
#[cfg(feature = "s3")]
/// # Errors
///
/// Returns an error if the persistence or storage operation fails.
pub async fn create_object_store(
    config: ObjectStoreConfig,
) -> Result<std::sync::Arc<dyn ObjectStore>, anyhow::Error> {
    use std::sync::Arc;
    let store = match config.provider {
        ObjectStoreProvider::AwsS3 {
            region,
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            let mut builder = S3Store::builder().region(region);
            if let (Some(access_key), Some(secret_key)) = (access_key_id, secret_access_key) {
                builder = builder.credentials(access_key, secret_key, session_token);
            }
            builder.build().await?
        }
        ObjectStoreProvider::CloudflareR2 {
            account_id,
            access_key_id,
            secret_access_key,
            endpoint,
        } => {
            let mut builder =
                S3Store::builder().for_r2(&account_id, access_key_id, secret_access_key);
            // Respect a caller-provided endpoint override even in the R2 path
            // (e.g. a regional alias or a mock URL used in tests).
            if let Some(endpoint) = endpoint {
                builder = builder.endpoint(Some(endpoint));
            }
            builder.build().await?
        }
        ObjectStoreProvider::GenericS3 {
            endpoint,
            region,
            path_style,
            access_key_id,
            secret_access_key,
            session_token,
        } => {
            let mut builder = S3Store::builder()
                .endpoint(Some(endpoint))
                .region(Some(region))
                .force_path_style(path_style)
                .credentials_provider_name("generic-s3");
            if let (Some(access_key), Some(secret_key)) = (access_key_id, secret_access_key) {
                builder = builder.credentials(access_key, secret_key, session_token);
            }
            builder.build().await?
        }
    };
    Ok(Arc::new(store))
}
