//! Configuration for object storage providers

use crate::data::storage::object::cache::CacheConfig;
use std::path::PathBuf;

/// Object storage provider configuration
#[derive(Debug, Clone)]
pub enum ObjectStoreProvider {
    /// AWS S3
    AwsS3 {
        /// AWS region (defaults to `AWS_REGION` env var or us-east-1)
        region: Option<String>,
        /// AWS access key ID (optional, uses default credentials if not set)
        access_key_id: Option<String>,
        /// AWS secret access key (optional, uses default credentials if not set)
        secret_access_key: Option<String>,
        /// AWS session token for temporary credentials (optional)
        session_token: Option<String>,
    },
    /// Cloudflare R2
    CloudflareR2 {
        /// Cloudflare account ID
        account_id: String,
        /// Access key ID
        access_key_id: String,
        /// Secret access key
        secret_access_key: String,
        /// Optional custom endpoint (defaults to R2 endpoint)
        endpoint: Option<String>,
    },
    /// Generic S3-compatible storage (`MinIO`, Wasabi, etc.)
    GenericS3 {
        /// Endpoint URL (e.g., <https://s3.example.com>)
        endpoint: String,
        /// Region
        region: String,
        /// Use path-style addressing (bucket.s3.amazonaws.com vs s3.amazonaws.com/bucket)
        path_style: bool,
        /// Access key ID
        access_key_id: Option<String>,
        /// Secret access key
        secret_access_key: Option<String>,
        /// Session token for temporary credentials (optional)
        session_token: Option<String>,
    },
}

/// Object storage configuration
#[derive(Debug, Clone)]
pub struct ObjectStoreConfig {
    pub provider: ObjectStoreProvider,
}

impl ObjectStoreConfig {
    /// Create config from environment variables
    ///
    /// Environment variables:
    /// - `OBJECT_STORE_PROVIDER`: "aws", "cloudflare", or "generic"
    /// - `OBJECT_STORE_ENDPOINT`: Endpoint URL (for generic or Cloudflare)
    /// - `OBJECT_STORE_REGION`: Region (for AWS or generic)
    /// - `OBJECT_STORE_PATH_STYLE`: "true" or "false" (for generic)
    /// - `CLOUDFLARE_ACCOUNT_ID`: Cloudflare account ID
    /// - `CLOUDFLARE_ACCESS_KEY_ID`: Cloudflare access key
    /// - `CLOUDFLARE_SECRET_ACCESS_KEY`: Cloudflare secret key
    /// - `AWS_ACCESS_KEY_ID`: AWS access key (optional, uses default credentials if not set)
    /// - `AWS_SECRET_ACCESS_KEY`: AWS secret key (optional)
    /// - `AWS_REGION`: AWS region (optional, defaults to us-east-1)
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub fn from_env() -> Result<Self, anyhow::Error> {
        let provider_str = caliche_contracts::credential_env::var("OBJECT_STORE_PROVIDER")
            .unwrap_or_else(|_| "aws".to_string())
            .to_lowercase();

        let provider = match provider_str.as_str() {
            "aws" | "s3" => {
                let region = caliche_contracts::credential_env::var("OBJECT_STORE_REGION")
                    .or_else(|_| caliche_contracts::credential_env::var("AWS_REGION"))
                    .ok();
                let access_key_id =
                    caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID").ok();
                let secret_access_key =
                    caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY").ok();
                let session_token =
                    caliche_contracts::credential_env::var("AWS_SESSION_TOKEN").ok();
                ObjectStoreProvider::AwsS3 {
                    region,
                    access_key_id,
                    secret_access_key,
                    session_token,
                }
            }
            "cloudflare" | "r2" => {
                let account_id = caliche_contracts::credential_env::var("CLOUDFLARE_ACCOUNT_ID")
                    .map_err(|_| anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID is required"))?;
                let access_key_id =
                    caliche_contracts::credential_env::var("CLOUDFLARE_ACCESS_KEY_ID")
                        .or_else(|_| caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID"))
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "CLOUDFLARE_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID is required"
                            )
                        })?;
                let secret_access_key =
                    caliche_contracts::credential_env::var("CLOUDFLARE_SECRET_ACCESS_KEY")
                        .or_else(|_| {
                            caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY")
                        })
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "CLOUDFLARE_SECRET_ACCESS_KEY or AWS_SECRET_ACCESS_KEY is required"
                            )
                        })?;
                let endpoint = caliche_contracts::credential_env::var("OBJECT_STORE_ENDPOINT").ok();

                ObjectStoreProvider::CloudflareR2 {
                    account_id,
                    access_key_id,
                    secret_access_key,
                    endpoint,
                }
            }
            "generic" => {
                let endpoint = caliche_contracts::credential_env::var("OBJECT_STORE_ENDPOINT")
                    .map_err(|_| {
                        anyhow::anyhow!("OBJECT_STORE_ENDPOINT is required for generic S3")
                    })?;
                let region = caliche_contracts::credential_env::var("OBJECT_STORE_REGION")
                    .unwrap_or_else(|_| "us-east-1".to_string());
                let path_style = match caliche_contracts::credential_env::var(
                    "OBJECT_STORE_PATH_STYLE",
                ) {
                    Ok(raw) => {
                        #[cfg(feature = "vector-store-s3-backend")]
                        {
                            dripstone::storage::parse_object_store_path_style(&raw)
                                .map_err(|e| anyhow::anyhow!("{e}"))?
                        }
                        #[cfg(not(feature = "vector-store-s3-backend"))]
                        {
                            match raw.trim().to_ascii_lowercase().as_str() {
                                "1" | "true" | "yes" | "on" => true,
                                "" | "0" | "false" | "no" | "off" => false,
                                other => {
                                    return Err(anyhow::anyhow!(
                                        "OBJECT_STORE_PATH_STYLE must be a boolean (true/false, 1/0, yes/no, on/off); got {other:?}"
                                    ));
                                }
                            }
                        }
                    }
                    Err(_) => false,
                };
                let access_key_id =
                    caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID").ok();
                let secret_access_key =
                    caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY").ok();
                let session_token =
                    caliche_contracts::credential_env::var("AWS_SESSION_TOKEN").ok();

                ObjectStoreProvider::GenericS3 {
                    endpoint,
                    region,
                    path_style,
                    access_key_id,
                    secret_access_key,
                    session_token,
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown OBJECT_STORE_PROVIDER: {provider_str}. Must be 'aws', 'cloudflare', or 'generic'"
                ));
            }
        };

        Ok(Self { provider })
    }

    /// Create config with AWS S3 provider
    #[must_use]
    pub const fn aws_s3(
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            provider: ObjectStoreProvider::AwsS3 {
                region,
                access_key_id,
                secret_access_key,
                session_token,
            },
        }
    }

    /// Create config with Cloudflare R2 provider
    #[must_use]
    pub const fn cloudflare_r2(
        account_id: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
    ) -> Self {
        Self {
            provider: ObjectStoreProvider::CloudflareR2 {
                account_id,
                access_key_id,
                secret_access_key,
                endpoint,
            },
        }
    }

    /// Create config with generic S3 provider
    #[must_use]
    pub const fn generic_s3(
        endpoint: String,
        region: String,
        path_style: bool,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            provider: ObjectStoreProvider::GenericS3 {
                endpoint,
                region,
                path_style,
                access_key_id,
                secret_access_key,
                session_token,
            },
        }
    }
}

impl Default for ObjectStoreConfig {
    fn default() -> Self {
        Self::from_env().unwrap_or_else(|error| {
            tracing::warn!(
                error = %error,
                "failed to load object store config from env; falling back to AWS defaults"
            );
            Self::aws_s3(None, None, None, None)
        })
    }
}

/// Storage backend type
#[derive(Debug, Clone)]
pub enum StorageBackendType {
    /// Local filesystem
    Filesystem,
    /// Object storage (S3/R2/generic S3)
    ObjectStore,
}

/// Unified storage configuration for templates and prompts
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Backend type
    pub backend: StorageBackendType,
    /// Object store configuration (required if backend is `ObjectStore`)
    pub object_store: Option<ObjectStoreConfig>,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Default bucket name (for object storage)
    pub default_bucket: Option<String>,
    /// Base path (for filesystem backend)
    pub base_path: Option<PathBuf>,
}

impl StorageConfig {
    /// Create a filesystem-based storage config
    pub fn filesystem(base_path: impl Into<PathBuf>) -> Self {
        Self {
            backend: StorageBackendType::Filesystem,
            object_store: None,
            cache: CacheConfig::default(),
            default_bucket: None,
            base_path: Some(base_path.into()),
        }
    }

    /// Create an object store-based storage config
    #[cfg(feature = "s3")]
    pub fn object_store(bucket: impl Into<String>, object_store_config: ObjectStoreConfig) -> Self {
        Self {
            backend: StorageBackendType::ObjectStore,
            object_store: Some(object_store_config),
            cache: CacheConfig::default(),
            default_bucket: Some(bucket.into()),
            base_path: None,
        }
    }

    /// Create an AWS S3 storage config
    #[cfg(feature = "s3")]
    pub fn aws_s3(
        bucket: impl Into<String>,
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self::object_store(
            bucket,
            ObjectStoreConfig::aws_s3(region, access_key_id, secret_access_key, session_token),
        )
    }

    /// Create a Cloudflare R2 storage config
    #[cfg(feature = "s3")]
    pub fn cloudflare_r2(
        bucket: impl Into<String>,
        account_id: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
    ) -> Self {
        Self::object_store(
            bucket,
            ObjectStoreConfig::cloudflare_r2(
                account_id,
                access_key_id,
                secret_access_key,
                endpoint,
            ),
        )
    }

    /// Create a generic S3 storage config
    #[cfg(feature = "s3")]
    pub fn generic_s3(
        bucket: impl Into<String>,
        endpoint: String,
        region: String,
        path_style: bool,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self::object_store(
            bucket,
            ObjectStoreConfig::generic_s3(
                endpoint,
                region,
                path_style,
                access_key_id,
                secret_access_key,
                session_token,
            ),
        )
    }

    /// Enable or disable caching
    #[must_use]
    pub fn with_cache(mut self, enabled: bool) -> Self {
        if enabled {
            self.cache = CacheConfig::enabled();
        } else {
            self.cache = CacheConfig::disabled();
        }
        self
    }

    /// Set cache TTL in seconds
    #[must_use]
    pub fn with_cache_ttl(mut self, ttl_seconds: u64) -> Self {
        self.cache = self.cache.with_ttl(ttl_seconds);
        self
    }

    /// Set cache max size in MB
    #[must_use]
    pub fn with_cache_max_size(mut self, max_size_mb: u64) -> Self {
        self.cache = self.cache.with_max_size(max_size_mb);
        self
    }

    /// Set cache directory
    #[must_use]
    pub fn with_cache_dir(mut self, cache_dir: PathBuf) -> Self {
        self.cache = self.cache.with_cache_dir(cache_dir);
        self
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self::filesystem(std::env::temp_dir().join("caliche_storage"))
    }
}
