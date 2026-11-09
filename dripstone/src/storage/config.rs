//! Configuration for object storage providers

use std::env;

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
        /// Endpoint URL (e.g., `https://s3.example.com`)
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

/// Parse `OBJECT_STORE_PATH_STYLE` / similar booleanish env values.
///
/// # Errors
///
/// Returns an error when `raw` is not a recognized boolean synonym.
pub fn parse_object_store_path_style(raw: &str) -> Result<bool, crate::errors::DripstoneError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "" | "0" | "false" | "no" | "off" => Ok(false),
        other => Err(crate::errors::DripstoneError::storage(format!(
            "OBJECT_STORE_PATH_STYLE must be a boolean (true/false, 1/0, yes/no, on/off); got {other:?}"
        ))),
    }
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
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn from_env() -> Result<Self, crate::errors::DripstoneError> {
        let provider_str = env::var("OBJECT_STORE_PROVIDER")
            .unwrap_or_else(|_| "aws".to_string())
            .to_lowercase();

        let provider = match provider_str.as_str() {
            "aws" | "s3" => {
                let region = env::var("OBJECT_STORE_REGION")
                    .or_else(|_| env::var("AWS_REGION"))
                    .ok();
                let access_key_id = env::var("AWS_ACCESS_KEY_ID").ok();
                let secret_access_key = env::var("AWS_SECRET_ACCESS_KEY").ok();
                let session_token = env::var("AWS_SESSION_TOKEN").ok();
                ObjectStoreProvider::AwsS3 {
                    region,
                    access_key_id,
                    secret_access_key,
                    session_token,
                }
            }
            "cloudflare" | "r2" => {
                let account_id = env::var("CLOUDFLARE_ACCOUNT_ID").map_err(|_| {
                    crate::errors::DripstoneError::storage("CLOUDFLARE_ACCOUNT_ID is required")
                })?;
                let access_key_id = env::var("CLOUDFLARE_ACCESS_KEY_ID")
                    .or_else(|_| env::var("AWS_ACCESS_KEY_ID"))
                    .map_err(|_| {
                        crate::errors::DripstoneError::storage(
                            "CLOUDFLARE_ACCESS_KEY_ID or AWS_ACCESS_KEY_ID is required",
                        )
                    })?;
                let secret_access_key = env::var("CLOUDFLARE_SECRET_ACCESS_KEY")
                    .or_else(|_| env::var("AWS_SECRET_ACCESS_KEY"))
                    .map_err(|_| {
                        crate::errors::DripstoneError::storage(
                            "CLOUDFLARE_SECRET_ACCESS_KEY or AWS_SECRET_ACCESS_KEY is required",
                        )
                    })?;
                let endpoint = env::var("OBJECT_STORE_ENDPOINT").ok();

                ObjectStoreProvider::CloudflareR2 {
                    account_id,
                    access_key_id,
                    secret_access_key,
                    endpoint,
                }
            }
            "generic" => {
                let endpoint = env::var("OBJECT_STORE_ENDPOINT").map_err(|_| {
                    crate::errors::DripstoneError::storage(
                        "OBJECT_STORE_ENDPOINT is required for generic S3",
                    )
                })?;
                let region =
                    env::var("OBJECT_STORE_REGION").unwrap_or_else(|_| "us-east-1".to_string());
                let path_style = match env::var("OBJECT_STORE_PATH_STYLE") {
                    Ok(raw) => parse_object_store_path_style(&raw)?,
                    Err(_) => false,
                };
                let access_key_id = env::var("AWS_ACCESS_KEY_ID").ok();
                let secret_access_key = env::var("AWS_SECRET_ACCESS_KEY").ok();
                let session_token = env::var("AWS_SESSION_TOKEN").ok();

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
                return Err(crate::errors::DripstoneError::storage(format!(
                    "Unknown OBJECT_STORE_PROVIDER: {provider_str}. Must be 'aws', 'cloudflare', or 'generic'"
                )));
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
        Self::from_env().unwrap_or_else(|_| {
            // Default to AWS S3 if env vars are not set
            Self::aws_s3(None, None, None, None)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::parse_object_store_path_style;

    #[test]
    fn path_style_accepts_boolean_synonyms() {
        for raw in ["1", "true", "YES", "on", " True "] {
            assert!(parse_object_store_path_style(raw).expect("trueish"));
        }
        for raw in ["0", "false", "NO", "off", "", " False "] {
            assert!(!parse_object_store_path_style(raw).expect("falseish"));
        }
    }

    #[test]
    fn path_style_rejects_junk() {
        assert!(parse_object_store_path_style("maybe").is_err());
    }
}
