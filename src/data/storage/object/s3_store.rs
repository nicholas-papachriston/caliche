//! Unified S3-compatible object storage.
//!
//! This module consolidates the previous `AwsS3Store`, `GenericS3Store`, and
//! `CloudflareR2Store` implementations into a single [`S3Store`] that is
//! built via an [`S3StoreBuilder`]. The three providers differ only in how
//! the underlying AWS SDK S3 [`Client`] is configured; once constructed the
//! read/write paths are identical.
//!
//! # Examples
//!
//! ```text
//! use caliche::data::storage::object::s3_store::S3Store;
//!
//! // AWS S3 with the default credential chain.
//! let store = S3Store::builder()
//!     .region(Some("us-east-1".into()))
//!     .build()
//!     .await?;
//!
//! // Cloudflare R2 with explicit credentials.
//! let r2 = S3Store::builder()
//!     .endpoint(Some("https://abc123.r2.cloudflarestorage.com".into()))
//!     .region(Some("auto".into()))
//!     .force_path_style(true)
//!     .credentials("AKID".into(), "SECRET".into(), None)
//!     .build()
//!     .await?;
//! ```

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_credential_types::Credentials as AwsCredentials;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::{Builder as S3ConfigBuilder, Region};
use bytes::Bytes;
use std::sync::Arc;
use std::time::SystemTime;

use super::{ObjectInfo, ObjectMetadata, ObjectStore};

/// Unified S3-compatible object store.
pub struct S3Store {
    client: Arc<Client>,
}

impl S3Store {
    /// Start building an [`S3Store`].
    #[must_use]
    pub fn builder() -> S3StoreBuilder {
        S3StoreBuilder::default()
    }

    /// Wrap an existing AWS SDK `Client` as an [`S3Store`]. Useful in tests
    /// and when callers manage their own SDK configuration.
    #[must_use]
    pub fn with_client(client: Client) -> Self {
        Self {
            client: Arc::new(client),
        }
    }
}

/// Default credential provider name reported to the SDK.
const DEFAULT_CREDENTIAL_PROVIDER_NAME: &str = "caliche-s3";

/// Builder for [`S3Store`]. Defaults match the behaviour of the previous
/// `AwsS3Store`: no custom endpoint, no forced path style, credentials
/// resolved via the AWS default credential chain.
#[derive(Debug, Clone)]
pub struct S3StoreBuilder {
    region: Option<String>,
    endpoint: Option<String>,
    force_path_style: bool,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    /// Static credential provider name reported to the SDK (used in signing
    /// metadata). Must be `'static` since the AWS SDK stores it verbatim.
    credentials_provider_name: &'static str,
}

impl Default for S3StoreBuilder {
    fn default() -> Self {
        Self {
            region: None,
            endpoint: None,
            force_path_style: false,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            credentials_provider_name: DEFAULT_CREDENTIAL_PROVIDER_NAME,
        }
    }
}

impl S3StoreBuilder {
    /// Set an explicit AWS region. When `None` the SDK uses its default
    /// resolver (env, profile, IMDS, …).
    #[must_use]
    pub fn region(mut self, region: Option<String>) -> Self {
        self.region = region;
        self
    }

    /// Set a custom endpoint URL (`MinIO`, Wasabi, R2, local mocks, …).
    #[must_use]
    pub fn endpoint(mut self, endpoint: Option<String>) -> Self {
        self.endpoint = endpoint;
        self
    }

    /// Force path-style addressing (`host/bucket/key`) instead of the
    /// virtual-hosted style (`bucket.host/key`). R2 and most MinIO-style
    /// deployments require this.
    #[must_use]
    pub const fn force_path_style(mut self, path_style: bool) -> Self {
        self.force_path_style = path_style;
        self
    }

    /// Supply explicit credentials. Takes precedence over the default
    /// credential chain. `session_token` is optional and only used for
    /// temporary credentials (STS, IAM Role, SSO session, …).
    #[must_use]
    pub fn credentials(
        mut self,
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    ) -> Self {
        self.access_key_id = Some(access_key_id);
        self.secret_access_key = Some(secret_access_key);
        self.session_token = session_token;
        self
    }

    /// Override the internal credential provider name. Primarily useful for
    /// diagnostics and audit logs where the provider name shows up in
    /// signing metadata. Defaults to `"caliche-s3"`.
    #[must_use]
    pub const fn credentials_provider_name(mut self, name: &'static str) -> Self {
        self.credentials_provider_name = name;
        self
    }

    /// Convenience constructor for Cloudflare R2. Builds the expected
    /// endpoint URL from `account_id`, forces path-style addressing, and
    /// pins the region to `"auto"`. R2 *requires* explicit credentials, so
    /// this shape deliberately takes them by value instead of `Option`.
    #[must_use]
    pub fn for_r2(
        mut self,
        account_id: &str,
        access_key_id: String,
        secret_access_key: String,
    ) -> Self {
        self.endpoint = Some(format!("https://{account_id}.r2.cloudflarestorage.com"));
        self.region = Some("auto".to_string());
        self.force_path_style = true;
        self.access_key_id = Some(access_key_id);
        self.secret_access_key = Some(secret_access_key);
        self.session_token = None;
        self.credentials_provider_name = "cloudflare-r2";
        self
    }

    /// Finalize the builder and produce an [`S3Store`].
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn build(self) -> Result<S3Store, anyhow::Error> {
        // When a custom endpoint is supplied (R2, MinIO, generic S3) we build
        // the SDK client directly from an explicit `Config`. Otherwise we use
        // the higher-level `aws_config::defaults` path so that the normal
        // credential resolution and region detection kick in.
        if self.endpoint.is_some() || self.force_path_style {
            let base_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
            let region = self.region.unwrap_or_else(|| "us-east-1".to_string());
            let mut s3_config = S3ConfigBuilder::default().region(Region::new(region));

            if let Some(bv) = base_config.behavior_version() {
                s3_config = s3_config.behavior_version(bv);
            }
            if let Some(endpoint) = &self.endpoint {
                s3_config = s3_config.endpoint_url(endpoint);
            }
            if self.force_path_style {
                s3_config = s3_config.force_path_style(true);
            }
            if let (Some(access_key), Some(secret_key)) =
                (self.access_key_id.clone(), self.secret_access_key.clone())
            {
                let credentials = AwsCredentials::new(
                    access_key,
                    secret_key,
                    self.session_token.clone(),
                    None,
                    self.credentials_provider_name,
                );
                s3_config = s3_config.credentials_provider(credentials);
            }

            let client = Client::from_conf(s3_config.build());
            return Ok(S3Store {
                client: Arc::new(client),
            });
        }

        let mut config_builder = aws_config::defaults(BehaviorVersion::latest());
        if let Some(region) = self.region {
            config_builder = config_builder.region(aws_config::Region::new(region));
        }
        if let (Some(access_key), Some(secret_key)) = (self.access_key_id, self.secret_access_key) {
            let credentials = AwsCredentials::new(
                access_key,
                secret_key,
                self.session_token,
                None,
                self.credentials_provider_name,
            );
            config_builder = config_builder.credentials_provider(credentials);
        }

        let config = config_builder.load().await;
        Ok(S3Store {
            client: Arc::new(Client::new(&config)),
        })
    }
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes, anyhow::Error> {
        let response = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get object from S3: {e}"))?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read object body: {e}"))?
            .into_bytes();

        Ok(bytes)
    }

    async fn put_object(
        &self,
        bucket: &str,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> Result<(), anyhow::Error> {
        let mut request = self
            .client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(data.into());

        if let Some(content_type) = content_type {
            request = request.content_type(content_type);
        }

        request
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to put object to S3: {e}"))?;
        Ok(())
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<(), anyhow::Error> {
        self.client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to delete object from S3: {e}"))?;
        Ok(())
    }

    async fn head_object(&self, bucket: &str, key: &str) -> Result<ObjectMetadata, anyhow::Error> {
        let response = self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to head object from S3: {e}"))?;

        Ok(ObjectMetadata {
            size: u64::try_from(response.content_length().unwrap_or(0)).unwrap_or(0),
            content_type: response.content_type().map(str::to_string),
            last_modified: response.last_modified().map(|dt| {
                let secs = u64::try_from(dt.secs().max(0)).unwrap_or(0);
                SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)
            }),
            etag: response.e_tag().map(str::to_string),
        })
    }

    async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<ObjectInfo>, anyhow::Error> {
        let mut request = self.client.list_objects_v2().bucket(bucket);
        if let Some(prefix) = prefix {
            request = request.prefix(prefix);
        }

        let mut objects = Vec::new();
        let mut continuation_token: Option<String> = None;
        loop {
            let mut req = request.clone();
            if let Some(token) = continuation_token.take() {
                req = req.continuation_token(token);
            }

            let response = req
                .send()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to list objects from S3: {e}"))?;

            for object in response.contents() {
                objects.push(ObjectInfo {
                    key: object.key().unwrap_or("").to_string(),
                    size: u64::try_from(object.size().unwrap_or(0)).unwrap_or(0),
                    last_modified: object.last_modified().map(|dt| {
                        let secs = u64::try_from(dt.secs().max(0)).unwrap_or(0);
                        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)
                    }),
                    etag: object.e_tag().map(str::to_string),
                });
            }

            if !response.is_truncated.unwrap_or(false) {
                break;
            }
            continuation_token = response.next_continuation_token().map(str::to_string);
        }

        Ok(objects)
    }
}
