//! Unified S3-compatible object storage.
//!
//! Consolidates the previous `AwsS3Store`, `GenericS3Store`, and
//! `CloudflareR2Store` implementations into a single [`S3Store`]. The three
//! providers differ only in how the underlying AWS SDK S3 [`Client`] is
//! configured; once constructed the read/write paths are identical.

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_credential_types::Credentials as AwsCredentials;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::{Builder as S3ConfigBuilder, Region};
use aws_sdk_s3::error::{ProvideErrorMetadata, SdkError};
use bytes::Bytes;
use std::error::Error as StdError;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::time::sleep;

use super::{
    OBJECT_STORE_PUT_MAX_ATTEMPTS, ObjectInfo, ObjectMetadata, ObjectStore,
    is_transient_object_store_error, object_store_put_backoff,
};

/// Unified S3-compatible object store.
pub struct S3Store {
    client: Arc<Client>,
    /// Label used in error messages (`S3`, `R2`, `S3-compatible storage`).
    provider_label: &'static str,
}

impl S3Store {
    /// Wrap an existing AWS SDK client.
    #[must_use]
    pub fn with_client(client: Client, provider_label: &'static str) -> Self {
        Self {
            client: Arc::new(client),
            provider_label,
        }
    }

    /// AWS S3 with the default credential chain (or explicit keys).
    ///
    /// # Errors
    ///
    /// Returns an error when the SDK client cannot be constructed.
    pub async fn aws(
        region: Option<String>,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Result<Self, crate::errors::DripstoneError> {
        // `Client::new` keeps the default credential chain (SSO profile, web
        // identity, IMDS). Building an `S3ConfigBuilder` from scratch drops that
        // provider and signs as anonymous when no static keys were passed.
        let mut loader = aws_config::defaults(BehaviorVersion::latest());
        if let Some(region) = region {
            loader = loader.region(Region::new(region));
        }
        if let (Some(access_key), Some(secret_key)) = (access_key_id, secret_access_key) {
            let credentials = AwsCredentials::new(
                &access_key,
                &secret_key,
                session_token,
                None,
                "aws-s3-store",
            );
            loader = loader.credentials_provider(credentials);
        }
        let sdk_config = loader.load().await;
        Ok(Self::with_client(Client::new(&sdk_config), "S3"))
    }

    /// Cloudflare R2 (S3-compatible with path-style addressing).
    ///
    /// # Errors
    ///
    /// Returns an error when the SDK client cannot be constructed.
    pub async fn cloudflare_r2(
        account_id: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
    ) -> Result<Self, crate::errors::DripstoneError> {
        let endpoint_url =
            endpoint.unwrap_or_else(|| format!("https://{account_id}.r2.cloudflarestorage.com"));

        let credentials = AwsCredentials::new(
            &access_key_id,
            &secret_access_key,
            None,
            None,
            "cloudflare-r2",
        );

        let base_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let mut s3_config_builder = S3ConfigBuilder::default()
            .region(Region::new("auto"))
            .endpoint_url(&endpoint_url)
            .force_path_style(true);

        if let Some(bv) = base_config.behavior_version() {
            s3_config_builder = s3_config_builder.behavior_version(bv);
        }
        s3_config_builder = s3_config_builder.credentials_provider(credentials);

        Ok(Self::with_client(
            Client::from_conf(s3_config_builder.build()),
            "R2",
        ))
    }

    /// Generic S3-compatible endpoint (`MinIO`, Wasabi, B2, …).
    ///
    /// # Errors
    ///
    /// Returns an error when the SDK client cannot be constructed.
    pub async fn generic(
        endpoint: String,
        region: String,
        path_style: bool,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Result<Self, crate::errors::DripstoneError> {
        let base_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let mut s3_config_builder = S3ConfigBuilder::default()
            .region(Region::new(region))
            .endpoint_url(&endpoint);

        if let Some(bv) = base_config.behavior_version() {
            s3_config_builder = s3_config_builder.behavior_version(bv);
        }

        if let (Some(access_key), Some(secret_key)) = (access_key_id, secret_access_key) {
            let credentials =
                AwsCredentials::new(&access_key, &secret_key, session_token, None, "generic-s3");
            s3_config_builder = s3_config_builder.credentials_provider(credentials);
        }

        if path_style {
            s3_config_builder = s3_config_builder.force_path_style(true);
        }

        Ok(Self::with_client(
            Client::from_conf(s3_config_builder.build()),
            "S3-compatible storage",
        ))
    }

    fn datetime_to_system_time(dt: &aws_sdk_s3::primitives::DateTime) -> SystemTime {
        let secs = u64::try_from(dt.secs().max(0)).unwrap_or(u64::MAX);
        SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs)
    }

    fn map_object_error<E>(
        &self,
        operation: &str,
        bucket: &str,
        key: &str,
        err: &SdkError<E>,
    ) -> crate::errors::DripstoneError
    where
        E: StdError + std::fmt::Debug + ProvideErrorMetadata + 'static,
    {
        let code = ProvideErrorMetadata::code(err);
        let (status, body) = service_http_detail(err);
        // `SdkError::ServiceError`'s Display is exactly "service error" and
        // drops the inner `NoSuchKey`. ColdForge treats that as a hard storage
        // failure on the first upsert unless we classify the code, source
        // chain, HTTP status, and body.
        let detail = sdk_error_detail(err, &body);
        if s3_response_is_missing_object(code, status, &detail) {
            // Include `NoSuchKey` so string matchers and [`DripstoneError::is_not_found`]
            // both treat a first-upsert missing index as empty.
            return crate::errors::DripstoneError::NotFound(format!("NoSuchKey: {bucket}/{key}"));
        }
        crate::errors::DripstoneError::storage(format!(
            "Failed to {operation} object from {}: {detail}",
            self.provider_label
        ))
    }
}

/// Classify an S3-compatible `GetObject` / `HeadObject` failure.
///
/// `aws-sdk-s3` formats every service failure as `service error`. The modeled
/// code is `NoSuchKey` (GET) or `NotFound` (HEAD). MinIO and Chainguard images
/// often leave that as an unhandled error whose `Display` is still only
/// `service error`, while `.code()`, the source chain, HTTP 404, or the XML
/// body carries `NoSuchKey`. `NoSuchBucket` stays a hard failure so a missing
/// bucket is not an empty index.
pub(crate) fn sdk_error_is_missing_object<E>(err: &SdkError<E>) -> bool
where
    E: StdError + std::fmt::Debug + ProvideErrorMetadata + 'static,
{
    let code = ProvideErrorMetadata::code(err);
    let (status, body) = service_http_detail(err);
    let detail = sdk_error_detail(err, &body);
    s3_response_is_missing_object(code, status, &detail)
}

pub(crate) fn s3_response_is_missing_object(
    code: Option<&str>,
    http_status: Option<u16>,
    detail: &str,
) -> bool {
    if let Some(code) = code {
        if code.eq_ignore_ascii_case("NoSuchBucket") {
            return false;
        }
        if code.eq_ignore_ascii_case("NoSuchKey") || code.eq_ignore_ascii_case("NotFound") {
            return true;
        }
    }
    let lower = detail.to_ascii_lowercase();
    if lower.contains("nosuchkey") || lower.contains("<code>notfound</code>") {
        return true;
    }
    if lower.contains("nosuchbucket") {
        return false;
    }
    http_status == Some(404)
}

fn sdk_error_detail<E>(err: &SdkError<E>, body: &str) -> String
where
    E: StdError + std::fmt::Debug + 'static,
{
    let mut detail = err.to_string();
    let mut current = StdError::source(err);
    let mut depth = 0u8;
    while let Some(source) = current {
        if depth == 4 {
            break;
        }
        detail.push('\n');
        detail.push_str(&source.to_string());
        current = StdError::source(source);
        depth += 1;
    }
    if !body.is_empty() {
        detail.push('\n');
        detail.push_str(body);
    }
    detail.push('\n');
    detail.push_str(&format!("{err:?}"));
    detail
}

fn service_http_detail<E>(err: &SdkError<E>) -> (Option<u16>, String) {
    let SdkError::ServiceError(service) = err else {
        return (None, String::new());
    };
    let status = service.raw().status().as_u16();
    let body = service
        .raw()
        .body()
        .bytes()
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_default();
    (Some(status), body)
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn get_object(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<Bytes, crate::errors::DripstoneError> {
        let response = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| self.map_object_error("get", bucket, key, &e))?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| {
                crate::errors::DripstoneError::storage(format!("Failed to read object body: {e}"))
            })?
            .into_bytes();

        Ok(bytes)
    }

    async fn put_object(
        &self,
        bucket: &str,
        key: &str,
        data: Bytes,
        content_type: Option<&str>,
    ) -> Result<(), crate::errors::DripstoneError> {
        let mut last_error: Option<String> = None;
        for attempt in 0..OBJECT_STORE_PUT_MAX_ATTEMPTS {
            let mut request = self
                .client
                .put_object()
                .bucket(bucket)
                .key(key)
                .body(data.clone().into());

            if let Some(content_type) = content_type {
                request = request.content_type(content_type);
            }

            match request.send().await {
                Ok(_) => return Ok(()),
                Err(error) => {
                    let message =
                        format!("Failed to put object to {}: {error}", self.provider_label);
                    let should_retry = attempt + 1 < OBJECT_STORE_PUT_MAX_ATTEMPTS
                        && is_transient_object_store_error(&message);
                    last_error = Some(message);
                    if should_retry {
                        sleep(object_store_put_backoff(attempt)).await;
                    }
                }
            }
        }

        Err(crate::errors::DripstoneError::storage(
            last_error
                .unwrap_or_else(|| format!("Failed to put object to {}", self.provider_label)),
        ))
    }

    async fn delete_object(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<(), crate::errors::DripstoneError> {
        self.client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                crate::errors::DripstoneError::storage(format!(
                    "Failed to delete object from {}: {e}",
                    self.provider_label
                ))
            })?;

        Ok(())
    }

    async fn head_object(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<ObjectMetadata, crate::errors::DripstoneError> {
        let response = self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| self.map_object_error("head", bucket, key, &e))?;

        Ok(ObjectMetadata {
            size: u64::try_from(response.content_length().unwrap_or(0)).unwrap_or(0),
            content_type: response
                .content_type()
                .map(std::string::ToString::to_string),
            last_modified: response.last_modified().map(Self::datetime_to_system_time),
            etag: response.e_tag().map(std::string::ToString::to_string),
        })
    }

    async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<ObjectInfo>, crate::errors::DripstoneError> {
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

            let response = req.send().await.map_err(|e| {
                crate::errors::DripstoneError::storage(format!(
                    "Failed to list objects from {}: {e}",
                    self.provider_label
                ))
            })?;

            for object in response.contents() {
                objects.push(ObjectInfo {
                    key: object.key().unwrap_or("").to_string(),
                    size: u64::try_from(object.size().unwrap_or(0)).unwrap_or(0),
                    last_modified: object.last_modified().map(Self::datetime_to_system_time),
                    etag: object.e_tag().map(std::string::ToString::to_string),
                });
            }

            if !response.is_truncated.unwrap_or(false) {
                break;
            }

            continuation_token = response
                .next_continuation_token()
                .map(std::string::ToString::to_string);
        }

        Ok(objects)
    }
}

#[cfg(test)]
mod tests {
    use super::{S3Store, s3_response_is_missing_object};

    #[test]
    fn modeled_nosuchkey_is_missing() {
        assert!(s3_response_is_missing_object(
            Some("NoSuchKey"),
            Some(404),
            "unhandled error"
        ));
    }

    #[test]
    fn head_not_found_code_is_missing() {
        assert!(s3_response_is_missing_object(
            Some("NotFound"),
            Some(404),
            ""
        ));
    }

    #[test]
    fn sdk_service_error_display_hides_nosuchkey() {
        // aws-sdk-s3 Display for every service failure is exactly "service error".
        // Wrapping that string alone must not count as not-found.
        assert!(!s3_response_is_missing_object(None, None, "service error"));
        // The code and the source chain (`NoSuchKey: ...`) are what ColdForge needs
        // on an empty index. Redis and Valkey hot tiers hit the same get_object path.
        assert!(s3_response_is_missing_object(
            Some("NoSuchKey"),
            Some(404),
            "service error"
        ));
        assert!(s3_response_is_missing_object(
            None,
            None,
            "service error\nNoSuchKey: The specified key does not exist."
        ));
    }

    #[test]
    fn minio_unhandled_404_without_display_code_is_missing() {
        // Chainguard/MinIO: Display is "unhandled error"; status is 404;
        // NoSuchKey shows up only in the XML body or Debug metadata.
        assert!(s3_response_is_missing_object(
            None,
            Some(404),
            "unhandled error"
        ));
        assert!(s3_response_is_missing_object(
            None,
            None,
            "ServiceError code: Some(\"NoSuchKey\") message: The specified key does not exist."
        ));
    }

    #[test]
    fn missing_bucket_is_not_a_missing_key() {
        assert!(!s3_response_is_missing_object(
            Some("NoSuchBucket"),
            Some(404),
            "<Code>NoSuchBucket</Code>"
        ));
    }

    #[test]
    fn unhandled_404_that_names_the_bucket_is_not_a_missing_key() {
        assert!(!s3_response_is_missing_object(
            None,
            Some(404),
            "<Code>NoSuchBucket</Code>"
        ));
    }

    #[test]
    fn access_denied_is_not_missing() {
        assert!(!s3_response_is_missing_object(
            Some("AccessDenied"),
            Some(403),
            "AccessDenied"
        ));
    }

    #[tokio::test]
    async fn unset_keys_keep_the_default_credential_chain() {
        // `S3ConfigBuilder` from scratch, even after `load_defaults`, drops the
        // credential provider and signs anonymous. `Client::new` keeps the chain
        // (env, SSO profile, web identity, IMDS). This must fail if that wiring
        // is removed and no live AWS call is required to see it.
        let store = S3Store::aws(Some("us-east-1".to_string()), None, None, None)
            .await
            .expect("aws client");
        let rendered = format!("{:?}", store.client.config().to_builder());
        assert!(
            rendered.contains("DefaultCredentialsChain"),
            "unset keys did not install DefaultCredentialsChain; an empty S3ConfigBuilder signs anonymous"
        );
    }
}
