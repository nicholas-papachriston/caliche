//! Shared SigV4 JSON HTTP signing for DynamoDB and S3 Vectors.
//!
//! Keeps protocol-specific headers in the call site while centralizing
//! credential loading, request signing, and response handling.

use std::time::SystemTime;

use aws_config::{BehaviorVersion, SdkConfig};
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use serde::de::DeserializeOwned;

use crate::errors::DripstoneError;

/// Options for loading an [`SdkConfig`] with optional static credentials.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AwsSdkConfigOpts<'a> {
    pub region: Option<&'a str>,
    pub endpoint_url: Option<&'a str>,
    pub access_key_id: Option<&'a str>,
    pub secret_access_key: Option<&'a str>,
    pub session_token: Option<&'a str>,
    pub credentials_provider_name: &'static str,
}

/// A fully signed HTTP request ready for `reqwest`.
#[derive(Debug, Clone)]
pub(crate) struct SignedHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

/// Load SDK config, honoring explicit static credentials when both key halves are set.
pub(crate) async fn load_sdk_config(
    opts: AwsSdkConfigOpts<'_>,
) -> Result<SdkConfig, DripstoneError> {
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    let region = opts.region.unwrap_or("us-east-1");
    loader = loader.region(aws_config::Region::new(region.to_string()));
    if let Some(endpoint) = opts.endpoint_url {
        loader = loader.endpoint_url(endpoint);
    }
    if let (Some(access_key), Some(secret_key)) = (opts.access_key_id, opts.secret_access_key) {
        let credentials = aws_credential_types::Credentials::new(
            access_key,
            secret_key,
            opts.session_token.map(str::to_string),
            None,
            opts.credentials_provider_name,
        );
        loader = loader.credentials_provider(credentials);
    }
    Ok(loader.load().await)
}

/// Sign a JSON HTTP request with SigV4.
pub(crate) async fn sign_json_http_request(
    method: &str,
    url: &str,
    region: &str,
    service: &str,
    aws_config: &SdkConfig,
    extra_headers: &[(&str, &str)],
    body_json: &str,
) -> Result<SignedHttpRequest, DripstoneError> {
    let parsed_url = url
        .parse::<http::Uri>()
        .map_err(|e| DripstoneError::storage(format!("Invalid URL: {e}")))?;
    let host = parsed_url
        .host()
        .ok_or_else(|| DripstoneError::storage(format!("URL missing host: {url}")))?
        .to_string();

    let credentials = aws_config
        .credentials_provider()
        .ok_or_else(|| DripstoneError::storage("No credentials provider available"))?
        .provide_credentials()
        .await
        .map_err(|e| DripstoneError::storage(format!("Failed to get credentials: {e}")))?;

    let signing_settings = SigningSettings::default();
    let identity = credentials.into();
    let signing_params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(service)
        .time(SystemTime::now())
        .settings(signing_settings)
        .build()
        .map_err(|e| DripstoneError::storage(format!("Failed to build signing params: {e}")))?
        .into();

    let mut headers_for_signing: Vec<(&str, &str)> = Vec::with_capacity(2 + extra_headers.len());
    headers_for_signing.push(("host", host.as_str()));
    for (name, value) in extra_headers {
        headers_for_signing.push((name, value));
    }

    let uri_str = parsed_url.to_string();
    let signable_request = SignableRequest::new(
        method,
        &uri_str,
        headers_for_signing.into_iter(),
        SignableBody::Bytes(body_json.as_bytes()),
    )
    .map_err(|e| DripstoneError::storage(format!("Failed to create signable request: {e}")))?;

    let (signing_instructions, _signature) = sign(signable_request, &signing_params)
        .map_err(|e| DripstoneError::storage(format!("Failed to sign request: {e}")))?
        .into_parts();

    let mut http_request = http::Request::builder()
        .method(method)
        .uri(&parsed_url)
        .header("Host", &host);
    for (name, value) in extra_headers {
        http_request = http_request.header(*name, *value);
    }
    let mut http_request = http_request
        .body(body_json.to_string())
        .map_err(|e| DripstoneError::storage(format!("Failed to build request: {e}")))?;
    signing_instructions.apply_to_request_http1x(&mut http_request);

    let headers = http_request
        .headers()
        .iter()
        .map(|(name, value)| {
            (
                name.as_str().to_string(),
                value.to_str().unwrap_or_default().to_string(),
            )
        })
        .collect();

    Ok(SignedHttpRequest {
        method: method.to_string(),
        url: url.to_string(),
        headers,
        body: body_json.to_string(),
    })
}

/// Send a signed JSON request and optionally deserialize the body.
///
/// When `empty_ok` is true, an empty successful body returns `Ok(None)`.
/// When false, an empty successful body is treated as `{}` for object deserialization
/// by returning `Ok(None)` and letting the caller substitute an empty object, or
/// the caller can use [`send_signed_json_value`].
pub(crate) async fn send_signed_json<R: DeserializeOwned>(
    http: &reqwest::Client,
    signed: SignedHttpRequest,
    empty_ok: bool,
) -> Result<Option<R>, DripstoneError> {
    let method = reqwest::Method::from_bytes(signed.method.as_bytes())
        .map_err(|e| DripstoneError::storage(format!("Invalid method: {e}")))?;
    let mut reqwest_request = http.request(method, &signed.url).body(signed.body);
    for (name, value) in &signed.headers {
        reqwest_request = reqwest_request.header(name, value);
    }

    let response = reqwest_request.send().await.map_err(|e| {
        DripstoneError::storage(format!(
            "Signed HTTP request failed for {}: {e}",
            signed.url
        ))
    })?;
    let status = response.status();
    let response_text = response
        .text()
        .await
        .map_err(|e| DripstoneError::storage(format!("Failed to read response: {e}")))?;
    if !status.is_success() {
        return Err(DripstoneError::storage(format!(
            "AWS API error ({status}) for {}: {response_text}",
            signed.url
        )));
    }
    if response_text.trim().is_empty() {
        return if empty_ok {
            Ok(None)
        } else {
            Err(DripstoneError::storage(format!(
                "Unexpected empty response for {}",
                signed.url
            )))
        };
    }
    let parsed = serde_json::from_str(&response_text)
        .map_err(|e| DripstoneError::storage(format!("Failed to parse response: {e}")))?;
    Ok(Some(parsed))
}

/// Like [`send_signed_json`], but empty bodies become `serde_json::Value::Object(Default)`.
#[cfg(feature = "dynamodb-backend")]
pub(crate) async fn send_signed_json_value(
    http: &reqwest::Client,
    signed: SignedHttpRequest,
) -> Result<serde_json::Value, DripstoneError> {
    match send_signed_json::<serde_json::Value>(http, signed, true).await? {
        Some(value) => Ok(value),
        None => Ok(serde_json::Value::Object(serde_json::Map::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::{AwsSdkConfigOpts, load_sdk_config, sign_json_http_request};

    #[tokio::test]
    async fn load_sdk_config_honors_static_credentials() {
        use aws_credential_types::provider::ProvideCredentials;

        let cfg = load_sdk_config(AwsSdkConfigOpts {
            region: Some("us-west-2"),
            endpoint_url: None,
            access_key_id: Some(concat!("AK", "IAEXAMPLE")),
            secret_access_key: Some("secret"),
            session_token: Some("token"),
            credentials_provider_name: "sigv4-http-test",
        })
        .await
        .expect("load");
        assert_eq!(
            cfg.region().map(aws_config::Region::as_ref),
            Some("us-west-2")
        );
        let creds = cfg
            .credentials_provider()
            .expect("provider")
            .provide_credentials()
            .await
            .expect("creds");
        assert_eq!(creds.access_key_id(), concat!("AK", "IAEXAMPLE"));
        assert_eq!(creds.secret_access_key(), "secret");
        assert_eq!(creds.session_token(), Some("token"));
    }

    #[tokio::test]
    async fn sign_json_http_request_sets_authorization() {
        use aws_credential_types::provider::ProvideCredentials;

        let cfg = load_sdk_config(AwsSdkConfigOpts {
            region: Some("us-east-1"),
            endpoint_url: None,
            access_key_id: Some(concat!("AK", "IAEXAMPLE")),
            secret_access_key: Some("secret"),
            session_token: None,
            credentials_provider_name: "sigv4-http-test",
        })
        .await
        .expect("load");
        // Touch provider so signing uses cached identity.
        let _ = cfg
            .credentials_provider()
            .expect("provider")
            .provide_credentials()
            .await
            .expect("creds");

        let signed = sign_json_http_request(
            "POST",
            "https://dynamodb.us-east-1.amazonaws.com",
            "us-east-1",
            "dynamodb",
            &cfg,
            &[
                ("content-type", "application/x-amz-json-1.0"),
                ("x-amz-target", "DynamoDB_20120810.DescribeTable"),
            ],
            r#"{"TableName":"t"}"#,
        )
        .await
        .expect("sign");

        assert!(
            signed
                .headers
                .iter()
                .any(|(k, v)| k.eq_ignore_ascii_case("authorization")
                    && v.contains("AWS4-HMAC-SHA256")),
            "missing SigV4 Authorization header: {:?}",
            signed.headers
        );
    }
}
