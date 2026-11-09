//! SigV4 JSON client for DynamoDB (including `SearchVectors`).
//!
//! Uses the DynamoDB JSON protocol (`application/x-amz-json-1.0`) so Dripstone can
//! call native vector APIs without requiring `aws-sdk-dynamodb` 1.120+ (MSRV 1.94).

use aws_config::SdkConfig;
use reqwest::Client;
use serde::Serialize;
use serde_json::Value;

use crate::aws::{
    AwsSdkConfigOpts, load_sdk_config, send_signed_json_value, sign_json_http_request,
};
use crate::errors::DripstoneError;

const API_VERSION: &str = "DynamoDB_20120810";
const SEARCH_VECTORS: &str = "SearchVectors";
const SEARCH_HOST_PREFIX: &str = "search-dynamodb";
const SEARCH_HOST_SUFFIX: &str = "api.aws";

fn resolve_endpoint(
    operation: &str,
    region: &str,
    data_endpoint: &str,
    data_overridden: bool,
    search_endpoint: Option<&str>,
) -> String {
    let is_search = operation == SEARCH_VECTORS || operation.ends_with(".SearchVectors");
    if !is_search {
        return data_endpoint.to_string();
    }
    if let Some(endpoint) = search_endpoint {
        return endpoint.trim_end_matches('/').to_string();
    }
    if data_overridden {
        return data_endpoint.to_string();
    }
    format!("https://{SEARCH_HOST_PREFIX}.{region}.{SEARCH_HOST_SUFFIX}")
}

#[derive(Debug, Clone)]
pub(crate) struct DynamoDbJsonClient {
    http: Client,
    aws_config: SdkConfig,
    region: String,
    endpoint: String,
    /// `Some` when the caller overrode the data-plane endpoint (LocalStack).
    /// Search then uses that same host. Live AWS uses the dedicated search host.
    data_endpoint_overridden: bool,
    search_endpoint: Option<String>,
}

impl DynamoDbJsonClient {
    pub(crate) async fn connect(
        region: Option<&str>,
        endpoint_url: Option<&str>,
        search_endpoint_url: Option<&str>,
        access_key_id: Option<&str>,
        secret_access_key: Option<&str>,
        session_token: Option<&str>,
    ) -> Result<Self, DripstoneError> {
        let region = region.unwrap_or("us-east-1").to_string();
        let aws_config = load_sdk_config(AwsSdkConfigOpts {
            region: Some(region.as_str()),
            endpoint_url,
            access_key_id,
            secret_access_key,
            session_token,
            credentials_provider_name: "dynamodb-vector-backend",
        })
        .await?;
        let endpoint = endpoint_url
            .map(str::to_string)
            .unwrap_or_else(|| format!("https://dynamodb.{region}.amazonaws.com"));
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| DripstoneError::storage(format!("Failed to create HTTP client: {e}")))?;
        Ok(Self {
            http,
            aws_config,
            region,
            endpoint,
            data_endpoint_overridden: endpoint_url.is_some(),
            search_endpoint: search_endpoint_url.map(str::to_string),
        })
    }

    fn endpoint_for(&self, operation: &str) -> String {
        resolve_endpoint(
            operation,
            &self.region,
            &self.endpoint,
            self.data_endpoint_overridden,
            self.search_endpoint.as_deref(),
        )
    }

    pub(crate) async fn call<B: Serialize + Sync>(
        &self,
        operation: &str,
        body: &B,
    ) -> Result<Value, DripstoneError> {
        let body_json = serde_json::to_string(body).map_err(|e| {
            DripstoneError::storage(format!("Failed to serialize DynamoDB request: {e}"))
        })?;
        self.send(operation, &body_json).await
    }

    async fn send(&self, operation: &str, body_json: &str) -> Result<Value, DripstoneError> {
        let target = if operation.contains('.') {
            operation.to_string()
        } else {
            format!("{API_VERSION}.{operation}")
        };
        let uri = self.endpoint_for(operation);
        let signed = sign_json_http_request(
            "POST",
            &uri,
            &self.region,
            "dynamodb",
            &self.aws_config,
            &[
                ("content-type", "application/x-amz-json-1.0"),
                ("x-amz-target", target.as_str()),
            ],
            body_json,
        )
        .await?;
        send_signed_json_value(&self.http, signed).await
    }
}

#[cfg(test)]
mod endpoint_tests {
    use super::resolve_endpoint;

    #[test]
    fn search_uses_the_dedicated_host_unless_overridden() {
        let data = "https://dynamodb.us-east-1.amazonaws.com";
        let search = resolve_endpoint("SearchVectors", "us-east-1", data, false, None);
        assert_eq!(search, "https://search-dynamodb.us-east-1.api.aws");
        let get = resolve_endpoint("GetItem", "us-east-1", data, false, None);
        assert_eq!(get, data);
        let local = "http://localhost:4566";
        let local_search = resolve_endpoint("SearchVectors", "us-east-1", local, true, None);
        assert_eq!(local_search, local);
        let custom = resolve_endpoint(
            "DynamoDB_20120810.SearchVectors",
            "us-east-1",
            data,
            false,
            Some("https://search.example/"),
        );
        assert_eq!(custom, "https://search.example");
    }
}
