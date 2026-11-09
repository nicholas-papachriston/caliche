//! HTTP transport for web backends, with an injectable mock for tests.

use super::error::WebError;
use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Mutex;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);
const USER_AGENT: &str = "Mozilla/5.0 (compatible; Caliche/1.0)";

/// JSON POST response from a web API.
#[derive(Debug, Clone)]
pub struct WebHttpResponse {
    pub status: u16,
    pub body: Value,
}

/// Outbound HTTP used by search and fetch backends.
#[async_trait]
pub trait WebHttpTransport: Send + Sync {
    async fn post_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &Value,
    ) -> Result<WebHttpResponse, WebError>;

    async fn get_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
    ) -> Result<WebHttpResponse, WebError>;

    async fn get_text(&self, url: &str) -> Result<String, WebError>;
}

/// Production transport: SSRF-checked reqwest client.
#[derive(Debug, Default)]
pub struct ReqwestWebTransport;

impl ReqwestWebTransport {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn client(url: &str) -> Result<reqwest::Client, WebError> {
        let policy = crate::gateway::egress::current_policy()
            .map_err(|err| WebError::BlockedUrl(err.to_string()))?;
        let validated = policy
            .validate(crate::gateway::egress::EgressClass::Tool, url)
            .map_err(|err| WebError::BlockedUrl(err.to_string()))?;
        validated
            .pinned_client_builder()
            .map_err(|err| WebError::message(err.to_string()))?
            .timeout(REQUEST_TIMEOUT)
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| WebError::message(err.to_string()))
    }
}

#[async_trait]
impl WebHttpTransport for ReqwestWebTransport {
    async fn post_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &Value,
    ) -> Result<WebHttpResponse, WebError> {
        let client = Self::client(url)?;
        let mut request = client.post(url).json(body);
        for (name, value) in headers {
            if crate::security::http_policy::is_forbidden_egress_header(name) {
                return Err(WebError::message(format!(
                    "HTTP header not allowed: {name}"
                )));
            }
            request = request.header(*name, *value);
        }
        let response = request
            .send()
            .await
            .map_err(|err| WebError::message(err.to_string()))?;
        let status = response.status().as_u16();
        let parsed = response.json::<Value>().await.unwrap_or(Value::Null);
        Ok(WebHttpResponse {
            status,
            body: parsed,
        })
    }

    async fn get_json(
        &self,
        url: &str,
        headers: &[(&str, &str)],
    ) -> Result<WebHttpResponse, WebError> {
        let client = Self::client(url)?;
        let mut request = client.get(url);
        for (name, value) in headers {
            if crate::security::http_policy::is_forbidden_egress_header(name) {
                return Err(WebError::message(format!(
                    "HTTP header not allowed: {name}"
                )));
            }
            request = request.header(*name, *value);
        }
        let response = request
            .send()
            .await
            .map_err(|err| WebError::message(err.to_string()))?;
        let status = response.status().as_u16();
        let parsed = response.json::<Value>().await.unwrap_or(Value::Null);
        Ok(WebHttpResponse {
            status,
            body: parsed,
        })
    }

    async fn get_text(&self, url: &str) -> Result<String, WebError> {
        crate::agent::tools::common::http::validate_http_url(url)
            .map_err(|err| WebError::BlockedUrl(err.to_string()))?;
        let client = Self::client(url)?;
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|err| WebError::message(err.to_string()))?;
        if !response.status().is_success() {
            return Err(WebError::http(
                "http",
                response.status().as_u16(),
                response.status().to_string(),
            ));
        }
        response
            .text()
            .await
            .map_err(|err| WebError::message(err.to_string()))
    }
}

/// In-memory transport for unit tests. Does not open sockets.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct MockWebTransport {
    posts: Mutex<HashMap<String, WebHttpResponse>>,
    get_jsons: Mutex<HashMap<String, WebHttpResponse>>,
    gets: Mutex<HashMap<String, String>>,
}

#[cfg(test)]
impl MockWebTransport {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_post(self, url: impl Into<String>, status: u16, body: Value) -> Self {
        self.posts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(url.into(), WebHttpResponse { status, body });
        self
    }

    #[must_use]
    pub fn with_get_json(self, url: impl Into<String>, status: u16, body: Value) -> Self {
        self.get_jsons
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(url.into(), WebHttpResponse { status, body });
        self
    }

    #[must_use]
    pub fn with_get(self, url: impl Into<String>, body: impl Into<String>) -> Self {
        self.gets
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .insert(url.into(), body.into());
        self
    }
}

#[cfg(test)]
#[async_trait]
impl WebHttpTransport for MockWebTransport {
    async fn post_json(
        &self,
        url: &str,
        _headers: &[(&str, &str)],
        _body: &Value,
    ) -> Result<WebHttpResponse, WebError> {
        self.posts
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(url)
            .cloned()
            .ok_or_else(|| WebError::message(format!("no mock POST for {url}")))
    }

    async fn get_json(
        &self,
        url: &str,
        _headers: &[(&str, &str)],
    ) -> Result<WebHttpResponse, WebError> {
        self.get_jsons
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(url)
            .cloned()
            .ok_or_else(|| WebError::message(format!("no mock GET JSON for {url}")))
    }

    async fn get_text(&self, url: &str) -> Result<String, WebError> {
        self.gets
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .get(url)
            .cloned()
            .ok_or_else(|| WebError::message(format!("no mock GET for {url}")))
    }
}
