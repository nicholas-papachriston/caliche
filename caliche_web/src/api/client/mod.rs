//! HTTP API client with SSE streaming support.

mod catalog;
mod chat;
mod documents;
mod fleet;
mod harness;
mod orchestration;
mod plugins;
mod policy;
mod session;
mod tunnel;

use anyhow::{Context, Result};
use reqwest::Client;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::types::StreamEventEnvelopeDto;
use crate::api::types::StreamingEvent;

/// API client for communicating with caliche_server
#[derive(Clone)]
pub struct ApiClient {
    pub(in crate::api::client) client: Client,
    pub(in crate::api::client) base_url: String,
    pub(in crate::api::client) rpc_id: Arc<AtomicU64>,
}

// Manual PartialEq - compare by base_url since reqwest::Client doesn't implement PartialEq
impl PartialEq for ApiClient {
    fn eq(&self, other: &Self) -> bool {
        self.base_url == other.base_url
    }
}

fn stream_event_from_envelope_value_impl(value: serde_json::Value) -> Result<StreamingEvent> {
    let envelope: StreamEventEnvelopeDto =
        serde_json::from_value(value).context("failed to parse stream event envelope")?;
    Ok(envelope.payload)
}

fn stream_event_from_envelope_str_impl(data: &str) -> Result<StreamingEvent> {
    let envelope: StreamEventEnvelopeDto =
        serde_json::from_str(data).context("failed to parse stream event envelope")?;
    Ok(envelope.payload)
}

fn done_stream_event_impl() -> StreamingEvent {
    StreamingEvent::done()
}

fn error_stream_event_impl(message: impl Into<String>) -> StreamingEvent {
    StreamingEvent::error(message)
}

/// Stable `HTTP <status> <body>` text so panes can tell a rejected token from other failures.
pub fn http_error_text(status: u16, body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        format!("HTTP {status}")
    } else {
        format!("HTTP {status} {body}")
    }
}

fn format_rpc_error_impl(method: &str, error: &serde_json::Value) -> String {
    let message = error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("RPC request failed");

    let lower = message.to_ascii_lowercase();
    if lower.contains("openai") && lower.contains("insufficient_quota") {
        return "OpenAI quota exceeded. Check your OpenAI plan/billing, update the API key, or switch to another provider/model.".to_string();
    }
    if lower.contains("openai") && (lower.contains("429") || lower.contains("rate limit")) {
        return "OpenAI rate limit exceeded. Retry later, use a lower-throughput model, or switch provider/model.".to_string();
    }

    if error.get("code").and_then(serde_json::Value::as_i64) == Some(-32601) {
        return format!("{method} failed: {message} (code -32601)");
    }

    format!("{method} failed: {message}")
}

#[cfg(any(test, feature = "test-utils"))]
pub fn stream_event_from_envelope_value(value: serde_json::Value) -> Result<StreamingEvent> {
    stream_event_from_envelope_value_impl(value)
}

#[cfg(not(any(test, feature = "test-utils")))]
pub(in crate::api::client) fn stream_event_from_envelope_value(
    value: serde_json::Value,
) -> Result<StreamingEvent> {
    stream_event_from_envelope_value_impl(value)
}

#[cfg(any(test, feature = "test-utils"))]
pub fn stream_event_from_envelope_str(data: &str) -> Result<StreamingEvent> {
    stream_event_from_envelope_str_impl(data)
}

#[cfg(not(any(test, feature = "test-utils")))]
pub(in crate::api::client) fn stream_event_from_envelope_str(data: &str) -> Result<StreamingEvent> {
    stream_event_from_envelope_str_impl(data)
}

#[cfg(any(test, feature = "test-utils"))]
pub fn done_stream_event() -> StreamingEvent {
    done_stream_event_impl()
}

#[cfg(not(any(test, feature = "test-utils")))]
pub(in crate::api::client) fn done_stream_event() -> StreamingEvent {
    done_stream_event_impl()
}

#[cfg(any(test, feature = "test-utils"))]
pub fn error_stream_event(message: impl Into<String>) -> StreamingEvent {
    error_stream_event_impl(message)
}

#[cfg(not(any(test, feature = "test-utils")))]
pub(in crate::api::client) fn error_stream_event(message: impl Into<String>) -> StreamingEvent {
    error_stream_event_impl(message)
}

#[cfg(any(test, feature = "test-utils"))]
pub fn format_rpc_error(method: &str, error: &serde_json::Value) -> String {
    format_rpc_error_impl(method, error)
}

#[cfg(not(any(test, feature = "test-utils")))]
pub(in crate::api::client) fn format_rpc_error(method: &str, error: &serde_json::Value) -> String {
    format_rpc_error_impl(method, error)
}

impl ApiClient {
    /// Create a new API client
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            rpc_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Check if the API server is healthy
    pub async fn health_check(&self) -> Result<bool> {
        let response = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await;

        match response {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    pub(in crate::api::client) async fn rpc_call<P, R>(&self, method: &str, params: &P) -> Result<R>
    where
        P: serde::Serialize + ?Sized,
        R: serde::de::DeserializeOwned,
    {
        let id = self.rpc_id.fetch_add(1, Ordering::Relaxed);
        let payload = serde_json::json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "method": method,
            "params": params,
            "id": id
        });

        let response = self
            .client
            .post(format!("{}/rpc", self.base_url))
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("Failed RPC call: {}", method))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "RPC transport failed for {method}: {}",
                http_error_text(status, &body)
            );
        }

        let raw: serde_json::Value = response
            .json()
            .await
            .with_context(|| format!("Failed to decode RPC response: {}", method))?;

        if let Some(error) = raw.get("error") {
            anyhow::bail!("{}", format_rpc_error(method, error));
        }

        let result = raw
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing RPC result field for method {}", method))?;
        serde_json::from_value(result)
            .with_context(|| format!("Failed to parse RPC result for {}", method))
    }

    /// True when `method` is registered. `-32601` is unavailable. Any other
    /// JSON-RPC error means the method exists (for example invalid params).
    pub async fn rpc_method_available(&self, method: &str) -> Result<bool> {
        let id = self.rpc_id.fetch_add(1, Ordering::Relaxed);
        let payload = serde_json::json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "method": method,
            "params": {},
            "id": id
        });
        let response = self
            .client
            .post(format!("{}/rpc", self.base_url))
            .json(&payload)
            .send()
            .await
            .with_context(|| format!("request failed: {method}"))?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{}", http_error_text(status, &body));
        }
        let raw: serde_json::Value = response
            .json()
            .await
            .with_context(|| format!("request failed: {method}"))?;
        let unavailable = raw
            .get("error")
            .and_then(|error| error.get("code"))
            .and_then(serde_json::Value::as_i64)
            == Some(-32601);
        Ok(!unavailable)
    }

    pub(in crate::api::client) fn is_method_not_found_for(
        err: &anyhow::Error,
        method: &str,
    ) -> bool {
        let msg = err.to_string();
        msg.contains(method) && (msg.contains("(code -32601)") || msg.contains("\"code\":-32601"))
    }
}

#[cfg(test)]
mod method_not_found_tests {
    use super::ApiClient;

    #[test]
    fn formatted_method_not_found_is_detected() {
        let err =
            anyhow::anyhow!("documents.listCollections failed: Method not found (code -32601)");
        assert!(ApiClient::is_method_not_found_for(
            &err,
            "documents.listCollections"
        ));
        let other =
            anyhow::anyhow!("documents.listCollections failed: Invalid params (code -32602)");
        assert!(!ApiClient::is_method_not_found_for(
            &other,
            "documents.listCollections"
        ));
    }
}

impl Default for ApiClient {
    fn default() -> Self {
        Self::new("http://localhost:3000")
    }
}
