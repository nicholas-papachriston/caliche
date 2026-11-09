//! Schema-defined durable stream backend adapter: implements DurableStreamBackend via HTTP.
//!
//! Maps append, read_after, last_id, delete_stream to endpoint requests/responses.

use crate::plugins::runtime::resolve_path;
use crate::plugins::schema::{
    ConnectionSpec, PluginDefinition, PluginSpec, StreamBackendPluginSpec,
};
use crate::streams::{DurableEvent, DurableStreamBackend, TrimPolicy};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

fn base_url(conn: &ConnectionSpec) -> Option<String> {
    match conn {
        ConnectionSpec::Inline { url } => Some(url.trim_end_matches('/').to_string()),
        ConnectionSpec::Named { .. } => None,
    }
}

#[derive(Debug)]
pub struct SchemaStreamBackendAdapter {
    _definition: Arc<PluginDefinition>,
    spec: StreamBackendPluginSpec,
    #[expect(
        dead_code,
        reason = "constructor retains the shared client; requests use a DNS-pinned client"
    )]
    http_client: Arc<reqwest::Client>,
    base_url: String,
}

impl SchemaStreamBackendAdapter {
    pub fn new(
        definition: Arc<PluginDefinition>,
        http_client: Arc<reqwest::Client>,
    ) -> Option<Self> {
        let spec = match &definition.spec {
            PluginSpec::StreamBackend(s) => s.clone(),
            _ => return None,
        };
        let base_url = base_url(&spec.connection)?;
        Some(Self {
            _definition: definition,
            spec,
            http_client,
            base_url,
        })
    }
}

#[async_trait]
impl DurableStreamBackend for SchemaStreamBackendAdapter {
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String> {
        let ep = match &self.spec.endpoints.append {
            Some(e) => e,
            None => return Err(anyhow!("stream backend append endpoint not configured")),
        };
        let path = ep.path.trim_start_matches('/');
        let url = format!("{}/{}", self.base_url, path);
        let body = serde_json::json!({
            "stream_key": stream_key,
            "event": event,
        });
        let resp = crate::plugins::adapters::pinned_request("POST", &url)
            .context("stream append pin")?
            .json(&body)
            .send()
            .await
            .context("stream append request")?;
        let status = resp.status();
        let text = resp.text().await.context("stream append response body")?;
        if !status.is_success() {
            return Err(anyhow!("stream append error {}: {}", status, text));
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
        let id_path = self.spec.id_path.as_deref().unwrap_or("id");
        let id = resolve_path(&value, id_path)
            .map_err(|e| anyhow!("append response id path: {}", e))?
            .as_str()
            .map(String::from)
            .ok_or_else(|| anyhow!("append response id must be string"))?;
        Ok(id)
    }

    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let ep = match &self.spec.endpoints.read_after {
            Some(e) => e,
            None => return Err(anyhow!("stream backend read_after endpoint not configured")),
        };
        let path = ep.path.trim_start_matches('/');
        let url = format!("{}/{}", self.base_url, path);
        let params: HashMap<&str, String> = [
            ("stream_key", stream_key.to_string()),
            ("after_id", after_id.to_string()),
            ("count", count.to_string()),
        ]
        .into();
        let resp = crate::plugins::adapters::pinned_request("GET", &url)
            .context("stream read_after pin")?
            .query(&params)
            .send()
            .await
            .context("stream read_after request")?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .context("stream read_after response body")?;
        if !status.is_success() {
            return Err(anyhow!("stream read_after error {}: {}", status, text));
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
        let events_path = self.spec.events_path.as_deref().unwrap_or("events");
        let event_id_path = self.spec.event_id_path.as_deref().unwrap_or("id");
        let events_value = resolve_path(&value, events_path)
            .map_err(|e| anyhow!("read_after events path: {}", e))?;
        let arr = events_value
            .as_array()
            .ok_or_else(|| anyhow!("read_after events must be array"))?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let id = resolve_path(item, event_id_path)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            let mut obj = item
                .as_object()
                .ok_or_else(|| anyhow!("read_after event item must be object"))?
                .clone();
            obj.remove(event_id_path);
            let event: DurableEvent = serde_json::from_value(serde_json::Value::Object(obj))
                .context("deserialize DurableEvent")?;
            out.push((id, event));
        }
        Ok(out)
    }

    async fn read_blocking(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let deadline = timeout.map(|d| tokio::time::Instant::now() + d);
        loop {
            let events = self.read_after(stream_key, after_id, count).await?;
            if !events.is_empty() {
                return Ok(events);
            }
            if let Some(dead) = deadline
                && tokio::time::Instant::now() >= dead
            {
                return Ok(Vec::new());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    async fn trim(&self, _stream_key: &str, _policy: TrimPolicy) -> Result<u64> {
        tracing::warn!(
            "schema stream backend does not support trim; remote backend controls retention"
        );
        Ok(0)
    }

    async fn last_id(&self, stream_key: &str) -> Result<Option<String>> {
        let ep = match &self.spec.endpoints.last_id {
            Some(e) => e,
            None => return Err(anyhow!("stream backend last_id endpoint not configured")),
        };
        let path = ep.path.trim_start_matches('/');
        let url = format!("{}/{}", self.base_url, path);
        let params: HashMap<&str, &str> = [("stream_key", stream_key)].into();
        let resp = crate::plugins::adapters::pinned_request("GET", &url)
            .context("stream last_id pin")?
            .query(&params)
            .send()
            .await
            .context("stream last_id request")?;
        let status = resp.status();
        let text = resp.text().await.context("stream last_id response body")?;
        if !status.is_success() {
            return Err(anyhow!("stream last_id error {}: {}", status, text));
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
        let id = value.get("id").and_then(|v| v.as_str().map(String::from));
        Ok(id)
    }

    async fn delete_stream(&self, stream_key: &str) -> Result<()> {
        let ep = match &self.spec.endpoints.delete_stream {
            Some(e) => e,
            None => {
                return Err(anyhow!(
                    "stream backend delete_stream endpoint not configured"
                ));
            }
        };
        let path = ep.path.trim_start_matches('/');
        let url = format!("{}/{}", self.base_url, path);
        let params: HashMap<&str, &str> = [("stream_key", stream_key)].into();
        let resp = crate::plugins::adapters::pinned_request("DELETE", &url)
            .context("stream delete_stream pin")?
            .query(&params)
            .send()
            .await
            .context("stream delete_stream request")?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!("stream delete_stream error: {}", text));
        }
        Ok(())
    }
}
