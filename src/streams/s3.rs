//! S3 backend for durable event streams.
//!
//! Stores one JSON object per event. Key = `{prefix}/{stream_key}/{event_id}.json`.
//! Event IDs are timestamp-nanos for ordering. Uses aws-sdk-s3.

use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::types::{DurableEvent, STREAM_START, TrimPolicy};
use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_sdk_s3::primitives::ByteStream;
use std::time::Duration;

/// Configuration for the S3 durable stream backend.
#[derive(Clone, Debug)]
pub struct S3DurableStreamConfig {
    pub bucket: String,
    pub prefix: String,
    pub region: Option<String>,
}

/// S3-backed durable stream backend.
///
/// One object per event. Event ID = timestamp_nanos (string) for monotonic ordering.
#[derive(Clone, Debug)]
pub struct S3DurableStreamBackend {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
}

impl S3DurableStreamBackend {
    /// Create a backend using default AWS config (env / instance profile).
    pub async fn new(config: S3DurableStreamConfig) -> Result<Self> {
        let mut config_builder = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = config.region.as_deref() {
            config_builder = config_builder.region(aws_config::Region::new(region.to_string()));
        }
        let cfg = config_builder.load().await;
        let client = aws_sdk_s3::Client::new(&cfg);
        Ok(Self {
            client,
            bucket: config.bucket,
            prefix: config.prefix,
        })
    }

    fn key(&self, stream_key: &str, event_id: &str) -> String {
        format!("{}/{}/{}.json", self.prefix, stream_key, event_id)
    }

    fn list_prefix(&self, stream_key: &str) -> String {
        format!("{}/{}/", self.prefix, stream_key)
    }

    /// Parse event id from key "prefix/stream_key/id.json".
    fn key_to_id(&self, stream_key: &str, key: &str) -> Option<String> {
        let prefix = self.list_prefix(stream_key);
        key.strip_prefix(&prefix)
            .and_then(|s| s.strip_suffix(".json"))
            .map(|s| s.to_string())
    }
}

#[async_trait]
impl DurableStreamBackend for S3DurableStreamBackend {
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String> {
        let id = chrono::Utc::now()
            .timestamp_nanos_opt()
            .map(|n| n.to_string())
            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis().to_string());
        let json = serde_json::to_vec(event).context("Serialize DurableEvent")?;
        let key = self.key(stream_key, &id);
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .body(ByteStream::from(json))
            .send()
            .await
            .context("S3 PutObject")?;
        Ok(id)
    }

    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let prefix = self.list_prefix(stream_key);
        let mut keys = Vec::new();
        let mut continuation_token: Option<String> = None;
        loop {
            let mut list = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix);
            if let Some(ref tok) = continuation_token {
                list = list.continuation_token(tok);
            }
            let out = list.send().await.context("S3 ListObjectsV2")?;
            for obj in out.contents().iter().flat_map(|o| o.key()) {
                if let Some(id) = self.key_to_id(stream_key, obj) {
                    keys.push((id, obj.to_string()));
                }
            }
            continuation_token = out.next_continuation_token().map(String::from);
            if continuation_token.is_none() {
                break;
            }
        }
        keys.sort_by(|a, b| a.0.cmp(&b.0));
        let start_ok = after_id == STREAM_START;
        let filtered: Vec<_> = keys
            .into_iter()
            .filter(|(id, _)| start_ok || id.as_str() > after_id)
            .take(count)
            .collect();
        let mut results = Vec::with_capacity(filtered.len());
        for (id, key) in filtered {
            let body = self
                .client
                .get_object()
                .bucket(&self.bucket)
                .key(&key)
                .send()
                .await
                .context("S3 GetObject")?
                .body
                .collect()
                .await
                .context("S3 body collect")?
                .into_bytes();
            let ev: DurableEvent =
                serde_json::from_slice(&body).context("Deserialize DurableEvent")?;
            results.push((id, ev));
        }
        Ok(results)
    }

    async fn read_blocking(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let deadline = timeout.map(|d| std::time::Instant::now() + d);
        let poll_interval = Duration::from_millis(100);
        loop {
            let out = self.read_after(stream_key, after_id, count).await?;
            if !out.is_empty() {
                return Ok(out);
            }
            if let Some(d) = deadline {
                if std::time::Instant::now() >= d {
                    return Ok(Vec::new());
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn trim(&self, stream_key: &str, policy: TrimPolicy) -> Result<u64> {
        let prefix = self.list_prefix(stream_key);
        let mut keys_with_id: Vec<(String, String)> = Vec::new();
        let mut continuation_token: Option<String> = None;
        loop {
            let mut list = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix);
            if let Some(ref tok) = continuation_token {
                list = list.continuation_token(tok);
            }
            let out = list.send().await.context("S3 ListObjectsV2")?;
            for obj in out.contents().iter().flat_map(|o| o.key()) {
                if let Some(id) = self.key_to_id(stream_key, obj) {
                    keys_with_id.push((id, obj.to_string()));
                }
            }
            continuation_token = out.next_continuation_token().map(String::from);
            if continuation_token.is_none() {
                break;
            }
        }
        keys_with_id.sort_by(|a, b| a.0.cmp(&b.0));
        let to_delete: Vec<String> = match policy {
            TrimPolicy::MaxLen(n) => {
                let len = keys_with_id.len();
                if len <= n {
                    Vec::new()
                } else {
                    keys_with_id
                        .into_iter()
                        .take(len - n)
                        .map(|(_, k)| k)
                        .collect()
                }
            }
            TrimPolicy::MinId(min_id) => keys_with_id
                .into_iter()
                .filter(|(id, _)| id.as_str() < min_id.as_str())
                .map(|(_, k)| k)
                .collect(),
            TrimPolicy::MaxAgeSecs(secs) => {
                let cutoff_nanos = chrono::Utc::now()
                    .timestamp_nanos_opt()
                    .map(|t| t.saturating_sub(secs as i64 * 1_000_000_000))
                    .unwrap_or(0);
                let cutoff_str = cutoff_nanos.to_string();
                keys_with_id
                    .into_iter()
                    .filter(|(id, _)| id.as_str() < cutoff_str.as_str())
                    .map(|(_, k)| k)
                    .collect()
            }
        };
        let mut deleted = 0u64;
        for chunk in to_delete.chunks(1000) {
            let objects: Vec<_> = chunk
                .iter()
                .map(|k| {
                    aws_sdk_s3::types::ObjectIdentifier::builder()
                        .key(k)
                        .build()
                        .context("Build ObjectIdentifier")
                })
                .collect::<Result<Vec<_>>>()?;
            let delete = aws_sdk_s3::types::Delete::builder()
                .set_objects(Some(objects))
                .build()
                .context("Build Delete")?;
            self.client
                .delete_objects()
                .bucket(&self.bucket)
                .delete(delete)
                .send()
                .await
                .context("S3 DeleteObjects")?;
            deleted += chunk.len() as u64;
        }
        Ok(deleted)
    }

    async fn last_id(&self, stream_key: &str) -> Result<Option<String>> {
        let mut keys = Vec::new();
        let mut continuation_token: Option<String> = None;
        let prefix = self.list_prefix(stream_key);
        loop {
            let mut list = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix);
            if let Some(ref tok) = continuation_token {
                list = list.continuation_token(tok);
            }
            let out = list.send().await.context("S3 ListObjectsV2")?;
            for obj in out.contents().iter().flat_map(|o| o.key()) {
                if let Some(id) = self.key_to_id(stream_key, obj) {
                    keys.push(id);
                }
            }
            continuation_token = out.next_continuation_token().map(String::from);
            if continuation_token.is_none() {
                break;
            }
        }
        keys.sort();
        Ok(keys.into_iter().next_back())
    }

    async fn delete_stream(&self, stream_key: &str) -> Result<()> {
        let prefix = self.list_prefix(stream_key);
        let mut continuation_token: Option<String> = None;
        loop {
            let mut list = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix);
            if let Some(ref tok) = continuation_token {
                list = list.continuation_token(tok);
            }
            let out = list.send().await.context("S3 ListObjectsV2")?;
            let keys: Vec<String> = out
                .contents()
                .iter()
                .filter_map(|o| o.key().map(String::from))
                .collect();
            if keys.is_empty() {
                break;
            }
            let objects: Vec<_> = keys
                .iter()
                .map(|k| {
                    aws_sdk_s3::types::ObjectIdentifier::builder()
                        .key(k)
                        .build()
                        .context("Build ObjectIdentifier")
                })
                .collect::<Result<Vec<_>>>()?;
            let delete = aws_sdk_s3::types::Delete::builder()
                .set_objects(Some(objects))
                .build()
                .context("Build Delete")?;
            self.client
                .delete_objects()
                .bucket(&self.bucket)
                .delete(delete)
                .send()
                .await
                .context("S3 DeleteObjects")?;
            continuation_token = out.next_continuation_token().map(String::from);
            if continuation_token.is_none() {
                break;
            }
        }
        Ok(())
    }
}
