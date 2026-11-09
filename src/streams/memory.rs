//! In-memory durable stream backend using a ring buffer per stream.
//!
//! Uses synthetic monotonic IDs (`{timestamp_ms}-{seq}`) and a configurable
//! capacity with automatic eviction. No external dependencies.

use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::types::{DurableEvent, STREAM_START, TrimPolicy};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{Notify, RwLock};

/// In-memory stream state: ring buffer and notifier for blocking reads.
struct StreamState {
    buf: RwLock<VecDeque<(String, DurableEvent)>>,
    next_seq: AtomicU64,
    notify: std::sync::Arc<Notify>,
    max_len: usize,
}

impl std::fmt::Debug for StreamState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamState")
            .field("next_seq", &self.next_seq.load(Ordering::Relaxed))
            .field("max_len", &self.max_len)
            .finish()
    }
}

/// In-memory durable stream backend.
///
/// Each stream key has its own ring buffer. Event IDs are `{timestamp_ms}-{seq}`.
#[derive(Debug)]
pub struct MemoryDurableStreamBackend {
    streams: RwLock<std::collections::HashMap<String, std::sync::Arc<StreamState>>>,
    default_max_len: usize,
}

impl MemoryDurableStreamBackend {
    /// Create a backend with the given default max events per stream.
    pub fn new(max_len: usize) -> Self {
        Self {
            streams: RwLock::new(std::collections::HashMap::new()),
            default_max_len: max_len,
        }
    }

    async fn stream_state(&self, stream_key: &str) -> std::sync::Arc<StreamState> {
        let mut streams = self.streams.write().await;
        streams
            .entry(stream_key.to_string())
            .or_insert_with(|| {
                std::sync::Arc::new(StreamState {
                    buf: RwLock::new(VecDeque::new()),
                    next_seq: AtomicU64::new(0),
                    notify: std::sync::Arc::new(Notify::new()),
                    max_len: self.default_max_len,
                })
            })
            .clone()
    }

    fn next_id(&self, state: &StreamState) -> String {
        let ts = chrono::Utc::now().timestamp_millis() as u64;
        let seq = state.next_seq.fetch_add(1, Ordering::Relaxed);
        format!("{}-{}", ts, seq)
    }
}

#[async_trait]
impl DurableStreamBackend for MemoryDurableStreamBackend {
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String> {
        let state = self.stream_state(stream_key).await;
        let id = self.next_id(&state);
        let event = event.clone();
        {
            let mut buf = state.buf.write().await;
            while buf.len() >= state.max_len {
                buf.pop_front();
            }
            buf.push_back((id.clone(), event));
        }
        state.notify.notify_waiters();
        Ok(id)
    }

    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let state = self.stream_state(stream_key).await;
        let buf = state.buf.read().await;
        let mut out = Vec::with_capacity(count.min(buf.len()));
        let mut seen = after_id == STREAM_START;
        for (id, ev) in buf.iter() {
            if seen {
                out.push((id.clone(), ev.clone()));
                if out.len() >= count {
                    break;
                }
            } else if *id == after_id {
                seen = true;
            }
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
        let state = self.stream_state(stream_key).await;
        loop {
            {
                let buf = state.buf.read().await;
                let mut out = Vec::new();
                let mut seen = after_id == STREAM_START;
                for (id, ev) in buf.iter() {
                    if seen {
                        out.push((id.clone(), ev.clone()));
                        if out.len() >= count {
                            break;
                        }
                    } else if *id == after_id {
                        seen = true;
                    }
                }
                if !out.is_empty() {
                    return Ok(out);
                }
            }
            let n = state.notify.clone();
            match timeout {
                Some(d) => {
                    tokio::time::timeout(d, n.notified()).await.ok();
                }
                None => {
                    n.notified().await;
                }
            }
        }
    }

    async fn trim(&self, stream_key: &str, policy: TrimPolicy) -> Result<u64> {
        let state = self.stream_state(stream_key).await;
        let mut buf = state.buf.write().await;
        let before = buf.len();
        match policy {
            TrimPolicy::MaxLen(n) => {
                while buf.len() > n {
                    buf.pop_front();
                }
            }
            TrimPolicy::MinId(min_id) => {
                while let Some((id, _)) = buf.front() {
                    if *id >= min_id {
                        break;
                    }
                    buf.pop_front();
                }
            }
            TrimPolicy::MaxAgeSecs(secs) => {
                let cutoff = chrono::Utc::now() - chrono::Duration::seconds(secs as i64);
                while let Some((_, ev)) = buf.front() {
                    if ev.timestamp > cutoff {
                        break;
                    }
                    buf.pop_front();
                }
            }
        }
        let removed = before.saturating_sub(buf.len()) as u64;
        Ok(removed)
    }

    async fn last_id(&self, stream_key: &str) -> Result<Option<String>> {
        let state = self.stream_state(stream_key).await;
        let buf = state.buf.read().await;
        Ok(buf.back().map(|(id, _)| id.clone()))
    }

    async fn delete_stream(&self, stream_key: &str) -> Result<()> {
        let mut streams = self.streams.write().await;
        streams.remove(stream_key);
        Ok(())
    }
}
