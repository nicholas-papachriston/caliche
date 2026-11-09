//! Reader for replaying and tailing durable event streams.

use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::types::DurableEvent;
use anyhow::Result;
use async_stream::stream;
use std::sync::Arc;
use std::time::Duration;

const REPLAY_BATCH: usize = 100;

/// Reads events from a durable stream (replay and blocking tail).
#[derive(Clone, Debug)]
pub struct DurableStreamReader {
    backend: Arc<dyn DurableStreamBackend>,
    stream_key: String,
}

impl DurableStreamReader {
    /// Create a reader for the given backend and stream key.
    pub fn new(backend: Arc<dyn DurableStreamBackend>, stream_key: impl Into<String>) -> Self {
        Self {
            backend,
            stream_key: stream_key.into(),
        }
    }

    /// Read up to `count` events after `after_id`.
    pub async fn read_after(
        &self,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        self.backend
            .read_after(&self.stream_key, after_id, count)
            .await
    }

    /// Block until new events arrive, or until `timeout` elapses.
    pub async fn read_blocking(
        &self,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>> {
        self.backend
            .read_blocking(&self.stream_key, after_id, count, timeout)
            .await
    }

    /// Return the last event ID in the stream, if any.
    pub async fn last_id(&self) -> Result<Option<String>> {
        self.backend.last_id(&self.stream_key).await
    }

    /// Stream all events from `from_id` to the current end of the stream (replay only).
    /// Use [STREAM_START] for `from_id` to replay from the beginning.
    pub fn replay_stream(
        self,
        from_id: String,
    ) -> impl futures_util::Stream<Item = Result<(String, DurableEvent)>> + Send + 'static {
        let backend = self.backend;
        let stream_key = self.stream_key;
        stream! {
            let mut cursor = from_id;
            loop {
                let batch = backend.read_after(&stream_key, &cursor, REPLAY_BATCH).await;
                let batch = match batch {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };
                if batch.is_empty() {
                    break;
                }
                let n = batch.len();
                for (id, ev) in batch {
                    cursor = id.clone();
                    yield Ok((id, ev));
                }
                if n < REPLAY_BATCH {
                    break;
                }
            }
        }
    }

    /// Stream events from `from_id`: first replay all existing events, then block and yield new ones (replay + live tail).
    /// If `tail_timeout` is `None`, only replays and ends; if `Some(d)`, after replay blocks with timeout `d` for new events.
    pub fn replay_then_tail_stream(
        self,
        from_id: String,
        tail_timeout: Option<Duration>,
    ) -> impl futures_util::Stream<Item = Result<(String, DurableEvent)>> + Send + 'static {
        let backend = self.backend;
        let stream_key = self.stream_key;
        stream! {
            let mut cursor = from_id;
            loop {
                let batch = backend.read_after(&stream_key, &cursor, REPLAY_BATCH).await;
                let batch = match batch {
                    Ok(b) => b,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                };
                let n = batch.len();
                for (id, ev) in batch {
                    cursor = id.clone();
                    yield Ok((id, ev));
                }
                if n < REPLAY_BATCH {
                    break;
                }
            }
            if let Some(timeout) = tail_timeout {
                loop {
                    let batch = backend
                        .read_blocking(&stream_key, &cursor, REPLAY_BATCH, Some(timeout))
                        .await;
                    let batch = match batch {
                        Ok(b) => b,
                        Err(e) => {
                            yield Err(e);
                            break;
                        }
                    };
                    for (id, ev) in batch {
                        cursor = id.clone();
                        yield Ok((id, ev));
                    }
                }
            }
        }
    }
}
