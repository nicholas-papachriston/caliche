//! Durable stream backend trait.
//!
//! Implementations persist event streams for replay and resume (Redis/Valkey,
//! in-memory ring buffer, or memory-mapped append log).

use crate::streams::types::{DurableEvent, TrimPolicy};
use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;

/// Backend for persisting and reading durable event streams.
///
/// Event IDs are opaque, monotonic strings (e.g. Redis stream IDs or
/// `{timestamp_ms}-{seq}`). Use [crate::streams::STREAM_START] for "from beginning".
#[async_trait]
pub trait DurableStreamBackend: Send + Sync + std::fmt::Debug {
    /// Append an event to the stream for the given key.
    /// Returns the assigned event ID (monotonic, opaque string).
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String>;

    /// Read up to `count` events after `after_id` (exclusive).
    /// Use [crate::streams::STREAM_START] for `after_id` to read from the beginning.
    /// Returns `(event_id, event)` pairs in order.
    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>>;

    /// Block until new events arrive after `after_id`, or until `timeout` elapses.
    /// If `timeout` is `None`, blocks indefinitely. Returns up to `count` events.
    async fn read_blocking(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>>;

    /// Trim the stream according to the policy. Returns the number of entries removed (best-effort).
    async fn trim(&self, stream_key: &str, policy: TrimPolicy) -> Result<u64>;

    /// Return the last event ID in the stream, if any (for resume).
    async fn last_id(&self, stream_key: &str) -> Result<Option<String>>;

    /// Delete the entire stream.
    async fn delete_stream(&self, stream_key: &str) -> Result<()>;
}
