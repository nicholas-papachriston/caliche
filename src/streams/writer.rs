//! Writer that appends events to a durable stream backend.

use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::reader::DurableStreamReader;
use crate::streams::types::DurableEvent;
use anyhow::Result;
use std::sync::Arc;

/// Writes events to a durable stream for a fixed stream key.
///
/// Used by [crate::harness::EventStream] when a backend is configured:
/// each emitted event is also appended to the backend.
#[derive(Clone, Debug)]
pub struct DurableStreamWriter {
    backend: Arc<dyn DurableStreamBackend>,
    stream_key: String,
}

impl DurableStreamWriter {
    /// Create a writer for the given backend and stream key.
    pub fn new(backend: Arc<dyn DurableStreamBackend>, stream_key: impl Into<String>) -> Self {
        Self {
            backend,
            stream_key: stream_key.into(),
        }
    }

    /// Append an event and return its assigned ID.
    pub async fn append(&self, event: &DurableEvent) -> Result<String> {
        self.backend.append(&self.stream_key, event).await
    }

    /// Create a reader for the same stream (replay and tail).
    pub fn reader(&self) -> DurableStreamReader {
        DurableStreamReader::new(self.backend.clone(), self.stream_key.clone())
    }
}
