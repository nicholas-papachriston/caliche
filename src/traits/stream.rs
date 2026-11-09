//! Stream transformer abstraction for transforming and filtering stream events.
//!
//! This module provides a unified interface for transforming stream events
//! during LLM streaming responses. Implement the `StreamTransformer` trait to
//! add custom transformations like filtering, rate limiting, or logging.

use crate::messages::StreamEvent;
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Trait for stream transformers that can transform or filter stream events
#[async_trait]
pub trait StreamTransformer: Send + Sync + std::fmt::Debug {
    /// Transform a stream event
    ///
    /// # Arguments
    /// * `event` - The stream event to transform
    ///
    /// # Returns
    /// * `Ok(Some(StreamEvent))` - Transformed event (None means filter out)
    /// * `Ok(None)` - Event should be filtered out
    /// * `Err` - Error during transformation
    async fn transform(&self, event: StreamEvent) -> Result<Option<StreamEvent>>;

    /// Get the name of this transformer (for debugging/logging)
    fn name(&self) -> &str;
}

/// Stream transformer that filters events by type or content
#[derive(Debug, Clone)]
pub struct FilterTransformer {
    allowed_types: Vec<String>,
    filter_empty_content: bool,
}

impl FilterTransformer {
    /// Create a new filter transformer
    ///
    /// # Arguments
    /// * `allowed_types` - List of event types to allow (e.g., ["text_chunk", "metadata"])
    /// * `filter_empty_content` - Whether to filter out events with empty content
    pub fn new(allowed_types: Vec<String>, filter_empty_content: bool) -> Self {
        Self {
            allowed_types,
            filter_empty_content,
        }
    }

    /// Create a transformer that only allows text chunks
    pub fn text_only() -> Self {
        Self::new(vec!["text_chunk".to_string()], true)
    }

    /// Create a transformer that filters out metadata events
    pub fn no_metadata() -> Self {
        Self::new(
            vec![
                "text_chunk".to_string(),
                "tool_call_start".to_string(),
                "tool_call_result".to_string(),
                "done".to_string(),
            ],
            false,
        )
    }
}

#[async_trait]
impl StreamTransformer for FilterTransformer {
    async fn transform(&self, event: StreamEvent) -> Result<Option<StreamEvent>> {
        let event_type = stream_event_name(&event);

        if !self.allowed_types.is_empty()
            && !self
                .allowed_types
                .iter()
                .any(|allowed| allowed == event_type)
        {
            return Ok(None);
        }

        // Filter empty content if enabled
        if self.filter_empty_content {
            match &event {
                StreamEvent::TextChunk { content } if content.is_empty() => return Ok(None),
                _ => {}
            }
        }

        Ok(Some(event))
    }

    fn name(&self) -> &str {
        "filter"
    }
}

fn stream_event_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::TextChunk { .. } => "text_chunk",
        StreamEvent::ThinkingChunk { .. } => "thinking_chunk",
        StreamEvent::ToolCallStart { .. } => "tool_call_start",
        StreamEvent::ToolCallResult { .. } => "tool_call_result",
        StreamEvent::AudioChunk { .. } => "audio_chunk",
        StreamEvent::ImageGenerated { .. } => "image_generated",
        StreamEvent::AttachmentProcessed { .. } => "attachment_processed",
        StreamEvent::Metadata { .. } => "metadata",
        StreamEvent::LimitReached { .. } => "limit_reached",
        StreamEvent::Compaction { .. } => "compaction",
        StreamEvent::PhaseTransition { .. } => "phase_transition",
        StreamEvent::Status { .. } => "status",
        StreamEvent::Done => "done",
        StreamEvent::AgentStart => "agent_start",
        StreamEvent::AgentEnd => "agent_end",
        StreamEvent::TurnStart => "turn_start",
        StreamEvent::TurnEnd => "turn_end",
    }
}

/// Stream transformer that rate limits stream chunks
#[derive(Debug)]
pub struct RateLimitTransformer {
    max_chunks_per_second: f64,
    last_chunk_time: tokio::sync::Mutex<Option<Instant>>,
    chunk_count: std::sync::atomic::AtomicU64,
    window_start: tokio::sync::Mutex<Instant>,
}

impl RateLimitTransformer {
    /// Create a new rate limit transformer
    ///
    /// # Arguments
    /// * `max_chunks_per_second` - Maximum number of chunks to allow per second
    pub fn new(max_chunks_per_second: f64) -> Self {
        Self {
            max_chunks_per_second,
            last_chunk_time: tokio::sync::Mutex::new(None),
            chunk_count: std::sync::atomic::AtomicU64::new(0),
            window_start: tokio::sync::Mutex::new(Instant::now()),
        }
    }
}

#[async_trait]
impl StreamTransformer for RateLimitTransformer {
    async fn transform(&self, event: StreamEvent) -> Result<Option<StreamEvent>> {
        // Only rate limit text chunks
        if matches!(event, StreamEvent::TextChunk { .. }) {
            let now = Instant::now();
            let mut window_start = self.window_start.lock().await;

            // Reset window if a second has passed
            if now.duration_since(*window_start) >= Duration::from_secs(1) {
                self.chunk_count
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                *window_start = now;
            }

            // Check rate limit
            let count = self
                .chunk_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            if count as f64 > self.max_chunks_per_second {
                // Rate limited - filter out this chunk
                return Ok(None);
            }

            // Update last chunk time
            *self.last_chunk_time.lock().await = Some(now);
        }

        Ok(Some(event))
    }

    fn name(&self) -> &str {
        "rate-limit"
    }
}

/// Stream transformer that logs all stream events
#[derive(Debug, Clone)]
pub struct LoggingTransformer {
    log_level: LogLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Debug,
    Info,
    Trace,
}

impl LoggingTransformer {
    /// Create a new logging transformer
    ///
    /// # Arguments
    /// * `log_level` - Log level to use
    pub fn new(log_level: LogLevel) -> Self {
        Self { log_level }
    }

    /// Create a debug-level logging transformer
    pub fn debug() -> Self {
        Self::new(LogLevel::Debug)
    }
}

#[async_trait]
impl StreamTransformer for LoggingTransformer {
    async fn transform(&self, event: StreamEvent) -> Result<Option<StreamEvent>> {
        match self.log_level {
            LogLevel::Debug => tracing::debug!("Stream event: {:?}", event),
            LogLevel::Info => tracing::info!("Stream event: {:?}", event),
            LogLevel::Trace => tracing::trace!("Stream event: {:?}", event),
        }
        Ok(Some(event))
    }

    fn name(&self) -> &str {
        "logging"
    }
}

/// Stream transformer that chains multiple transformers
#[derive(Debug)]
pub struct ChainedTransformer {
    transformers: Vec<Arc<dyn StreamTransformer>>,
}

impl ChainedTransformer {
    /// Create a new chained transformer
    pub fn new(transformers: Vec<Arc<dyn StreamTransformer>>) -> Self {
        Self { transformers }
    }
}

#[async_trait]
impl StreamTransformer for ChainedTransformer {
    async fn transform(&self, mut event: StreamEvent) -> Result<Option<StreamEvent>> {
        for transformer in &self.transformers {
            match transformer.transform(event).await? {
                Some(e) => event = e,
                None => return Ok(None), // Filtered out by a transformer
            }
        }
        Ok(Some(event))
    }

    fn name(&self) -> &str {
        "chained"
    }
}

/// No-op stream transformer that passes events through unchanged
#[derive(Debug, Clone)]
pub struct NoOpTransformer;

impl NoOpTransformer {
    /// Create a new no-op transformer
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StreamTransformer for NoOpTransformer {
    async fn transform(&self, event: StreamEvent) -> Result<Option<StreamEvent>> {
        Ok(Some(event))
    }

    fn name(&self) -> &str {
        "no-op"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_filter_transformer() {
        let transformer = FilterTransformer::text_only();

        let text_event = StreamEvent::TextChunk {
            content: "hello".to_string(),
        };
        let result = transformer
            .transform(text_event.clone())
            .await
            .expect("transform should succeed");
        assert!(result.is_some());

        let metadata_event = StreamEvent::Metadata {
            usage: None,
            finish_reason: None,
            metrics: None,
        };
        let result = transformer
            .transform(metadata_event)
            .await
            .expect("transform should succeed");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_logging_transformer() {
        let transformer = LoggingTransformer::debug();
        let event = StreamEvent::TextChunk {
            content: "test".to_string(),
        };
        let result = transformer
            .transform(event)
            .await
            .expect("transform should succeed");
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_chained_transformer() {
        let filter = Arc::new(FilterTransformer::text_only());
        let logger = Arc::new(LoggingTransformer::debug());

        let chain = ChainedTransformer::new(vec![filter, logger]);

        let text_event = StreamEvent::TextChunk {
            content: "hello".to_string(),
        };
        let result = chain
            .transform(text_event)
            .await
            .expect("chained transform should succeed");
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_no_op_transformer() {
        let transformer = NoOpTransformer::new();
        let event = StreamEvent::TextChunk {
            content: "test".to_string(),
        };
        let result = transformer
            .transform(event)
            .await
            .expect("transform should succeed");
        match result {
            Some(StreamEvent::TextChunk { content }) => assert_eq!(content, "test"),
            _ => panic!("Expected TextChunk event"),
        }
    }
}
