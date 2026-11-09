use crate::agent::core::loop_detection::{TextLoopDetector, ToolLoopDetector};
use crate::messages::{PerformanceMetrics, StreamEvent, TokenUsage};
use anyhow::Result;
use futures_util::Stream;
use std::time::Duration;

const STREAM_EVENT_TIMEOUT_DEFAULT_SECS: u64 = 900;

/// Threshold for warning about high tool-call counts in a single turn.
const HIGH_TOOL_CALL_WARNING_THRESHOLD: u32 = 20;

/// Typed stream processing errors that callers can match on for recovery.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error("Stream timed out after {elapsed_secs}s waiting for next event")]
    Timeout {
        elapsed_secs: u64,
        partial_text: String,
        usage: TokenUsage,
        tool_calls_completed: u32,
    },
    #[error("Operation cancelled by user")]
    Cancelled,
    #[error("Loop detected: {0}")]
    LoopDetected(String),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl StreamError {
    /// Returns `true` if this error is a stream timeout (recoverable).
    #[must_use]
    pub const fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }

    /// Returns `true` if this error was a user cancellation.
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

#[derive(Debug)]
pub struct StreamResult {
    pub text: String,
    pub usage: TokenUsage,
    pub tools_used: bool,
    /// True when the stream ended on [`StreamEvent::LimitReached`].
    pub limit_reached: bool,
    /// Cap reported by [`StreamEvent::LimitReached`], when present.
    pub max_iterations: Option<usize>,
}

pub trait EventSink: Send {
    fn on_text_chunk(&mut self, text: &str);
    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value);
    fn on_tool_complete(&mut self, name: &str, result: &str);
    fn on_tool_complete_with_payload(
        &mut self,
        name: &str,
        result: &str,
        _payload: Option<&serde_json::Value>,
    ) {
        self.on_tool_complete(name, result);
    }
    fn on_metadata(&mut self, _usage: Option<&TokenUsage>, _metrics: Option<&PerformanceMetrics>) {}
    fn on_thinking(&mut self, content: &str);
    fn on_complete(&mut self, usage: &TokenUsage);
    fn on_phase_transition(&mut self, _phase_index: usize, _phase_total: usize, _tools: &[String]) {
    }
    /// A progress note while nothing else streams (see [`StreamEvent::Status`]).
    fn on_status(&mut self, _message: &str) {}
    fn on_audio_chunk(&mut self, _data: &[u8], _format: &str) {}
    fn on_image_generated(&mut self, _data: &[u8], _media_type: &str) {}
    fn on_attachment_processed(&mut self, _attachment_type: &str, _filename: &str, _summary: &str) {
    }
    fn on_turn_start(&mut self) {}
    fn on_turn_complete(&mut self) {}
    fn on_agent_start(&mut self) {}
    fn on_agent_end(&mut self) {}
    fn on_compaction(
        &mut self,
        _tokens_before: usize,
        _tokens_after: usize,
        _messages_removed: usize,
        _summary: Option<&str>,
    ) {
    }
    fn on_limit_reached(&mut self, _max_iterations: usize, _usage: &TokenUsage) {}
}

pub trait CancellationSignal: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

struct StreamProgress {
    text: String,
    usage: TokenUsage,
    tools_used: bool,
    tool_call_count: u32,
    high_tool_call_warned: bool,
    limit_reached: bool,
    max_iterations: Option<usize>,
    loop_detector: TextLoopDetector,
    tool_loop_detector: ToolLoopDetector,
}

impl Default for StreamProgress {
    fn default() -> Self {
        Self {
            text: String::new(),
            usage: TokenUsage::default(),
            tools_used: false,
            tool_call_count: 0,
            high_tool_call_warned: false,
            limit_reached: false,
            max_iterations: None,
            loop_detector: TextLoopDetector::new_default(),
            tool_loop_detector: ToolLoopDetector::new_default(),
        }
    }
}

impl StreamProgress {
    fn into_result(self) -> StreamResult {
        StreamResult {
            text: self.text,
            usage: self.usage,
            tools_used: self.tools_used,
            limit_reached: self.limit_reached,
            max_iterations: self.max_iterations,
        }
    }
}

pub struct StreamProcessor;

impl StreamProcessor {
    fn stream_event_timeout() -> Duration {
        let secs = std::env::var("CALICHE_STREAM_EVENT_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(STREAM_EVENT_TIMEOUT_DEFAULT_SECS);
        Duration::from_secs(secs)
    }

    pub async fn process_stream<S, F>(
        mut stream: S,
        sink: &mut dyn EventSink,
        cancellation: Option<&dyn CancellationSignal>,
        mut on_usage: F,
    ) -> Result<StreamResult>
    where
        S: Stream<Item = Result<StreamEvent>> + Unpin,
        F: FnMut(&TokenUsage),
    {
        use futures_util::StreamExt;

        let mut progress = StreamProgress::default();
        let stream_timeout = Self::stream_event_timeout();
        let cancellation_poll_interval = Duration::from_millis(200);

        loop {
            if let Some(token) = cancellation
                && token.is_cancelled()
            {
                return Err(StreamError::Cancelled.into());
            }

            let wait_started = std::time::Instant::now();
            let event_result = loop {
                if let Some(token) = cancellation
                    && token.is_cancelled()
                {
                    return Err(StreamError::Cancelled.into());
                }

                let elapsed = wait_started.elapsed();
                if elapsed >= stream_timeout {
                    return Err(StreamError::Timeout {
                        elapsed_secs: stream_timeout.as_secs(),
                        partial_text: progress.text,
                        usage: progress.usage,
                        tool_calls_completed: progress.tool_call_count,
                    }
                    .into());
                }

                let remaining = stream_timeout
                    .checked_sub(elapsed)
                    .unwrap_or_else(|| Duration::from_secs(0));
                let wait_window = std::cmp::min(remaining, cancellation_poll_interval);

                match tokio::time::timeout(wait_window, stream.next()).await {
                    Ok(Some(ev)) => break ev,
                    Ok(None) => {
                        on_usage(&progress.usage);
                        sink.on_complete(&progress.usage);
                        return Ok(progress.into_result());
                    }
                    Err(_) => {
                        continue;
                    }
                }
            };

            if let Some(token) = cancellation
                && token.is_cancelled()
            {
                return Err(StreamError::Cancelled.into());
            }

            if apply_stream_event(&mut progress, sink, event_result)? {
                break;
            }
        }

        on_usage(&progress.usage);
        sink.on_complete(&progress.usage);
        Ok(progress.into_result())
    }

    pub fn dispatch_event(sink: &mut dyn EventSink, event: StreamEvent) {
        match event {
            StreamEvent::TextChunk { content } => sink.on_text_chunk(&content),
            StreamEvent::ThinkingChunk { content, .. } => sink.on_thinking(&content),
            StreamEvent::Metadata { usage, metrics, .. } => {
                sink.on_metadata(usage.as_ref(), metrics.as_ref());
            }
            StreamEvent::ToolCallStart {
                name, arguments, ..
            } => sink.on_tool_start(&name, &arguments),
            StreamEvent::ToolCallResult {
                name,
                result,
                typed_payload,
                ..
            } => {
                sink.on_tool_complete_with_payload(&name, &result, typed_payload.as_ref());
            }
            StreamEvent::AudioChunk { data, format } => sink.on_audio_chunk(&data, &format),
            StreamEvent::ImageGenerated {
                data, media_type, ..
            } => sink.on_image_generated(&data, &media_type),
            StreamEvent::AttachmentProcessed {
                attachment_type,
                filename,
                summary,
                ..
            } => sink.on_attachment_processed(&attachment_type, &filename, &summary),
            StreamEvent::Done => {}
            StreamEvent::LimitReached {
                max_iterations,
                usage,
            } => sink.on_limit_reached(max_iterations, &usage),
            StreamEvent::Compaction {
                tokens_before,
                tokens_after,
                messages_removed,
                summary,
                report: _,
            } => sink.on_compaction(
                tokens_before,
                tokens_after,
                messages_removed,
                summary.as_deref(),
            ),
            StreamEvent::AgentStart => sink.on_agent_start(),
            StreamEvent::AgentEnd => sink.on_agent_end(),
            StreamEvent::TurnStart => sink.on_turn_start(),
            StreamEvent::TurnEnd => sink.on_turn_complete(),
            StreamEvent::PhaseTransition {
                phase_index,
                phase_total,
                tool_names,
            } => sink.on_phase_transition(phase_index, phase_total, &tool_names),
            StreamEvent::Status { message } => sink.on_status(&message),
        }
    }
}

fn apply_stream_event(
    progress: &mut StreamProgress,
    sink: &mut dyn EventSink,
    event: Result<StreamEvent>,
) -> Result<bool> {
    match event {
        Ok(StreamEvent::TextChunk { content }) => apply_text_chunk(progress, sink, &content)?,
        Ok(StreamEvent::Metadata { usage, metrics, .. }) => {
            if let Some(usage) = usage.as_ref() {
                progress.usage = *usage;
            }
            sink.on_metadata(usage.as_ref(), metrics.as_ref());
        }
        Ok(StreamEvent::ToolCallStart {
            name, arguments, ..
        }) => apply_tool_start(progress, sink, &name, &arguments)?,
        Ok(StreamEvent::ToolCallResult {
            name,
            result,
            typed_payload,
            ..
        }) => apply_tool_result(progress, sink, &name, &result, typed_payload.as_ref()),
        Ok(StreamEvent::Done) => return Ok(true),
        Ok(StreamEvent::LimitReached {
            max_iterations,
            usage,
        }) => {
            progress.usage = usage;
            progress.limit_reached = true;
            progress.max_iterations = Some(max_iterations);
            StreamProcessor::dispatch_event(
                sink,
                StreamEvent::LimitReached {
                    max_iterations,
                    usage: progress.usage,
                },
            );
            return Ok(true);
        }
        Ok(other) => StreamProcessor::dispatch_event(sink, other),
        Err(error) => return Err(error.context("Stream event processing failed")),
    }
    Ok(false)
}

fn apply_text_chunk(
    progress: &mut StreamProgress,
    sink: &mut dyn EventSink,
    content: &str,
) -> Result<()> {
    progress.text.push_str(content);
    progress.loop_detector.on_text(content);
    if let Some(repeated) = progress.loop_detector.detect() {
        sink.on_text_chunk("\n");
        return Err(StreamError::LoopDetected(format!(
            "model repeated {:?} {} or more times without \
             calling any tools.  Aborting turn.  Try rephrasing or use /clear.",
            repeated,
            progress.loop_detector.repeat_threshold()
        ))
        .into());
    }
    sink.on_text_chunk(content);
    Ok(())
}

fn apply_tool_start(
    progress: &mut StreamProgress,
    sink: &mut dyn EventSink,
    name: &str,
    arguments: &serde_json::Value,
) -> Result<()> {
    progress.tools_used = true;
    progress.tool_call_count += 1;
    progress.loop_detector.on_tool_call();
    if !progress.high_tool_call_warned
        && progress.tool_call_count >= HIGH_TOOL_CALL_WARNING_THRESHOLD
    {
        progress.high_tool_call_warned = true;
        tracing::warn!(
            "High tool call count ({}) in single turn — \
             model may be stuck in an inefficient loop",
            progress.tool_call_count
        );
    }
    if let Some(signature) = progress.tool_loop_detector.on_tool_call(name, arguments) {
        sink.on_text_chunk("\n");
        return Err(StreamError::LoopDetected(format!(
            "repeated tool call signature {:?} {} or more times in one turn.",
            signature,
            progress.tool_loop_detector.repeat_threshold()
        ))
        .into());
    }
    sink.on_tool_start(name, arguments);
    Ok(())
}

fn apply_tool_result(
    progress: &mut StreamProgress,
    sink: &mut dyn EventSink,
    name: &str,
    result: &str,
    typed_payload: Option<&serde_json::Value>,
) {
    progress.tools_used = true;
    if !result.contains("\"success\":false") && !result.starts_with("Error") {
        progress.tool_loop_detector.reset();
    }
    sink.on_tool_complete_with_payload(name, result, typed_payload);
}

#[cfg(test)]
mod tests {
    use super::{EventSink, StreamProcessor, StreamResult};
    use crate::messages::{StreamEvent, TokenUsage};
    use anyhow::Result;
    use futures_util::stream;

    #[derive(Default)]
    struct RecordingSink {
        chunks: Vec<String>,
        limit_reached: Option<usize>,
        completed: bool,
        agent_started: bool,
    }

    impl EventSink for RecordingSink {
        fn on_text_chunk(&mut self, text: &str) {
            self.chunks.push(text.to_string());
        }

        fn on_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}

        fn on_tool_complete(&mut self, _name: &str, _result: &str) {}

        fn on_thinking(&mut self, _content: &str) {}

        fn on_complete(&mut self, _usage: &TokenUsage) {
            self.completed = true;
        }

        fn on_limit_reached(&mut self, max_iterations: usize, _usage: &TokenUsage) {
            self.limit_reached = Some(max_iterations);
        }

        fn on_agent_start(&mut self) {
            self.agent_started = true;
        }
    }

    fn usage() -> TokenUsage {
        TokenUsage {
            prompt_tokens: Some(1),
            completion_tokens: Some(2),
            total_tokens: Some(3),
            embedding_tokens: None,
            context_window: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            last_prompt_tokens: None,
        }
    }

    #[tokio::test]
    async fn process_stream_marks_limit_reached_separately_from_done() {
        let events = vec![
            Ok(StreamEvent::TextChunk {
                content: "partial work".to_string(),
            }),
            Ok(StreamEvent::LimitReached {
                max_iterations: 100,
                usage: usage(),
            }),
        ];
        let mut sink = RecordingSink::default();
        let result: StreamResult =
            StreamProcessor::process_stream(stream::iter(events), &mut sink, None, |_| {})
                .await
                .expect("limit stream");

        assert_eq!(result.text, "partial work");
        assert!(result.limit_reached);
        assert_eq!(result.max_iterations, Some(100));
        assert_eq!(sink.limit_reached, Some(100));
        assert!(sink.completed);
    }

    #[tokio::test]
    async fn process_stream_done_is_not_a_limit() {
        let events: Vec<Result<StreamEvent>> = vec![
            Ok(StreamEvent::TextChunk {
                content: "done".to_string(),
            }),
            Ok(StreamEvent::Done),
        ];
        let mut sink = RecordingSink::default();
        let result = StreamProcessor::process_stream(stream::iter(events), &mut sink, None, |_| {})
            .await
            .expect("done stream");

        assert!(!result.limit_reached);
        assert_eq!(result.max_iterations, None);
        assert!(sink.limit_reached.is_none());
    }

    #[test]
    fn dispatch_event_forwards_limit_reached() {
        let mut sink = RecordingSink::default();
        StreamProcessor::dispatch_event(
            &mut sink,
            StreamEvent::LimitReached {
                max_iterations: 40,
                usage: usage(),
            },
        );
        assert_eq!(sink.limit_reached, Some(40));
        assert!(!sink.completed);
    }

    #[tokio::test]
    async fn process_stream_forwards_agent_start_and_keeps_text() {
        let events = vec![
            Ok(StreamEvent::AgentStart),
            Ok(StreamEvent::TextChunk {
                content: "hi".to_string(),
            }),
            Ok(StreamEvent::Done),
        ];
        let mut sink = RecordingSink::default();
        let result = StreamProcessor::process_stream(stream::iter(events), &mut sink, None, |_| {})
            .await
            .expect("agent start stream");

        assert!(sink.agent_started);
        assert_eq!(result.text, "hi");
        assert!(!result.limit_reached);
        assert!(sink.completed);
    }
}
