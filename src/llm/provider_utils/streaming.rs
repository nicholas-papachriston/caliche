//! Streaming utilities for LLM providers.
//!
//! Provides shared SSE line parsing, iteration, and tool-call accumulation used by
//! Anthropic, Bedrock, Azure `OpenAI`, Cohere, and OpenAI-compatible providers.

use crate::llm::streaming::ByteLineBuffer;
use crate::messages::StreamEvent;
use futures_util::{Stream, TryStreamExt};
use reqwest::Response;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::time::Duration;

const SSE_STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// Tracks in-progress tool calls during streaming. Use with Anthropic/Bedrock (`content_block`
/// events) or OpenAI-compatible (`tool_calls` delta) providers.
pub struct StreamingToolCallAccumulator {
    in_progress: HashMap<u32, ToolCallInProgress>,
    emitted_ids: HashSet<String>,
}

struct ToolCallInProgress {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl StreamingToolCallAccumulator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            in_progress: HashMap::new(),
            emitted_ids: HashSet::new(),
        }
    }

    /// Merge partial tool-call data. Creates an entry if missing.
    /// For Anthropic/Bedrock: call with (id, name) on start, (_, _, args) on delta.
    /// For OpenAI-compat: call with any combination as fragments arrive.
    pub fn accumulate(
        &mut self,
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) {
        let entry = self
            .in_progress
            .entry(index)
            .or_insert_with(|| ToolCallInProgress {
                id: None,
                name: None,
                arguments: String::new(),
            });
        if let Some(id) = id {
            entry.id = Some(id);
        }
        if let Some(name) = name {
            entry.name = Some(name);
        }
        if let Some(args) = arguments {
            entry.arguments.push_str(&args);
        }
    }

    /// Merge tool-call data when the provider sends cumulative argument snapshots
    /// (e.g. Google Gemini streaming) rather than strict deltas.
    pub fn accumulate_cumulative(
        &mut self,
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) {
        let entry = self
            .in_progress
            .entry(index)
            .or_insert_with(|| ToolCallInProgress {
                id: None,
                name: None,
                arguments: String::new(),
            });
        if let Some(id) = id {
            entry.id = Some(id);
        }
        if let Some(name) = name {
            entry.name = Some(name);
        }
        if let Some(args) = arguments
            && (entry.arguments.is_empty() || args.starts_with(&entry.arguments))
        {
            entry.arguments = args;
        }
    }

    /// Finalize a tool call by index.
    /// - `require_arguments`: if true and arguments are empty, returns None without removing.
    /// - Skips if name is empty or id was already emitted.
    pub fn finalize(&mut self, index: u32, require_arguments: bool) -> Option<StreamEvent> {
        let builder = self.in_progress.get(&index)?;
        let id = builder
            .id
            .clone()
            .unwrap_or_else(|| format!("call_{index}"));
        let name = builder.name.clone().unwrap_or_default();

        if name.is_empty() {
            return None;
        }
        if self.emitted_ids.contains(&id) {
            self.in_progress.remove(&index);
            return None;
        }
        if require_arguments && builder.arguments.trim().is_empty() {
            return None;
        }

        let arguments = builder.arguments.clone();
        self.in_progress.remove(&index);

        self.emitted_ids.insert(id.clone());
        match serde_json::from_str::<serde_json::Value>(&arguments) {
            Ok(arguments_value) => Some(StreamEvent::ToolCallStart {
                id: beef::lean::Cow::owned(id),
                name: beef::lean::Cow::owned(name),
                arguments: arguments_value,
                thought_signature: None,
                action_id: None,
            }),
            Err(e) => {
                // Provider sent tool arguments as a string that failed to parse as JSON
                // (e.g. truncated, malformed, or double-encoded). Wrap in {"raw": "<string>"}
                // so downstream can try to normalize; see ToolRegistry::normalize_tool_arguments.
                tracing::debug!(
                    tool = %name,
                    parse_error = %e,
                    args_preview = %arguments.chars().take(200).collect::<String>(),
                    "tool call arguments not valid JSON, wrapped as raw"
                );
                Some(StreamEvent::ToolCallStart {
                    id: beef::lean::Cow::owned(id),
                    name: beef::lean::Cow::owned(name),
                    arguments: serde_json::json!({ "raw": arguments }),
                    thought_signature: None,
                    action_id: None,
                })
            }
        }
    }

    /// Returns whether a tool call at `index` is still in progress.
    #[must_use]
    pub fn has_in_progress(&self, index: u32) -> bool {
        self.in_progress.contains_key(&index)
    }

    /// Finalize all remaining tool calls (e.g. on EOF).
    pub fn finalize_all(&mut self, require_arguments: bool) -> Vec<StreamEvent> {
        let indices: Vec<u32> = self.in_progress.keys().copied().collect();
        let mut out = Vec::new();
        for index in indices {
            if let Some(evt) = self.finalize(index, require_arguments) {
                out.push(evt);
            }
        }
        out
    }
}

impl Default for StreamingToolCallAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse SSE data line and extract the data field.
#[must_use]
pub fn parse_sse_data_line(line: &str) -> Option<&str> {
    line.strip_prefix("data: ")
}

/// Check if an SSE line indicates the end of the stream.
#[must_use]
pub fn is_stream_done(line: &str) -> bool {
    line == "data: [DONE]" || line.trim() == "[DONE]"
}

/// SSE data line: either JSON content or stream end.
#[derive(Debug, Clone)]
pub enum SseDataLine {
    /// Data line (content after "data: " prefix, excluding [DONE])
    Data(String),
    /// Stream terminated with [DONE] or EOF
    Done,
}

/// How long an SSE response body may stay silent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SseLimits {
    /// When the first byte must have arrived. `None` applies
    /// [`Self::stall`] to it too.
    pub first_byte: Option<FirstByteDeadline>,
    /// Limit for each gap once bytes flow.
    pub stall: Duration,
}

/// The first-byte deadline of one request, and the limit it came from (for
/// the error message).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstByteDeadline {
    pub at: tokio::time::Instant,
    pub limit: Duration,
}

impl SseLimits {
    /// Stall limit used by every provider that sets none.
    pub const DEFAULT_STALL: Duration = SSE_STALL_TIMEOUT;

    /// The stall limit for every gap, the first byte included.
    pub const STALL_ONLY: Self = Self {
        first_byte: None,
        stall: SSE_STALL_TIMEOUT,
    };
}

/// The error when a response sends no byte within `limit` (a local model
/// may still be loading or reading the prompt).
#[must_use]
pub fn no_first_token_error(limit: Duration) -> String {
    format!(
        "no first token after {}s (model still loading?)",
        limit.as_secs()
    )
}

fn stalled_error(limit: Duration) -> String {
    format!(
        "SSE stream stalled: no data received for {} seconds",
        limit.as_secs()
    )
}

/// Iterate over SSE data lines from an HTTP response body.
///
/// Yields `Data(json_str)` for each "data: {...}" line and `Done` when "[DONE]" or stream ends.
/// Returns a pinned boxed stream so it implements `Unpin` for use with `StreamExt::next`.
#[must_use]
pub fn sse_data_lines(
    response: Response,
) -> Pin<Box<dyn Stream<Item = Result<SseDataLine, anyhow::Error>> + Send>> {
    sse_data_lines_within(response, SseLimits::STALL_ONLY)
}

/// [`sse_data_lines`] with its own silence limits.
#[must_use]
pub fn sse_data_lines_within(
    response: Response,
    limits: SseLimits,
) -> Pin<Box<dyn Stream<Item = Result<SseDataLine, anyhow::Error>> + Send>> {
    Box::pin(async_stream::stream! {
        let mut stream = response.bytes_stream();
        let mut buffer = ByteLineBuffer::new();
        let mut first_byte = limits.first_byte;

        loop {
            let deadline = first_byte.map_or_else(
                || tokio::time::Instant::now() + limits.stall,
                |first| first.at,
            );
            match tokio::time::timeout_at(deadline, stream.try_next()).await {
                Ok(Ok(Some(chunk))) => {
                    first_byte = None;
                    buffer.push_bytes(chunk.as_ref() as &[u8]);

                    while let Some(line_result) = buffer.next_line() {
                        let line = match line_result {
                            Ok(s) => s,
                            Err(e) => {
                                yield Err(anyhow::anyhow!("Failed to decode chunk: {e}"));
                                return;
                            }
                        };
                        if line.is_empty() {
                            continue;
                        }

                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if json_str == "[DONE]" {
                                yield Ok(SseDataLine::Done);
                                return;
                            }
                            yield Ok(SseDataLine::Data(json_str.to_string()));
                        }
                    }
                }
                Ok(Ok(None)) => {
                    yield Ok(SseDataLine::Done);
                    return;
                }
                Ok(Err(e)) => {
                    yield Err(anyhow::Error::new(e).context("reading the response body failed"));
                    return;
                }
                Err(_elapsed) => {
                    let message = match first_byte {
                        Some(first) => no_first_token_error(first.limit),
                        None => stalled_error(limits.stall),
                    };
                    yield Err(anyhow::anyhow!(message));
                    return;
                }
            }
        }
    })
}
