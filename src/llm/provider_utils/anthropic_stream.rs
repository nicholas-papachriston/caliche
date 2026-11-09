//! Shared Anthropic-style streaming event parser for content blocks and tool calls.
//!
//! Anthropic direct API and Bedrock Claude models share the same `content_block_*`
//! event structure. This module centralizes text/tool-call accumulation so both
//! providers stay in sync.

use crate::llm::provider_utils::StreamingToolCallAccumulator;
use crate::messages::StreamEvent;

/// Normalized Anthropic-style stream event for feeding the shared parser.
#[derive(Debug, Clone, Copy)]
pub enum AnthropicStreamFeed<'a> {
    ContentBlockDelta {
        index: Option<u32>,
        text: Option<&'a str>,
        thinking: Option<&'a str>,
        partial_json: Option<&'a str>,
    },
    ContentBlockStart {
        index: Option<u32>,
        block_type: &'a str,
        tool_id: Option<&'a str>,
        tool_name: Option<&'a str>,
    },
    ContentBlockStop {
        index: Option<u32>,
    },
}

/// Parses Anthropic-style content block streaming events into [`StreamEvent`]s.
pub struct AnthropicStreamParser {
    tool_call_accumulator: StreamingToolCallAccumulator,
}

impl AnthropicStreamParser {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tool_call_accumulator: StreamingToolCallAccumulator::new(),
        }
    }

    /// Process one stream event. Returns zero or more output events.
    pub fn feed(&mut self, event: AnthropicStreamFeed<'_>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        match event {
            AnthropicStreamFeed::ContentBlockDelta {
                index,
                text,
                thinking,
                partial_json,
            } => {
                if let Some(thinking) = thinking
                    && !thinking.is_empty()
                {
                    out.push(StreamEvent::ThinkingChunk {
                        content: thinking.to_string(),
                        action_id: None,
                    });
                }
                if let Some(text) = text
                    && !text.is_empty()
                {
                    out.push(StreamEvent::TextChunk {
                        content: text.to_string(),
                    });
                }
                if let (Some(partial_json), Some(index)) = (partial_json, index) {
                    self.tool_call_accumulator.accumulate(
                        index,
                        None,
                        None,
                        Some(partial_json.to_string()),
                    );
                }
            }
            AnthropicStreamFeed::ContentBlockStart {
                index,
                block_type,
                tool_id,
                tool_name,
            } => {
                if block_type == "tool_use" {
                    if let Some(index) = index {
                        if let (Some(id), Some(name)) = (tool_id, tool_name) {
                            tracing::info!(
                                "Started tracking tool call: index={}, id={}, name={}",
                                index,
                                id,
                                name
                            );
                            self.tool_call_accumulator.accumulate(
                                index,
                                Some(id.to_string()),
                                Some(name.to_string()),
                                None,
                            );
                        } else {
                            tracing::warn!(
                                "content_block_start for tool_use at index {:?} missing id or name. id={:?}, name={:?}",
                                index,
                                tool_id,
                                tool_name
                            );
                        }
                    } else {
                        tracing::warn!("content_block_start for tool_use missing index");
                    }
                } else if block_type == "redacted_thinking" {
                    out.push(StreamEvent::ThinkingChunk {
                        content: "[redacted]".to_string(),
                        action_id: None,
                    });
                }
            }
            AnthropicStreamFeed::ContentBlockStop { index } => {
                if let Some(index) = index {
                    if let Some(evt) = self.tool_call_accumulator.finalize(index, false) {
                        out.push(evt);
                    } else {
                        tracing::debug!(
                            "content_block_stop for index {} but no tool call in progress",
                            index
                        );
                    }
                }
            }
        }
        out
    }

    /// Flush any remaining in-progress tool calls (`require_arguments=true` then `false`).
    pub fn finish_tool_calls(&mut self) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        out.extend(self.tool_call_accumulator.finalize_all(true));
        out.extend(self.tool_call_accumulator.finalize_all(false));
        out
    }

    /// Complete stream shutdown: flush tool calls and emit [`StreamEvent::Done`].
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut out = self.finish_tool_calls();
        out.push(StreamEvent::Done);
        out
    }
}

impl Default for AnthropicStreamParser {
    fn default() -> Self {
        Self::new()
    }
}
