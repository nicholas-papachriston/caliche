//! Streaming response handler for real-time text output.
//!
//! This module processes streaming LLM responses and emits
//! text chunks and code blocks as events.

use crate::harness::events::{AgentEvent, EventStream};
use std::sync::Arc;
use tokio::sync::Mutex;

/// State for tracking code block parsing
#[derive(Debug, Clone)]
struct CodeBlockState {
    in_code_block: bool,
    language: Option<String>,
    content: String,
    fence_char: char,
}

impl Default for CodeBlockState {
    fn default() -> Self {
        Self {
            in_code_block: false,
            language: None,
            content: String::new(),
            fence_char: '`',
        }
    }
}

/// Handler for streaming LLM responses
pub struct StreamingResponseHandler {
    event_stream: EventStream,
    buffer: Arc<Mutex<String>>,
    code_block_state: Arc<Mutex<CodeBlockState>>,
    emit_text_chunks: bool,
    emit_code_blocks: bool,
}

impl StreamingResponseHandler {
    /// Create a new streaming handler
    pub fn new(event_stream: EventStream) -> Self {
        Self {
            event_stream,
            buffer: Arc::new(Mutex::new(String::new())),
            code_block_state: Arc::new(Mutex::new(CodeBlockState::default())),
            emit_text_chunks: true,
            emit_code_blocks: true,
        }
    }

    /// Configure whether to emit text chunk events
    pub fn with_text_chunks(mut self, enabled: bool) -> Self {
        self.emit_text_chunks = enabled;
        self
    }

    /// Configure whether to emit code block events
    pub fn with_code_blocks(mut self, enabled: bool) -> Self {
        self.emit_code_blocks = enabled;
        self
    }

    /// Process a chunk of streaming text
    pub async fn process_chunk(&self, chunk: &str) {
        // Add to buffer
        {
            let mut buffer = self.buffer.lock().await;
            buffer.push_str(chunk);
        }

        // Emit text chunk event
        if self.emit_text_chunks && !chunk.is_empty() {
            self.event_stream.emit(AgentEvent::TextChunk {
                content: chunk.to_string(),
            });
        }

        // Process for code blocks if enabled
        if self.emit_code_blocks {
            self.process_code_blocks(chunk).await;
        }
    }

    /// Process text looking for code blocks
    async fn process_code_blocks(&self, chunk: &str) {
        let mut state = self.code_block_state.lock().await;

        for line in chunk.lines() {
            let trimmed = line.trim();

            // Check for code fence start
            if !state.in_code_block && (trimmed.starts_with("```") || trimmed.starts_with("~~~")) {
                state.in_code_block = true;
                state.fence_char = trimmed.chars().next().unwrap_or('`');
                state.content.clear();

                // Extract language
                let fence = if state.fence_char == '`' {
                    "```"
                } else {
                    "~~~"
                };
                let lang = trimmed.strip_prefix(fence).unwrap_or("").trim();
                state.language = if lang.is_empty() {
                    None
                } else {
                    Some(lang.to_string())
                };
                continue;
            }

            // Check for code fence end
            if state.in_code_block {
                let fence = format!(
                    "{}{}{}",
                    state.fence_char, state.fence_char, state.fence_char
                );
                if trimmed == fence || trimmed.starts_with(&fence) {
                    // Emit code block event
                    if !state.content.is_empty() {
                        self.event_stream.emit(AgentEvent::CodeBlock {
                            language: state.language.clone(),
                            content: state.content.clone(),
                        });
                    }

                    // Reset state
                    state.in_code_block = false;
                    state.language = None;
                    state.content.clear();
                    continue;
                }

                // Add line to code content
                if !state.content.is_empty() {
                    state.content.push('\n');
                }
                state.content.push_str(line);
            }
        }
    }

    /// Get the full accumulated buffer
    pub async fn full_content(&self) -> String {
        self.buffer.lock().await.clone()
    }

    /// Clear the buffer and state
    pub async fn reset(&self) {
        self.buffer.lock().await.clear();
        *self.code_block_state.lock().await = CodeBlockState::default();
    }

    /// Finalize streaming and emit any remaining code block
    pub async fn finalize(&self) {
        let state = self.code_block_state.lock().await;

        // If we're still in a code block, emit it
        if state.in_code_block && !state.content.is_empty() {
            self.event_stream.emit(AgentEvent::CodeBlock {
                language: state.language.clone(),
                content: state.content.clone(),
            });
        }
    }
}

impl std::fmt::Debug for StreamingResponseHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingResponseHandler")
            .field("emit_text_chunks", &self.emit_text_chunks)
            .field("emit_code_blocks", &self.emit_code_blocks)
            .finish()
    }
}

/// Extract code blocks from a complete response
pub fn extract_code_blocks(content: &str) -> Vec<(Option<String>, String)> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current_lang: Option<String> = None;
    let mut current_content = String::new();
    // Static fence string stored once when entering a block, avoiding repeated allocations.
    let mut current_fence: &str = "```";

    for line in content.lines() {
        let trimmed = line.trim();

        if !in_block && (trimmed.starts_with("```") || trimmed.starts_with("~~~")) {
            in_block = true;
            current_fence = if trimmed.starts_with('`') {
                "```"
            } else {
                "~~~"
            };
            let lang = trimmed.strip_prefix(current_fence).unwrap_or("").trim();
            current_lang = if lang.is_empty() {
                None
            } else {
                Some(lang.to_string())
            };
            current_content.clear();
            continue;
        }

        if in_block {
            if trimmed == current_fence || trimmed.starts_with(current_fence) {
                // Use take/mem::take to move values out without cloning.
                let lang = current_lang.take();
                let block_content = std::mem::take(&mut current_content);
                if !block_content.is_empty() {
                    blocks.push((lang, block_content));
                }
                in_block = false;
                continue;
            }

            if !current_content.is_empty() {
                current_content.push('\n');
            }
            current_content.push_str(line);
        }
    }

    // Handle unclosed code block
    if in_block && !current_content.is_empty() {
        blocks.push((current_lang, current_content));
    }

    blocks
}

/// Detect thinking/reasoning blocks in content
pub fn detect_thinking_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();

    // Look for <thinking> tags
    let mut in_thinking = false;
    let mut current = String::new();

    for line in content.lines() {
        if line.contains("<thinking>") || line.contains("<antThinking>") {
            in_thinking = true;
            current.clear();
            continue;
        }

        if line.contains("</thinking>") || line.contains("</antThinking>") {
            if !current.is_empty() {
                blocks.push(current.trim().to_string());
            }
            in_thinking = false;
            current.clear();
            continue;
        }

        if in_thinking {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }

    blocks
}
