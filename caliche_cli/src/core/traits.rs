//! Core traits for input/output abstraction.

use crate::core::slash_commands::SlashCommandResult;
use anyhow::Result;
use async_trait::async_trait;
use caliche::observability::OutputVerbosity;

/// Trait for input sources (terminal, test, etc.)
#[async_trait]
pub trait InputSource: Send {
    /// Read a line of input
    async fn read_line(&mut self, prompt: &str) -> Result<Option<String>>;

    /// Seed the next prompt with prefilled text.
    fn set_prefill_buffer(&mut self, _text: String) {}

    /// Drop stdin bytes left over from a finished agent turn.
    ///
    /// Call this after a streaming turn so Enter pressed during the turn does
    /// not become the next prompt submission. Slash commands and blank lines
    /// must not set this.
    fn request_stdin_drain(&mut self) {}

    /// Add a line to history
    fn add_history(&mut self, line: &str);

    /// Save history to disk
    fn save_history(&mut self) -> Result<()>;
}

/// Trait for output renderers (REPL, JSON, etc.)
pub trait OutputRenderer: Send {
    /// Handle a command result (popup, status, etc.)
    fn handle_command_result(&mut self, result: SlashCommandResult);

    /// Render a text chunk during streaming
    fn render_chunk(&mut self, text: &str);

    /// Render tool execution start
    fn render_tool_start(&mut self, name: &str, args: &serde_json::Value);

    /// Render tool execution completion
    fn render_tool_complete(&mut self, name: &str, result: &str);

    /// Render tool execution completion with structured metadata when available.
    fn render_tool_complete_with_payload(
        &mut self,
        name: &str,
        result: &str,
        _payload: Option<&serde_json::Value>,
    ) {
        self.render_tool_complete(name, result);
    }

    /// Render tool execution error
    fn render_tool_error(&mut self, name: &str, error: &str);

    /// Render thinking/reasoning
    fn render_thinking(&mut self, content: &str);

    /// Close a streamed thinking block before assistant text or tools.
    fn finish_thinking(&mut self) {}

    /// Render token usage after response completion
    fn render_token_usage(&mut self, usage: &caliche::messages::TokenUsage);

    /// Set generation-only duration for the next token usage render.
    fn set_generation_duration_ms(&mut self, _duration_ms: Option<u64>) {}

    /// Show error message
    fn show_error(&mut self, message: &str);

    /// Show status message
    fn show_status(&mut self, message: &str);

    /// Clear the output
    fn clear(&mut self);

    /// Show a transient "thinking" indicator before the first stream event (optional).
    fn show_thinking_indicator(&mut self) {}

    /// Clear the thinking indicator when the first chunk or tool event arrives (optional).
    fn clear_thinking_indicator(&mut self) {}

    /// Replace the thinking indicator with `message`, for example while a
    /// local model loads. The next output clears it like the indicator.
    fn show_waiting_status(&mut self, message: &str) {
        self.show_status(message);
    }

    /// Flush any buffered output and end the current turn on a clean line.
    fn finish_turn(&mut self) {}

    /// Set verbose display mode (show full tool output).
    fn set_verbose(&mut self, _v: bool) {}

    /// Set compact display mode (suppress tool args/results).
    fn set_compact(&mut self, _v: bool) {}

    /// Set REPL output verbosity (tools, thinking, cache detail).
    fn set_output_verbosity(&mut self, _verbosity: OutputVerbosity) {}

    /// Current REPL output verbosity.
    fn output_verbosity(&self) -> OutputVerbosity {
        OutputVerbosity::Normal
    }

    /// Update session-level token tracking for display in usage lines.
    fn set_session_token_info(&mut self, _total_tokens: u64, _turn_count: u32) {}
}
