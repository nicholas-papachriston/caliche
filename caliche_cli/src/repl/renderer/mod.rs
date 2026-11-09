//! REPL renderer implementation for OutputRenderer trait.

mod panels;
mod text;

use text::{
    BLUE, BOLD, CYAN, DIM, GREEN, RED, RESET, YELLOW, prompt_cache_usage_parts, truncate_str,
    turn_rule,
};

use crate::core::slash_commands::SlashCommandResult;
use crate::core::traits::OutputRenderer;
use crate::repl::io::OutputHandler;
use caliche::observability::OutputVerbosity;
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::time::{Duration, Instant};

/// The indicator shown before the first stream event of a turn.
pub(crate) const THINKING_INDICATOR: &str = "Thinking...";

/// Max length for one-line tool result preview (use /verbose for full output).
const TOOL_PREVIEW_MAX: usize = 200;

/// Max length for tool args summary (truncate verbose JSON).
const TOOL_ARGS_MAX: usize = 80;
const STREAM_FLUSH_INTERVAL: Duration = Duration::from_millis(40);
const STREAM_FLUSH_BYTES: usize = 256;

pub struct ReplRenderer {
    output: Box<dyn OutputHandler>,
    tool_outputs: HashMap<String, String>,
    tool_call_counter: u32,
    tool_keys_by_name: HashMap<String, VecDeque<String>>,
    active_tool_calls: usize,
    /// True when `render_chunk` left the cursor mid-line (no trailing newline).
    needs_newline: bool,
    /// Buffered streaming text used when output is routed through reedline's external printer.
    pending_chunk: String,
    /// Last time a partial chunk was flushed while streaming.
    last_chunk_flush_at: Option<Instant>,
    /// Current tool call round (incremented after each batch of parallel calls).
    tool_round: u32,
    /// How much turn detail to print.
    verbosity: OutputVerbosity,
    /// True while a thinking/reasoning stream is open.
    thinking_open: bool,
    /// Cumulative session token count (set by engine after each turn).
    session_total_tokens: u64,
    /// Number of completed turns in this session.
    session_turn_count: u32,
    /// Generation-only duration for the current turn, excluding tool execution.
    generation_duration_ms: Option<u64>,
}

impl ReplRenderer {
    /// Create a new REPL renderer
    pub fn new(output: Box<dyn OutputHandler>) -> Self {
        Self {
            output,
            tool_outputs: HashMap::new(),
            tool_call_counter: 0,
            tool_keys_by_name: HashMap::new(),
            active_tool_calls: 0,
            needs_newline: false,
            pending_chunk: String::new(),
            last_chunk_flush_at: None,
            tool_round: 0,
            verbosity: OutputVerbosity::Normal,
            thinking_open: false,
            session_total_tokens: 0,
            session_turn_count: 0,
            generation_duration_ms: None,
        }
    }

    /// Ensure stdout is on a fresh line before printing structured output.
    fn ensure_newline(&mut self) {
        self.flush_pending_chunk(true);
        if self.needs_newline {
            crate::repl::stdout_sync::with_stdout(|out| {
                if let Err(err) = out.write_all(b"\n") {
                    tracing::warn!(target: "repl::renderer", "failed to write newline: {}", err);
                    return;
                }
                if let Err(err) = out.flush() {
                    tracing::warn!(target: "repl::renderer", "failed to flush newline: {}", err);
                }
            });
            self.needs_newline = false;
        }
    }

    fn flush_pending_chunk(&mut self, force: bool) {
        if self.pending_chunk.is_empty() {
            return;
        }

        let flush_all = |pending: &mut String| {
            let text = std::mem::take(pending);
            crate::repl::stdout_sync::with_stdout(|out| {
                use std::io::Write;
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
            });
        };

        if force {
            flush_all(&mut self.pending_chunk);
            self.last_chunk_flush_at = Some(Instant::now());
            return;
        }

        // Flush complete lines
        while let Some(idx) = self.pending_chunk.find('\n') {
            let line_end = idx + 1; // include the newline
            let line: String = self.pending_chunk.drain(..line_end).collect();
            crate::repl::stdout_sync::with_stdout(|out| {
                use std::io::Write;
                let _ = out.write_all(line.as_bytes());
                let _ = out.flush();
            });
        }

        let now = Instant::now();
        let should_flush_partial = self.pending_chunk.len() >= STREAM_FLUSH_BYTES
            || self
                .last_chunk_flush_at
                .is_none_or(|last| now.duration_since(last) >= STREAM_FLUSH_INTERVAL);
        if should_flush_partial && !self.pending_chunk.is_empty() {
            flush_all(&mut self.pending_chunk);
            self.last_chunk_flush_at = Some(now);
        }
    }

    fn begin_turn(&mut self) {
        self.flush_pending_chunk(true);
        self.needs_newline = false;
    }

    fn close_thinking_stream(&mut self) {
        if !self.thinking_open {
            return;
        }
        self.render_chunk(RESET);
        self.thinking_open = false;
    }

    fn finish_turn_output(&mut self) {
        self.close_thinking_stream();
        self.flush_pending_chunk(true);
        if self.needs_newline {
            crate::repl::stdout_sync::write_newline();
            self.needs_newline = false;
        }
        self.last_chunk_flush_at = None;
    }

    /// Summarize tool args for display; avoid dumping full JSON (e.g. fs edit operations).
    /// Prioritizes showing op/pattern/action for fs/workspace tools.
    fn summarize_tool_args(args: &serde_json::Value) -> String {
        let raw = args.to_string();
        if raw.len() <= TOOL_ARGS_MAX {
            return raw;
        }
        let obj = match args.as_object() {
            Some(o) => o,
            None => return format!("{}...", truncate_str(&raw, TOOL_ARGS_MAX.saturating_sub(3))),
        };
        let mut parts = Vec::new();

        // Priority fields for fs tool: show op first
        if let Some(op) = obj.get("op").and_then(|v| v.as_str()) {
            parts.push(format!("op={}", op));
        }

        // Priority field for workspace tool: show action
        if let Some(action) = obj.get("action").and_then(|v| v.as_str()) {
            parts.push(format!("action={}", action));
        }

        // Show pattern for grep operations
        if let Some(pattern) = obj.get("pattern").and_then(|v| v.as_str()) {
            let p_preview = if pattern.len() > 30 {
                format!("{}...", truncate_str(pattern, 27))
            } else {
                pattern.to_string()
            };
            parts.push(format!("pattern=\"{}\"", p_preview));
        }

        // Show path (but not if it would be redundant with op=grep at root)
        if let Some(path) = obj.get("path").and_then(|v| v.as_str()) {
            // Only show path if it's meaningful (not just "." when we have other info)
            let is_root = path == "." || path == "./";
            let already_has_content = parts.len() >= 2 || obj.contains_key("pattern");
            if !is_root || !already_has_content {
                parts.push(format!("path={}", path));
            }
        }

        // Fallback for edit operations
        if let Some(ops) = obj.get("operations").and_then(|v| v.as_array()) {
            parts.push(format!("{} ops", ops.len()));
        }

        // Fallback for query-based reads
        if let Some(q) = obj.get("query").and_then(|v| v.as_str()) {
            let q_preview = if q.len() > 30 {
                format!("{}...", truncate_str(q, 27))
            } else {
                q.to_string()
            };
            parts.push(format!("query=\"{}\"", q_preview));
        }

        if parts.is_empty() {
            format!("{}...", truncate_str(&raw, TOOL_ARGS_MAX.saturating_sub(3)))
        } else {
            parts.join(" | ")
        }
    }

    fn tool_style(name: &str, args: &serde_json::Value) -> (&'static str, &'static str) {
        if name == "fs" {
            match args.get("op").and_then(|v| v.as_str()) {
                Some("read") => ("read", BLUE),
                Some("grep") | Some("glob") | Some("ls") => ("scan", BLUE),
                Some("edit") => ("patch", YELLOW),
                Some("write") => ("write", YELLOW),
                Some("delete") => ("remove", RED),
                _ => ("file", CYAN),
            }
        } else if name.contains("search") || name.contains("grep") {
            ("scan", BLUE)
        } else if name.contains("bash") || name.contains("shell") || name.contains("exec") {
            ("exec", YELLOW)
        } else if name.contains("web") || name.contains("fetch") {
            ("fetch", CYAN)
        } else {
            ("run", CYAN)
        }
    }

    fn result_preview(result: &str) -> (&'static str, &'static str, String) {
        let trimmed = result.trim();
        if trimmed.starts_with("Tool call validation failed") {
            return ("blocked", YELLOW, trimmed.to_string());
        }
        if trimmed.starts_with("Error:") || trimmed.starts_with("Error ") {
            return ("error", RED, trimmed.to_string());
        }
        if trimmed.is_empty() {
            return ("ok", GREEN, "completed".to_string());
        }

        let one_line = trimmed.replace('\n', " ");
        let max_bytes = TOOL_PREVIEW_MAX.saturating_sub(40);
        let preview = if one_line.len() > max_bytes {
            format!(
                "{}… (use /verbose for full output)",
                truncate_str(&one_line, max_bytes)
            )
        } else {
            one_line
        };
        ("ok", GREEN, preview)
    }

    fn tool_duration_ms(payload: Option<&serde_json::Value>) -> Option<u64> {
        payload
            .and_then(|value| value.get("duration_ms"))
            .and_then(serde_json::Value::as_u64)
    }

    fn format_duration(duration_ms: u64) -> String {
        if duration_ms >= 1_000 {
            format!("{:.1}s", duration_ms as f64 / 1_000.0)
        } else {
            format!("{duration_ms}ms")
        }
    }

    fn format_tokens_per_second(completion_tokens: u32, duration_ms: u64) -> Option<String> {
        if completion_tokens == 0 || duration_ms == 0 {
            return None;
        }
        let seconds = duration_ms as f64 / 1_000.0;
        if seconds <= f64::EPSILON {
            return None;
        }
        Some(format!("{:.1} tok/s", completion_tokens as f64 / seconds))
    }

    fn render_tool_complete_with_duration(
        &mut self,
        name: &str,
        result: &str,
        duration_ms: Option<u64>,
    ) {
        self.ensure_newline();

        let key = self
            .tool_keys_by_name
            .get_mut(name)
            .and_then(VecDeque::pop_front);
        if let Some(key) = key.as_ref()
            && let Some(output) = self.tool_outputs.get_mut(key)
        {
            output.push_str(result);
        }

        self.active_tool_calls = self.active_tool_calls.saturating_sub(1);
        if let Some(key) = key {
            self.tool_outputs.remove(&key);
        }

        let duration = duration_ms.map(Self::format_duration);
        if !self.verbosity.show_tools() {
            return;
        }
        if self.verbosity.show_full_tool_output() {
            let (status, color, _) = Self::result_preview(result);
            let label = duration
                .as_deref()
                .map_or_else(|| status.to_string(), |ms| format!("{status} {ms}"));
            self.output
                .print(&format!("      {}{}{}", color, label, RESET));
            for line in result.lines() {
                self.output
                    .print(&format!("      {}{}{}", DIM, line, RESET));
            }
        } else {
            let (status, color, preview) = Self::result_preview(result);
            let label = duration
                .as_deref()
                .map_or_else(|| status.to_string(), |ms| format!("{status} {ms}"));
            self.output.print(&format!(
                "      {}{}{}  {}{}{}",
                color, label, RESET, DIM, preview, RESET
            ));
        }
    }
}

impl ReplRenderer {
    /// Show "Thinking..." before first stream chunk (called from engine).
    pub fn show_thinking_indicator_impl(&mut self) {
        self.ensure_newline();
        self.write_indicator_line(THINKING_INDICATOR);
    }

    /// Write `text` dimmed on the indicator line, replacing what is there.
    /// Without terminal control, print it as a line instead.
    fn write_indicator_line(&mut self, text: &str) {
        if !crate::repl::terminal::supports_ansi_terminal_control() {
            self.output.print(text);
            return;
        }
        crate::repl::stdout_sync::with_stdout(|out| {
            if let Err(err) = write!(out, "\r\x1b[K\x1b[90m{text}\x1b[0m") {
                tracing::warn!(
                    target: "repl::renderer",
                    "failed to write thinking indicator: {}",
                    err
                );
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(
                    target: "repl::renderer",
                    "failed to flush thinking indicator: {}",
                    err
                );
            }
        });
    }

    /// Clear the thinking line so the first chunk overwrites it.
    pub fn clear_thinking_indicator_impl(&mut self) {
        if !crate::repl::terminal::supports_ansi_terminal_control() {
            return;
        }
        crate::repl::stdout_sync::with_stdout(|out| {
            if let Err(err) = out.write_all(b"\r\x1b[K") {
                tracing::warn!(
                    target: "repl::renderer",
                    "failed to clear thinking indicator: {}",
                    err
                );
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(
                    target: "repl::renderer",
                    "failed to flush cleared thinking indicator: {}",
                    err
                );
            }
        });
    }
}

impl OutputRenderer for ReplRenderer {
    fn handle_command_result(&mut self, result: SlashCommandResult) {
        match result {
            SlashCommandResult::Continue => {}
            SlashCommandResult::Exit => {}
            SlashCommandResult::ShowHelp(commands) => panels::render_help(self, &commands),
            SlashCommandResult::ShowPopup { title, content } => {
                self.output.print(&format!("\n=== {} ===", title));
                for line in content.lines() {
                    self.output.print(line);
                }
                self.output.print("");
            }
            SlashCommandResult::ShowList {
                title,
                items,
                descriptions,
            } => panels::render_list(self, &title, &items, &descriptions),
            SlashCommandResult::StatusMessage(msg) => {
                self.output.print_success(&msg);
            }
            SlashCommandResult::Error(msg) => {
                self.output.print_error(&msg);
            }
            SlashCommandResult::ModeChanged(mode) => {
                self.output
                    .print_success(&format!("Switched to {} mode", mode.name()));
            }
            SlashCommandResult::ShowTokenUsage {
                session_tokens,
                context_tokens,
                cache_creation_tokens,
                cache_read_tokens,
                session_prompt_tokens,
            } => panels::render_token_usage_popup(
                self,
                session_tokens,
                context_tokens,
                cache_creation_tokens,
                cache_read_tokens,
                session_prompt_tokens,
            ),
            SlashCommandResult::ShowBudget {
                session_tokens,
                context_tokens,
            } => {
                use crate::cost::format_tokens;
                self.output.print(&format!(
                    "Token Budget\n============\n\nSession Tokens: {}\nContext Tokens: {}\n\nMax context varies by model.",
                    format_tokens(session_tokens),
                    format_tokens(context_tokens)
                ));
            }
            SlashCommandResult::ShowStats {
                total_tokens,
                message_count,
                provider,
                model,
                quality_summary,
            } => {
                self.output.print(&format!(
                    "Statistics\n==========\n\nMessages: {}\nTokens: {}\nProvider: {}\nModel: {}\n\n{}",
                    message_count, total_tokens, provider, model, quality_summary
                ));
            }
            SlashCommandResult::EditorContent(_) => {
                // Editor content is handled specially in the REPL engine
                // This shouldn't be rendered directly
            }
        }
    }

    fn render_chunk(&mut self, text: &str) {
        // Always write directly to stdout during streaming.
        // The external printer routes through a bounded crossbeam channel
        // consumed by the reedline event loop, which is NOT running while we
        // stream. Using it here deadlocks once the channel fills up.
        crate::repl::stdout_sync::with_stdout(|out| {
            if let Err(err) = out.write_all(text.as_bytes()) {
                tracing::warn!(target: "repl::renderer", "failed to render chunk: {}", err);
                return;
            }
            if let Err(err) = out.flush() {
                tracing::warn!(target: "repl::renderer", "failed to flush chunk: {}", err);
            }
        });
        self.needs_newline = !text.is_empty() && !text.ends_with('\n');
    }

    fn render_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        self.close_thinking_stream();
        self.ensure_newline();

        if self.active_tool_calls == 0 {
            self.tool_round += 1;
            if self.verbosity.show_tools() {
                self.output
                    .print(&format!("  {}Round {}{}", DIM, self.tool_round, RESET));
            }
        }

        self.tool_call_counter += 1;
        let key = format!("{}#{}", name, self.tool_call_counter);
        self.tool_outputs.insert(key.clone(), String::new());
        self.tool_keys_by_name
            .entry(name.to_string())
            .or_default()
            .push_back(key);
        self.active_tool_calls += 1;

        let ordinal = self.tool_call_counter;
        if !self.verbosity.show_tools() {
            return;
        }
        let (badge, color) = Self::tool_style(name, args);
        let title = format!(
            "    {}{:02}{}  {}{}{}  {}{}{}",
            DIM, ordinal, RESET, color, badge, RESET, BOLD, name, RESET
        );
        self.output.print(&title);
        if self.verbosity.show_tool_args() {
            let summary = Self::summarize_tool_args(args);
            if !summary.is_empty() {
                self.output
                    .print(&format!("      {}{}{}", DIM, summary, RESET));
            }
        }
    }

    fn render_tool_complete(&mut self, name: &str, result: &str) {
        self.render_tool_complete_with_duration(name, result, None);
    }

    fn render_tool_complete_with_payload(
        &mut self,
        name: &str,
        result: &str,
        payload: Option<&serde_json::Value>,
    ) {
        self.render_tool_complete_with_duration(name, result, Self::tool_duration_ms(payload));
    }

    fn render_tool_error(&mut self, name: &str, error: &str) {
        self.ensure_newline();
        self.output
            .print_error(&format!("      failed {}: {}", name, error));
        let key = self
            .tool_keys_by_name
            .get_mut(name)
            .and_then(VecDeque::pop_front);
        self.active_tool_calls = self.active_tool_calls.saturating_sub(1);
        if let Some(key) = key {
            self.tool_outputs.remove(&key);
        }
    }

    fn render_thinking(&mut self, content: &str) {
        if !self.verbosity.show_thinking() || content.is_empty() {
            return;
        }
        if !self.thinking_open {
            self.ensure_newline();
            self.output.print(&format!("  {}thinking{}", DIM, RESET));
            self.thinking_open = true;
            self.render_chunk(DIM);
        }
        self.render_chunk(content);
    }

    fn render_token_usage(&mut self, usage: &caliche::messages::TokenUsage) {
        use crate::cost::format_tokens;

        self.close_thinking_stream();
        if !self.verbosity.show_usage() {
            self.generation_duration_ms = None;
            return;
        }

        self.ensure_newline();
        let mut parts = Vec::new();

        if let Some(prompt) = usage.prompt_tokens {
            parts.push(format!("prompt: {}", format_tokens(prompt as u64)));
        }
        if let Some(last) = usage.last_prompt_tokens
            && usage.prompt_tokens.is_some_and(|sum| sum != last)
        {
            parts.push(format!("last prompt: {}", format_tokens(u64::from(last))));
        }

        if let Some(completion) = usage.completion_tokens {
            parts.push(format!("completion: {}", format_tokens(completion as u64)));
            if let Some(duration_ms) = self.generation_duration_ms
                && let Some(rate) = Self::format_tokens_per_second(completion, duration_ms)
            {
                parts.push(format!(
                    "speed: {} ({})",
                    rate,
                    Self::format_duration(duration_ms)
                ));
            }
        }

        if let Some(embedding) = usage.embedding_tokens {
            parts.push(format!("embedding: {}", format_tokens(embedding as u64)));
        }

        if let Some(total) = usage.total_tokens {
            parts.push(format!("total: {}", format_tokens(total as u64)));
        }

        parts.extend(prompt_cache_usage_parts(usage, self.verbosity));

        // `prompt_tokens` sums every model call in the turn. The window
        // percentage is the latest request, which is the live context size.
        let context_tokens = usage.last_prompt_tokens.or(usage.prompt_tokens);
        if let (Some(prompt), Some(window)) = (context_tokens, usage.context_window) {
            let pct = (prompt as f64 / window as f64) * 100.0;
            parts.push(format!(
                "context: {:.0}% of {}",
                pct,
                format_tokens(window as u64)
            ));
        }

        if !parts.is_empty() {
            let token_info = parts.join(" | ");
            self.output
                .print(&format!("{}{}{}", DIM, token_info, RESET));
            if self.session_total_tokens > 0 {
                self.output.print(&format!(
                    "{}session total: {} tokens ({} turn{}){}",
                    DIM,
                    format_tokens(self.session_total_tokens),
                    self.session_turn_count,
                    if self.session_turn_count == 1 {
                        ""
                    } else {
                        "s"
                    },
                    RESET,
                ));
            }
            self.output.print(&turn_rule());
            self.output.print("");
        }
        self.generation_duration_ms = None;
    }

    fn set_generation_duration_ms(&mut self, duration_ms: Option<u64>) {
        self.generation_duration_ms = duration_ms;
    }

    fn show_error(&mut self, message: &str) {
        self.ensure_newline();
        self.output.print_error(message);
    }

    fn show_status(&mut self, message: &str) {
        self.ensure_newline();
        // Dim gray keeps status messages visually light — they're informational,
        // not success confirmations, and shouldn't outshine the agent's output.
        self.output.print(&format!("{}{}{}", DIM, message, RESET));
    }

    fn clear(&mut self) {
        self.output.clear();
    }

    fn show_thinking_indicator(&mut self) {
        self.begin_turn();
        self.show_thinking_indicator_impl();
    }

    fn clear_thinking_indicator(&mut self) {
        self.clear_thinking_indicator_impl();
    }

    fn show_waiting_status(&mut self, message: &str) {
        self.write_indicator_line(message);
    }

    fn finish_turn(&mut self) {
        self.finish_turn_output();
    }

    fn finish_thinking(&mut self) {
        self.close_thinking_stream();
    }

    fn set_verbose(&mut self, v: bool) {
        if v {
            self.verbosity = OutputVerbosity::Verbose;
        } else if self.verbosity == OutputVerbosity::Quiet {
            self.verbosity = OutputVerbosity::Quiet;
        } else {
            self.verbosity = OutputVerbosity::Normal;
        }
    }

    fn set_compact(&mut self, v: bool) {
        self.verbosity = if v {
            OutputVerbosity::Quiet
        } else if self.verbosity == OutputVerbosity::Quiet {
            OutputVerbosity::Normal
        } else {
            self.verbosity
        };
    }

    fn set_output_verbosity(&mut self, verbosity: OutputVerbosity) {
        self.verbosity = verbosity;
    }

    fn output_verbosity(&self) -> OutputVerbosity {
        self.verbosity
    }

    fn set_session_token_info(&mut self, total_tokens: u64, turn_count: u32) {
        self.session_total_tokens = total_tokens;
        self.session_turn_count = turn_count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::slash_commands::SlashCommandResult;
    use crate::core::traits::OutputRenderer;
    use parking_lot::Mutex;
    use std::sync::Arc;

    struct CaptureOutput {
        lines: Arc<Mutex<Vec<String>>>,
    }

    impl OutputHandler for CaptureOutput {
        fn print(&mut self, message: &str) {
            self.lines.lock().push(message.to_string());
        }

        fn print_error(&mut self, message: &str) {
            self.lines.lock().push(format!("error:{message}"));
        }

        fn print_success(&mut self, message: &str) {
            self.lines.lock().push(format!("ok:{message}"));
        }

        fn clear(&mut self) {}
    }

    #[test]
    fn help_command_prints_available_commands() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let mut renderer = ReplRenderer::new(Box::new(CaptureOutput {
            lines: Arc::clone(&lines),
        }));
        renderer.handle_command_result(SlashCommandResult::ShowHelp(vec![
            ("/help".to_string(), "Show help".to_string()),
            ("/exit".to_string(), "Exit".to_string()),
        ]));
        let printed = lines.lock().join("\n");
        assert!(
            printed.contains("Available Commands"),
            "help output was empty or missing the command list: {printed:?}"
        );
        assert!(printed.contains("/help"));
        assert!(printed.contains("/exit"));
    }

    #[test]
    fn thinking_is_hidden_at_normal_verbosity() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let mut renderer = ReplRenderer::new(Box::new(CaptureOutput {
            lines: Arc::clone(&lines),
        }));
        renderer.render_thinking("secret plan");
        assert!(
            lines.lock().is_empty(),
            "normal verbosity must hide thinking text"
        );
        renderer.set_output_verbosity(OutputVerbosity::Thinking);
        renderer.render_thinking("secret plan");
        let printed = lines.lock().join("\n");
        assert!(
            printed.contains("thinking"),
            "thinking verbosity must label the thinking stream: {printed:?}"
        );
        renderer.set_output_verbosity(OutputVerbosity::Verbose);
        renderer.render_thinking("secret plan");
        let printed = lines.lock().join("\n");
        assert!(
            printed.contains("thinking"),
            "verbose mode must label the thinking stream: {printed:?}"
        );
    }

    #[test]
    fn token_usage_includes_cache_hit_ratio() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let mut renderer = ReplRenderer::new(Box::new(CaptureOutput {
            lines: Arc::clone(&lines),
        }));
        let usage = caliche::messages::TokenUsage::from_counts(Some(130), Some(4), Some(134))
            .with_cache_tokens(Some(20), Some(100));
        renderer.render_token_usage(&usage);
        let printed = lines.lock().join("\n");
        assert!(printed.contains("cache hit:"));
        assert!(printed.contains("cache read:"));
        assert!(printed.contains("uncached:"));
    }

    #[test]
    fn context_percent_uses_the_latest_prompt() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let mut renderer = ReplRenderer::new(Box::new(CaptureOutput {
            lines: Arc::clone(&lines),
        }));
        let mut usage =
            caliche::messages::TokenUsage::from_counts(Some(1_400_000), Some(25_000), None);
        usage.last_prompt_tokens = Some(20_000);
        usage.context_window = Some(200_000);
        renderer.render_token_usage(&usage);
        let printed = lines.lock().join("\n");
        assert!(
            printed.contains("context: 10% of 200.0K"),
            "context percent used the turn sum: {printed:?}"
        );
        assert!(printed.contains("last prompt: 20.0K"));
        assert!(printed.contains("prompt: 1.4M"));
    }
}
