//! Stream event sink for REPL turns (direct and harness paths).

use crate::core::stream::EventSink;
use crate::core::traits::OutputRenderer;
use caliche::agent::tools::names::{TOOL_FILESYSTEM, fs_op, is_fs_write_op};

/// Combined sink used for both direct streaming and harness streaming (clears thinking indicator on first event).
pub(crate) struct CombinedSink<'a, R: OutputRenderer> {
    pub(crate) renderer: &'a mut R,
    pub(crate) tool_args: &'a mut std::collections::HashMap<String, serde_json::Value>,
    pub(crate) first_event: &'a mut bool,
    pub(crate) tool_calls_started: &'a mut u64,
    pub(crate) tool_calls_succeeded: &'a mut u64,
    pub(crate) tool_calls_failed: &'a mut u64,
    pub(crate) tool_retries_attempted: &'a mut u64,
    pub(crate) tool_retries_succeeded: &'a mut u64,
    pub(crate) generation_duration_ms: &'a mut Option<u64>,
    pub(crate) session_total_tokens: u64,
    pub(crate) session_turn_count: u32,
    /// Thinking text kept for session persistence. Display is gated separately.
    pub(crate) thinking: &'a mut String,
}

impl<'a, R: OutputRenderer> EventSink for CombinedSink<'a, R> {
    fn on_text_chunk(&mut self, text: &str) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        self.renderer.finish_thinking();
        self.renderer.render_chunk(text);
    }

    fn on_tool_start(&mut self, name: &str, args: &serde_json::Value) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        self.renderer.render_tool_start(name, args);
        *self.tool_calls_started += 1;
        if name == TOOL_FILESYSTEM {
            let Some(op) = fs_op(args) else {
                return;
            };
            if is_fs_write_op(op)
                && let Some(path_str) = args.get("path").and_then(|v| v.as_str())
            {
                let tool_key = format!("{}:{}:{}", name, op, path_str);
                self.tool_args.insert(tool_key, args.clone());
            }
        }
    }

    fn on_tool_complete(&mut self, name: &str, result: &str) {
        self.on_tool_complete_with_payload(name, result, None);
    }

    fn on_tool_complete_with_payload(
        &mut self,
        name: &str,
        result: &str,
        payload: Option<&serde_json::Value>,
    ) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        self.renderer
            .render_tool_complete_with_payload(name, result, payload);
        let trimmed = result.trim();
        let is_failure = trimmed.starts_with("Error")
            || trimmed.contains("\"success\":false")
            || trimmed.contains("Tool call validation failed");
        if let Some(idx) = trimmed.find("[tool retry]: attempted ") {
            let suffix = &trimmed[idx + "[tool retry]: attempted ".len()..];
            if let Some(raw_num) = suffix.split_whitespace().next()
                && let Ok(n) = raw_num.parse::<u64>()
            {
                *self.tool_retries_attempted += n;
                if !is_failure {
                    *self.tool_retries_succeeded += 1;
                }
            }
        }
        if is_failure {
            *self.tool_calls_failed += 1;
        } else {
            *self.tool_calls_succeeded += 1;
        }
    }

    fn on_metadata(
        &mut self,
        _usage: Option<&caliche::messages::TokenUsage>,
        metrics: Option<&caliche::messages::PerformanceMetrics>,
    ) {
        *self.generation_duration_ms = metrics.and_then(|m| m.llm_duration_ms);
    }

    fn on_status(&mut self, message: &str) {
        self.renderer.show_waiting_status(message);
        // The next output clears the status line as it clears the indicator.
        *self.first_event = true;
    }

    fn on_thinking(&mut self, content: &str) {
        self.thinking.push_str(content);
        if !self.renderer.output_verbosity().show_thinking() {
            return;
        }
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        self.renderer.render_thinking(content);
    }

    fn on_complete(&mut self, usage: &caliche::messages::TokenUsage) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        // Include current turn's tokens in the session total for display.
        let turn_tokens =
            usage.prompt_tokens.unwrap_or(0) as u64 + usage.completion_tokens.unwrap_or(0) as u64;
        self.renderer.set_session_token_info(
            self.session_total_tokens + turn_tokens,
            self.session_turn_count + 1,
        );
        self.renderer
            .set_generation_duration_ms(*self.generation_duration_ms);
        self.renderer.render_token_usage(usage);
        if let Some(summary) = agent_process_summary() {
            self.renderer.show_status(&summary);
        }
    }

    fn on_phase_transition(&mut self, phase_index: usize, phase_total: usize, tools: &[String]) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        let tools_str = tools.join(", ");
        self.renderer.show_status(&format!(
            "Phase {} / {}: {}",
            phase_index + 1,
            phase_total,
            if tools_str.is_empty() {
                "planning".to_string()
            } else {
                tools_str
            }
        ));
    }

    fn on_turn_start(&mut self) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        if self.renderer.output_verbosity().show_lifecycle() {
            self.renderer.show_status("turn.start");
        }
    }

    fn on_turn_complete(&mut self) {
        if self.renderer.output_verbosity().show_lifecycle() {
            self.renderer.show_status("turn.complete");
        }
    }

    fn on_agent_start(&mut self) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        if self.renderer.output_verbosity().show_lifecycle() {
            self.renderer.show_status("agent.start");
        }
    }

    fn on_agent_end(&mut self) {
        if self.renderer.output_verbosity().show_lifecycle() {
            self.renderer.show_status("agent.end");
        }
    }

    fn on_limit_reached(&mut self, _max_iterations: usize, _usage: &caliche::messages::TokenUsage) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
    }

    fn on_compaction(
        &mut self,
        tokens_before: usize,
        tokens_after: usize,
        messages_removed: usize,
        summary: Option<&str>,
    ) {
        if *self.first_event {
            self.renderer.clear_thinking_indicator();
            *self.first_event = false;
        }
        let mut msg = format!(
            "compaction: {tokens_before}→{tokens_after} tokens, removed {messages_removed} message(s)"
        );
        if let Some(summary) = summary.filter(|s| !s.is_empty()) {
            msg.push_str(": ");
            msg.push_str(summary);
        }
        self.renderer.show_status(&msg);
    }
}

pub(crate) fn agent_process_summary() -> Option<String> {
    let processes = caliche::agent::tools::process_tracker::tracked_processes_for_turn_end();
    if processes.is_empty() {
        return None;
    }

    let mut lines = Vec::with_capacity(processes.len() + 1);
    lines.push("Agent-spawned processes still need attention:".to_string());
    for process in processes.iter().take(8) {
        let state = match process.state {
            caliche::agent::tools::process_tracker::TrackedProcessState::Active => "active",
            caliche::agent::tools::process_tracker::TrackedProcessState::Lingering => "lingering",
        };
        let location = process
            .cwd
            .as_ref()
            .map(|path| format!(" cwd={}", path.display()))
            .unwrap_or_default();
        let pid = process
            .pid
            .map(|pid| format!(" pid={pid}"))
            .unwrap_or_default();
        let pgid = process
            .process_group_id
            .map(|pgid| format!(" pgid={pgid}"))
            .unwrap_or_default();
        let live_pids = if process.live_pids.is_empty() {
            String::new()
        } else {
            format!(" live_pids={:?}", process.live_pids)
        };
        lines.push(format!(
            "- #{id} {state} tool={tool}{pid}{pgid}{live_pids} elapsed={elapsed}s{location}: {command}",
            id = process.id,
            tool = process.tool_name,
            elapsed = process.elapsed_secs,
            command = process.command,
        ));
    }
    if processes.len() > 8 {
        lines.push(format!(
            "- ... {} more tracked process(es)",
            processes.len() - 8
        ));
    }
    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::CombinedSink;
    use crate::core::slash_commands::SlashCommandResult;
    use crate::core::stream::EventSink;
    use crate::core::traits::OutputRenderer;
    use caliche::observability::OutputVerbosity;

    struct Rec {
        chunks: Vec<String>,
        thinking: Vec<String>,
        verbosity: OutputVerbosity,
    }

    impl OutputRenderer for Rec {
        fn handle_command_result(&mut self, _result: SlashCommandResult) {}
        fn render_chunk(&mut self, text: &str) {
            self.chunks.push(text.to_string());
        }
        fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
        fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
        fn render_tool_error(&mut self, _name: &str, _error: &str) {}
        fn render_thinking(&mut self, content: &str) {
            self.thinking.push(content.to_string());
        }
        fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
        fn show_error(&mut self, _message: &str) {}
        fn show_status(&mut self, _message: &str) {}
        fn clear(&mut self) {}
        fn output_verbosity(&self) -> OutputVerbosity {
            self.verbosity
        }
    }

    fn sink<'a>(
        renderer: &'a mut Rec,
        thinking: &'a mut String,
        first_event: &'a mut bool,
        tool_args: &'a mut std::collections::HashMap<String, serde_json::Value>,
        started: &'a mut u64,
        succeeded: &'a mut u64,
        failed: &'a mut u64,
        retries: &'a mut u64,
        retries_ok: &'a mut u64,
        duration: &'a mut Option<u64>,
    ) -> CombinedSink<'a, Rec> {
        CombinedSink {
            renderer,
            tool_args,
            first_event,
            tool_calls_started: started,
            tool_calls_succeeded: succeeded,
            tool_calls_failed: failed,
            tool_retries_attempted: retries,
            tool_retries_succeeded: retries_ok,
            generation_duration_ms: duration,
            session_total_tokens: 0,
            session_turn_count: 0,
            thinking,
        }
    }

    #[test]
    fn normal_verbosity_records_thinking_without_displaying_it() {
        let mut captured = String::new();
        let mut first_event = true;
        let mut tool_args = std::collections::HashMap::new();
        let mut started = 0;
        let mut succeeded = 0;
        let mut failed = 0;
        let mut retries = 0;
        let mut retries_ok = 0;
        let mut duration = None;
        let mut renderer = Rec {
            chunks: Vec::new(),
            thinking: Vec::new(),
            verbosity: OutputVerbosity::Normal,
        };
        {
            let mut events = sink(
                &mut renderer,
                &mut captured,
                &mut first_event,
                &mut tool_args,
                &mut started,
                &mut succeeded,
                &mut failed,
                &mut retries,
                &mut retries_ok,
                &mut duration,
            );
            events.on_thinking("hidden cot");
            events.on_text_chunk("visible");
        }
        assert_eq!(captured, "hidden cot");
        assert!(renderer.thinking.is_empty());
        assert_eq!(renderer.chunks, vec!["visible".to_string()]);
    }
}
