//! Agent message turns (stream, harness, decomposition).

#[path = "turn_helpers.rs"]
mod turn_helpers;

use super::ReplEngine;
use super::stream_sink::{CombinedSink, agent_process_summary};
use super::turn_quality::{
    IncompleteTurnAction, inject_continue_prompt, record_tool_call_metrics, surface_incomplete_turn,
};
use crate::client::sse_event_payload;
use crate::context_refs::{ContextResolver, ResolvedContext};
use crate::core::stream::StreamProcessor;
use crate::core::traits::{InputSource, OutputRenderer};
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::hooks::{CONTINUE_TASK_PROMPT, is_steering_only_query};
use caliche::messages::ACTION_ID_FIELD;
use caliche::messages::stream_contracts::sse_name;
use serde_json::Value;
use std::path::PathBuf;

/// Read streamed assistant text from a session SSE frame.
///
/// Session events wrap harness payloads as `{version, type, payload}`.
/// Flat JSON from tests and the POST `/turn` fallback is also accepted.
pub(crate) fn thin_client_streamed_text<'a>(event_name: &str, data: &'a Value) -> Option<&'a str> {
    let body = sse_event_payload(data);
    match event_name {
        sse_name::AGENT_THINKING | "text_chunk" | "code_block" => {
            body.get("content").and_then(Value::as_str)
        }
        _ => None,
    }
}

/// Provider and model to post when the prompt label differs from the remote session.
///
/// A matching label returns `None`, so the turn body omits both fields and the
/// remote session keeps the model it already has.
fn thin_client_selection<'a>(
    applied: &Option<(String, String)>,
    provider: &'a str,
    model: &'a str,
) -> Option<(&'a str, &'a str)> {
    let unchanged = applied
        .as_ref()
        .is_some_and(|(applied_provider, applied_model)| {
            applied_provider == provider && applied_model == model
        });
    if unchanged {
        None
    } else {
        Some((provider, model))
    }
}

/// Read the completed-turn assistant text from a session SSE frame.
pub(crate) fn thin_client_final_response<'a>(event_name: &str, data: &'a Value) -> Option<&'a str> {
    let body = sse_event_payload(data);
    match event_name {
        sse_name::TURN_COMPLETE => body.get("response").and_then(Value::as_str),
        "done" => body.get("final_response").and_then(Value::as_str),
        _ => None,
    }
}

#[derive(Default)]
struct ThinClientTurnState {
    response: String,
    usage: caliche::messages::TokenUsage,
    duration_ms: u64,
    streamed_any: bool,
    limit_reached: bool,
    max_iterations: Option<usize>,
    open_thinking: Option<String>,
    /// Chain-of-thought kept for persistence even when the TUI hides it.
    thinking: String,
    announced_thinking: bool,
    /// A `status` line holds the indicator line.
    status_line: bool,
}

fn persisted_thin_client_message(turn: &ThinClientTurnState) -> Option<caliche::messages::Message> {
    if turn.response.trim().is_empty() {
        return None;
    }
    Some(caliche::messages::Message::assistant_with_reasoning(
        &turn.response,
        &turn.thinking,
    ))
}

fn record_thin_client_thinking<R: OutputRenderer>(
    renderer: &mut R,
    turn: &mut ThinClientTurnState,
    body: &Value,
    content: &str,
) {
    turn.thinking.push_str(content);
    if !renderer.output_verbosity().show_thinking() {
        return;
    }
    if !turn.announced_thinking {
        renderer.clear_thinking_indicator();
        turn.announced_thinking = true;
    }
    let action_id = traced_action_id(body);
    if !action_id.is_empty() && turn.open_thinking.as_deref() != Some(action_id.as_str()) {
        close_open_thinking(renderer, turn);
        renderer.show_status(&format!("thinking.start {}", display_action_id(&action_id)));
        turn.open_thinking = Some(action_id);
    }
    if !content.is_empty() {
        renderer.render_thinking(content);
    }
}

fn parse_token_usage_value(value: &Value) -> caliche::messages::TokenUsage {
    caliche::messages::TokenUsage {
        prompt_tokens: value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        completion_tokens: value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        embedding_tokens: value
            .get("embedding_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        context_window: value
            .get("context_window")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        cache_creation_tokens: value
            .get("cache_creation_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        cache_read_tokens: value
            .get("cache_read_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        last_prompt_tokens: value
            .get("last_prompt_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
    }
}

fn close_open_thinking<R: OutputRenderer>(renderer: &mut R, turn: &mut ThinClientTurnState) {
    if let Some(id) = turn.open_thinking.take() {
        renderer.show_status(&format!("thinking.end {}", display_action_id(&id)));
    }
}

/// Status-line suffix for a UUID v7 action id.
///
/// The full id stays on SSE, hooks, and the JSON outcome log. The TUI shows
/// the last 8 hex digits of the final UUID segment so a row stays scannable.
fn display_action_id(id: &str) -> String {
    if id.is_empty() {
        return String::new();
    }
    if let Some((head, tail)) = id.rsplit_once('-')
        && !head.is_empty()
        && !tail.is_empty()
        && tail.chars().all(|ch| ch.is_ascii_hexdigit())
    {
        return last_chars(tail, 8);
    }
    let compact: String = id.chars().filter(|ch| *ch != '-').collect();
    if compact.len() >= 8 && compact.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return last_chars(&compact, 8);
    }
    last_chars(id, 12)
}

fn last_chars(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    text.chars().skip(count - max_chars).collect()
}

fn traced_action_id(body: &Value) -> String {
    body.get(ACTION_ID_FIELD)
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .or_else(|| body.get("tool_id").and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

/// Session event for [`caliche::harness::AgentEvent::Status`].
const STATUS_EVENT: &str = "status";

fn apply_thin_client_sse_event<R: OutputRenderer>(
    renderer: &mut R,
    turn: &mut ThinClientTurnState,
    event_name: &str,
    data: &Value,
) {
    let body = sse_event_payload(data);
    if event_name == STATUS_EVENT {
        if let Some(message) = body.get("message").and_then(Value::as_str) {
            renderer.show_waiting_status(message);
            turn.status_line = true;
        }
        return;
    }
    if std::mem::take(&mut turn.status_line) {
        // The model answered: the status line goes back to the indicator
        // that the existing events clear.
        if turn.streamed_any {
            renderer.clear_thinking_indicator();
        } else {
            renderer.show_waiting_status(crate::repl::renderer::THINKING_INDICATOR);
        }
    }
    match event_name {
        sse_name::AGENT_THINKING | "text_chunk" | "code_block" => {
            if let Some(content) = thin_client_streamed_text(event_name, data) {
                if event_name == sse_name::AGENT_THINKING {
                    record_thin_client_thinking(renderer, turn, body, content);
                } else {
                    close_open_thinking(renderer, turn);
                    if !turn.streamed_any {
                        renderer.clear_thinking_indicator();
                    }
                    turn.streamed_any = true;
                    renderer.render_chunk(content);
                }
            }
        }
        sse_name::TOOL_START => {
            close_open_thinking(renderer, turn);
            let name = body
                .get("tool_name")
                .and_then(Value::as_str)
                .unwrap_or("tool");
            let action_id = traced_action_id(body);
            renderer.show_status(&format!(
                "tool.start {} {name}",
                display_action_id(&action_id)
            ));
        }
        sse_name::TOOL_END | "tool_error" => {
            close_open_thinking(renderer, turn);
            let name = body
                .get("tool_name")
                .and_then(Value::as_str)
                .unwrap_or("tool");
            let ok = event_name != "tool_error"
                && body.get("success").and_then(Value::as_bool).unwrap_or(true);
            let action_id = traced_action_id(body);
            renderer.show_status(&format!(
                "tool.end {} {name} ok={ok}",
                display_action_id(&action_id)
            ));
        }
        sse_name::SESSION_UPDATED => {
            renderer.show_status("session.updated");
        }
        sse_name::TURN_COMPLETE | "done" => {
            close_open_thinking(renderer, turn);
            if let Some(text) = thin_client_final_response(event_name, data)
                && !(event_name == "done" && !turn.response.trim().is_empty())
            {
                turn.response = text.to_string();
            }
            if event_name == sse_name::TURN_COMPLETE {
                if let Some(u) = body.get("usage") {
                    turn.usage = parse_token_usage_value(u);
                }
                turn.duration_ms = body.get("duration_ms").and_then(Value::as_u64).unwrap_or(0);
            }
        }
        "limit_reached" => {
            turn.limit_reached = true;
            turn.max_iterations = body
                .get("max_iterations")
                .and_then(Value::as_u64)
                .map(|n| n as usize);
            if let Some(u) = body.get("usage") {
                turn.usage = parse_token_usage_value(u);
            }
        }
        _ => {}
    }
}

impl<I: InputSource + Clone + 'static, R: OutputRenderer> ReplEngine<I, R> {
    pub(crate) async fn resolve_context_for_message(
        service: &AgentService,
        message: &str,
    ) -> ResolvedContext {
        ContextResolver::from_service(service)
            .resolve_prompt(message)
            .await
    }

    /// Single path for: apply driver + skills to service, run turn (harness or direct), restore prompt, autosave/notify.
    /// Call this from custom prompt, editor content, and main message flows to avoid duplication.
    pub(crate) async fn run_message_turn(
        &mut self,
        mut harness: Option<&mut caliche::AgentHarness>,
        service: &mut AgentService,
        message: &str,
        tool_execution_driver: &Option<String>,
        auto_skill: bool,
        use_harness_turns: bool,
        auto_decompose: bool,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
        session_manager: &mut crate::core::session::SessionManager,
        notify_ctx: Option<(&str, &str)>,
        turn_attachments: Vec<caliche::messages::Attachment>,
        turn_allowed_paths: Vec<PathBuf>,
    ) -> Result<()> {
        if let Some(token) = cancellation_token {
            token.reset();
        }

        if let Some(ref mut deferred) = self.deferred_workspace
            && let Some(msg) = deferred.poll_apply(service).await
        {
            self.renderer.show_status(&msg);
        }

        let skills = turn_helpers::detect_turn_skills(service, message, auto_skill);
        let apply_local_skills = turn_helpers::apply_local_turn_skills(self.thin_client.is_some());
        {
            let svc = harness.as_mut().map(|h| h.service_mut()).unwrap_or(service);
            turn_helpers::apply_turn_driver(svc, tool_execution_driver.as_deref(), message);
            if apply_local_skills {
                turn_helpers::apply_skills_to_service(
                    &mut self.renderer,
                    &self.state.original_system_prompt,
                    svc,
                    &skills,
                );
            }
        }
        self.state.save_undo_state();
        let turn_result = self
            .dispatch_message_turn(
                &mut harness,
                service,
                message,
                cancellation_token,
                turn_attachments,
                turn_allowed_paths,
                use_harness_turns,
                auto_decompose,
            )
            .await;
        if apply_local_skills && !skills.is_empty() {
            let s = harness.as_mut().map(|h| h.service_mut()).unwrap_or(service);
            turn_helpers::restore_system_prompt(s, &self.state.original_system_prompt);
        }
        self.handle_turn_outcome(
            &mut harness,
            service,
            turn_result,
            session_manager,
            notify_ctx,
        )
        .await;

        self.input.request_stdin_drain();
        Ok(())
    }

    /// Run a complex message through harness with decomposition: planning turn then sub-tasks with [1/N] progress.
    pub(crate) async fn process_message_via_harness_decomposed(
        &mut self,
        harness: &mut caliche::AgentHarness,
        message: &str,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
    ) -> Result<()> {
        use caliche::messages::Message;

        self.state.add_message(Message::user(message));
        self.renderer.show_thinking_indicator();

        let planning_prompt = crate::core::decomposition::planning_prompt(message);
        let plan_result = harness.run_turn(&planning_prompt).await?;
        let subtasks = crate::core::decomposition::parse_subtasks(&plan_result.response);

        if subtasks.len() < 2 {
            self.renderer.clear_thinking_indicator();
            let turn_result = harness.run_turn(message).await?;
            if !turn_result.response.trim().is_empty() {
                self.renderer.render_chunk(&turn_result.response);
            }
            let turn_tok = turn_result.usage.prompt_tokens.unwrap_or(0) as u64
                + turn_result.usage.completion_tokens.unwrap_or(0) as u64;
            self.renderer.set_session_token_info(
                self.state.total_tokens + turn_tok,
                self.state.turn_count + 1,
            );
            self.renderer
                .set_generation_duration_ms(Some(turn_result.duration_ms));
            self.renderer.render_token_usage(&turn_result.usage);
            self.renderer
                .show_status(&turn_result.discipline.summary_line());
            self.state.update_token_usage(&turn_result.usage);
            if !turn_result.response.trim().is_empty() {
                self.state.add_message(Message::assistant_with_reasoning(
                    &turn_result.response,
                    &turn_result.reasoning_content,
                ));
            }
            let action = surface_incomplete_turn(
                &mut self.renderer,
                &mut self.state.quality_metrics,
                &turn_result.response,
                turn_result.tools_used,
                turn_result.limit_reached && !turn_result.task_complete,
                None,
                false,
            );
            if action == IncompleteTurnAction::InjectContinue {
                persist_injected_continue(&mut self.state.messages, harness.service_mut()).await;
            }
            self.renderer.finish_turn();
            return Ok(());
        }

        let n = subtasks.len();
        let mut final_response = String::new();
        let mut last_usage = plan_result.usage.clone();
        let mut last_duration_ms = plan_result.duration_ms;
        let mut last_discipline = plan_result.discipline.clone();

        let mut last_limit_reached = false;
        let mut last_tools_used = false;
        let mut last_reasoning = String::new();

        for (i, sub) in subtasks.iter().enumerate() {
            if let Some(token) = cancellation_token {
                if token.is_cancelled() {
                    self.renderer.clear_thinking_indicator();
                    return Err(anyhow::anyhow!("Operation cancelled by user"));
                }
            }
            self.renderer
                .show_status(&format!("[{} / {}] {}", i + 1, n, sub));
            let turn_result = harness.run_turn(sub).await?;
            final_response = turn_result.response;
            last_usage = turn_result.usage;
            last_duration_ms = turn_result.duration_ms;
            last_discipline = turn_result.discipline;
            last_limit_reached = turn_result.limit_reached && !turn_result.task_complete;
            last_tools_used = turn_result.tools_used;
            last_reasoning = turn_result.reasoning_content;
        }

        self.renderer.clear_thinking_indicator();
        if !final_response.trim().is_empty() {
            self.renderer.render_chunk(&final_response);
        }
        let turn_tok = last_usage.prompt_tokens.unwrap_or(0) as u64
            + last_usage.completion_tokens.unwrap_or(0) as u64;
        self.renderer.set_session_token_info(
            self.state.total_tokens + turn_tok,
            self.state.turn_count + 1,
        );
        self.renderer
            .set_generation_duration_ms(Some(last_duration_ms));
        self.renderer.render_token_usage(&last_usage);
        self.renderer.show_status(&last_discipline.summary_line());
        self.state.update_token_usage(&last_usage);
        if !final_response.trim().is_empty() {
            self.state.add_message(Message::assistant_with_reasoning(
                &final_response,
                &last_reasoning,
            ));
        }
        let action = surface_incomplete_turn(
            &mut self.renderer,
            &mut self.state.quality_metrics,
            &final_response,
            last_tools_used,
            last_limit_reached,
            None,
            false,
        );
        if action == IncompleteTurnAction::InjectContinue {
            persist_injected_continue(&mut self.state.messages, harness.service_mut()).await;
        }
        self.renderer.finish_turn();
        Ok(())
    }

    /// Process a message through the thin-client HTTP+SSE path (ROADMAP 2.5).
    pub(crate) async fn process_message_via_thin_client(
        &mut self,
        client: &crate::client::AgentClient,
        session_id: &str,
        message: &str,
        allow_auto_continue: bool,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
    ) -> Result<()> {
        use caliche::messages::Message;

        self.state.add_message(Message::user(message));
        self.renderer.show_thinking_indicator();

        let mut turn = ThinClientTurnState::default();
        let provider = self.state.token_budget.provider.clone();
        let model = self.state.token_budget.model.clone();
        let selection = thin_client_selection(&self.remote_turn_selection, &provider, &model);
        let posted_selection = selection.is_some();
        let context_window = self
            .state
            .context_window_override
            .filter(|window| self.remote_context_window != Some(*window));
        if let Err(error) = client
            .run_turn_stream_with_cancel(
                session_id,
                message,
                selection,
                context_window,
                cancellation_token,
                |event_name, data| {
                    apply_thin_client_sse_event(&mut self.renderer, &mut turn, event_name, data);
                },
            )
            .await
        {
            self.renderer.clear_thinking_indicator();
            if turn_helpers::turn_was_cancelled(&error) {
                self.reopen_thin_session_after_cancel(client).await;
            }
            return Err(turn_helpers::thin_client_turn_error(error));
        }
        if posted_selection {
            self.remote_turn_selection = Some((provider, model));
        }
        if context_window.is_some() {
            self.remote_context_window = context_window;
        }

        if !turn.streamed_any {
            self.renderer.clear_thinking_indicator();
        }
        if !turn.streamed_any && !turn.response.trim().is_empty() {
            self.renderer.render_chunk(&turn.response);
        }
        if let Some(message) = persisted_thin_client_message(&turn) {
            self.state.add_message(message);
        }

        let turn_tok = turn.usage.prompt_tokens.unwrap_or(0) as u64
            + turn.usage.completion_tokens.unwrap_or(0) as u64;
        self.renderer.set_session_token_info(
            self.state.total_tokens + turn_tok,
            self.state.turn_count + 1,
        );
        self.renderer
            .set_generation_duration_ms(Some(turn.duration_ms));
        self.renderer.render_token_usage(&turn.usage);
        self.state.update_token_usage(&turn.usage);

        let action = surface_incomplete_turn(
            &mut self.renderer,
            &mut self.state.quality_metrics,
            &turn.response,
            turn.streamed_any,
            turn.limit_reached,
            turn.max_iterations,
            allow_auto_continue,
        );
        match action {
            IncompleteTurnAction::AutoContinue => {
                return Box::pin(self.process_message_via_thin_client(
                    client,
                    session_id,
                    CONTINUE_TASK_PROMPT,
                    false,
                    cancellation_token,
                ))
                .await;
            }
            IncompleteTurnAction::InjectContinue => {
                inject_continue_prompt(&mut self.state.messages);
            }
            IncompleteTurnAction::None => {}
        }

        self.renderer.finish_turn();
        Ok(())
    }

    /// Process a message through the AgentHarness (checkpointing, state tracking, event stream).
    /// Used when memory.use_harness_turns is true. Streams response to the same sink as non-harness path.
    pub(crate) async fn process_message_via_harness(
        &mut self,
        harness: &mut caliche::AgentHarness,
        message: &str,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
        mut turn_attachments: Vec<caliche::messages::Attachment>,
        allow_auto_continue: bool,
    ) -> Result<()> {
        let mut pending_attachments = std::mem::take(&mut self.state.pending_attachments);
        pending_attachments.append(&mut turn_attachments);

        turn_helpers::record_user_message_with_attachments(
            &mut self.state,
            message,
            &pending_attachments,
        );
        self.renderer.show_thinking_indicator();

        if self.max_new_doc_files > 0 {
            caliche::agent::tools::file_creation_context::set_context(
                self.docs_created.clone(),
                self.max_new_doc_files,
            );
        }

        self.refresh_extra_allowed_paths(message, Vec::new()).await;

        let mut events = harness.subscribe();
        let mut stats = turn_helpers::HarnessStreamStats::default();
        let turn_result = turn_helpers::run_harness_turn_stream(
            &mut self.renderer,
            harness,
            message,
            pending_attachments,
            cancellation_token,
            self.state.total_tokens,
            self.state.turn_count,
            &mut stats,
        )
        .await?;

        self.docs_created
            .extend(caliche::agent::tools::file_creation_context::take_files_created());

        turn_helpers::drain_harness_events(&mut self.renderer, &mut events);
        self.record_harness_metrics_and_response(&stats, &turn_result);
        let action = self.surface_harness_incomplete(&turn_result, allow_auto_continue);
        if self
            .apply_incomplete_turn_action(harness, action, cancellation_token)
            .await?
        {
            return Ok(());
        }

        turn_helpers::maybe_auto_compact_after_turn(
            &mut self.renderer,
            &mut self.state,
            harness.service_mut(),
        )
        .await;

        self.renderer.finish_turn();
        Ok(())
    }

    /// Process a message to the agent
    pub(crate) async fn process_message(
        &mut self,
        service: &mut AgentService,
        message: &str,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
        mut turn_attachments: Vec<caliche::messages::Attachment>,
        turn_allowed_paths: Vec<PathBuf>,
        allow_auto_continue: bool,
    ) -> Result<()> {
        // Drain any pending media attachments queued via /attach
        let mut pending_attachments = std::mem::take(&mut self.state.pending_attachments);
        pending_attachments.append(&mut turn_attachments);

        // Add user message to state before streaming
        use caliche::messages::Message;
        if pending_attachments.is_empty() {
            self.state.add_message(Message::user(message));
        } else {
            let names: Vec<String> = pending_attachments
                .iter()
                .filter_map(|a| a.filename.clone())
                .collect();
            let note = if names.is_empty() {
                format!(
                    "[{} attachment(s)]\n\n{}",
                    pending_attachments.len(),
                    message
                )
            } else {
                format!("[attached: {}]\n\n{}", names.join(", "), message)
            };
            self.state.add_message(Message::user(&note));
        }

        // Enforce the per-session doc-file creation limit.
        if self.max_new_doc_files > 0 {
            caliche::agent::tools::file_creation_context::set_context(
                self.docs_created.clone(),
                self.max_new_doc_files,
            );
        }

        // Refresh the per-turn extra-allowed paths from the user's message.
        if !is_steering_only_query(message) {
            let paths = caliche::agent::hooks::extract_paths_from_text(message);
            let mut guard = self.extra_allowed.write().await;
            guard.clear();
            for p in paths {
                guard.insert(p);
            }
            for path in turn_allowed_paths {
                guard.insert(path);
            }
        }

        // Stream the chat. When a cancellation token is available, pass a
        // shared cancel flag so deeper tool-execution paths can short-circuit.
        let cancel_flag = cancellation_token.map(|t| t.shared_cancelled_flag());
        let mut stream: std::pin::Pin<
            Box<
                dyn futures_util::Stream<Item = anyhow::Result<caliche::messages::StreamEvent>>
                    + Send
                    + '_,
            >,
        > = if pending_attachments.is_empty() {
            service.stream_chat_with_cancel_flag(message, cancel_flag)
        } else {
            service.stream_chat_with_attachments_and_cancel_flag(
                message,
                pending_attachments,
                cancel_flag,
            )
        };

        // Show "Thinking..." until first stream event
        self.renderer.show_thinking_indicator();

        let mut tool_args = std::collections::HashMap::new();
        let mut first_event = true;
        let mut tool_calls_started = 0_u64;
        let mut tool_calls_succeeded = 0_u64;
        let mut tool_calls_failed = 0_u64;
        let mut tool_retries_attempted = 0_u64;
        let mut tool_retries_succeeded = 0_u64;
        let mut generation_duration_ms = None;
        let mut captured_thinking = String::new();
        let mut sink = CombinedSink {
            renderer: &mut self.renderer,
            tool_args: &mut tool_args,
            first_event: &mut first_event,
            tool_calls_started: &mut tool_calls_started,
            tool_calls_succeeded: &mut tool_calls_succeeded,
            tool_calls_failed: &mut tool_calls_failed,
            tool_retries_attempted: &mut tool_retries_attempted,
            tool_retries_succeeded: &mut tool_retries_succeeded,
            generation_duration_ms: &mut generation_duration_ms,
            session_total_tokens: self.state.total_tokens,
            session_turn_count: self.state.turn_count,
            thinking: &mut captured_thinking,
        };

        let result = match StreamProcessor::process_stream(
            &mut stream,
            &mut self.state,
            &mut sink,
            cancellation_token,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                if first_event {
                    self.renderer.clear_thinking_indicator();
                }
                if let Some(summary) = agent_process_summary() {
                    self.renderer.show_status(&summary);
                }
                drop(stream);
                return Err(e);
            }
        };

        if first_event {
            self.renderer.clear_thinking_indicator();
        }
        record_tool_call_metrics(
            &mut self.state.quality_metrics,
            tool_calls_started,
            tool_calls_succeeded,
            tool_calls_failed,
            tool_retries_attempted,
            tool_retries_succeeded,
        );

        // Sync any newly created doc files back into the session tracker.
        self.docs_created
            .extend(caliche::agent::tools::file_creation_context::take_files_created());

        // Drop stream to release borrow on service
        drop(stream);

        let action = surface_incomplete_turn(
            &mut self.renderer,
            &mut self.state.quality_metrics,
            &result.text,
            result.tools_used,
            result.limit_reached,
            result.max_iterations,
            allow_auto_continue,
        );

        if !result.text.trim().is_empty() {
            self.state.add_message(Message::assistant_with_reasoning(
                &result.text,
                &captured_thinking,
            ));
        }

        match action {
            IncompleteTurnAction::AutoContinue => {
                return Box::pin(self.process_message(
                    service,
                    CONTINUE_TASK_PROMPT,
                    cancellation_token,
                    Vec::new(),
                    Vec::new(),
                    false,
                ))
                .await;
            }
            IncompleteTurnAction::InjectContinue => {
                persist_injected_continue(&mut self.state.messages, service).await;
            }
            IncompleteTurnAction::None => {}
        }

        if self.state.should_auto_compact() {
            use crate::core::autocompact::maybe_auto_compact;
            match maybe_auto_compact(&mut self.state, service).await {
                Ok(true) => {
                    self.renderer.show_status(
                        "Auto-compact: Conversation summarized to stay within context limits.",
                    );
                }
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!("Auto-compact failed: {}", e);
                    self.renderer.show_error(&format!(
                        "Auto-compact failed: {}. Use /compress to manually compress.",
                        e
                    ));
                }
            }
        }

        self.renderer.finish_turn();

        Ok(())
    }
}

async fn append_service_memory(service: &AgentService, messages: Vec<caliche::messages::Message>) {
    if messages.is_empty() {
        return;
    }
    let Some(memory_arc) = service.memory() else {
        return;
    };
    let mut memory = memory_arc.write().await;
    if let Err(e) = memory.add_messages(messages).await {
        tracing::warn!("Failed to sync messages to service memory: {}", e);
    }
}

async fn persist_injected_continue(
    messages: &mut Vec<caliche::messages::Message>,
    service: &AgentService,
) {
    inject_continue_prompt(messages);
    append_service_memory(
        service,
        vec![caliche::messages::Message::user(CONTINUE_TASK_PROMPT)],
    )
    .await;
}

#[cfg(test)]
#[path = "message_turn_tests.rs"]
mod tests;
