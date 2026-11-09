//! Named helpers for REPL message turns (keeps parent CCN under 20).

use super::super::ReplEngine;
use super::super::stream_sink::CombinedSink;
use super::super::turn_quality::{
    IncompleteTurnAction, handle_turn_error, record_tool_call_metrics, surface_incomplete_turn,
};
use crate::bootstrap;
use crate::core::stream::{EventSink, StreamProcessor};
use crate::core::traits::{InputSource, OutputRenderer};
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::hooks::{CONTINUE_TASK_PROMPT, is_steering_only_query};
use caliche::agent::skills::{Skill, SkillIndexEntry};
use caliche::harness::{AgentEvent, BroadcastResync};
use caliche::messages::{Message, StreamEvent};
use std::path::PathBuf;
use tokio::sync::broadcast::error::TryRecvError;

pub(super) fn detect_turn_skills(
    service: &AgentService,
    message: &str,
    auto_skill: bool,
) -> Vec<Skill> {
    if !auto_skill {
        return Vec::new();
    }
    caliche::agent::skills::detect_skills_with_catalogs(
        message,
        &[
            caliche::agent::skills::discovered_skills(),
            service.plugin_skills(),
        ],
    )
}

fn skill_status_message(names: &[&str]) -> String {
    if names.len() == 1 {
        format!("[skill: {}]", names[0])
    } else {
        format!("[skills: {}]", names.join(", "))
    }
}

/// Local skill prompts stay on this process. A thin-client turn runs on the
/// server, which owns skill discovery, so applying them here would print a
/// label the remote session never sees.
#[must_use]
pub(super) fn apply_local_turn_skills(thin_client: bool) -> bool {
    !thin_client
}

pub(super) fn apply_skills_to_service<R: OutputRenderer>(
    renderer: &mut R,
    original_system_prompt: &str,
    svc: &mut AgentService,
    skills: &[Skill],
) {
    if skills.is_empty() {
        return;
    }
    let mut composed = original_system_prompt.to_string();
    let mut names = Vec::new();
    for skill in skills {
        if let Some(entry) = SkillIndexEntry::from_skill(skill) {
            names.push(skill.name.as_str());
            svc.agent_mut().register_skill_index(entry);
        } else if let Some(ref add) = skill.system_prompt_addition {
            composed.push_str("\n\n");
            composed.push_str(add);
            names.push(skill.name.as_str());
        }
    }
    svc.agent_mut().set_system_prompt(composed);
    if !names.is_empty() {
        renderer.show_status(&skill_status_message(&names));
    }
}

pub(super) fn apply_turn_driver(
    svc: &mut AgentService,
    tool_execution_driver: Option<&str>,
    message: &str,
) {
    if let Some(driver) =
        bootstrap::resolve_tool_execution_driver_for_message(tool_execution_driver, message)
    {
        svc.agent_mut().set_tool_execution_driver(driver);
    }
}

pub(super) fn thin_client_turn_error(error: anyhow::Error) -> anyhow::Error {
    let text = error.to_string();
    if text.contains("Turn failed:") || text.contains("cancelled by user") {
        return error;
    }
    anyhow::anyhow!("thin-client turn failed: {text}")
}

pub(super) fn turn_was_cancelled(error: &anyhow::Error) -> bool {
    error.to_string().contains("cancelled by user")
}

pub(super) fn restore_system_prompt(svc: &mut AgentService, original: &str) {
    svc.agent_mut().set_system_prompt(original.to_string());
}

pub(super) fn record_user_message_with_attachments(
    state: &mut crate::core::state::InteractiveState,
    message: &str,
    pending_attachments: &[caliche::messages::Attachment],
) {
    if pending_attachments.is_empty() {
        state.add_message(Message::user(message));
        return;
    }
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
    state.add_message(Message::user(&note));
}

fn dispatch_harness_stream_event<R: OutputRenderer>(
    sink: &mut CombinedSink<'_, R>,
    ev: anyhow::Result<StreamEvent>,
) {
    if let Ok(event) = ev {
        StreamProcessor::dispatch_event(sink, event);
    }
}

pub(super) struct HarnessStreamStats {
    pub first_event: bool,
    pub tool_calls_started: u64,
    pub tool_calls_succeeded: u64,
    pub tool_calls_failed: u64,
    pub tool_retries_attempted: u64,
    pub tool_retries_succeeded: u64,
    pub generation_duration_ms: Option<u64>,
    /// Chain-of-thought captured for REPL persistence, independent of display.
    pub thinking: String,
}

impl Default for HarnessStreamStats {
    fn default() -> Self {
        Self {
            first_event: true,
            tool_calls_started: 0,
            tool_calls_succeeded: 0,
            tool_calls_failed: 0,
            tool_retries_attempted: 0,
            tool_retries_succeeded: 0,
            generation_duration_ms: None,
            thinking: String::new(),
        }
    }
}

pub(super) async fn run_harness_turn_stream<R: OutputRenderer>(
    renderer: &mut R,
    harness: &mut caliche::AgentHarness,
    message: &str,
    pending_attachments: Vec<caliche::messages::Attachment>,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
    session_total_tokens: u64,
    session_turn_count: u32,
    stats: &mut HarnessStreamStats,
) -> Result<caliche::harness::HarnessTurnResult> {
    let mut tool_args = std::collections::HashMap::new();
    let mut sink = CombinedSink {
        renderer,
        tool_args: &mut tool_args,
        first_event: &mut stats.first_event,
        tool_calls_started: &mut stats.tool_calls_started,
        tool_calls_succeeded: &mut stats.tool_calls_succeeded,
        tool_calls_failed: &mut stats.tool_calls_failed,
        tool_retries_attempted: &mut stats.tool_retries_attempted,
        tool_retries_succeeded: &mut stats.tool_retries_succeeded,
        generation_duration_ms: &mut stats.generation_duration_ms,
        session_total_tokens,
        session_turn_count,
        thinking: &mut stats.thinking,
    };

    let result = if let Some(token) = cancellation_token {
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                crate::repl::stdout_sync::write_newline();
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
            r = harness.run_turn_stream_with_attachments(
                message,
                pending_attachments,
                |ev| dispatch_harness_stream_event(&mut sink, ev),
            ) => r,
        }
    } else {
        harness
            .run_turn_stream_with_attachments(message, pending_attachments, |ev| {
                dispatch_harness_stream_event(&mut sink, ev);
            })
            .await
    };

    if let Ok(ref turn) = result {
        sink.on_complete(&turn.usage);
    }
    result
}

/// Show every event still queued for this turn.
///
/// A lag shows the resync line and keeps draining, so the retained events
/// after the hole (checkpoints, file writes, rollbacks) still print.
pub(super) fn drain_harness_events<R: OutputRenderer>(
    renderer: &mut R,
    events: &mut tokio::sync::broadcast::Receiver<AgentEvent>,
) {
    loop {
        match events.try_recv() {
            Ok(ev) => show_harness_event(renderer, &ev),
            Err(TryRecvError::Lagged(skipped)) => {
                renderer.show_status(BroadcastResync::from_skipped(skipped).cli_line().trim_end());
            }
            Err(TryRecvError::Empty | TryRecvError::Closed) => break,
        }
    }
}

fn show_harness_event<R: OutputRenderer>(renderer: &mut R, ev: &AgentEvent) {
    match ev {
        AgentEvent::CheckpointCreated {
            checkpoint_id,
            label,
            file_count,
            ..
        } => {
            renderer.show_status(&format!(
                "Checkpoint '{label}' created: {checkpoint_id} ({file_count} file(s))"
            ));
        }
        AgentEvent::FileWrite { path, is_new, .. } => {
            let action = if *is_new { "Created" } else { "Modified" };
            renderer.show_status(&format!("{action} {}", path.display()));
        }
        AgentEvent::RolledBack {
            checkpoint_id,
            files_restored,
            ..
        } => show_rollback_status(renderer, checkpoint_id, files_restored),
        _ => {}
    }
}

fn show_rollback_status<R: OutputRenderer>(
    renderer: &mut R,
    checkpoint_id: &str,
    files_restored: &[PathBuf],
) {
    if files_restored.is_empty() {
        renderer.show_status(&format!("Rolled back to checkpoint: {checkpoint_id}"));
        return;
    }
    renderer.show_status(&format!(
        "Rolled back to {checkpoint_id} ({} file(s) restored)",
        files_restored.len()
    ));
}

pub(super) async fn maybe_auto_compact_after_turn<R: OutputRenderer>(
    renderer: &mut R,
    state: &mut crate::core::state::InteractiveState,
    service: &mut AgentService,
) {
    if !state.should_auto_compact() {
        return;
    }
    use crate::core::autocompact::maybe_auto_compact;
    match maybe_auto_compact(state, service).await {
        Ok(true) => {
            renderer.show_status(
                "Auto-compact: Conversation summarized to stay within context limits.",
            );
        }
        Ok(false) => {}
        Err(e) => {
            tracing::warn!("Auto-compact failed: {}", e);
            renderer.show_error(&format!(
                "Auto-compact failed: {e}. Use /compress to manually compress."
            ));
        }
    }
}

pub(super) fn notify_turn_complete(
    notify_command: Option<&str>,
    notify_ctx: Option<(&str, &str)>,
    status: &str,
) {
    let Some((session_id, prompt_preview)) = notify_ctx else {
        return;
    };
    let Some(cmd) = notify_command else {
        return;
    };
    crate::notify::fire_notify_command(cmd, status, prompt_preview, session_id, None);
}

pub(super) fn error_notify_status(is_cancelled: bool, is_timeout: bool) -> &'static str {
    if is_cancelled {
        "interrupted"
    } else if is_timeout {
        "timeout"
    } else {
        "error"
    }
}

impl<I: InputSource + Clone + 'static, R: OutputRenderer> ReplEngine<I, R> {
    pub(super) async fn refresh_extra_allowed_paths(&self, message: &str, extra: Vec<PathBuf>) {
        if is_steering_only_query(message) {
            return;
        }
        let paths = caliche::agent::hooks::extract_paths_from_text(message);
        let mut guard = self.extra_allowed.write().await;
        guard.clear();
        for p in paths {
            guard.insert(p);
        }
        for path in extra {
            guard.insert(path);
        }
    }

    pub(super) async fn reopen_thin_session_after_cancel(
        &mut self,
        client: &crate::client::AgentClient,
    ) {
        match client.create_session().await {
            Ok(id) => {
                self.thin_session_id = Some(id);
                self.remote_turn_selection = None;
                self.remote_context_window = None;
            }
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "could not open a replacement session after cancel"
                );
                self.thin_session_id = None;
                self.remote_turn_selection = None;
                self.remote_context_window = None;
            }
        }
    }

    pub(super) async fn dispatch_message_turn(
        &mut self,
        harness: &mut Option<&mut caliche::AgentHarness>,
        service: &mut AgentService,
        message: &str,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
        turn_attachments: Vec<caliche::messages::Attachment>,
        turn_allowed_paths: Vec<PathBuf>,
        use_harness_turns: bool,
        auto_decompose: bool,
    ) -> Result<()> {
        let thin_turn = self.thin_client.clone().zip(self.thin_session_id.clone());
        if let Some((client, remote_session_id)) = thin_turn {
            return self
                .process_message_via_thin_client(
                    &client,
                    &remote_session_id,
                    message,
                    true,
                    cancellation_token,
                )
                .await;
        }
        if use_harness_turns {
            return self
                .dispatch_harness_or_direct(
                    harness,
                    service,
                    message,
                    cancellation_token,
                    turn_attachments,
                    turn_allowed_paths,
                    auto_decompose,
                )
                .await;
        }
        let svc = harness.as_mut().map(|h| h.service_mut()).unwrap_or(service);
        self.process_message(
            svc,
            message,
            cancellation_token,
            turn_attachments,
            turn_allowed_paths,
            true,
        )
        .await
    }

    async fn dispatch_harness_or_direct(
        &mut self,
        harness: &mut Option<&mut caliche::AgentHarness>,
        service: &mut AgentService,
        message: &str,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
        turn_attachments: Vec<caliche::messages::Attachment>,
        turn_allowed_paths: Vec<PathBuf>,
        auto_decompose: bool,
    ) -> Result<()> {
        if let Some(h) = harness {
            if auto_decompose && crate::core::decomposition::is_complex_message(message) {
                return self
                    .process_message_via_harness_decomposed(h, message, cancellation_token)
                    .await;
            }
            return self
                .process_message_via_harness(h, message, cancellation_token, turn_attachments, true)
                .await;
        }
        let svc = harness.as_mut().map(|h| h.service_mut()).unwrap_or(service);
        self.process_message(
            svc,
            message,
            cancellation_token,
            turn_attachments,
            turn_allowed_paths,
            true,
        )
        .await
    }

    pub(super) async fn handle_turn_outcome(
        &mut self,
        harness: &mut Option<&mut caliche::AgentHarness>,
        service: &mut AgentService,
        turn_result: Result<()>,
        session_manager: &mut crate::core::session::SessionManager,
        notify_ctx: Option<(&str, &str)>,
    ) {
        match turn_result {
            Ok(()) => self.on_turn_success(session_manager, notify_ctx).await,
            Err(e) => {
                self.on_turn_error(harness, service, e, notify_ctx).await;
            }
        }
    }

    async fn on_turn_success(
        &mut self,
        session_manager: &mut crate::core::session::SessionManager,
        notify_ctx: Option<(&str, &str)>,
    ) {
        if self.state.should_autosave(30) {
            if let Err(e) = session_manager.maybe_autosave(&self.state).await {
                tracing::warn!(error = %e, "failed to autosave REPL session");
            }
            self.state.mark_autosaved();
        }
        notify_turn_complete(self.notify_command.as_deref(), notify_ctx, "success");
    }

    async fn on_turn_error(
        &mut self,
        harness: &mut Option<&mut caliche::AgentHarness>,
        service: &mut AgentService,
        error: anyhow::Error,
        notify_ctx: Option<(&str, &str)>,
    ) {
        let memory_from = self.state.messages.len();
        let outcome = handle_turn_error(
            &mut self.renderer,
            &mut self.state.quality_metrics,
            &mut self.state.messages,
            &error,
        );
        if outcome.continuation_injected {
            let svc = harness.as_mut().map(|h| h.service_mut()).unwrap_or(service);
            let continuation = self.state.messages[memory_from..].to_vec();
            super::append_service_memory(svc, continuation).await;
        }
        notify_turn_complete(
            self.notify_command.as_deref(),
            notify_ctx,
            error_notify_status(outcome.is_cancelled, outcome.is_timeout),
        );
    }

    pub(super) async fn apply_incomplete_turn_action(
        &mut self,
        harness: &mut caliche::AgentHarness,
        action: IncompleteTurnAction,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
    ) -> Result<bool> {
        match action {
            IncompleteTurnAction::AutoContinue => {
                Box::pin(self.process_message_via_harness(
                    harness,
                    CONTINUE_TASK_PROMPT,
                    cancellation_token,
                    Vec::new(),
                    false,
                ))
                .await?;
                return Ok(true);
            }
            IncompleteTurnAction::InjectContinue => {
                super::persist_injected_continue(&mut self.state.messages, harness.service_mut())
                    .await;
            }
            IncompleteTurnAction::None => {}
        }
        Ok(false)
    }

    pub(super) fn record_harness_metrics_and_response(
        &mut self,
        stats: &HarnessStreamStats,
        turn_result: &caliche::harness::HarnessTurnResult,
    ) {
        if stats.first_event {
            self.renderer.clear_thinking_indicator();
        }
        record_tool_call_metrics(
            &mut self.state.quality_metrics,
            stats.tool_calls_started,
            stats.tool_calls_succeeded,
            stats.tool_calls_failed,
            stats.tool_retries_attempted,
            stats.tool_retries_succeeded,
        );
        self.state.update_token_usage(&turn_result.usage);
        self.renderer
            .show_status(&turn_result.discipline.summary_line());
        if !turn_result.response.trim().is_empty() {
            self.state.add_message(Message::assistant_with_reasoning(
                &turn_result.response,
                &stats.thinking,
            ));
        }
    }

    pub(super) fn surface_harness_incomplete(
        &mut self,
        turn_result: &caliche::harness::HarnessTurnResult,
        allow_auto_continue: bool,
    ) -> IncompleteTurnAction {
        surface_incomplete_turn(
            &mut self.renderer,
            &mut self.state.quality_metrics,
            &turn_result.response,
            turn_result.tools_used,
            turn_result.limit_reached && !turn_result.task_complete,
            None,
            allow_auto_continue,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_local_turn_skills, apply_skills_to_service, drain_harness_events};
    use crate::core::slash_commands::SlashCommandResult;
    use crate::core::traits::OutputRenderer;
    use caliche::agent::skills::Skill;
    use caliche::harness::AgentEvent;
    use caliche::llm::ProviderType;
    use caliche::testing::MockProvider;
    use caliche::{Agent, AgentService, ToolRegistry};
    use std::sync::Arc;

    struct StatusRenderer {
        status: Vec<String>,
    }

    impl OutputRenderer for StatusRenderer {
        fn handle_command_result(&mut self, _result: SlashCommandResult) {}
        fn render_chunk(&mut self, _text: &str) {}
        fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
        fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
        fn render_tool_error(&mut self, _name: &str, _error: &str) {}
        fn render_thinking(&mut self, _content: &str) {}
        fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
        fn show_error(&mut self, _message: &str) {}
        fn show_status(&mut self, message: &str) {
            self.status.push(message.to_string());
        }
        fn clear(&mut self) {}
    }

    fn file_write(path: &str) -> AgentEvent {
        AgentEvent::FileWrite {
            path: path.into(),
            diff_preview: None,
            is_new: false,
            checkpoint_id: None,
        }
    }

    #[test]
    fn a_lagged_drain_shows_the_resync_and_the_retained_events() {
        let (tx, mut rx) = tokio::sync::broadcast::channel(2);
        for path in ["a", "b", "c", "d"] {
            tx.send(file_write(path)).expect("send");
        }
        let mut renderer = StatusRenderer { status: Vec::new() };
        drain_harness_events(&mut renderer, &mut rx);
        assert_eq!(
            renderer.status,
            [
                "[resync] skipped 2 events; re-read session state before trusting later events",
                "Modified c",
                "Modified d",
            ]
        );
    }

    #[test]
    fn thin_client_does_not_apply_local_skills() {
        assert!(!apply_local_turn_skills(true));
        assert!(apply_local_turn_skills(false));
    }

    #[tokio::test]
    async fn local_skill_label_is_skipped_when_the_turn_is_remote() {
        let provider = Arc::new(MockProvider::new(vec!["ok".to_string()]));
        let agent = Agent::new(provider, Arc::new(ToolRegistry::new()));
        let mut service =
            AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("service");
        service.agent_mut().set_system_prompt("base".to_string());
        let skill = Skill::builder("research")
            .system_prompt_addition("Follow the research skill.")
            .build();
        let mut renderer = StatusRenderer { status: Vec::new() };
        if apply_local_turn_skills(true) {
            apply_skills_to_service(&mut renderer, "base", &mut service, &[skill.clone()]);
        }
        assert!(renderer.status.is_empty());
        assert_eq!(service.agent().system_prompt(), "base");

        if apply_local_turn_skills(false) {
            apply_skills_to_service(&mut renderer, "base", &mut service, &[skill]);
        }
        assert_eq!(renderer.status, vec!["[skill: research]".to_string()]);
        assert!(
            service
                .agent()
                .system_prompt()
                .contains("Follow the research skill.")
        );
    }
}
