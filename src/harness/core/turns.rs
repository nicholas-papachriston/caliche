//! Core Agent Harness implementation.
//!
//! The AgentHarness wraps AgentService with enhanced capabilities for
//! event streaming, context management, and state tracking.

use super::types::{AgentHarness, HarnessTurnResult, TaskResult};
use crate::agent::hooks::CONTINUE_TASK_PROMPT;
use crate::harness::events::AgentEvent;
use crate::harness::metrics::HarnessDisciplineMetrics;
use crate::messages::{StreamEvent, TokenUsage};
use anyhow::Result;
use futures_util::StreamExt;
use std::time::Instant;

impl AgentHarness {
    /// Run a complete task, potentially spanning multiple turns.
    ///
    /// Each turn runs the inner tool loop to a final reply. Extra outer turns
    /// are used only when that inner loop hit its iteration cap.
    pub async fn run_task(&mut self, task: &str) -> Result<TaskResult> {
        let start = Instant::now();
        let mut files_modified = Vec::new();

        // Emit session start event
        self.event_stream.emit(AgentEvent::session_start(
            &self.session_id,
            Some(
                self.context_manager
                    .read()
                    .await
                    .workspace_root()
                    .to_path_buf(),
            ),
        ));

        // Create checkpoint if auto-checkpoint is enabled
        let checkpoint_id = if self.config.auto_checkpoint {
            let id = self
                .state_manager
                .write()
                .await
                .checkpoint(&format!("task_start_{}", self.turn_count));
            self.event_stream.emit(AgentEvent::CheckpointCreated {
                checkpoint_id: id.to_string(),
                label: "task_start".to_string(),
                file_count: 0,
                timestamp: chrono::Utc::now(),
            });
            Some(id.to_string())
        } else {
            None
        };

        let mut final_response = String::new();
        let initial_turn = self.turn_count;
        let mut continue_prompt: Option<&'static str> = None;

        while self.turn_count - initial_turn < self.config.max_turns {
            let input = continue_prompt.unwrap_or(task);
            let turn_result = self.run_turn(input).await?;
            final_response.clone_from(&turn_result.response);

            // Track files modified in this turn
            let new_files = self
                .state_manager
                .read()
                .await
                .files_modified_since(initial_turn);
            for file in new_files {
                if !files_modified.contains(&file) {
                    files_modified.push(file);
                }
            }

            // `chat()` already ran tools until a final answer. Repeat the original
            // task only when the inner loop was cut off by the iteration cap.
            if !turn_result.limit_reached || turn_result.task_complete {
                break;
            }
            continue_prompt = Some(CONTINUE_TASK_PROMPT);
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Emit done event
        self.event_stream.emit(AgentEvent::done(
            &final_response,
            self.total_usage,
            duration_ms,
        ));

        self.event_stream.emit(AgentEvent::session_end(
            &self.session_id,
            self.turn_count - initial_turn,
            self.total_usage,
        ));

        let turns = self.turn_count - initial_turn;
        let discipline = HarnessDisciplineMetrics::for_task(
            &self.total_usage,
            turns,
            self.total_tool_calls,
            self.total_compactions,
            self.total_readonly_filter_hits,
            Some(self.service.provider()),
            Some(self.service.model()),
        );
        tracing::info!(
            turn_count = turns,
            turn_tokens = ?self.total_usage.total_tokens,
            turn_tool_calls = self.total_tool_calls,
            turn_compactions = self.total_compactions,
            readonly_filter_hits = self.total_readonly_filter_hits,
            task_cost_estimate = ?discipline.estimated_cost_usd,
            "harness task discipline metrics"
        );

        Ok(TaskResult {
            response: final_response,
            total_usage: self.total_usage,
            turns,
            duration_ms,
            files_modified,
            checkpoint_id,
            discipline,
        })
    }

    /// Run a single turn of conversation.
    pub async fn run_turn(&mut self, input: &str) -> Result<HarnessTurnResult> {
        if self.is_hard_stopped() {
            anyhow::bail!("session is hard-stopped; refusing new turns");
        }
        self.phase = crate::harness::suspend::SessionPhase::InTurn;
        let start = Instant::now();
        self.turn_count += 1;
        let prompt_preview: String = input.chars().take(100).collect();

        self.diff_tracker
            .write()
            .await
            .start_message(uuid::Uuid::now_v7());

        // Emit turn start event
        self.event_stream
            .emit(AgentEvent::turn_start(self.turn_count, input));

        // Execute the turn
        let response = self.service.chat(input).await;

        let duration_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(chat_response) => {
                let response_text = chat_response.text().to_string();
                let reasoning_content = chat_response.reasoning_content().to_string();
                let usage = *chat_response.usage();
                let tools_used = chat_response.tools_used;
                let tool_calls = chat_response.tools_executed.len();
                self.total_tool_calls = self.total_tool_calls.saturating_add(tool_calls);

                // Update accumulated usage
                self.total_usage = (self.total_usage + usage).ensure_total_tokens();

                self.replace_history_from_memory(input, &response_text)
                    .await;

                // Emit turn complete event
                self.event_stream.emit(AgentEvent::TurnComplete {
                    turn_id: self.turn_count,
                    response: response_text.clone(),
                    usage,
                    duration_ms,
                    timestamp: chrono::Utc::now(),
                });

                self.fire_notify_command(
                    "success",
                    &prompt_preview,
                    usage.total_tokens.map(u64::from),
                );

                let limit_reached = chat_response.limit_reached();
                let task_complete = !limit_reached || self.detect_task_completion(&response_text);

                self.diff_tracker.write().await.finish_message();
                self.sync_session_diff_state().await?;

                let readonly_hits = self.service.agent().take_readonly_filter_hits();
                self.total_readonly_filter_hits = self
                    .total_readonly_filter_hits
                    .saturating_add(readonly_hits);
                let discipline = HarnessDisciplineMetrics::for_turn(
                    &usage,
                    tool_calls,
                    0,
                    readonly_hits,
                    Some(self.service.provider()),
                    Some(self.service.model()),
                );
                tracing::debug!(
                    turn_index = self.turn_count,
                    turn_tokens = ?usage.total_tokens,
                    turn_tool_calls = tool_calls,
                    readonly_filter_hits = readonly_hits,
                    task_cost_estimate = ?discipline.estimated_cost_usd,
                    "harness turn discipline metrics"
                );

                let mut result = HarnessTurnResult {
                    response: response_text,
                    usage,
                    tools_used,
                    duration_ms,
                    task_complete,
                    limit_reached,
                    discipline,
                    input_id: None,
                    reasoning_content,
                };
                self.finish_turn_side_effects(&mut result);
                Ok(result)
            }
            Err(e) => {
                self.finish_turn_side_effects_after_error();
                self.diff_tracker.write().await.finish_message();
                if let Err(sync_err) = self.sync_session_diff_state().await {
                    tracing::warn!(error = %sync_err, "failed to persist session diff state after turn error");
                }
                self.event_stream
                    .emit(AgentEvent::error(e.to_string(), true));
                self.fire_notify_command("error", &prompt_preview, None);
                Err(e)
            }
        }
    }

    /// Run a single turn with streaming: events are forwarded to `on_event`, then state is updated and `HarnessTurnResult` returned.
    /// Use this when the caller wants to render chunks and tool calls in real time while still getting harness checkpointing and history.
    pub async fn run_turn_stream<F>(
        &mut self,
        input: &str,
        on_event: F,
    ) -> Result<HarnessTurnResult>
    where
        F: FnMut(anyhow::Result<StreamEvent>),
    {
        self.run_turn_stream_with_cancel(input, None, on_event)
            .await
    }

    /// Stream one turn, stopping when `cancel` is set.
    ///
    /// The flag is the same `AtomicBool` the agent stream loop already polls
    /// between provider chunks. Callers that can drop this future (HTTP cancel)
    /// should do that as well; a blocked provider read only notices the flag
    /// on the next poll.
    pub async fn run_turn_stream_with_cancel<F>(
        &mut self,
        input: &str,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        on_event: F,
    ) -> Result<HarnessTurnResult>
    where
        F: FnMut(anyhow::Result<StreamEvent>),
    {
        self.run_turn_stream_with_attachments_and_cancel(input, Vec::new(), cancel, on_event)
            .await
    }

    /// Stream a turn, attaching multimodal files to the user message when present.
    ///
    /// Conversation context comes from service memory so tool-call transcripts
    /// survive across turns (truncated user/assistant-only history does not).
    pub async fn run_turn_stream_with_attachments<F>(
        &mut self,
        input: &str,
        attachments: Vec<crate::messages::Attachment>,
        on_event: F,
    ) -> Result<HarnessTurnResult>
    where
        F: FnMut(anyhow::Result<StreamEvent>),
    {
        self.run_turn_stream_with_attachments_and_cancel(input, attachments, None, on_event)
            .await
    }

    /// [`Self::run_turn_stream_with_attachments`] with an optional cancel flag.
    pub async fn run_turn_stream_with_attachments_and_cancel<F>(
        &mut self,
        input: &str,
        attachments: Vec<crate::messages::Attachment>,
        cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        mut on_event: F,
    ) -> Result<HarnessTurnResult>
    where
        F: FnMut(anyhow::Result<StreamEvent>),
    {
        if turn_cancel_requested(cancel.as_ref()) {
            anyhow::bail!("session turn cancelled");
        }
        let start = Instant::now();
        self.turn_count += 1;
        let prompt_preview: String = input.chars().take(100).collect();

        self.diff_tracker
            .write()
            .await
            .start_message(uuid::Uuid::now_v7());

        self.event_stream
            .emit(AgentEvent::turn_start(self.turn_count, input));

        let mut stream = if attachments.is_empty() {
            self.service
                .stream_chat_with_cancel_flag(input, cancel.clone())
        } else {
            self.service.stream_chat_with_attachments_and_cancel_flag(
                input,
                attachments,
                cancel.clone(),
            )
        };

        let mut accumulated_text = String::new();
        let mut final_usage = TokenUsage::default();
        let mut tools_used = false;
        let mut tool_calls = 0usize;
        let mut compactions = 0usize;
        let mut limit_reached = false;

        loop {
            if turn_cancel_requested(cancel.as_ref()) {
                self.event_stream.emit(AgentEvent::error(
                    "session turn cancelled".to_string(),
                    true,
                ));
                self.diff_tracker.write().await.finish_message();
                if let Err(sync_err) = self.sync_session_diff_state().await {
                    tracing::warn!(error = %sync_err, "failed to persist session diff state after turn cancel");
                }
                anyhow::bail!("session turn cancelled");
            }
            let Some(event_result) = stream.next().await else {
                break;
            };
            on_event(
                event_result
                    .as_ref()
                    .map(std::clone::Clone::clone)
                    .map_err(|e| anyhow::anyhow!("{}", e)),
            );
            if let Ok(ref event) = event_result {
                self.emit_public_stream_event(event);
            }
            match event_result {
                Ok(StreamEvent::TextChunk { content }) => {
                    accumulated_text.push_str(&content);
                }
                Ok(StreamEvent::Metadata { usage: Some(u), .. }) => {
                    final_usage = u;
                }
                Ok(StreamEvent::Metadata { usage: None, .. }) => {}
                Ok(StreamEvent::ToolCallStart { .. }) => {
                    tools_used = true;
                    tool_calls = tool_calls.saturating_add(1);
                }
                Ok(StreamEvent::ToolCallResult { .. }) => {
                    tools_used = true;
                }
                Ok(StreamEvent::Compaction { .. }) => {
                    compactions = compactions.saturating_add(1);
                }
                Ok(StreamEvent::Done) => break,
                Ok(StreamEvent::LimitReached { usage, .. }) => {
                    final_usage = usage;
                    limit_reached = true;
                    break;
                }
                Err(e) => {
                    self.event_stream
                        .emit(AgentEvent::error(e.to_string(), true));
                    self.fire_notify_command("error", &prompt_preview, None);
                    self.diff_tracker.write().await.finish_message();
                    if let Err(sync_err) = self.sync_session_diff_state().await {
                        tracing::warn!(error = %sync_err, "failed to persist session diff state after stream error");
                    }
                    return Err(e);
                }
                _ => {}
            }
        }
        drop(stream);

        self.diff_tracker.write().await.finish_message();
        self.sync_session_diff_state().await?;

        let duration_ms = start.elapsed().as_millis() as u64;
        self.total_usage = (self.total_usage + final_usage).ensure_total_tokens();
        self.total_tool_calls = self.total_tool_calls.saturating_add(tool_calls);
        self.total_compactions = self.total_compactions.saturating_add(compactions);
        let readonly_hits = self.service.agent().take_readonly_filter_hits();
        self.total_readonly_filter_hits = self
            .total_readonly_filter_hits
            .saturating_add(readonly_hits);
        self.replace_history_from_memory(input, &accumulated_text)
            .await;

        self.event_stream.emit(AgentEvent::TurnComplete {
            turn_id: self.turn_count,
            response: accumulated_text.clone(),
            usage: final_usage,
            duration_ms,
            timestamp: chrono::Utc::now(),
        });

        self.fire_notify_command(
            "success",
            &prompt_preview,
            final_usage.total_tokens.map(u64::from),
        );

        let task_complete = !limit_reached || self.detect_task_completion(&accumulated_text);
        let discipline = HarnessDisciplineMetrics::for_turn(
            &final_usage,
            tool_calls,
            compactions,
            readonly_hits,
            Some(self.service.provider()),
            Some(self.service.model()),
        );

        let mut result = HarnessTurnResult {
            response: accumulated_text,
            usage: final_usage,
            tools_used,
            duration_ms,
            task_complete,
            limit_reached,
            discipline,
            input_id: None,
            // Stream REPL persists sink thinking. In-loop stream replay is a
            // separate follow-up and is not copied onto this result.
            reasoning_content: String::new(),
        };
        self.finish_turn_side_effects(&mut result);
        Ok(result)
    }
}

fn turn_cancel_requested(cancel: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>) -> bool {
    cancel.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Relaxed))
}
