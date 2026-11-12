use super::ToolCallingConfig;
use super::chat::{StepResult, get_last_assistant_response};
use super::chat_helpers::{
    IterationLimiter, SOFT_WRAP_UP_MESSAGE, TokenUsageAccumulator, iteration_limit_reached_suffix,
    should_inject_soft_wrap,
};
use super::message_history::arc_push_message;
use crate::agent::core::tool_ordering::{TOOL_ORDERING_THRESHOLD, order_tools};
use crate::agent::hooks::{CompletionContext, ToolCallSummary, TurnContext};
use crate::agent::kernel::turn::Handoff;
use crate::errors::AgentError;
use crate::messages::{Message, MessageRole, PerformanceMetrics, StepMetrics};
use std::collections::VecDeque;
use std::sync::Arc;

const RECENT_TOOL_CAP: usize = 30;

impl super::Agent {
    /// Internal method to process a complete message history through the agent loop.
    #[tracing::instrument(
        name = "agent.session",
        skip_all,
        fields(
            agent.session.message_count = messages.len(),
            agent.session.max_iterations = self.max_iterations,
        )
    )]
    pub(super) async fn chat_with_messages(
        &self,
        messages: Vec<Message>,
        tool_cfg: ToolCallingConfig,
    ) -> Result<crate::messages::ChatResponse, AgentError> {
        let mut messages = messages;
        let expected_growth = if tool_cfg.use_native { 10 } else { 4 };
        messages.reserve(expected_growth);
        let mut messages_arc = Arc::new(messages);
        let mut total_usage = TokenUsageAccumulator::default();
        total_usage.set_context_window(self.cached_context_window);
        let mut limiter = IterationLimiter::new(self.max_iterations);
        let mut tools_used = false;
        let mut tools_executed = std::collections::HashSet::new();

        let request_start = std::time::Instant::now();
        let mut step_count = 0u32;
        let mut total_llm_ms: u64 = 0;
        let mut total_tool_ms: u64 = 0;
        let mut steps_detail: Vec<StepMetrics> = Vec::new();
        let mut first_llm_ms: Option<u64> = None;
        let mut recent_tool_calls: VecDeque<ToolCallSummary> = VecDeque::new();
        let mut injections_this_session: usize = 0;

        let mut turn_ctx = TurnContext {
            session_id: self.session_id().unwrap_or("default"),
            turn_count: step_count as usize,
            workspace_root: self.workspace_root(),
        };
        messages_arc = self.run_session_start_hooks(messages_arc, &turn_ctx).await;

        let (query, pre_ordered_schemas) =
            pre_order_tool_schemas(self, &messages_arc, &tool_cfg).await;
        let failure_breaker = std::sync::Arc::new(std::sync::Mutex::new(
            crate::agent::tools::failure_breaker::IdenticalFailureBreaker::new(),
        ));

        loop {
            if limiter.tick() {
                return Ok(iteration_limit_chat_response(
                    &messages_arc,
                    limiter.limit(),
                    tools_executed,
                    tools_used,
                    total_usage.to_token_usage(),
                    request_start,
                    total_llm_ms,
                    total_tool_ms,
                    step_count,
                    first_llm_ms,
                    steps_detail,
                ));
            }
            step_count += 1;
            turn_ctx.turn_count = step_count as usize;

            if should_inject_soft_wrap(step_count as usize, limiter.limit()) {
                tracing::warn!(
                    "High step count ({} of {}). Injecting wrap-up nudge.",
                    step_count,
                    limiter.limit()
                );
                messages_arc = arc_push_message(
                    messages_arc,
                    Message::new(MessageRole::System, SOFT_WRAP_UP_MESSAGE),
                );
            }

            tracing::info!(
                "Step {}: Starting agent step ({} messages in history)",
                step_count,
                messages_arc.len()
            );

            let step_result = crate::agent::tools::failure_breaker::with_turn_scope(
                Arc::clone(&failure_breaker),
                Arc::clone(&self.action_ids),
                self.step(
                    Arc::clone(&messages_arc),
                    &tool_cfg,
                    pre_ordered_schemas.as_ref(),
                    &query,
                    step_count as usize,
                    turn_ctx.session_id,
                ),
            )
            .await?;

            collect_tool_activity(
                &step_result,
                &mut tools_used,
                &mut tools_executed,
                &mut recent_tool_calls,
            );

            #[cfg(feature = "learning-loop")]
            apply_learning_loop(self, &step_result).await;

            total_llm_ms = total_llm_ms.saturating_add(step_result.llm_duration_ms);
            total_tool_ms = total_tool_ms.saturating_add(step_result.tool_duration_ms);
            if first_llm_ms.is_none() {
                first_llm_ms = Some(step_result.llm_duration_ms);
            }
            steps_detail.push(StepMetrics {
                step: step_count,
                llm_duration_ms: step_result.llm_duration_ms,
                tool_duration_ms: step_result.tool_duration_ms,
                tools_executed: step_result
                    .tool_results
                    .iter()
                    .map(|(n, _)| n.clone())
                    .collect(),
            });

            total_usage.add_usage(&step_result.usage);
            let total_usage_snapshot = total_usage.to_token_usage();

            tracing::info!(
                "Step {} token usage: +{:?} tokens (cumulative: {:?} total)",
                step_count,
                step_result.usage.total_tokens,
                total_usage_snapshot.total_tokens
            );

            messages_arc = step_result.updated_history;

            tracing::info!(
                "Step {} completed: {} tool calls, response length: {} chars, pending tools: {}",
                step_count,
                step_result.tool_calls.len(),
                step_result.response_text.len(),
                step_result.has_pending_tool_calls
            );

            if !step_result.has_pending_tool_calls {
                if continue_after_pending_tools(
                    self,
                    &mut messages_arc,
                    &turn_ctx,
                    step_count as usize,
                    &mut recent_tool_calls,
                    &mut injections_this_session,
                    step_result.response_text.as_ref().as_str(),
                )
                .await
                {
                    continue;
                }

                let final_response_text =
                    super::message_history::arc_unwrap_or_clone(step_result.response_text);
                let reasoning = last_assistant_reasoning(&messages_arc);
                return Ok(complete_chat_response(
                    final_response_text,
                    total_usage_snapshot,
                    tools_used,
                    tools_executed,
                    request_start,
                    total_llm_ms,
                    total_tool_ms,
                    step_count,
                    first_llm_ms,
                    steps_detail,
                )
                .with_reasoning_content(reasoning));
            }
        }
    }

    /// Execute chat tracking handoff tool calls. Used by [`Agent::execute_turn`].
    pub(super) async fn chat_with_messages_and_track_handoff(
        &self,
        messages: Vec<Message>,
        tool_cfg: ToolCallingConfig,
    ) -> Result<(String, crate::messages::TokenUsage, Option<Handoff>), AgentError> {
        let mut messages_arc = Arc::new(messages);
        let mut total_usage = TokenUsageAccumulator::default();
        total_usage.set_context_window(self.cached_context_window);
        let mut limiter = IterationLimiter::new(self.max_iterations);
        let mut detected_handoff: Option<Handoff> = None;
        let mut step_count: usize = 0;
        let mut recent_tool_calls: VecDeque<ToolCallSummary> = VecDeque::new();
        let mut injections_this_session: usize = 0;

        let mut turn_ctx = TurnContext {
            session_id: self.session_id().unwrap_or("default"),
            turn_count: step_count,
            workspace_root: self.workspace_root(),
        };
        messages_arc = self.run_session_start_hooks(messages_arc, &turn_ctx).await;
        let failure_breaker = std::sync::Arc::new(std::sync::Mutex::new(
            crate::agent::tools::failure_breaker::IdenticalFailureBreaker::new(),
        ));

        loop {
            if limiter.tick() {
                tracing::warn!(
                    "Maximum iterations ({}) reached in handoff tracking, returning partial result",
                    limiter.limit()
                );
                let partial_text = build_iteration_limit_message(&messages_arc, limiter.limit());
                return Ok((partial_text, total_usage.to_token_usage(), None));
            }

            step_count += 1;
            turn_ctx.turn_count = step_count;

            if should_inject_soft_wrap(step_count, limiter.limit()) {
                messages_arc = arc_push_message(
                    messages_arc,
                    Message::new(MessageRole::System, SOFT_WRAP_UP_MESSAGE),
                );
            }

            let handoff_query = Self::extract_last_user_query(&messages_arc).into_owned();
            let step_result = crate::agent::tools::failure_breaker::with_turn_scope(
                Arc::clone(&failure_breaker),
                Arc::clone(&self.action_ids),
                self.step(
                    Arc::clone(&messages_arc),
                    &tool_cfg,
                    None,
                    &handoff_query,
                    step_count,
                    turn_ctx.session_id,
                ),
            )
            .await?;

            record_recent_tool_calls(&step_result, &mut recent_tool_calls);

            total_usage.add_usage(&step_result.usage);
            let total_usage_snapshot = total_usage.to_token_usage();

            tracing::info!(
                "Handoff tracking step token usage: +{:?} tokens (cumulative: {:?} total)",
                step_result.usage.total_tokens,
                total_usage_snapshot.total_tokens
            );

            messages_arc = step_result.updated_history;

            if let Some(handoff) = Self::check_for_handoff_in_results(&step_result.tool_results) {
                detected_handoff = Some(handoff);
                let final_response_text =
                    super::message_history::arc_unwrap_or_clone(step_result.response_text);
                tracing::info!(
                    "Total request token usage - prompt: {:?}, completion: {:?}, embedding: {:?}, total: {:?}",
                    total_usage_snapshot.prompt_tokens,
                    total_usage_snapshot.completion_tokens,
                    total_usage_snapshot.embedding_tokens,
                    total_usage_snapshot.total_tokens
                );
                return Ok((final_response_text, total_usage_snapshot, detected_handoff));
            }

            if !step_result.has_pending_tool_calls {
                if continue_after_pending_tools(
                    self,
                    &mut messages_arc,
                    &turn_ctx,
                    step_count,
                    &mut recent_tool_calls,
                    &mut injections_this_session,
                    step_result.response_text.as_ref().as_str(),
                )
                .await
                {
                    continue;
                }
                let final_response_text =
                    super::message_history::arc_unwrap_or_clone(step_result.response_text);
                tracing::info!(
                    "Total request token usage - prompt: {:?}, completion: {:?}, embedding: {:?}, total: {:?}",
                    total_usage_snapshot.prompt_tokens,
                    total_usage_snapshot.completion_tokens,
                    total_usage_snapshot.embedding_tokens,
                    total_usage_snapshot.total_tokens
                );
                return Ok((final_response_text, total_usage_snapshot, detected_handoff));
            }
        }
    }
}

async fn pre_order_tool_schemas(
    agent: &super::Agent,
    messages: &[Message],
    tool_cfg: &ToolCallingConfig,
) -> (String, Option<Arc<Vec<crate::messages::ToolSchema>>>) {
    if !tool_cfg.use_native || tool_cfg.tool_schemas.len() <= TOOL_ORDERING_THRESHOLD {
        return (String::new(), None);
    }
    let query = messages
        .iter()
        .rev()
        .find_map(|m| {
            if matches!(m.role, MessageRole::User) {
                m.content.as_text().map(std::string::ToString::to_string)
            } else {
                None
            }
        })
        .unwrap_or_default();
    let schemas = order_tools(
        &tool_cfg.tool_schemas,
        &query,
        agent.tool_ordering_embedding_provider.as_ref(),
    )
    .await;
    (query, Some(schemas))
}

fn iteration_limit_chat_response(
    messages: &[Message],
    limit: usize,
    tools_executed: std::collections::HashSet<String>,
    tools_used: bool,
    usage: crate::messages::TokenUsage,
    request_start: std::time::Instant,
    total_llm_ms: u64,
    total_tool_ms: u64,
    step_count: u32,
    first_llm_ms: Option<u64>,
    steps_detail: Vec<StepMetrics>,
) -> crate::messages::ChatResponse {
    tracing::warn!(
        "Maximum iterations ({}) reached, returning partial result",
        limit
    );
    let partial_text = build_iteration_limit_message(messages, limit);
    let tools_executed_vec: Vec<String> = tools_executed.into_iter().collect();
    let total_duration_ms = request_start.elapsed().as_millis() as u64;
    let _chat_request_span = tracing::info_span!(
        "chat_request",
        step_count = step_count,
        total_ms = total_duration_ms
    )
    .entered();
    let metrics = build_metrics(
        total_duration_ms,
        total_llm_ms,
        total_tool_ms,
        step_count,
        first_llm_ms,
        steps_detail,
    );
    crate::messages::ChatResponse::new(partial_text, usage, tools_used)
        .with_tools_executed(tools_executed_vec)
        .with_limit_reached()
        .with_metrics(metrics)
        .with_reasoning_content(last_assistant_reasoning(messages))
}

fn last_assistant_reasoning(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|message| message.role == MessageRole::Assistant)
        .and_then(Message::reasoning_content)
        .unwrap_or("")
        .to_string()
}

fn complete_chat_response(
    final_response_text: String,
    usage: crate::messages::TokenUsage,
    tools_used: bool,
    tools_executed: std::collections::HashSet<String>,
    request_start: std::time::Instant,
    total_llm_ms: u64,
    total_tool_ms: u64,
    step_count: u32,
    first_llm_ms: Option<u64>,
    steps_detail: Vec<StepMetrics>,
) -> crate::messages::ChatResponse {
    let total_duration_ms = request_start.elapsed().as_millis() as u64;
    tracing::info!(
        "Request completed in {}ms ({} steps) - Token usage: prompt={:?}, completion={:?}, embedding={:?}, total={:?}",
        total_duration_ms,
        step_count,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.embedding_tokens,
        usage.total_tokens
    );
    let _chat_request_span = tracing::info_span!(
        "chat_request",
        step_count = step_count,
        total_ms = total_duration_ms
    )
    .entered();
    let tools_executed_vec: Vec<String> = tools_executed.into_iter().collect();
    let metrics = build_metrics(
        total_duration_ms,
        total_llm_ms,
        total_tool_ms,
        step_count,
        first_llm_ms,
        steps_detail,
    );
    crate::messages::ChatResponse::new(final_response_text, usage, tools_used)
        .with_tools_executed(tools_executed_vec)
        .with_metrics(metrics)
}

async fn continue_after_pending_tools(
    agent: &super::Agent,
    messages_arc: &mut Arc<Vec<Message>>,
    turn_ctx: &TurnContext<'_>,
    turn_count: usize,
    recent_tool_calls: &mut VecDeque<ToolCallSummary>,
    injections_this_session: &mut usize,
    last_assistant_response: &str,
) -> bool {
    let comp_ctx = CompletionContext {
        session_id: turn_ctx.session_id,
        turn_count,
        recent_tool_calls: recent_tool_calls.make_contiguous(),
        message_history_len: messages_arc.len(),
        injections_this_session: *injections_this_session,
        last_user_message: messages_arc
            .iter()
            .rev()
            .find(|m| matches!(m.role, MessageRole::User))
            .and_then(|m| m.content.as_text()),
        last_assistant_response: Some(last_assistant_response),
    };
    if let Some(msg) = agent.run_completion_hooks(&comp_ctx).await {
        *injections_this_session += 1;
        *messages_arc = arc_push_message(
            Arc::clone(messages_arc),
            Message::new(MessageRole::System, msg),
        );
        tracing::info!(
            "Pre-completion hook injected verification prompt (injection #{})",
            injections_this_session
        );
        return true;
    }
    append_follow_up_messages(agent, messages_arc).await
}

async fn append_follow_up_messages(
    agent: &super::Agent,
    messages_arc: &mut Arc<Vec<Message>>,
) -> bool {
    let Some(ref provider) = agent.steering_followup_provider else {
        return false;
    };
    let follow_up = provider.get_follow_up_messages().await;
    let n = follow_up.len();
    if n == 0 {
        return false;
    }
    for msg in follow_up {
        *messages_arc = arc_push_message(Arc::clone(messages_arc), msg);
    }
    tracing::info!("Follow-up messages queued ({}), continuing turn", n);
    true
}

#[cfg(feature = "learning-loop")]
async fn apply_learning_loop(agent: &super::Agent, step_result: &StepResult) {
    let Some(ref ll) = agent.learning_loop else {
        return;
    };
    for (tool_name, result) in &step_result.tool_results {
        record_learning_tool_result(ll, agent.workspace_root(), step_result, tool_name, result)
            .await;
    }
}

#[cfg(feature = "learning-loop")]
async fn record_learning_tool_result(
    ll: &crate::agent::learning::LearningLoop,
    workspace: Option<&std::path::Path>,
    step_result: &StepResult,
    tool_name: &str,
    result: &crate::agent::tools::tool::ToolResult,
) {
    let arguments_summary = step_result
        .tool_calls
        .iter()
        .find(|tc| tc.name.as_ref() == tool_name)
        .map(|tc| summarize_tool_input(tool_name, &tc.arguments))
        .unwrap_or_else(|| tool_name.to_string());
    let error_message = if result.success {
        String::new()
    } else {
        result
            .error
            .as_ref()
            .map_or_else(|| "tool call failed".to_string(), |e| e.clone())
    };
    let record = crate::agent::learning::ToolCallRecord {
        tool_name: tool_name.to_string(),
        arguments_summary,
        success: result.success,
        duration_ms: step_result.tool_duration_ms,
        timestamp: chrono::Utc::now(),
    };
    let needs_checkpoint = ll.record_and_check(record).await;
    if result.success {
        ll.mark_last_error_recovered().await;
    } else if !error_message.is_empty() {
        let mut tracker = ll.episode_tracker.write().await;
        tracker.amend_last_error_message(error_message);
    }
    if needs_checkpoint {
        run_learning_checkpoint(ll, workspace).await;
    }
}

#[cfg(feature = "learning-loop")]
async fn run_learning_checkpoint(
    ll: &crate::agent::learning::LearningLoop,
    workspace: Option<&std::path::Path>,
) {
    let existing_skills = collect_existing_skills(ll).await;
    let actions = match ll.evaluate_checkpoint(&existing_skills).await {
        Ok(actions) => actions,
        Err(err) => {
            tracing::warn!(error = %err, "learning checkpoint evaluation failed");
            return;
        }
    };
    let mut outcome = crate::agent::learning::EpisodeOutcome::PartialSuccess;
    for action in actions {
        if apply_checkpoint_action(ll, workspace, action).await {
            outcome = crate::agent::learning::EpisodeOutcome::Success;
        }
    }
    ll.finalize_current_episode(outcome).await;
}

#[cfg(feature = "learning-loop")]
async fn apply_checkpoint_action(
    ll: &crate::agent::learning::LearningLoop,
    workspace: Option<&std::path::Path>,
    action: crate::agent::learning::checkpoint::CheckpointAction,
) -> bool {
    match action {
        crate::agent::learning::checkpoint::CheckpointAction::MemoryNudge(facts) => {
            apply_memory_nudge(ll, workspace, &facts).await;
            false
        }
        crate::agent::learning::checkpoint::CheckpointAction::CreateSkill {
            definition: def,
            report,
        } => persist_created_skill(ll, def, &report).await,
        crate::agent::learning::checkpoint::CheckpointAction::PatchSkill {
            skill_id,
            patch,
            report,
        } => persist_skill_patch(ll, &skill_id, &patch, &report).await,
        crate::agent::learning::checkpoint::CheckpointAction::Skip => false,
    }
}

#[cfg(feature = "learning-loop")]
async fn apply_memory_nudge(
    ll: &crate::agent::learning::LearningLoop,
    workspace: Option<&std::path::Path>,
    facts: &[String],
) {
    let Some(ref path) = ll.config.memory_file_path else {
        return;
    };
    let store = crate::agent::learning::memory_nudge::MemoryStore::new(path, workspace);
    if let Err(e) = store.append_facts(facts).await {
        tracing::warn!("Memory nudge failed: {e}");
    }
}

#[cfg(feature = "learning-loop")]
async fn persist_created_skill(
    ll: &crate::agent::learning::LearningLoop,
    def: Box<crate::plugins::schema::PluginDefinition>,
    report: &crate::agent::learning::weakness::DistillReport,
) -> bool {
    let Some(ref store) = ll.plugin_store else {
        tracing::debug!(
            skill_name = %def.name,
            candidate = report.candidate,
            "learning loop proposed skill but no plugin store is configured"
        );
        return false;
    };
    if let Err(err) = store.put(&def).await {
        tracing::warn!(
            error = %err,
            skill_name = %def.name,
            "failed to persist auto-created skill"
        );
        return false;
    }
    tracing::info!(
        skill_id = %def.id,
        skill_name = %def.name,
        candidate = report.candidate,
        weakness = report.weakness,
        specificity = report.specificity,
        holdout_extent = report.holdout_extent,
        "learning loop created skill"
    );
    ll.mark_skill_created().await;
    true
}

#[cfg(feature = "learning-loop")]
async fn persist_skill_patch(
    ll: &crate::agent::learning::LearningLoop,
    skill_id: &str,
    patch: &crate::agent::learning::checkpoint::SkillPatch,
    report: &crate::agent::learning::weakness::DistillReport,
) -> bool {
    let Some(ref store) = ll.plugin_store else {
        return false;
    };
    match store.get_by_id(skill_id).await {
        Ok(Some(mut existing)) => {
            apply_skill_patch(&mut existing, patch);
            existing.updated_at = chrono::Utc::now();
            if let Err(err) = store.put(&existing).await {
                tracing::warn!(
                    error = %err,
                    %skill_id,
                    "failed to persist skill patch"
                );
                false
            } else {
                tracing::info!(
                    %skill_id,
                    candidate = report.candidate,
                    weakness = report.weakness,
                    holdout_extent = report.holdout_extent,
                    "learning loop patched skill"
                );
                true
            }
        }
        Ok(None) => {
            tracing::warn!(%skill_id, "skill patch targeted missing skill");
            false
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                %skill_id,
                "failed to load skill for patching"
            );
            false
        }
    }
}

/// Build the text returned when the iteration limit is reached.
fn build_iteration_limit_message(messages: &[Message], limit: usize) -> String {
    let suffix = iteration_limit_reached_suffix(limit);
    let base = get_last_assistant_response(messages).unwrap_or_else(|| suffix.clone());
    if base.contains(&suffix) {
        base
    } else {
        format!("{}\n\n{suffix}", base.trim_end())
    }
}

/// Build a [`PerformanceMetrics`] value.
const fn build_metrics(
    total_duration_ms: u64,
    total_llm_ms: u64,
    total_tool_ms: u64,
    step_count: u32,
    first_llm_ms: Option<u64>,
    steps: Vec<StepMetrics>,
) -> PerformanceMetrics {
    let overhead_ms = total_duration_ms
        .saturating_sub(total_llm_ms)
        .saturating_sub(total_tool_ms);
    PerformanceMetrics {
        total_duration_ms: Some(total_duration_ms),
        llm_duration_ms: Some(total_llm_ms),
        tool_duration_ms: Some(total_tool_ms),
        overhead_ms: Some(overhead_ms),
        step_count: Some(step_count),
        time_to_first_token_ms: first_llm_ms,
        steps: Some(steps),
    }
}

/// Append this step's tool calls to the rolling window used by completion hooks.
fn record_recent_tool_calls(step: &StepResult, recent_tool_calls: &mut VecDeque<ToolCallSummary>) {
    for (tool_name, _) in &step.tool_results {
        let input_summary = step
            .tool_calls
            .iter()
            .find(|tc| tc.name.as_ref() == tool_name.as_str())
            .map_or_else(
                || tool_name.clone(),
                |tc| summarize_tool_input(tool_name, &tc.arguments),
            );
        recent_tool_calls.push_back(ToolCallSummary {
            tool_name: tool_name.clone(),
            input_summary,
        });
        if recent_tool_calls.len() > RECENT_TOOL_CAP {
            recent_tool_calls.pop_front();
        }
    }
}

/// Record tool activity from a step into the mutable accumulators.
fn collect_tool_activity(
    step: &StepResult,
    tools_used: &mut bool,
    tools_executed: &mut std::collections::HashSet<String>,
    recent_tool_calls: &mut VecDeque<ToolCallSummary>,
) {
    if !step.tool_calls.is_empty() {
        *tools_used = true;
    }
    for (tool_name, _) in &step.tool_results {
        tools_executed.insert(tool_name.clone());
    }
    record_recent_tool_calls(step, recent_tool_calls);
}

fn summarize_tool_input(tool_name: &str, args: &serde_json::Value) -> String {
    match tool_name {
        "bash" => args
            .get("command")
            .and_then(|v| v.as_str())
            .map_or_else(|| tool_name.to_string(), |s| s.chars().take(120).collect()),
        "fs" => {
            let operation = args
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("?");
            format!("{operation}: {path}")
        }
        _ => serde_json::to_string(args)
            .map_or_else(|_| tool_name.to_string(), |s| s.chars().take(120).collect()),
    }
}

#[cfg(feature = "learning-loop")]
async fn collect_existing_skills(
    ll: &crate::agent::learning::LearningLoop,
) -> Vec<crate::agent::learning::SkillSummary> {
    let Some(ref store) = ll.plugin_store else {
        return Vec::new();
    };
    match store.load_all().await {
        Ok(defs) => defs
            .into_iter()
            .filter(|def| matches!(def.plugin_type, crate::plugins::schema::PluginType::Skill))
            .map(|def| crate::agent::learning::SkillSummary {
                id: def.id,
                name: def.name,
                description: def.description,
                usage_count: 0,
                last_used: def.updated_at,
            })
            .collect(),
        Err(err) => {
            tracing::warn!(error = %err, "failed to load skills for learning checkpoint");
            Vec::new()
        }
    }
}

#[cfg(feature = "learning-loop")]
fn apply_skill_patch(
    definition: &mut crate::plugins::schema::PluginDefinition,
    patch: &crate::agent::learning::checkpoint::SkillPatch,
) {
    if let Some(ref desc) = patch.new_description {
        definition.description = desc.clone();
    }
    if let crate::plugins::schema::PluginSpec::Skill(ref mut spec) = definition.spec {
        if let Some(ref prompt) = patch.new_system_prompt_template
            && !spec.system_prompt_template.contains(prompt.trim())
        {
            spec.system_prompt_template.push_str(prompt);
        }
        for tool in &patch.additional_required_tools {
            if !spec.required_tools.iter().any(|t| t == tool) {
                spec.required_tools.push(tool.clone());
            }
        }
    }
}

pub async fn chat_with_messages_impl(
    agent: &super::Agent,
    messages: Vec<Message>,
    tool_cfg: ToolCallingConfig,
) -> Result<crate::messages::ChatResponse, AgentError> {
    agent.chat_with_messages(messages, tool_cfg).await
}

pub async fn chat_with_messages_and_track_handoff_impl(
    agent: &super::Agent,
    messages: Vec<Message>,
    tool_cfg: ToolCallingConfig,
) -> Result<(String, crate::messages::TokenUsage, Option<Handoff>), AgentError> {
    agent
        .chat_with_messages_and_track_handoff(messages, tool_cfg)
        .await
}
