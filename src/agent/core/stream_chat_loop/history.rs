use crate::agent::core::chat_helpers::TokenUsageAccumulator;
use crate::agent::core::{Agent, context_window_guard, message_history, tool_output};
use crate::agent::memory::MemorySnapshot;
#[cfg(feature = "compaction")]
use crate::messages::StreamEvent;
use crate::messages::{Message, MessageRole, PerformanceMetrics};
use std::collections::HashSet;
use std::sync::Arc;

use super::PersistMemory;

pub(crate) async fn persist_stream_memory(memory: &PersistMemory, history: &[Message]) {
    let Some(memory) = memory else {
        return;
    };
    let messages = message_history::sanitize_history_for_request(history.iter().cloned());
    let mut guard = memory.write().await;
    let metadata = match guard.save_state().await {
        Ok(snapshot) => snapshot.metadata,
        Err(error) => {
            tracing::warn!(error = %error, "failed to read memory snapshot before persist");
            serde_json::Value::Null
        }
    };
    if let Err(error) = guard
        .load_state(MemorySnapshot { messages, metadata })
        .await
    {
        tracing::warn!(error = %error, "failed to persist streamed turn to memory");
    }
}

/// Maximum size for tool outputs before truncation (in characters).
/// ~20K chars ~= 5K tokens. Keeps stream events manageable for CLI display
/// while retaining enough content for the model to reason over.
const MAX_TOOL_OUTPUT_SIZE: usize = 20_000;

/// Threshold above which to apply tool-specific summarization instead of raw truncation.
const SMART_SUMMARIZE_THRESHOLD: usize = 8_000;

/// Summarize tool output before including it in the prompt/history.
pub(crate) fn summarize_tool_output(content: &str, tool_name: &str) -> String {
    tool_output::summarize_tool_output(
        content,
        tool_name,
        tool_output::ToolOutputLimits {
            max_chars: MAX_TOOL_OUTPUT_SIZE,
            summarize_threshold: SMART_SUMMARIZE_THRESHOLD,
            head_chars: 3_000,
            tail_chars: 1_000,
            annotate: true,
        },
    )
}

pub(crate) fn tool_result_payload_with_duration(
    payload: Option<serde_json::Value>,
    duration_ms: u64,
) -> serde_json::Value {
    let mut payload = payload.unwrap_or_else(|| serde_json::json!({}));
    let Some(obj) = payload.as_object_mut() else {
        return serde_json::json!({
            "payload": payload,
            "duration_ms": duration_ms,
        });
    };
    obj.insert(
        "duration_ms".to_string(),
        serde_json::Value::Number(serde_json::Number::from(duration_ms)),
    );
    payload
}

pub(crate) fn elapsed_millis_u64(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

const BUDGET_PRUNING_MESSAGE_THRESHOLD: usize = 10;

/// When compaction is disabled: prune history if over budget and return (new Arc, None, false).
/// When compaction is enabled: run `ContextWindowGuard::check_and_compact`; returns
/// (history, optional event, `should_stop`).
#[cfg(feature = "compaction")]
pub(crate) async fn apply_compaction_or_prune(
    agent: &Agent,
    conversation_history: Arc<Vec<Message>>,
    tool_schema_tokens: usize,
    usage: &TokenUsageAccumulator,
    max_iterations: usize,
) -> Result<
    (
        Arc<Vec<Message>>,
        Option<crate::messages::StreamEvent>,
        bool,
    ),
    anyhow::Error,
> {
    use context_window_guard::GuardResult;
    if let Some(ref guard) = agent.compaction_guard {
        let result = guard
            .check_and_compact(
                conversation_history.as_ref(),
                agent.max_context_tokens,
                tool_schema_tokens,
                agent.token_counter.as_ref(),
            )
            .await?;
        match result {
            GuardResult::WithinBudget => Ok((conversation_history, None, false)),
            GuardResult::Compacted(r) => {
                agent.store_compaction_report(r.report.clone());
                let ev = StreamEvent::Compaction {
                    tokens_before: r.tokens_before,
                    tokens_after: r.tokens_after,
                    messages_removed: r.messages_removed,
                    summary: r.summary,
                    report: (!r.report.is_empty()).then_some(r.report),
                };
                Ok((Arc::new(r.messages), Some(ev), false))
            }
            GuardResult::OverBudget {
                estimated_tokens,
                budget,
            } => {
                tracing::warn!(
                    "Context over budget: {} tokens > {}; ending stream",
                    estimated_tokens,
                    budget
                );
                let ev = StreamEvent::LimitReached {
                    max_iterations,
                    usage: usage.to_token_usage(),
                };
                Ok((conversation_history, Some(ev), true))
            }
        }
    } else {
        let pruned =
            prune_stream_history_if_needed(agent, conversation_history, tool_schema_tokens);
        Ok((pruned, None, false))
    }
}

/// If conversation history exceeds message threshold and token budget, prune and return new Arc.
/// Otherwise returns the same Arc.
pub(crate) fn prune_stream_history_if_needed(
    agent: &Agent,
    conversation_history: std::sync::Arc<Vec<Message>>,
    tool_schema_tokens: usize,
) -> std::sync::Arc<Vec<Message>> {
    if conversation_history.len() < BUDGET_PRUNING_MESSAGE_THRESHOLD {
        return conversation_history;
    }
    let estimated_tokens =
        Agent::estimate_context_tokens(agent.token_counter.as_ref(), conversation_history.as_ref());
    let effective_tokens = estimated_tokens.saturating_add(tool_schema_tokens);
    let compaction_threshold = agent.max_context_tokens.saturating_mul(95) / 100;
    let prune_target = compaction_threshold.saturating_sub(tool_schema_tokens);

    if effective_tokens <= compaction_threshold {
        return conversation_history;
    }

    tracing::info!(
        "Auto-compacting conversation history: {} message tokens (+{} tool schema tokens) > {} (95% of budget)",
        estimated_tokens,
        tool_schema_tokens,
        compaction_threshold
    );
    let mut pruned = (*conversation_history).clone();
    Agent::prune_history_for_budget(agent.token_counter.as_ref(), &mut pruned, prune_target);
    std::sync::Arc::new(pruned)
}

/// Build final metadata and done events for the stream.
pub(crate) fn build_final_metadata_events(
    total_usage: &TokenUsageAccumulator,
    request_start: std::time::Instant,
    total_llm_ms: u64,
    total_tool_ms: u64,
    iteration_count: u32,
    actual_ttfb_ms: Option<u64>,
    finish_reason: Option<beef::lean::Cow<'static, str>>,
) -> (crate::messages::StreamEvent, crate::messages::StreamEvent) {
    let total_duration_ms = elapsed_millis_u64(request_start);
    let overhead_ms = total_duration_ms
        .saturating_sub(total_llm_ms)
        .saturating_sub(total_tool_ms);
    let metrics = PerformanceMetrics {
        total_duration_ms: Some(total_duration_ms),
        llm_duration_ms: Some(total_llm_ms),
        tool_duration_ms: Some(total_tool_ms),
        overhead_ms: Some(overhead_ms),
        step_count: Some(iteration_count),
        time_to_first_token_ms: actual_ttfb_ms,
        steps: None,
    };
    let metadata = crate::messages::StreamEvent::Metadata {
        usage: Some(total_usage.to_token_usage()),
        finish_reason,
        metrics: Some(metrics),
    };
    let done = crate::messages::StreamEvent::Done;
    (metadata, done)
}

pub(crate) fn last_user_query(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| {
            if matches!(m.role, MessageRole::User) {
                m.content.as_text().map(ToString::to_string)
            } else {
                None
            }
        })
        .unwrap_or_default()
}

pub(crate) fn order_native_stream_schemas(
    use_native: bool,
    tool_cfg_schemas_for_filter: &Arc<Vec<crate::messages::ToolSchema>>,
    tool_schemas_all: &Arc<Vec<crate::messages::ToolSchema>>,
    ordered_all: &Arc<Vec<crate::messages::ToolSchema>>,
) -> Arc<Vec<crate::messages::ToolSchema>> {
    if !use_native {
        return Arc::clone(tool_cfg_schemas_for_filter);
    }
    if std::sync::Arc::ptr_eq(tool_cfg_schemas_for_filter, tool_schemas_all) {
        return Arc::clone(ordered_all);
    }
    let cfg_names: HashSet<&str> = tool_cfg_schemas_for_filter
        .iter()
        .map(|s| s.name.as_ref())
        .collect();
    let filtered: Vec<_> = ordered_all
        .iter()
        .filter(|s| cfg_names.contains(s.name.as_ref()))
        .cloned()
        .collect();
    Arc::new(filtered)
}

pub(crate) fn schemas_for_current_phase(
    ordered_for_filtering: &Arc<Vec<crate::messages::ToolSchema>>,
    allowed: HashSet<String>,
) -> Arc<Vec<crate::messages::ToolSchema>> {
    let filtered: Vec<_> = ordered_for_filtering
        .iter()
        .filter(|s| allowed.contains(s.name.as_ref()))
        .cloned()
        .collect();
    if filtered.is_empty() {
        Arc::clone(ordered_for_filtering)
    } else {
        Arc::new(filtered)
    }
}
