use crate::agent::core::chat_helpers::{
    IndexedToolCall, SOFT_WRAP_UP_MESSAGE, TokenUsageAccumulator, advance_phase_if_needed,
    current_phase_allowed_tools, ensure_phase_plan_for_query, get_current_phase_display,
    iteration_limit_reached_suffix, length_truncated_tool_error, should_inject_soft_wrap,
};
use crate::agent::core::{Agent, ToolIntelligenceState, tool_extraction, tool_output};
use crate::messages::{Message, MessageRole};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::history::{
    apply_compaction_or_prune, build_final_metadata_events, schemas_for_current_phase,
    summarize_tool_output, tool_result_payload_with_duration,
};

pub(crate) fn recover_text_tool_calls(
    accumulated_text: &str,
    accumulated_tool_calls: &[crate::messages::ToolCallInfo],
    use_native: bool,
) -> (Vec<crate::messages::ToolCallInfo>, bool) {
    if !accumulated_tool_calls.is_empty() {
        return (Vec::new(), false);
    }
    let text_based = tool_extraction::extract_tool_calls_from_text(accumulated_text);
    if text_based.is_empty() {
        return (Vec::new(), false);
    }
    tracing::info!(
        extracted_count = text_based.len(),
        use_native,
        "Extracted tool calls from assistant text"
    );
    (tool_extraction::tool_calls_as_infos(&text_based), true)
}

pub(crate) fn text_tool_start_events(
    recovered_infos: &[crate::messages::ToolCallInfo],
) -> Vec<crate::messages::StreamEvent> {
    recovered_infos
        .iter()
        .map(|tc| crate::messages::StreamEvent::ToolCallStart {
            id: tc.id.clone(),
            name: tc.name.clone(),
            arguments: tc.arguments.clone(),
            thought_signature: None,
            action_id: tc.action_id.clone(),
        })
        .collect()
}

pub(crate) fn assistant_history_content(
    use_native: bool,
    has_tool_calls: bool,
    recovered_from_text: bool,
    recovered_infos: &[crate::messages::ToolCallInfo],
    accumulated_tool_calls: &[crate::messages::ToolCallInfo],
    accumulated_text: String,
) -> crate::messages::MessageContent {
    if use_native && has_tool_calls {
        let history_tool_calls = if recovered_from_text {
            recovered_infos.to_vec()
        } else {
            accumulated_tool_calls.to_vec()
        };
        crate::messages::MessageContent::Structured {
            text: if accumulated_text.is_empty() {
                None
            } else {
                Some(Arc::new(accumulated_text))
            },
            tool_calls: Some(history_tool_calls),
            tool_call_id: None,
        }
    } else {
        crate::messages::MessageContent::text(accumulated_text)
    }
}

pub(crate) fn indexed_calls_from_infos(
    infos: Vec<crate::messages::ToolCallInfo>,
) -> Vec<IndexedToolCall> {
    infos
        .into_iter()
        .enumerate()
        .map(|(idx, tc)| IndexedToolCall {
            idx,
            id: tc.id.to_string(),
            name: tc.name.to_string(),
            arguments: tc.arguments,
            action_id: tc.action_id,
        })
        .collect()
}

pub(crate) fn truncated_refusal_events(
    refusals: &[(usize, String, String, crate::agent::tools::tool::ToolResult)],
    action_ids: &HashMap<String, String>,
) -> Vec<crate::messages::StreamEvent> {
    refusals
        .iter()
        .map(
            |(_idx, id, name, result)| crate::messages::StreamEvent::ToolCallResult {
                name: beef::lean::Cow::owned(name.clone()),
                result: result
                    .error
                    .clone()
                    .unwrap_or_else(|| length_truncated_tool_error(name)),
                typed_payload: Some(serde_json::json!({
                    "refused": true,
                    "reason": "length_truncated",
                })),
                id: Some(id.clone()),
                action_id: action_ids.get(id).cloned(),
            },
        )
        .collect()
}

pub(crate) fn invalid_tool_events(
    invalid: &[(usize, String, String, beef::lean::Cow<'static, str>)],
    action_ids: &HashMap<String, String>,
) -> Vec<crate::messages::StreamEvent> {
    invalid
        .iter()
        .map(|(_idx, tool_call_id, tool_name, error_msg)| {
            crate::messages::StreamEvent::ToolCallResult {
                name: beef::lean::Cow::owned(tool_name.clone()),
                result: format!("Tool call validation failed: {error_msg}"),
                typed_payload: None,
                id: Some(tool_call_id.clone()),
                action_id: action_ids.get(tool_call_id).cloned(),
            }
        })
        .collect()
}

pub(crate) type ExecutedTool = (
    usize,
    String,
    String,
    serde_json::Value,
    u64,
    crate::agent::tools::tool::ToolResult,
);

pub(crate) fn split_fresh_and_cached_calls(
    valid: &[(usize, String, Arc<crate::agent::tools::tool::ToolCall>)],
    turn_tool_cache: &HashMap<(String, String), String>,
    turn_dup_count: &mut HashMap<(String, String), u32>,
) -> (
    Vec<(usize, String, Arc<crate::agent::tools::tool::ToolCall>)>,
    HashMap<usize, (String, String)>,
    Vec<ExecutedTool>,
) {
    let mut fresh_calls = Vec::new();
    let mut fresh_cache_keys = HashMap::new();
    let mut cached_results = Vec::new();
    for (idx, id, tool_call) in valid {
        let cache_key = (
            tool_call.name.to_string(),
            serde_json::to_string(&tool_call.arguments).unwrap_or_default(),
        );
        if let Some(prev) = turn_tool_cache.get(&cache_key) {
            cached_results.push(cached_duplicate_result(
                *idx,
                id,
                tool_call,
                &cache_key,
                prev,
                turn_dup_count,
            ));
        } else {
            fresh_cache_keys.insert(*idx, cache_key);
            fresh_calls.push((*idx, id.clone(), Arc::clone(tool_call)));
        }
    }
    (fresh_calls, fresh_cache_keys, cached_results)
}

pub(crate) fn cached_duplicate_result(
    idx: usize,
    id: &str,
    tool_call: &crate::agent::tools::tool::ToolCall,
    cache_key: &(String, String),
    prev: &str,
    turn_dup_count: &mut HashMap<(String, String), u32>,
) -> ExecutedTool {
    let count = turn_dup_count
        .entry(cache_key.clone())
        .and_modify(|c| *c = c.saturating_add(1))
        .or_insert(1);
    tracing::info!(
        "Dedup: tool {}({}) already executed this turn (hit #{}), returning cached result",
        tool_call.name,
        caliche_contracts::text::truncate_utf8(&cache_key.1, 80),
        *count,
    );
    let header = if *count >= 2 {
        format!(
            "[REPEATED DUPLICATE -- hit #{count}. STOP re-issuing this exact `{name}` call: the result is already in your context from earlier in this turn. Reason from what you have or call a DIFFERENT tool / different arguments.]\n",
            name = tool_call.name,
        )
    } else {
        format!(
            "[Cached: identical `{name}` call already executed this turn -- using cached result. Do NOT re-issue this exact call again; the content is in your context.]\n",
            name = tool_call.name,
        )
    };
    (
        idx,
        id.to_string(),
        tool_call.name.to_string(),
        tool_call.arguments.clone(),
        0,
        crate::agent::tools::tool::ToolResult::success(format!("{header}{prev}")),
    )
}

pub(crate) fn cache_fresh_exec_results(
    executed: &[ExecutedTool],
    fresh_cache_keys: &mut HashMap<usize, (String, String)>,
    turn_tool_cache: &mut HashMap<(String, String), String>,
) {
    for (idx, _id, name, _args, _dur, result) in executed {
        if result.success
            && !result.content.is_empty()
            && let Some(cache_key) = fresh_cache_keys.remove(idx)
        {
            debug_assert_eq!(cache_key.0, *name);
            turn_tool_cache
                .entry(cache_key)
                .or_insert_with(|| result.content.clone());
        }
    }
}

pub(crate) fn executed_tool_result_event(
    tool_call_id: &str,
    tool_name: &str,
    duration_ms: u64,
    tool_result: &crate::agent::tools::tool::ToolResult,
    action_id: Option<String>,
) -> crate::messages::StreamEvent {
    let result_text = if tool_result.success {
        if tool_result.content.is_empty() {
            "Tool executed successfully (no output)".to_string()
        } else {
            summarize_tool_output(&tool_result.content, tool_name)
        }
    } else {
        format!(
            "Error: {}",
            tool_result.error.as_deref().unwrap_or("Unknown error")
        )
    };
    crate::messages::StreamEvent::ToolCallResult {
        name: beef::lean::Cow::owned(tool_name.to_string()),
        result: result_text,
        typed_payload: Some(tool_result_payload_with_duration(
            tool_result.typed_payload.clone(),
            duration_ms,
        )),
        id: Some(tool_call_id.to_string()),
        action_id,
    }
}

pub(crate) fn summarize_history_tool_output(
    content: &str,
    tool_name: &str,
    max_chars: usize,
    annotate: bool,
) -> String {
    tool_output::summarize_tool_output(
        content,
        tool_name,
        tool_output::ToolOutputLimits {
            max_chars,
            summarize_threshold: max_chars,
            head_chars: max_chars * 4 / 5,
            tail_chars: max_chars / 10,
            annotate,
        },
    )
}

#[cfg(feature = "compaction")]
pub(crate) struct CompactEmit {
    pub(crate) history: Arc<Vec<Message>>,
    pub(crate) events: Vec<crate::messages::StreamEvent>,
    pub(crate) stop: bool,
    pub(crate) error: Option<anyhow::Error>,
}

#[cfg(feature = "compaction")]
pub(crate) async fn compact_emit(
    agent: &Agent,
    conversation_history: Arc<Vec<Message>>,
    tool_schema_tokens: usize,
    usage: &TokenUsageAccumulator,
) -> CompactEmit {
    let history_keep = Arc::clone(&conversation_history);
    match compact_action_from_result(
        apply_compaction_or_prune(
            agent,
            conversation_history,
            tool_schema_tokens,
            usage,
            agent.max_iterations,
        )
        .await,
    ) {
        CompactAction::Keep { history, event } => CompactEmit {
            history,
            events: event.into_iter().collect(),
            stop: false,
            error: None,
        },
        CompactAction::Stop { history, event } => CompactEmit {
            history,
            events: vec![event],
            stop: true,
            error: None,
        },
        CompactAction::Err(error) => CompactEmit {
            history: history_keep,
            events: Vec::new(),
            stop: false,
            error: Some(error),
        },
    }
}

#[cfg(feature = "compaction")]
pub(crate) enum CompactAction {
    Keep {
        history: Arc<Vec<Message>>,
        event: Option<crate::messages::StreamEvent>,
    },
    Stop {
        history: Arc<Vec<Message>>,
        event: crate::messages::StreamEvent,
    },
    Err(anyhow::Error),
}

#[cfg(feature = "compaction")]
pub(crate) fn compact_action_from_result(
    result: Result<
        (
            Arc<Vec<Message>>,
            Option<crate::messages::StreamEvent>,
            bool,
        ),
        anyhow::Error,
    >,
) -> CompactAction {
    match result {
        Ok((history, event, stop)) => {
            if let Some(ev) = event {
                if stop {
                    CompactAction::Stop { history, event: ev }
                } else {
                    CompactAction::Keep {
                        history,
                        event: Some(ev),
                    }
                }
            } else {
                CompactAction::Keep {
                    history,
                    event: None,
                }
            }
        }
        Err(e) => CompactAction::Err(e),
    }
}

pub(crate) fn iteration_limit_events(
    limit: usize,
    usage: crate::messages::TokenUsage,
) -> [crate::messages::StreamEvent; 3] {
    tracing::warn!(
        "Maximum iterations ({limit}) reached in streaming chat, returning partial result"
    );
    let suffix = iteration_limit_reached_suffix(limit);
    [
        crate::messages::StreamEvent::TextChunk {
            content: format!("\n\n{suffix}"),
        },
        crate::messages::StreamEvent::AgentEnd,
        crate::messages::StreamEvent::LimitReached {
            max_iterations: limit,
            usage,
        },
    ]
}

pub(crate) enum NoToolsAction {
    ContinueWithPhase(Option<crate::messages::StreamEvent>),
    ContinueFollowUp,
    Stop(crate::messages::StreamEvent, crate::messages::StreamEvent),
}

pub(crate) async fn no_tools_action(
    agent: &Agent,
    phased_enabled: bool,
    tool_intelligence: &Arc<tokio::sync::Mutex<ToolIntelligenceState>>,
    conversation_history: &mut Arc<Vec<Message>>,
    total_usage: &TokenUsageAccumulator,
    request_start: std::time::Instant,
    total_llm_ms: u64,
    total_tool_ms: u64,
    iteration_count: u32,
    ttfb_ms: Option<u64>,
    finish_reason: Option<beef::lean::Cow<'static, str>>,
) -> NoToolsAction {
    if phased_enabled {
        let advanced = advance_phase_if_needed(phased_enabled, tool_intelligence).await;
        if advanced {
            let event =
                get_current_phase_display(tool_intelligence)
                    .await
                    .map(
                        |(idx, total, tools)| crate::messages::StreamEvent::PhaseTransition {
                            phase_index: idx,
                            phase_total: total,
                            tool_names: tools,
                        },
                    );
            return NoToolsAction::ContinueWithPhase(event);
        }
    }
    if append_stream_follow_ups(agent, conversation_history).await {
        return NoToolsAction::ContinueFollowUp;
    }
    let (metadata_ev, done_ev) = build_final_metadata_events(
        total_usage,
        request_start,
        total_llm_ms,
        total_tool_ms,
        iteration_count,
        ttfb_ms,
        finish_reason,
    );
    NoToolsAction::Stop(metadata_ev, done_ev)
}

pub(crate) async fn append_stream_follow_ups(
    agent: &Agent,
    conversation_history: &mut Arc<Vec<Message>>,
) -> bool {
    let Some(ref provider) = agent.steering_followup_provider else {
        return false;
    };
    let follow_up = provider.get_follow_up_messages().await;
    let n = follow_up.len();
    if n == 0 {
        return false;
    }
    let history = std::sync::Arc::make_mut(conversation_history);
    for msg in follow_up {
        history.push(msg);
    }
    tracing::info!("Follow-up messages queued ({n}), continuing stream turn");
    true
}

pub(crate) async fn init_phase_event(
    phased_enabled: bool,
    phase_plan_initialized: &mut bool,
    query: &str,
    ordered_for_planning: &Arc<Vec<crate::messages::ToolSchema>>,
    allowed_tools: Option<&Arc<HashSet<String>>>,
    tool_intelligence: &Arc<tokio::sync::Mutex<ToolIntelligenceState>>,
) -> anyhow::Result<Option<crate::messages::StreamEvent>> {
    if !phased_enabled || *phase_plan_initialized {
        return Ok(None);
    }
    ensure_phase_plan_for_query(
        phased_enabled,
        query,
        ordered_for_planning,
        allowed_tools,
        tool_intelligence,
    )
    .await
    .map_err(anyhow::Error::from)?;
    *phase_plan_initialized = true;
    Ok(get_current_phase_display(tool_intelligence)
        .await
        .map(
            |(idx, total, tools)| crate::messages::StreamEvent::PhaseTransition {
                phase_index: idx,
                phase_total: total,
                tool_names: tools,
            },
        ))
}

pub(crate) fn request_tools_option(
    use_native: bool,
    supports_text_based_tools: bool,
    tool_schemas_for_request: &Arc<Vec<crate::messages::ToolSchema>>,
) -> Option<Arc<Vec<crate::messages::ToolSchema>>> {
    if use_native || supports_text_based_tools {
        Some(Arc::clone(tool_schemas_for_request))
    } else {
        None
    }
}

pub(crate) fn estimate_stream_schema_tokens(
    agent: &Agent,
    use_native: bool,
    supports_text_based_tools: bool,
) -> usize {
    if use_native || supports_text_based_tools {
        agent
            .tools
            .estimate_schema_tokens(agent.token_counter.as_ref())
    } else {
        0
    }
}

pub(crate) fn inject_soft_wrap(
    conversation_history: &mut Arc<Vec<Message>>,
    iteration_count: usize,
    max_iterations: usize,
) {
    if !should_inject_soft_wrap(iteration_count, max_iterations) {
        return;
    }
    let history = std::sync::Arc::make_mut(conversation_history);
    history.push(Message {
        role: MessageRole::System,
        content: crate::messages::MessageContent::text(SOFT_WRAP_UP_MESSAGE.to_string()),
        metadata: None,
    });
}

#[cfg(not(feature = "compaction"))]
pub(crate) fn maybe_prune_periodic(
    agent: &Agent,
    conversation_history: Arc<Vec<Message>>,
    tool_schema_tokens: usize,
    iteration_count: u32,
) -> Arc<Vec<Message>> {
    const PRUNE_STREAM_EVERY_N_ITERATIONS: u32 = 5;
    if iteration_count > 1 && iteration_count % PRUNE_STREAM_EVERY_N_ITERATIONS == 0 {
        prune_stream_history_if_needed(agent, conversation_history, tool_schema_tokens)
    } else {
        conversation_history
    }
}

pub(crate) async fn stream_request_schemas(
    use_native: bool,
    supports_text_based_tools: bool,
    phased_enabled: bool,
    ordered_for_filtering: &Arc<Vec<crate::messages::ToolSchema>>,
    tool_intelligence: &Arc<tokio::sync::Mutex<ToolIntelligenceState>>,
) -> Arc<Vec<crate::messages::ToolSchema>> {
    if !(use_native || supports_text_based_tools) {
        return Arc::new(Vec::new());
    }
    if !phased_enabled {
        return Arc::clone(ordered_for_filtering);
    }
    let allowed = current_phase_allowed_tools(phased_enabled, tool_intelligence)
        .await
        .unwrap_or_else(HashSet::new);
    schemas_for_current_phase(ordered_for_filtering, allowed)
}

pub(crate) fn build_stream_request(
    agent: &Agent,
    messages: Arc<Vec<Message>>,
    iteration_count: u32,
    use_native: bool,
    supports_text_based_tools: bool,
    schemas: &Arc<Vec<crate::messages::ToolSchema>>,
) -> crate::messages::CompletionRequest {
    crate::messages::CompletionRequest {
        messages,
        max_tokens: agent.max_tokens.or(Some(4096)),
        temperature: agent.temperature.or(Some(0.7)),
        reasoning_effort: agent
            .reasoning_effort_slot
            .as_ref()
            .and_then(|slot| slot.current()),
        tools: request_tools_option(use_native, supports_text_based_tools, schemas),
        tool_choice: agent.tool_choice_for_iteration(iteration_count as usize, use_native),
        response_format: None,
    }
}
