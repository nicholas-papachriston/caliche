use crate::agent::core::Agent;
use crate::agent::core::chat_helpers::{
    IndexedToolCall, IterationLimiter, TokenUsageAccumulator, append_invalid_tool_messages_native,
    append_owned_tool_results_to_history, append_tool_results_to_history, execute_tools_parallel,
    is_length_finish_reason, refuse_length_truncated_calls, validate_and_filter_tool_calls,
};
use crate::messages::{Message, MessageRole};
use futures_util::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;

use super::PersistMemory;
use super::history::*;
use super::llm::*;
use super::tools::*;

pub(crate) struct StreamLoopCtx<'a> {
    pub(crate) agent: &'a Agent,
    pub(crate) persist_memory: &'a PersistMemory,
    pub(crate) cancellation_flag: Option<&'a std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub(crate) conversation_history: Arc<Vec<Message>>,
    pub(crate) limiter: IterationLimiter,
    pub(crate) iteration_count: u32,
    pub(crate) phase_plan_initialized: bool,
    pub(crate) total_usage: TokenUsageAccumulator,
    pub(crate) total_llm_ms: u64,
    pub(crate) total_tool_ms: u64,
    pub(crate) first_llm_ms: Option<u64>,
    pub(crate) actual_ttfb_ms: Option<u64>,
    pub(crate) turn_tool_cache: HashMap<(String, String), String>,
    pub(crate) turn_dup_count: HashMap<(String, String), u32>,
    pub(crate) failure_breaker: std::sync::Arc<
        std::sync::Mutex<crate::agent::tools::failure_breaker::IdenticalFailureBreaker>,
    >,
    pub(crate) use_native: bool,
    pub(crate) supports_text_based_tools: bool,
    pub(crate) phased_enabled: bool,
    pub(crate) tool_schema_tokens: usize,
    pub(crate) query: String,
    pub(crate) ordered_for_planning: Arc<Vec<crate::messages::ToolSchema>>,
    pub(crate) ordered_for_filtering: Arc<Vec<crate::messages::ToolSchema>>,
    pub(crate) request_start: std::time::Instant,
    pub(crate) readonly_filter_hits: Arc<std::sync::atomic::AtomicUsize>,
}

pub(crate) enum PrepTurn {
    Ready {
        events: Vec<crate::messages::StreamEvent>,
        schemas: Arc<Vec<crate::messages::ToolSchema>>,
    },
    LimitReached([crate::messages::StreamEvent; 3]),
    CompactStop {
        events: Vec<crate::messages::StreamEvent>,
    },
    Fail(anyhow::Error),
}

pub(crate) enum AfterLlmAction {
    Continue,
    Return,
    Fail(anyhow::Error),
}

pub(crate) struct AfterLlm {
    pub(crate) events: Vec<crate::messages::StreamEvent>,
    pub(crate) action: AfterLlmAction,
    pub(crate) extra_tool_ms: u64,
}

pub(crate) enum TurnItem {
    Event(crate::messages::StreamEvent),
    Break,
    Return,
    Error(anyhow::Error),
}

pub(crate) enum TurnEndKind {
    Continue,
    Break,
    Return,
}

pub(crate) async fn prepare_stream_turn(ctx: &mut StreamLoopCtx<'_>) -> PrepTurn {
    if ctx.limiter.tick() {
        return PrepTurn::LimitReached(iteration_limit_events(
            ctx.limiter.limit(),
            ctx.total_usage.to_token_usage(),
        ));
    }
    ctx.iteration_count = ctx.iteration_count.saturating_add(1);
    let mut events = vec![crate::messages::StreamEvent::TurnStart];
    inject_soft_wrap(
        &mut ctx.conversation_history,
        ctx.iteration_count as usize,
        ctx.agent.max_iterations,
    );

    #[cfg(not(feature = "compaction"))]
    {
        ctx.conversation_history = maybe_prune_periodic(
            ctx.agent,
            Arc::clone(&ctx.conversation_history),
            ctx.tool_schema_tokens,
            ctx.iteration_count,
        );
    }

    #[cfg(feature = "compaction")]
    {
        let compact = compact_emit(
            ctx.agent,
            Arc::clone(&ctx.conversation_history),
            ctx.tool_schema_tokens,
            &ctx.total_usage,
        )
        .await;
        ctx.conversation_history = compact.history;
        events.extend(compact.events);
        if let Some(error) = compact.error {
            return PrepTurn::Fail(error);
        }
        if compact.stop {
            return PrepTurn::CompactStop { events };
        }
    }

    match init_phase_event(
        ctx.phased_enabled,
        &mut ctx.phase_plan_initialized,
        &ctx.query,
        &ctx.ordered_for_planning,
        ctx.agent.allowed_tools.as_ref(),
        &ctx.agent.tool_intelligence,
    )
    .await
    {
        Ok(Some(ev)) => events.push(ev),
        Ok(None) => {}
        Err(e) => return PrepTurn::Fail(e),
    }

    if let Err(e) = ctx
        .agent
        .ensure_context_window_available(ctx.conversation_history.as_ref(), ctx.tool_schema_tokens)
    {
        return PrepTurn::Fail(anyhow::Error::new(e));
    }

    tracing::info!(
        "Streaming step {} - cumulative tokens: {}",
        ctx.iteration_count,
        ctx.total_usage.total_tokens()
    );

    let schemas = stream_request_schemas(
        ctx.use_native,
        ctx.supports_text_based_tools,
        ctx.phased_enabled,
        &ctx.ordered_for_filtering,
        &ctx.agent.tool_intelligence,
    )
    .await;
    PrepTurn::Ready { events, schemas }
}

pub(crate) fn append_invalid_stream_events(
    use_native: bool,
    invalid: &[(usize, String, String, beef::lean::Cow<'static, str>)],
    conversation_history: &mut Arc<Vec<Message>>,
    events: &mut Vec<crate::messages::StreamEvent>,
    action_ids: &std::collections::HashMap<String, String>,
) {
    if !use_native || invalid.is_empty() {
        return;
    }
    events.extend(invalid_tool_events(invalid, action_ids));
    let history = std::sync::Arc::make_mut(conversation_history);
    append_invalid_tool_messages_native(history, invalid);
}

pub(crate) fn truncated_stream_turn(
    conversation_history: &mut Arc<Vec<Message>>,
    use_native: bool,
    indexed_calls: Vec<IndexedToolCall>,
) -> AfterLlm {
    tracing::warn!(
        "Refusing {} tool call(s): assistant finish_reason indicates length truncation",
        indexed_calls.len()
    );
    let action_ids = action_ids_from_calls(&indexed_calls);
    let refusals = refuse_length_truncated_calls(&indexed_calls);
    let events = truncated_refusal_events(&refusals, &action_ids);
    let history = std::sync::Arc::make_mut(conversation_history);
    append_owned_tool_results_to_history(history, use_native, &refusals);
    AfterLlm {
        events,
        action: AfterLlmAction::Continue,
        extra_tool_ms: 0,
    }
}

pub(crate) fn record_allowed_tool_denials(
    readonly_filter_hits: &Arc<std::sync::atomic::AtomicUsize>,
    denials: usize,
) {
    if denials == 0 {
        return;
    }
    readonly_filter_hits.fetch_add(denials, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn collect_executed_result_events(
    executed: &[ExecutedTool],
    total_usage: &mut TokenUsageAccumulator,
    action_ids: &std::collections::HashMap<String, String>,
) -> (u64, Vec<crate::messages::StreamEvent>) {
    let iter_tool_ms: u64 = executed
        .iter()
        .map(|(_idx, _id, _name, _args, dur, _res)| *dur)
        .sum();
    let mut events = Vec::with_capacity(executed.len());
    for (_idx, tool_call_id, tool_name, _args, duration_ms, tool_result) in executed {
        if let Some(tool_usage) = tool_result.usage.as_ref() {
            total_usage.add_usage(tool_usage);
        }
        events.push(executed_tool_result_event(
            tool_call_id,
            tool_name,
            *duration_ms,
            tool_result,
            action_ids.get(tool_call_id).cloned(),
        ));
    }
    (iter_tool_ms, events)
}

fn action_ids_from_calls(
    calls: &[crate::agent::core::chat_helpers::IndexedToolCall],
) -> std::collections::HashMap<String, String> {
    calls
        .iter()
        .filter_map(|call| {
            call.action_id
                .as_ref()
                .map(|action_id| (call.id.clone(), action_id.clone()))
        })
        .collect()
}

fn publish_action_ids(
    agent: &crate::agent::core::Agent,
    calls: &[crate::agent::core::chat_helpers::IndexedToolCall],
) {
    let mut ids = agent
        .action_ids
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for call in calls {
        if let Some(action_id) = &call.action_id {
            ids.insert(call.id.clone(), action_id.clone());
        }
    }
}

fn ensure_action_ids(infos: &mut [crate::messages::ToolCallInfo]) {
    for info in infos.iter_mut() {
        if info.action_id.as_ref().is_none_or(String::is_empty) {
            info.action_id = Some(crate::utils::time_id::generate_time_id().to_string());
        }
    }
}

pub(crate) fn append_executed_history(
    conversation_history: &mut Arc<Vec<Message>>,
    use_native: bool,
    executed: &[ExecutedTool],
    max_chars: usize,
    annotate: bool,
) {
    let results_for_history: Vec<(usize, &str, &str, &crate::agent::tools::tool::ToolResult)> =
        executed
            .iter()
            .map(|(idx, id, name, _args, _dur, result)| (*idx, id.as_str(), name.as_str(), result))
            .collect();
    let history = std::sync::Arc::make_mut(conversation_history);
    append_tool_results_to_history(
        history,
        use_native,
        &results_for_history,
        |content, tool_name| summarize_history_tool_output(content, tool_name, max_chars, annotate),
    );
}

pub(crate) async fn after_no_tools(
    ctx: &mut StreamLoopCtx<'_>,
    finish_reason: Option<beef::lean::Cow<'static, str>>,
) -> AfterLlm {
    match no_tools_action(
        ctx.agent,
        ctx.phased_enabled,
        &ctx.agent.tool_intelligence,
        &mut ctx.conversation_history,
        &ctx.total_usage,
        ctx.request_start,
        ctx.total_llm_ms,
        ctx.total_tool_ms,
        ctx.iteration_count,
        ctx.actual_ttfb_ms.or(ctx.first_llm_ms),
        finish_reason,
    )
    .await
    {
        NoToolsAction::ContinueWithPhase(event) => AfterLlm {
            events: event.into_iter().collect(),
            action: AfterLlmAction::Continue,
            extra_tool_ms: 0,
        },
        NoToolsAction::ContinueFollowUp => AfterLlm {
            events: Vec::new(),
            action: AfterLlmAction::Continue,
            extra_tool_ms: 0,
        },
        NoToolsAction::Stop(metadata_ev, done_ev) => {
            persist_stream_memory(ctx.persist_memory, &ctx.conversation_history).await;
            AfterLlm {
                events: vec![crate::messages::StreamEvent::TurnEnd, metadata_ev, done_ev],
                action: AfterLlmAction::Return,
                extra_tool_ms: 0,
            }
        }
    }
}

pub(crate) async fn execute_stream_tools(
    ctx: &mut StreamLoopCtx<'_>,
    accumulated_tool_calls: Vec<crate::messages::ToolCallInfo>,
    recovered_infos: Vec<crate::messages::ToolCallInfo>,
    finish_reason: Option<beef::lean::Cow<'static, str>>,
) -> AfterLlm {
    let tool_call_infos_for_execution = if accumulated_tool_calls.is_empty() {
        recovered_infos
    } else {
        accumulated_tool_calls
    };
    let indexed_calls = indexed_calls_from_infos(tool_call_infos_for_execution);
    let action_ids = action_ids_from_calls(&indexed_calls);
    if is_length_finish_reason(finish_reason.as_deref()) && !indexed_calls.is_empty() {
        return truncated_stream_turn(&mut ctx.conversation_history, ctx.use_native, indexed_calls);
    }
    publish_action_ids(ctx.agent, &indexed_calls);

    let validation = match validate_and_filter_tool_calls(
        indexed_calls,
        &ctx.agent.tools,
        ctx.agent.allowed_tools.as_ref(),
        ctx.agent.permission_mode,
        ctx.agent.tool_execution_driver,
        &ctx.agent.tool_intelligence,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return AfterLlm {
                events: Vec::new(),
                action: AfterLlmAction::Fail(e.into()),
                extra_tool_ms: 0,
            };
        }
    };
    record_allowed_tool_denials(&ctx.readonly_filter_hits, validation.allowed_tools_denials);

    let mut events = Vec::new();
    append_invalid_stream_events(
        ctx.use_native,
        &validation.invalid,
        &mut ctx.conversation_history,
        &mut events,
        &action_ids,
    );
    if validation.valid.is_empty() {
        return AfterLlm {
            events,
            action: AfterLlmAction::Continue,
            extra_tool_ms: 0,
        };
    }

    let (fresh_calls, mut fresh_cache_keys, mut cached_results) = split_fresh_and_cached_calls(
        &validation.valid,
        &ctx.turn_tool_cache,
        &mut ctx.turn_dup_count,
    );

    let mut exec = crate::agent::tools::failure_breaker::with_turn_scope(
        std::sync::Arc::clone(&ctx.failure_breaker),
        std::sync::Arc::clone(&ctx.agent.action_ids),
        execute_tools_parallel(
            crate::agent::core::chat_helpers::tool_pipeline::ExecuteToolsParallelParams {
                valid_tool_calls: fresh_calls,
                tools: &ctx.agent.tools,
                hooks: &ctx.agent.hooks,
                cancellation_flag: ctx.cancellation_flag,
                user_context: ctx.agent.user_context.as_ref(),
                metrics: ctx.agent.metrics.as_ref(),
                tool_intelligence: &ctx.agent.tool_intelligence,
                approval_resolver: ctx.agent.approval_resolver.clone(),
                amendment_persister: ctx.agent.amendment_persister.clone(),
                permission_mode: ctx.agent.permission_mode,
                workspace_root: ctx.agent.workspace_root_buf(),
                #[cfg(feature = "tools-rag")]
                workspace_rag: ctx.agent.workspace_rag.as_deref(),
                #[cfg(feature = "tools-rag")]
                workspace_rag_top_k: ctx.agent.workspace_rag_top_k,
                #[cfg(feature = "tools-rag")]
                workspace_rag_min_similarity: ctx.agent.workspace_rag_min_similarity,
                #[cfg(feature = "agentfs")]
                agentfs_tool_logger: ctx.agent.agentfs_tool_logger.as_deref(),
                #[cfg(feature = "sandbox")]
                sandbox: ctx
                    .agent
                    .sandbox
                    .as_deref()
                    .map(|h| h as &dyn caliche_sandbox::SandboxHandleOps),
                operation_manager: ctx.agent.operation_manager.as_deref(),
                operation_translators: ctx.agent.operation_translators.as_deref(),
            },
        ),
    )
    .await;

    cache_fresh_exec_results(
        &exec.executed,
        &mut fresh_cache_keys,
        &mut ctx.turn_tool_cache,
    );
    exec.executed.append(&mut cached_results);
    exec.executed.sort_by_key(|(idx, ..)| *idx);

    append_invalid_stream_events(
        ctx.use_native,
        &exec.denied,
        &mut ctx.conversation_history,
        &mut events,
        &action_ids,
    );

    let (iter_tool_ms, result_events) =
        collect_executed_result_events(&exec.executed, &mut ctx.total_usage, &action_ids);
    events.extend(result_events);
    append_executed_history(
        &mut ctx.conversation_history,
        ctx.use_native,
        &exec.executed,
        ctx.agent.max_tool_output_in_history,
        ctx.agent.annotate_tool_output_truncation,
    );
    events.push(crate::messages::StreamEvent::TurnEnd);
    AfterLlm {
        events,
        action: AfterLlmAction::Continue,
        extra_tool_ms: iter_tool_ms,
    }
}

pub(crate) fn push_assistant_history(
    conversation_history: &mut Arc<Vec<Message>>,
    use_native: bool,
    has_tool_calls: bool,
    recovered_from_text: bool,
    recovered_infos: &[crate::messages::ToolCallInfo],
    accumulated_tool_calls: &[crate::messages::ToolCallInfo],
    accumulated_text: String,
    accumulated_thinking: String,
) {
    let assistant_content = assistant_history_content(
        use_native,
        has_tool_calls,
        recovered_from_text,
        recovered_infos,
        accumulated_tool_calls,
        accumulated_text,
    );
    let history = std::sync::Arc::make_mut(conversation_history);
    let assistant_message = Message {
        role: MessageRole::Assistant,
        content: assistant_content,
        metadata: None,
    }
    .with_reasoning_content(accumulated_thinking);
    history.push(assistant_message);
}

pub(crate) async fn finish_after_llm(
    ctx: &mut StreamLoopCtx<'_>,
    mut acc: LlmStreamAcc,
) -> AfterLlm {
    let finish_reason = acc.finish_reason.take();
    let (mut recovered_infos, recovered_from_text) = recover_text_tool_calls(
        &acc.accumulated_text,
        &acc.accumulated_tool_calls,
        ctx.use_native,
    );
    ensure_action_ids(&mut recovered_infos);
    let mut events = text_tool_start_events(&recovered_infos);
    let has_tool_calls = !acc.accumulated_tool_calls.is_empty() || recovered_from_text;
    let accumulated_text = std::mem::take(&mut acc.accumulated_text);
    let accumulated_thinking = std::mem::take(&mut acc.accumulated_thinking);
    let mut accumulated_tool_calls = std::mem::take(&mut acc.accumulated_tool_calls);
    ensure_action_ids(&mut accumulated_tool_calls);
    push_assistant_history(
        &mut ctx.conversation_history,
        ctx.use_native,
        has_tool_calls,
        recovered_from_text,
        &recovered_infos,
        &accumulated_tool_calls,
        accumulated_text,
        accumulated_thinking,
    );
    if !has_tool_calls {
        let mut out = after_no_tools(ctx, finish_reason).await;
        events.append(&mut out.events);
        out.events = events;
        return out;
    }
    let mut out =
        execute_stream_tools(ctx, accumulated_tool_calls, recovered_infos, finish_reason).await;
    events.append(&mut out.events);
    out.events = events;
    out
}

pub(crate) fn stream_one_turn<'a>(
    ctx: &'a mut StreamLoopCtx<'_>,
) -> impl futures_util::Stream<Item = TurnItem> + 'a {
    async_stream::stream! {
        match prepare_stream_turn(ctx).await {
            PrepTurn::LimitReached(events) => {
                yield TurnItem::Event(events[0].clone());
                yield TurnItem::Event(events[1].clone());
                persist_stream_memory(ctx.persist_memory, &ctx.conversation_history).await;
                yield TurnItem::Event(events[2].clone());
                yield TurnItem::Break;
                return;
            }
            PrepTurn::CompactStop { events } => {
                for ev in events {
                    yield TurnItem::Event(ev);
                }
                persist_stream_memory(ctx.persist_memory, &ctx.conversation_history).await;
                yield TurnItem::Break;
                return;
            }
            PrepTurn::Fail(e) => {
                yield TurnItem::Error(e);
                return;
            }
            PrepTurn::Ready { events, schemas } => {
                for ev in events {
                    yield TurnItem::Event(ev);
                }
                let llm_start = std::time::Instant::now();
                let request = build_stream_request(
                    ctx.agent,
                    Arc::clone(&ctx.conversation_history),
                    ctx.iteration_count,
                    ctx.use_native,
                    ctx.supports_text_based_tools,
                    &schemas,
                );
                let stream = ctx.agent.llm_provider.stream(request);
                let usage_policy = ctx.agent.llm_provider.streaming_usage_policy();
                let mut llm_turn = Box::pin(consume_llm_turn(
                    stream,
                    ctx.cancellation_flag,
                    usage_policy,
                    llm_start,
                ));
                let mut acc = None;
                while let Some(item) = llm_turn.next().await {
                    match item {
                        LlmTurnItem::Chunk(ev) => yield TurnItem::Event(ev),
                        LlmTurnItem::Finished(finished) => acc = Some(finished),
                        LlmTurnItem::Cancelled => {
                            persist_stream_memory(ctx.persist_memory, &ctx.conversation_history).await;
                            yield TurnItem::Error(anyhow::anyhow!("Stream cancelled by user"));
                            return;
                        }
                        LlmTurnItem::Error(e) => {
                            yield TurnItem::Error(e);
                            return;
                        }
                    }
                }
                let Some(acc) = acc else {
                    return;
                };
                add_acc_llm_duration(
                    &mut ctx.total_llm_ms,
                    &mut ctx.first_llm_ms,
                    acc.llm_duration_ms,
                );
                ctx.actual_ttfb_ms = ctx.actual_ttfb_ms.or(acc.actual_ttfb_ms);
                tracing::debug!(
                    "Inner stream loop completed for step {} ({} text chars, {} tool calls)",
                    ctx.iteration_count,
                    acc.accumulated_text.len(),
                    acc.accumulated_tool_calls.len(),
                );
                add_stream_usage(&mut ctx.total_usage, acc.final_usage.as_ref());
                let out = finish_after_llm(ctx, acc).await;
                ctx.total_tool_ms = ctx.total_tool_ms.saturating_add(out.extra_tool_ms);
                for ev in out.events {
                    yield TurnItem::Event(ev);
                }
                match out.action {
                    AfterLlmAction::Continue => {}
                    AfterLlmAction::Return => yield TurnItem::Return,
                    AfterLlmAction::Fail(e) => yield TurnItem::Error(e),
                }
            }
        }
    }
}
