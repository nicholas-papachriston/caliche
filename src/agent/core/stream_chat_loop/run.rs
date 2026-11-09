use crate::agent::ToolExecutionDriver;
use crate::agent::core::chat_helpers::{IterationLimiter, TokenUsageAccumulator};
use crate::agent::core::tool_ordering::order_tools;
use crate::agent::core::{Agent, ToolCallingConfig};
use crate::messages::Message;
use futures_util::StreamExt;
use std::collections::HashMap;

use super::PersistMemory;
use super::history::*;
use super::tools::*;
use super::turn::*;

/// Internal method to stream chat with complete message history
pub(crate) fn stream_chat_with_messages<'a>(
    agent: &'a Agent,
    messages: Vec<Message>,
    tool_cfg: ToolCallingConfig,
    cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    persist_memory: PersistMemory,
) -> std::pin::Pin<
    Box<dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>> + Send + 'a>,
> {
    let ToolCallingConfig {
        tool_schemas: tool_cfg_schemas_for_filter,
        use_native,
        include_tool_descriptions,
        supports_text_based_tools,
    } = tool_cfg;
    let query = last_user_query(&messages);
    let tool_schemas_all = agent.tools.get_tool_schemas();
    let tool_ordering_embedder = agent.tool_ordering_embedding_provider.clone();
    let readonly_filter_hits = agent.readonly_filter_hits_counter();
    let phased_enabled =
        matches!(
            agent.tool_execution_driver,
            ToolExecutionDriver::PlannerFirstPhased
        ) && (use_native || include_tool_descriptions || supports_text_based_tools);

    Box::pin(async_stream::stream! {
        let ordered_all = order_tools(
            &tool_schemas_all,
            &query,
            tool_ordering_embedder.as_ref(),
        )
        .await;

        let tool_schemas_ordered = order_native_stream_schemas(
            use_native,
            &tool_cfg_schemas_for_filter,
            &tool_schemas_all,
            &ordered_all,
        );
        let mut total_usage = TokenUsageAccumulator::default();
        total_usage.set_context_window(agent.cached_context_window);
        let request_start = std::time::Instant::now();
        let mut messages = messages;
        messages.reserve(10);
        let mut conversation_history = std::sync::Arc::new(messages);
        let tool_schema_tokens =
            estimate_stream_schema_tokens(agent, use_native, supports_text_based_tools);

        #[cfg(feature = "compaction")]
        {
            let compact = compact_emit(
                agent,
                conversation_history,
                tool_schema_tokens,
                &total_usage,
            )
            .await;
            conversation_history = compact.history;
            for ev in compact.events {
                yield Ok(ev);
            }
            if let Some(e) = compact.error {
                yield Err(e);
                return;
            }
            if compact.stop {
                persist_stream_memory(&persist_memory, &conversation_history).await;
                return;
            }
        }
        #[cfg(not(feature = "compaction"))]
        {
            conversation_history =
                prune_stream_history_if_needed(agent, conversation_history, tool_schema_tokens);
        }

        yield Ok(crate::messages::StreamEvent::AgentStart);

        let mut ctx = StreamLoopCtx {
            agent,
            persist_memory: &persist_memory,
            cancellation_flag: cancellation_flag.as_ref(),
            conversation_history,
            limiter: IterationLimiter::new(agent.max_iterations),
            iteration_count: 0,
            phase_plan_initialized: !phased_enabled,
            total_usage,
            total_llm_ms: 0,
            total_tool_ms: 0,
            first_llm_ms: None,
            actual_ttfb_ms: None,
            turn_tool_cache: HashMap::new(),
            turn_dup_count: HashMap::new(),
            failure_breaker: std::sync::Arc::new(std::sync::Mutex::new(
                crate::agent::tools::failure_breaker::IdenticalFailureBreaker::new(),
            )),
            use_native,
            supports_text_based_tools,
            phased_enabled,
            tool_schema_tokens,
            query,
            ordered_for_planning: ordered_all,
            ordered_for_filtering: tool_schemas_ordered,
            request_start,
            readonly_filter_hits,
        };

        loop {
            let mut turn = Box::pin(stream_one_turn(&mut ctx));
            let mut end = TurnEndKind::Continue;
            while let Some(item) = turn.next().await {
                match item {
                    TurnItem::Event(ev) => yield Ok(ev),
                    TurnItem::Break => end = TurnEndKind::Break,
                    TurnItem::Return => end = TurnEndKind::Return,
                    TurnItem::Error(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
            match end {
                TurnEndKind::Continue => {}
                TurnEndKind::Break => break,
                TurnEndKind::Return => return,
            }
        }

        persist_stream_memory(&persist_memory, &ctx.conversation_history).await;
        yield Ok(crate::messages::StreamEvent::AgentEnd);

        let (metadata_ev, done_ev) = build_final_metadata_events(
            &ctx.total_usage,
            ctx.request_start,
            ctx.total_llm_ms,
            ctx.total_tool_ms,
            ctx.iteration_count,
            ctx.actual_ttfb_ms.or(ctx.first_llm_ms),
            None,
        );
        yield Ok(metadata_ev);
        yield Ok(done_ev);
    })
}
