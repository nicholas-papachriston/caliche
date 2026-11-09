use crate::agent::core::chat_helpers::TokenUsageAccumulator;
use futures_util::StreamExt;

use super::history::elapsed_millis_u64;

pub(crate) struct LlmStreamAcc {
    pub(crate) accumulated_text: String,
    pub(crate) accumulated_thinking: String,
    pub(crate) accumulated_tool_calls: Vec<crate::messages::ToolCallInfo>,
    pub(crate) final_usage: Option<crate::messages::TokenUsage>,
    pub(crate) finish_reason: Option<beef::lean::Cow<'static, str>>,
    pub(crate) actual_ttfb_ms: Option<u64>,
    pub(crate) llm_duration_ms: Option<u64>,
    pub(crate) thinking_action_id: Option<String>,
}

pub(crate) enum ProviderEventAction {
    Forward(crate::messages::StreamEvent),
    Done,
    Ignore,
}

pub(crate) fn apply_llm_stream_event(
    event: crate::messages::StreamEvent,
    acc: &mut LlmStreamAcc,
    llm_start: std::time::Instant,
    usage_policy: crate::llm::UsageMergePolicy,
) -> ProviderEventAction {
    match event {
        crate::messages::StreamEvent::TextChunk { content } => {
            note_ttfb(acc, llm_start);
            end_thinking_segment(acc);
            acc.accumulated_text.push_str(&content);
            ProviderEventAction::Forward(crate::messages::StreamEvent::TextChunk { content })
        }
        crate::messages::StreamEvent::ThinkingChunk { content, action_id } => {
            note_ttfb(acc, llm_start);
            acc.accumulated_thinking.push_str(&content);
            let action_id = thinking_action_id(acc, action_id);
            ProviderEventAction::Forward(crate::messages::StreamEvent::ThinkingChunk {
                content,
                action_id: Some(action_id),
            })
        }
        crate::messages::StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            thought_signature,
            action_id,
        } => {
            end_thinking_segment(acc);
            let action_id = action_id.unwrap_or_else(new_action_id);
            acc.accumulated_tool_calls
                .push(crate::messages::ToolCallInfo {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    thought_signature: thought_signature.clone(),
                    action_id: Some(action_id.clone()),
                    arguments_json: None,
                });
            ProviderEventAction::Forward(crate::messages::StreamEvent::ToolCallStart {
                id,
                name,
                arguments,
                thought_signature,
                action_id: Some(action_id),
            })
        }
        crate::messages::StreamEvent::Metadata {
            usage,
            finish_reason: fr,
            metrics: _,
        } => {
            merge_stream_usage(&mut acc.final_usage, usage, usage_policy);
            acc.finish_reason = fr.or_else(|| acc.finish_reason.take());
            ProviderEventAction::Ignore
        }
        crate::messages::StreamEvent::Done => {
            end_thinking_segment(acc);
            acc.llm_duration_ms = Some(elapsed_millis_u64(llm_start));
            ProviderEventAction::Done
        }
        status @ crate::messages::StreamEvent::Status { .. } => {
            ProviderEventAction::Forward(status)
        }
        _ => ProviderEventAction::Ignore,
    }
}

fn new_action_id() -> String {
    crate::utils::time_id::generate_time_id().to_string()
}

fn end_thinking_segment(acc: &mut LlmStreamAcc) {
    acc.thinking_action_id = None;
}

fn thinking_action_id(acc: &mut LlmStreamAcc, stamped: Option<String>) -> String {
    if let Some(id) = stamped.filter(|id| !id.is_empty()) {
        acc.thinking_action_id = Some(id.clone());
        return id;
    }
    if let Some(id) = acc.thinking_action_id.clone() {
        return id;
    }
    let id = new_action_id();
    acc.thinking_action_id = Some(id.clone());
    id
}

pub(crate) fn note_ttfb(acc: &mut LlmStreamAcc, llm_start: std::time::Instant) {
    if acc.actual_ttfb_ms.is_none() {
        acc.actual_ttfb_ms = Some(elapsed_millis_u64(llm_start));
    }
}

pub(crate) enum LlmTurnItem {
    Chunk(crate::messages::StreamEvent),
    Finished(LlmStreamAcc),
    Cancelled,
    Error(anyhow::Error),
}

/// Resolves once `flag` is set, checking it every [`CANCEL_POLL`]; never
/// without a flag. Lets Ctrl+C end a long wait for the first token.
async fn cancel_requested(flag: Option<&std::sync::Arc<std::sync::atomic::AtomicBool>>) {
    let Some(flag) = flag else {
        return std::future::pending().await;
    };
    while !flag.load(std::sync::atomic::Ordering::Relaxed) {
        tokio::time::sleep(CANCEL_POLL).await;
    }
}

const CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(100);

pub(crate) fn consume_llm_turn<'a>(
    mut stream: std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + 'a,
        >,
    >,
    cancellation_flag: Option<&'a std::sync::Arc<std::sync::atomic::AtomicBool>>,
    usage_policy: crate::llm::UsageMergePolicy,
    llm_start: std::time::Instant,
) -> impl futures_util::Stream<Item = LlmTurnItem> + 'a {
    async_stream::stream! {
        let mut acc = LlmStreamAcc {
            accumulated_text: String::with_capacity(2048),
            accumulated_thinking: String::new(),
            accumulated_tool_calls: Vec::new(),
            final_usage: None,
            finish_reason: None,
            actual_ttfb_ms: None,
            llm_duration_ms: None,
            thinking_action_id: None,
        };
        loop {
            if let Some(cancel_flag) = cancellation_flag
                && cancel_flag.load(std::sync::atomic::Ordering::Relaxed)
            {
                yield LlmTurnItem::Cancelled;
                return;
            }
            let next = tokio::select! {
                biased;
                () = cancel_requested(cancellation_flag) => {
                    yield LlmTurnItem::Cancelled;
                    return;
                }
                next = stream.next() => next,
            };
            let Some(event_result) = next else {
                yield LlmTurnItem::Finished(acc);
                return;
            };
            match event_result {
                Ok(event) => match apply_llm_stream_event(event, &mut acc, llm_start, usage_policy) {
                    ProviderEventAction::Forward(ev) => yield LlmTurnItem::Chunk(ev),
                    ProviderEventAction::Done => {
                        yield LlmTurnItem::Finished(acc);
                        return;
                    }
                    ProviderEventAction::Ignore => {}
                },
                Err(e) => {
                    yield LlmTurnItem::Error(anyhow::anyhow!("Stream error: {e:#}"));
                    return;
                }
            }
        }
    }
}

pub(crate) fn merge_stream_usage(
    final_usage: &mut Option<crate::messages::TokenUsage>,
    usage: Option<crate::messages::TokenUsage>,
    usage_policy: crate::llm::UsageMergePolicy,
) {
    let Some(new_usage) = usage else {
        return;
    };
    *final_usage = Some(match (final_usage.take(), usage_policy) {
        (None, _) => new_usage,
        (Some(prev), crate::llm::UsageMergePolicy::Delta) => prev + new_usage,
        (Some(_), crate::llm::UsageMergePolicy::Cumulative) => new_usage,
    });
}

pub(crate) fn add_acc_llm_duration(
    total_llm_ms: &mut u64,
    first_llm_ms: &mut Option<u64>,
    duration_ms: Option<u64>,
) {
    let Some(ms) = duration_ms else {
        return;
    };
    *total_llm_ms = total_llm_ms.saturating_add(ms);
    if first_llm_ms.is_none() {
        *first_llm_ms = Some(ms);
    }
}

pub(crate) fn add_stream_usage(
    total_usage: &mut TokenUsageAccumulator,
    final_usage: Option<&crate::messages::TokenUsage>,
) {
    if let Some(usage) = final_usage {
        total_usage.add_usage(usage);
        tracing::info!(
            "Stream iteration token usage - prompt: {}, completion: {}, embedding: {}, total: {}",
            usage.prompt_tokens.unwrap_or(0),
            usage.completion_tokens.unwrap_or(0),
            usage.embedding_tokens.unwrap_or(0),
            usage.total_tokens.unwrap_or(0)
        );
    } else {
        tracing::warn!("No token usage found in stream response");
    }
}

#[cfg(test)]
mod tests {
    use super::{LlmStreamAcc, ProviderEventAction, apply_llm_stream_event, new_action_id};
    use crate::llm::UsageMergePolicy;
    use crate::messages::StreamEvent;
    use std::time::Instant;
    use uuid::Version;

    fn acc() -> LlmStreamAcc {
        LlmStreamAcc {
            accumulated_text: String::new(),
            accumulated_thinking: String::new(),
            accumulated_tool_calls: Vec::new(),
            final_usage: None,
            finish_reason: None,
            actual_ttfb_ms: None,
            llm_duration_ms: None,
            thinking_action_id: None,
        }
    }

    fn forward(event: StreamEvent, acc: &mut LlmStreamAcc) -> StreamEvent {
        match apply_llm_stream_event(event, acc, Instant::now(), UsageMergePolicy::Cumulative) {
            ProviderEventAction::Forward(event) => event,
            ProviderEventAction::Done | ProviderEventAction::Ignore => {
                panic!("expected forward")
            }
        }
    }

    #[test]
    fn thinking_burst_and_tool_call_share_time_ordered_ids() {
        let mut acc = acc();
        let first = forward(
            StreamEvent::ThinkingChunk {
                content: "a".into(),
                action_id: None,
            },
            &mut acc,
        );
        let second = forward(
            StreamEvent::ThinkingChunk {
                content: "b".into(),
                action_id: None,
            },
            &mut acc,
        );
        let StreamEvent::ThinkingChunk {
            action_id: Some(first_id),
            ..
        } = first
        else {
            panic!("missing thinking id");
        };
        let StreamEvent::ThinkingChunk {
            action_id: Some(second_id),
            ..
        } = second
        else {
            panic!("missing thinking id");
        };
        assert_eq!(first_id, second_id);
        assert_eq!(
            uuid::Uuid::parse_str(&first_id)
                .expect("uuid")
                .get_version(),
            Some(Version::SortRand)
        );

        let start = forward(
            StreamEvent::ToolCallStart {
                id: beef::lean::Cow::borrowed("call_1"),
                name: beef::lean::Cow::borrowed("bash"),
                arguments: serde_json::json!({}),
                thought_signature: None,
                action_id: None,
            },
            &mut acc,
        );
        let StreamEvent::ToolCallStart {
            action_id: Some(tool_id),
            ..
        } = start
        else {
            panic!("missing tool action id");
        };
        assert_ne!(tool_id, first_id);
        assert_eq!(
            acc.accumulated_tool_calls[0].action_id.as_deref(),
            Some(tool_id.as_str())
        );
        let again = new_action_id();
        assert_ne!(again, tool_id);
    }
}
