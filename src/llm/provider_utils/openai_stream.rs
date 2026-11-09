//! Shared OpenAI-compatible SSE stream parser.
//!
//! Extracted from `openai_compat::provider` so Azure and other OpenAI-shaped
//! providers can reuse the same streaming logic without duplicating tool-call
//! accumulation and finish-reason deferral.

use crate::llm::OpenAICompatConfig;
use crate::llm::openai_compat::{
    OpenAICompatResponse, OpenAICompatStreamChoice, OpenAICompatStreamChunk, OpenAICompatUsage,
};
use crate::llm::provider_utils::{
    SseDataLine, SseLimits, StreamingToolCallAccumulator, sse_data_lines_within,
};
use crate::messages::{StreamEvent, TokenUsage};
use beef::lean::Cow;
use futures_util::StreamExt;
use tokio::sync::mpsc;

/// Convert OpenAI-compatible usage to [`TokenUsage`].
#[must_use]
pub fn openai_usage_to_token_usage(usage: &OpenAICompatUsage) -> TokenUsage {
    let mapped = TokenUsage::from_counts(
        Some(usage.prompt_tokens),
        usage.completion_tokens,
        Some(usage.total_tokens),
    );
    crate::llm::prompt_cache::token_usage_with_openai_cached(
        mapped,
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
    )
}

/// Events queued between the SSE parser and the stream consumer.
///
/// 32 slots cover one socket read that expands into several deltas (text,
/// reasoning, tool-call fragments, usage). When every slot is full,
/// [`mpsc::Sender::send`] waits. The parser stops reading the HTTP body, so
/// the connection applies backpressure. Each chunk is delivered while the
/// consumer is alive. A closed consumer ends the parse.
pub const OPENAI_STREAM_CHANNEL_CAPACITY: usize = 32;

type StreamTx = mpsc::Sender<anyhow::Result<StreamEvent>>;

/// Queue used by the OpenAI-compatible streaming provider.
#[must_use]
pub fn openai_stream_channel() -> (StreamTx, mpsc::Receiver<anyhow::Result<StreamEvent>>) {
    mpsc::channel(OPENAI_STREAM_CHANNEL_CAPACITY)
}

async fn send_stream_event(tx: &StreamTx, event: anyhow::Result<StreamEvent>) -> bool {
    tx.send(event).await.is_ok()
}

async fn send_openai_choice<C: OpenAICompatConfig>(
    choice: OpenAICompatStreamChoice,
    config: &C,
    accumulator: &mut StreamingToolCallAccumulator,
    deferred_finish_reason: &mut Option<String>,
    tx: &StreamTx,
) -> bool {
    if let Some(content) = choice.delta.content
        && !content.is_empty()
    {
        let processed = config.post_process_output(&content);
        if !processed.is_empty()
            && !send_stream_event(
                tx,
                Ok(StreamEvent::TextChunk {
                    content: processed.into_owned(),
                }),
            )
            .await
        {
            return false;
        }
    }
    if let Some(reasoning) = choice.delta.reasoning_content
        && !reasoning.is_empty()
        && !send_stream_event(
            tx,
            Ok(StreamEvent::ThinkingChunk {
                content: reasoning,
                action_id: None,
            }),
        )
        .await
    {
        return false;
    }
    for tool_call in choice.delta.tool_calls.unwrap_or_default() {
        accumulator.accumulate(
            tool_call.index,
            tool_call.id.map(beef::generic::Cow::into_owned),
            tool_call.function.name.map(beef::generic::Cow::into_owned),
            tool_call.function.arguments,
        );
    }
    if let Some(finish_reason) = choice.finish_reason {
        if finish_reason == "tool_calls" || finish_reason == "function_call" {
            *deferred_finish_reason = Some(finish_reason);
        } else {
            return send_stream_event(
                tx,
                Ok(StreamEvent::Metadata {
                    usage: None,
                    finish_reason: Some(Cow::owned(finish_reason)),
                    metrics: None,
                }),
            )
            .await;
        }
    }
    true
}

async fn send_openai_stream_json<C: OpenAICompatConfig>(
    json: &str,
    config: &C,
    provider_name: &'static str,
    cache_key: Option<&str>,
    accumulator: &mut StreamingToolCallAccumulator,
    deferred_finish_reason: &mut Option<String>,
    tx: &StreamTx,
) -> bool {
    match serde_json::from_str::<OpenAICompatStreamChunk>(json) {
        Ok(chunk) => {
            if let Some(usage) = chunk.usage {
                let usage = openai_usage_to_token_usage(&usage);
                crate::llm::provider_utils::log_prompt_cache_usage(
                    Some(&usage),
                    provider_name,
                    cache_key,
                );
                if !send_stream_event(
                    tx,
                    Ok(StreamEvent::Metadata {
                        usage: Some(usage),
                        finish_reason: None,
                        metrics: None,
                    }),
                )
                .await
                {
                    return false;
                }
            }
            for choice in chunk.choices {
                if !send_openai_choice(choice, config, accumulator, deferred_finish_reason, tx)
                    .await
                {
                    return false;
                }
            }
            true
        }
        Err(chunk_error) => {
            if let Ok(final_response) = serde_json::from_str::<OpenAICompatResponse>(json)
                && let Some(usage) = final_response.usage
            {
                let usage = openai_usage_to_token_usage(&usage);
                crate::llm::provider_utils::log_prompt_cache_usage(
                    Some(&usage),
                    provider_name,
                    cache_key,
                );
                return send_stream_event(
                    tx,
                    Ok(StreamEvent::Metadata {
                        usage: Some(usage),
                        finish_reason: final_response
                            .choices
                            .first()
                            .and_then(|choice| choice.finish_reason.clone().map(Cow::owned)),
                        metrics: None,
                    }),
                )
                .await;
            }
            tracing::debug!(
                provider = provider_name,
                error = %chunk_error,
                data_preview = %json.chars().take(300).collect::<String>(),
                "SSE chunk not parseable as stream delta or final response"
            );
            true
        }
    }
}

/// Parses an OpenAI-compatible SSE response and sends events to `tx`.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn parse_sse_stream_to_channel<C: OpenAICompatConfig>(
    response: reqwest::Response,
    config: C,
    provider_name: &'static str,
    cache_key: Option<String>,
    limits: SseLimits,
    tx: StreamTx,
) -> anyhow::Result<()> {
    let mut sse_stream = sse_data_lines_within(response, limits);
    let mut tool_call_accumulator = StreamingToolCallAccumulator::new();
    let mut deferred_tool_finish_reason: Option<String> = None;

    loop {
        // A dropped receiver (the turn was cancelled) ends the read, which
        // closes the connection so a local server stops the work.
        let next = tokio::select! {
            biased;
            () = tx.closed() => return Ok(()),
            next = sse_stream.next() => next,
        };
        let Some(result) = next else {
            break;
        };
        let json_str = match result {
            Ok(SseDataLine::Done) => break,
            Ok(SseDataLine::Data(s)) => s,
            Err(e) => {
                let _ = send_stream_event(&tx, Err(e)).await;
                return Ok(());
            }
        };

        if !send_openai_stream_json(
            &json_str,
            &config,
            provider_name,
            cache_key.as_deref(),
            &mut tool_call_accumulator,
            &mut deferred_tool_finish_reason,
            &tx,
        )
        .await
        {
            return Ok(());
        }
    }

    for evt in tool_call_accumulator.finalize_all(true) {
        if !send_stream_event(&tx, Ok(evt)).await {
            return Ok(());
        }
    }
    for evt in tool_call_accumulator.finalize_all(false) {
        if !send_stream_event(&tx, Ok(evt)).await {
            return Ok(());
        }
    }
    if let Some(finish_reason) = deferred_tool_finish_reason
        && !send_stream_event(
            &tx,
            Ok(StreamEvent::Metadata {
                usage: None,
                finish_reason: Some(Cow::owned(finish_reason)),
                metrics: None,
            }),
        )
        .await
    {
        return Ok(());
    }

    let _ = send_stream_event(&tx, Ok(StreamEvent::Done)).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{OPENAI_STREAM_CHANNEL_CAPACITY, openai_stream_channel, send_stream_event};
    use crate::messages::StreamEvent;
    use std::time::Duration;

    fn chunk(n: usize) -> StreamEvent {
        StreamEvent::TextChunk {
            content: n.to_string(),
        }
    }

    #[tokio::test]
    async fn openai_stream_channel_waits_when_full_and_delivers_every_chunk() {
        let (tx, mut rx) = openai_stream_channel();
        for n in 0..OPENAI_STREAM_CHANNEL_CAPACITY {
            tx.try_send(Ok(chunk(n))).expect("fill");
        }

        let extra = OPENAI_STREAM_CHANNEL_CAPACITY;
        let waiting = tx.clone();
        let mut pending = std::pin::pin!(send_stream_event(&waiting, Ok(chunk(extra))));
        let blocked = tokio::time::timeout(Duration::from_millis(40), &mut pending).await;
        assert!(blocked.is_err(), "send must wait while the channel is full");

        let mut delivered = Vec::new();
        delivered.push(rx.recv().await.expect("first"));
        assert!(pending.await, "the waiting chunk is accepted");
        while let Ok(event) = rx.try_recv() {
            delivered.push(event);
        }

        assert_eq!(delivered.len(), OPENAI_STREAM_CHANNEL_CAPACITY + 1);
        let texts: Vec<String> = delivered
            .into_iter()
            .map(|event| match event.expect("chunk") {
                StreamEvent::TextChunk { content } => content,
                other => panic!("unexpected event {other:?}"),
            })
            .collect();
        let expected: Vec<String> = (0..=OPENAI_STREAM_CHANNEL_CAPACITY)
            .map(|n| n.to_string())
            .collect();
        assert_eq!(texts, expected);
    }
}
