//! Chat streaming (SSE). When the HTTP client disconnects, the response body stream is dropped and
//! this module's `async_stream` stops polling the provider stream, which stops further work on that
//! connection. In-flight provider-side work may still complete depending on the LLM client.

use crate::AppState;
use crate::handlers::chat_contracts::{dto_history_to_core, dto_mode_to_core};
use crate::handlers::chat_history::trim_trailing_non_conversational_turns;
use crate::handlers::chat_service::resolve_request_service;
use crate::handlers::multipart_chat::parse_multipart_chat;
use crate::middleware::OptionalUserContext;
use crate::transport::sse::{encode_sse_error, encode_sse_event, envelope_from_stream_event};
use async_stream::stream;
use axum::{
    Json,
    body::Body,
    extract::{Multipart, State},
    http::{StatusCode, header},
    response::Response,
};
use caliche::messages::Message;
use caliche::messages::public_sse_event_name;
use caliche_contracts::chat::ChatRequestDto;
use futures_util::StreamExt;
use std::sync::Arc;
use tracing::Instrument;
use uuid::Uuid;

/// POST /api/chat/stream/multipart - Stream chat responses with file attachments.
pub async fn chat_stream_multipart_handler(
    State(state): State<AppState>,
    OptionalUserContext(user_context): OptionalUserContext,
    multipart: Multipart,
) -> Result<Response<Body>, (StatusCode, String)> {
    let parsed = parse_multipart_chat(multipart).await?;
    chat_stream_handler_with_history(
        State(state),
        OptionalUserContext(user_context),
        Json(parsed.request),
        parsed.history_override,
    )
    .await
}

/// Produce a boxed stream of [`caliche::messages::StreamEvent`] results for
/// the given request. Shared between [`chat_stream_handler`] (HTTP SSE) and
/// the JSON-RPC `chat.stream` method so both paths deliver real token-level
/// streaming instead of synthetic chunks after a full response.
pub async fn chat_stream_events(
    state: AppState,
    user_context: Option<crate::middleware::UserContext>,
    request: ChatRequestDto,
) -> Result<
    futures_util::stream::BoxStream<'static, anyhow::Result<caliche::messages::StreamEvent>>,
    (StatusCode, String),
> {
    chat_stream_events_with_history(state, user_context, request, None).await
}

async fn chat_stream_events_with_history(
    state: AppState,
    user_context: Option<crate::middleware::UserContext>,
    request: ChatRequestDto,
    history_override: Option<Vec<Message>>,
) -> Result<
    futures_util::stream::BoxStream<'static, anyhow::Result<caliche::messages::StreamEvent>>,
    (StatusCode, String),
> {
    let request_id = Uuid::now_v7().to_string();
    tracing::info!(request_id = %request_id, "Building chat stream");

    let context_window_override = request.context_window.or(state.context_window);

    let (mut service, provider_type, model) =
        resolve_request_service(&state, user_context.as_ref(), &request).await?;

    if let Some(context_window) = context_window_override {
        service
            .set_context_window(context_window)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    }

    service.set_metrics(state.tool_metrics.clone());

    if let Some(ref ctx) = user_context {
        let agent_ctx = caliche::UserContext {
            user_id: ctx.user_id.clone(),
            tenant_id: ctx.tenant_id.clone(),
            email: ctx.email.clone(),
            roles: ctx.roles.clone(),
            custom_claims: ctx.custom_claims.clone(),
        };
        service.set_user_context(Some(agent_ctx));
    }

    let effective_mode = dto_mode_to_core(request.mode);
    if effective_mode.is_readonly() {
        use caliche::agent::modes::{readonly_blocked_tools, readonly_mode_prompt};
        let readonly_tools: Vec<String> = service
            .list_tools()
            .iter()
            .map(|(name, _, _)| name.to_string())
            .filter(|name| !readonly_blocked_tools().contains(&name.as_str()))
            .collect();
        service.agent_mut().set_allowed_tools(Some(readonly_tools));
        if let Some(mode_prompt) = readonly_mode_prompt(effective_mode) {
            let original = service.agent().system_prompt().to_string();
            let new_prompt = format!("{mode_prompt}\n\n{original}");
            service.agent_mut().set_system_prompt(new_prompt);
        }
    }

    tracing::info!(
        request_id = %request_id,
        provider = ?provider_type,
        model = %model,
        tools = ?request.tools,
        mode = %effective_mode,
        "Chat stream context prepared"
    );

    let message = request.message;
    let history = history_override
        .or_else(|| {
            let history = dto_history_to_core(request.messages);
            (!history.is_empty()).then_some(history)
        })
        .map(trim_trailing_non_conversational_turns);

    #[cfg(feature = "router")]
    let spend_tracker = state.spend_tracker.clone();
    #[cfg(feature = "router")]
    let tenant_id = user_context.as_ref().and_then(|ctx| ctx.tenant_id.clone());

    let service_arc = Arc::new(service);
    let event_stream = stream! {
        let _service_arc = service_arc;
        let agent_ref = _service_arc.agent();
        let mut provider_stream = if let Some(hist) = history {
            agent_ref.stream_chat_with_history(&message, hist)
        } else {
            agent_ref.stream_chat_with_tools(&message)
        };

        let mut final_usage: Option<caliche::messages::TokenUsage> = None;
        use futures_util::StreamExt;
        while let Some(item) = provider_stream.next().await {
            if let Ok(ref event) = item {
                match event {
                    caliche::messages::StreamEvent::Metadata { usage: Some(usage), .. } => {
                        final_usage = Some(usage.clone());
                    }
                    caliche::messages::StreamEvent::LimitReached { usage, .. } => {
                        final_usage = Some(usage.clone());
                    }
                    _ => {}
                }
            }
            yield item;
        }

        #[cfg(feature = "router")]
        if let (Some(tracker), Some(tid), Some(usage)) = (spend_tracker, tenant_id, final_usage)
        {
            let tokens = usage.total_tokens.unwrap_or(0) as u64;
            if tokens > 0 {
                tracker.record_tokens(&tid, tokens).await;
            }
        }
    };

    Ok(Box::pin(event_stream))
}

/// POST /api/chat/stream - Stream chat responses as Server-Sent Events
pub async fn chat_stream_handler(
    State(state): State<AppState>,
    OptionalUserContext(user_context): OptionalUserContext,
    Json(request): Json<ChatRequestDto>,
) -> Result<Response<Body>, (StatusCode, String)> {
    chat_stream_handler_with_history(
        State(state),
        OptionalUserContext(user_context),
        Json(request),
        None,
    )
    .await
}

async fn chat_stream_handler_with_history(
    State(state): State<AppState>,
    OptionalUserContext(user_context): OptionalUserContext,
    Json(request): Json<ChatRequestDto>,
    history_override: Option<Vec<Message>>,
) -> Result<Response<Body>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "api_chat_stream", request_id = %request_id);

    async move {
        tracing::info!("Received streaming chat request");

        let stream = chat_stream_events_with_history(state, user_context, request, history_override).await?;

        let sse_stream = stream.map(move |event_result| {
            // Pre-allocate buffer for SSE formatting (typical event is ~200-500 bytes)
            let mut sse_data = String::with_capacity(256);
            match event_result {
                Ok(event) => {
                    let event_name = public_sse_event_name(&event);
                    match envelope_from_stream_event(event, None)
                        .and_then(|v| encode_sse_event(event_name, &v))
                    {
                        Ok(framed) => {
                            sse_data.push_str(&framed);
                        }
                        Err(e) => {
                            tracing::error!("Failed to serialize stream event: {}", e);
                            match encode_sse_error(&e.to_string()) {
                                Ok(error_json) => sse_data.push_str(&error_json),
                                Err(_) => sse_data.push_str(
                                    "data: {\"version\":\"1\",\"type\":\"error\",\"payload\":{\"type\":\"error\",\"error\":\"stream serialization failed\"}}\n\n",
                                ),
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    match encode_sse_error(&e.to_string()) {
                        Ok(error_json) => sse_data.push_str(&error_json),
                        Err(_) => sse_data.push_str(
                            "data: {\"version\":\"1\",\"type\":\"error\",\"payload\":{\"type\":\"error\",\"error\":\"stream failed\"}}\n\n",
                        ),
                    }
                }
            };
            Ok::<_, std::io::Error>(axum::body::Bytes::from(sse_data))
        });

        // Create response with SSE headers
        let response = Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(Body::from_stream(sse_stream))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to create response: {}", e),
                )
            })?;

        Ok(response)
    }
    .instrument(span)
    .await
}
