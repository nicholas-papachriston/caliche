use crate::handlers;
use crate::middleware::OptionalUserContext;
use crate::models::ChatEstimateRequest;
use crate::rpc::connection::ConnectionCtx;
use crate::rpc::dispatch::{MethodKind, MethodRouter};
use crate::rpc::helpers::{into_value_response, map_handler_err, parse_params, to_rpc_value};
use crate::rpc::streaming::StreamNotifier;
use crate::rpc::types::JsonRpcError;
use axum::Json;
use axum::extract::{Query, State};
use caliche_contracts::{
    chat::ChatRequestDto, rpc::methods as rpc_methods, settings::AgentSettingsDto,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;

pub fn register(router: &mut MethodRouter) {
    register_chat(router);
    register_settings(router);
    register_metrics_files(router);
}

fn register_chat(router: &mut MethodRouter) {
    router.register(
        rpc_methods::CHAT_SEND,
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { Box::pin(chat_send(params, ctx)).await },
    );

    router.register(
        rpc_methods::CHAT_STREAM,
        MethodKind::Streaming,
        |params, id, ctx: ConnectionCtx| async move { chat_stream(params, id, ctx).await },
    );

    router.register(
        "chat.estimate",
        MethodKind::RequestResponse,
        |params, _id, _ctx: ConnectionCtx| async move { chat_estimate(params).await },
    );

    router.register(
        "chat.cancel",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { chat_cancel(ctx) },
    );
}

fn register_settings(router: &mut MethodRouter) {
    router.register(
        rpc_methods::SETTINGS_GET,
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { settings_get(ctx).await },
    );

    router.register(
        rpc_methods::SETTINGS_UPDATE,
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { settings_update(params, ctx).await },
    );

    router.register(
        rpc_methods::TOOLS_LIST,
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { tools_list(ctx).await },
    );

    router.register(
        rpc_methods::MODELS_LIST,
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { models_list(ctx).await },
    );

    router.register(
        rpc_methods::MODELS_LIST_BY_PROVIDER,
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { models_list_by_provider(params, ctx).await },
    );

    router.register(
        "storage.options",
        MethodKind::RequestResponse,
        |_params, _id, _ctx: ConnectionCtx| async move { storage_options().await },
    );
}

fn register_metrics_files(router: &mut MethodRouter) {
    router.register(
        "metrics.listTools",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { metrics_list_tools(params, ctx).await },
    );

    router.register(
        "metrics.getTool",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { metrics_get_tool(params, ctx).await },
    );

    router.register(
        "metrics.getCache",
        MethodKind::RequestResponse,
        |_params, _id, ctx: ConnectionCtx| async move { metrics_get_cache(ctx).await },
    );

    router.register(
        "files.enabled",
        MethodKind::RequestResponse,
        |_params, _id, _ctx: ConnectionCtx| async move { files_enabled().await },
    );

    router.register(
        "files.list",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { files_list(params, ctx).await },
    );

    router.register(
        "files.read",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { files_read(params, ctx).await },
    );
}

async fn chat_send(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<ChatRequestDto>(params)?;
    let Json(response) = handlers::chat::chat_handler(
        State(ctx.app_state),
        OptionalUserContext(ctx.user_context),
        Json(request),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn chat_stream(
    params: Value,
    id: Option<Value>,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    // Reuse the streaming helper that backs `POST /api/chat/stream` so
    // RPC subscribers receive real token-level events instead of
    // synthetic chunks after a completed non-streaming call.
    let request = parse_params::<ChatRequestDto>(params)?;
    let mut event_stream = handlers::chat_stream::chat_stream_events(
        ctx.app_state.clone(),
        ctx.user_context.clone(),
        request,
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;

    let request_id = id.unwrap_or_else(|| json!(null));
    let notifier =
        StreamNotifier::new(ctx.sink.clone(), request_id, rpc_methods::CHAT_STREAM_EVENT);
    let mut emitted_events: u32 = 0;
    let mut final_usage: Option<caliche::messages::TokenUsage> = None;
    let cancel = ctx.cancel_token.clone();

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                tracing::info!(target: "caliche_server::rpc::chat_stream", "chat.stream cancelled");
                break;
            }
            maybe_event = event_stream.next() => {
                let Some(event_result) = maybe_event else { break; };
                match event_result {
                    Ok(event) => {
                        apply_stream_usage(&event, &mut final_usage);
                        if !notify_stream_event(&notifier, event).await {
                            break;
                        }
                        emitted_events = emitted_events.saturating_add(1);
                    }
                    Err(e) => {
                        notify_stream_error(&notifier, e).await?;
                        break;
                    }
                }
            }
        }
    }

    let done_event = crate::transport::sse::envelope_from_serializable(&json!({"type": "done"}))
        .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;
    notifier
        .notify(done_event)
        .await
        .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;

    Ok(json!({
        "emitted_events": emitted_events,
        "usage": final_usage,
    }))
}

fn apply_stream_usage(
    event: &caliche::messages::StreamEvent,
    final_usage: &mut Option<caliche::messages::TokenUsage>,
) {
    if let caliche::messages::StreamEvent::Metadata { usage: Some(u), .. } = event {
        *final_usage = Some(u.clone());
    }
    if let caliche::messages::StreamEvent::LimitReached { usage, .. } = event {
        *final_usage = Some(usage.clone());
    }
}

async fn notify_stream_event(
    notifier: &StreamNotifier,
    event: caliche::messages::StreamEvent,
) -> bool {
    let payload = match crate::transport::sse::envelope_from_stream_event(event, None) {
        Ok(v) => v,
        Err(e) => json!({
            "type": "error",
            "error": e.to_string(),
        }),
    };
    if let Err(e) = notifier.notify(payload).await {
        tracing::warn!(target: "caliche_server::rpc::chat_stream", error = %e, "failed to deliver chat.stream event; client likely gone");
        return false;
    }
    true
}

async fn notify_stream_error(
    notifier: &StreamNotifier,
    error: impl std::fmt::Display,
) -> Result<(), JsonRpcError> {
    let error_event = crate::transport::sse::envelope_from_serializable(&json!({
        "type": "error",
        "error": error.to_string(),
    }))
    .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;
    notifier
        .notify(error_event)
        .await
        .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;
    Ok(())
}

async fn chat_estimate(params: Value) -> Result<Value, JsonRpcError> {
    let request = parse_params::<ChatEstimateRequest>(params)?;
    let Json(response) = handlers::estimate::estimate_chat_handler(Json(request))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

fn chat_cancel(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    // Trip the connection cancellation token so any in-flight
    // streaming method bails out at the next `tokio::select!` poll.
    ctx.cancel_token.cancel();
    Ok(json!({"cancelled": true}))
}

async fn settings_get(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::settings::get_settings_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn settings_update(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let settings = parse_params::<AgentSettingsDto>(params)?;
    let Json(response) =
        handlers::settings::put_settings_handler(State(ctx.app_state), Json(settings))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn tools_list(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::tools::tools_handler(State(ctx.app_state))
        .await
        .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn models_list(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = Box::pin(handlers::models::all_models_handler(State(ctx.app_state))).await;
    to_rpc_value(response)
}

async fn models_list_by_provider(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let mut query = HashMap::new();
    if let Some(provider) = params.get("provider").and_then(Value::as_str) {
        query.insert("provider".to_string(), provider.to_string());
    }
    let Json(response) = Box::pin(handlers::models::models_handler(
        State(ctx.app_state),
        Query(query),
    ))
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn storage_options() -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::storage::storage_options_handler().await;
    to_rpc_value(response)
}

async fn metrics_list_tools(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let query = handlers::metrics::MetricsQuery {
        tool: params
            .get("tool")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    };
    let Json(response) =
        handlers::metrics::get_tools_metrics_handler(State(ctx.app_state), Query(query))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn metrics_get_tool(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return Err(JsonRpcError::invalid_params(Some(
            json!({"missing":"name"}),
        )));
    };
    let Json(response) = handlers::metrics::get_tool_metrics_handler(
        State(ctx.app_state),
        axum::extract::Path(name.to_string()),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn metrics_get_cache(ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let Json(response) = handlers::metrics::get_cache_metrics_handler(State(ctx.app_state)).await;
    to_rpc_value(response)
}

async fn files_enabled() -> Result<Value, JsonRpcError> {
    let response = handlers::files::files_enabled_handler().await;
    into_value_response(response).await
}

async fn files_list(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let query = handlers::files::ListDirQuery {
        path: params
            .get("path")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    };
    let response = handlers::files::list_files_handler(State(ctx.app_state), Query(query)).await;
    into_value_response(response).await
}

async fn files_read(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let query = parse_params::<handlers::files::ReadFileQuery>(params)?;
    let response = handlers::files::read_file_handler(State(ctx.app_state), Query(query)).await;
    into_value_response(response).await
}
