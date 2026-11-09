use crate::handlers;
use crate::middleware::OptionalUserContext;
use crate::rpc::connection::{ConnectionCtx, ResponseSink};
use crate::rpc::dispatch::{MethodKind, MethodRouter};
use crate::rpc::helpers::{
    json_handler_to_value, map_handler_err, parse_params, require_authorized_session,
    require_param_str, to_rpc_value,
};
use crate::rpc::types::JsonRpcError;
use axum::Json;
use axum::extract::{Path, Query, State};
use caliche::harness::{AgentEvent, BroadcastResync, CodingTaskRequest, broadcast_tail};
use caliche_contracts::{rpc::methods as rpc_methods, settings::InteractionModeDto};
use futures_util::StreamExt;
use futures_util::stream::Stream;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

fn rpc_user(ctx: &ConnectionCtx) -> OptionalUserContext {
    OptionalUserContext(ctx.user_context.clone())
}

pub fn register(router: &mut MethodRouter) {
    register_coding(router);
    register_sessions_core(router);
    register_sessions_history(router);
    register_sessions_mode(router);
    register_sessions_context(router);
    register_memory(router);
    register_session_events(router);
}

fn register_coding(router: &mut MethodRouter) {
    router.register(
        rpc_methods::CODING_RUN,
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { coding_run(params, ctx).await },
    );
    router.register(
        "agent.execute",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { agent_execute(params, ctx).await },
    );
    router.register(
        "agent.enhanced",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { agent_enhanced(params, ctx).await },
    );
}

fn register_sessions_core(router: &mut MethodRouter) {
    router.register(
        "sessions.create",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_create(params, ctx).await },
    );
    router.register(
        "sessions.get",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_get(params, ctx).await },
    );
    router.register(
        "sessions.delete",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_delete(params, ctx).await },
    );
    router.register(
        "sessions.executeTurn",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_execute_turn(params, ctx).await },
    );
    router.register(
        "sessions.executeTask",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_execute_task(params, ctx).await },
    );
}

fn register_sessions_history(router: &mut MethodRouter) {
    router.register(
        "sessions.rollback",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_rollback(params, ctx).await },
    );
    router.register(
        "sessions.checkpoint",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_checkpoint(params, ctx).await },
    );
    router.register(
        "sessions.undo",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_undo(params, ctx).await },
    );
    router.register(
        "sessions.diff",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_diff(params, ctx).await },
    );
    router.register(
        "sessions.revert",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_revert(params, ctx).await },
    );
    router.register(
        "sessions.unrevert",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_unrevert(params, ctx).await },
    );
    router.register(
        "sessions.clearHistory",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_clear_history(params, ctx).await },
    );
}

fn register_sessions_mode(router: &mut MethodRouter) {
    router.register(
        "sessions.getMode",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_get_mode(params, ctx).await },
    );
    router.register(
        "sessions.setMode",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_set_mode(params, ctx).await },
    );
    router.register(
        "sessions.getPlan",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_get_plan(params, ctx).await },
    );
    router.register(
        "sessions.clearPlan",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_clear_plan(params, ctx).await },
    );
}

fn register_sessions_context(router: &mut MethodRouter) {
    router.register(
        "sessions.getContext",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_get_context(params, ctx).await },
    );
    router.register(
        "sessions.trackFile",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_track_file(params, ctx).await },
    );
    router.register(
        "sessions.refreshGit",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_refresh_git(params, ctx).await },
    );
    router.register(
        "sessions.detectProject",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { sessions_detect_project(params, ctx).await },
    );
    router.register(
        "sessions.getRelevantContext",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move {
            sessions_get_relevant_context(params, ctx).await
        },
    );
}

fn register_memory(router: &mut MethodRouter) {
    router.register(
        "memory.stats",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { memory_stats(params, ctx).await },
    );
    router.register(
        "memory.inspect",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { memory_inspect(params, ctx).await },
    );
    router.register(
        "memory.search",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { memory_search(params, ctx).await },
    );
    router.register(
        "memory.clear",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { memory_clear(params, ctx).await },
    );
    router.register(
        "memory.export",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { memory_export(params, ctx).await },
    );
    router.register(
        "memory.import",
        MethodKind::RequestResponse,
        |params, _id, ctx: ConnectionCtx| async move { memory_import(params, ctx).await },
    );
}

fn register_session_events(router: &mut MethodRouter) {
    router.register(
        "sessions.subscribe",
        MethodKind::Subscription,
        |params, _id, ctx: ConnectionCtx| async move { sessions_subscribe(params, ctx).await },
    );
    router.register(
        "sessions.unsubscribe",
        MethodKind::Subscription,
        |params, _id, ctx: ConnectionCtx| async move { sessions_unsubscribe(params, ctx).await },
    );
}

async fn coding_run(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = params
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    let request_value = params.get("request").cloned().unwrap_or(params);
    let request = parse_params::<CodingTaskRequest>(request_value)?;
    let response = Box::pin(
        ctx.sessions_state
            .execute_coding_task(session_id.as_deref(), request),
    )
    .await
    .map_err(|e| {
        JsonRpcError::internal_error(Some(json!({
            "error": e.to_string()
        })))
    })?;
    to_rpc_value(response)
}

async fn agent_execute(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::agent_stream::AgentTaskRequest>(params)?;
    let Json(response) = Box::pin(handlers::agent_stream::execute_task_handler(
        State(ctx.agent_state),
        Json(request),
    ))
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn agent_enhanced(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::agent_stream::EnhancedAgentRequest>(params)?;
    let Json(response) =
        handlers::agent_stream::enhanced_agent_handler(State(ctx.agent_state), Json(request))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_create(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let request = parse_params::<handlers::sessions::CreateSessionRequest>(params)?;
    let user = rpc_user(&ctx);
    let Json(response) = Box::pin(handlers::sessions::create_session_handler(
        State(ctx.sessions_state),
        user,
        Json(request),
    ))
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_get(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    json_handler_to_value(
        handlers::sessions::get_session_handler(State(ctx.sessions_state), Path(session_id)).await,
    )
}

async fn sessions_delete(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let status =
        handlers::sessions::delete_session_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(s, m)| map_handler_err(s, m))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn sessions_execute_turn(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let input = require_param_str(&params, "input")?;
    let provider = optional_param_string(&params, "provider");
    let model = optional_param_string(&params, "model");
    json_handler_to_value(
        handlers::sessions::execute_turn_handler(
            State(ctx.sessions_state),
            Path(session_id),
            Json(handlers::sessions::ExecuteTurnRequest {
                input,
                provider,
                model,
                context_window: params
                    .get("context_window")
                    .and_then(Value::as_u64)
                    .and_then(|window| usize::try_from(window).ok()),
            }),
        )
        .await,
    )
}

fn optional_param_string(params: &Value, key: &str) -> Option<String> {
    params
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

async fn sessions_execute_task(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let task = require_param_str(&params, "task")?;
    json_handler_to_value(
        handlers::sessions::execute_task_handler(
            State(ctx.sessions_state),
            Path(session_id),
            Json(handlers::sessions::ExecuteTaskRequest { task }),
        )
        .await,
    )
}

async fn sessions_rollback(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let checkpoint_id = require_param_str(&params, "checkpoint_id")?;
    let Json(response) = handlers::sessions::rollback_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Json(handlers::sessions::RollbackRequest { checkpoint_id }),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_checkpoint(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let label = require_param_str(&params, "label")?;
    let Json(response) = handlers::sessions::checkpoint_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Json(handlers::sessions::CheckpointRequest { label }),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_undo(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::sessions::undo_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_diff(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let query = handlers::sessions::DiffQuery {
        message_id: params
            .get("message_id")
            .and_then(Value::as_str)
            .and_then(|s| uuid::Uuid::parse_str(s).ok()),
    };
    let Json(response) =
        handlers::sessions::diff_handler(State(ctx.sessions_state), Path(session_id), Query(query))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_revert(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let request = handlers::sessions::RevertRequest {
        message_id: params
            .get("message_id")
            .and_then(Value::as_str)
            .and_then(|s| uuid::Uuid::parse_str(s).ok()),
    };
    let Json(response) = handlers::sessions::revert_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Json(request),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn sessions_unrevert(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::sessions::unrevert_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_clear_history(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let status =
        handlers::sessions::clear_history_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn sessions_get_mode(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::sessions::get_session_mode_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_set_mode(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let mode_value = params
        .get("mode")
        .cloned()
        .ok_or_else(|| JsonRpcError::invalid_params(Some(json!({"missing":"mode"}))))?;
    let mode: InteractionModeDto = serde_json::from_value(mode_value)
        .map_err(|e| JsonRpcError::invalid_params(Some(json!({"error": e.to_string()}))))?;
    let Json(response) = handlers::sessions::set_session_mode_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Json(handlers::sessions::SetModeRequest { mode }),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_get_plan(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::sessions::get_session_plan_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_clear_plan(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let status =
        handlers::sessions::clear_session_plan_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn sessions_get_context(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::context::get_context_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_track_file(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let path = require_param_str(&params, "path")?;
    let Json(response) = handlers::context::track_file_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Json(handlers::context::TrackFileRequest { path }),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_refresh_git(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::context::refresh_git_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn sessions_detect_project(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::context::detect_project_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(response)
}

async fn sessions_get_relevant_context(
    params: Value,
    ctx: ConnectionCtx,
) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let query_str = require_param_str(&params, "query")?;
    let query = handlers::context::RelevantContextQuery { query: query_str };
    let Json(response) = handlers::context::get_relevant_context_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Query(query),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn memory_stats(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::memory::memory_stats_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn memory_inspect(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::memory::memory_inspect_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn memory_search(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let query = handlers::memory::SearchQuery {
        q: params
            .get("q")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    };
    let Json(response) = handlers::memory::memory_search_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Query(query),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn memory_clear(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let status =
        handlers::memory::memory_clear_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn memory_export(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let Json(response) =
        handlers::memory::memory_export_handler(State(ctx.sessions_state), Path(session_id))
            .await
            .map_err(|(status, message)| map_handler_err(status, message))?;
    to_rpc_value(response)
}

async fn memory_import(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    let session_id = require_authorized_session(&ctx, &params).await?;
    let snapshot: caliche::agent::memory::MemorySnapshot = parse_params(
        params
            .get("snapshot")
            .cloned()
            .ok_or_else(|| JsonRpcError::invalid_params(Some(json!({"missing":"snapshot"}))))?,
    )?;
    let status = handlers::memory::memory_import_handler(
        State(ctx.sessions_state),
        Path(session_id),
        Json(snapshot),
    )
    .await
    .map_err(|(status, message)| map_handler_err(status, message))?;
    Ok(json!({"status": status.as_u16()}))
}

async fn sessions_subscribe(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    ctx.ensure_ws_for_subscriptions()
        .map_err(|e| JsonRpcError::server_error(-32006, e.to_string(), None))?;
    let session_id = require_authorized_session(&ctx, &params).await?;
    let stream = ctx
        .sessions_state
        .subscribe_events(&session_id)
        .await
        .map_err(|e| JsonRpcError::internal_error(Some(json!({"error": e.to_string()}))))?;

    let key = format!("session:{session_id}");
    {
        let mut subs = ctx.subscriptions.write().await;
        subs.insert(key);
    }

    spawn_session_event_forwarder(&ctx, &session_id, stream);
    Ok(json!({"subscribed": true, "session_id": session_id}))
}

fn event_notification(session_id: &str, event: &AgentEvent) -> Option<Value> {
    let payload = serde_json::to_value(event).ok()?;
    Some(json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "method": rpc_methods::SESSIONS_EVENT,
        "params": {
            "session_id": session_id,
            "event": payload
        }
    }))
}

fn resync_notification(session_id: &str, resync: BroadcastResync) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "method": BroadcastResync::RPC_METHOD,
        "params": {
            "session_id": session_id,
            "skipped": resync.skipped
        }
    })
}

fn session_notifications(
    rx: tokio::sync::broadcast::Receiver<AgentEvent>,
    session_id: String,
) -> impl Stream<Item = Value> + Send {
    broadcast_tail(rx).filter_map(move |item| {
        let note = match item {
            Ok(event) => event_notification(&session_id, &event),
            Err(resync) => Some(resync_notification(&session_id, resync)),
        };
        futures_util::future::ready(note)
    })
}

fn spawn_session_event_forwarder(
    ctx: &ConnectionCtx,
    session_id: &str,
    rx: tokio::sync::broadcast::Receiver<AgentEvent>,
) {
    let sink = ctx.sink.clone();
    let subscriptions = ctx.subscriptions.clone();
    let session_id_for_task = session_id.to_string();
    tokio::spawn(run_session_forwarder(
        sink,
        subscriptions,
        session_id_for_task,
        rx,
    ));
}

async fn run_session_forwarder(
    sink: ResponseSink,
    subscriptions: Arc<RwLock<HashSet<String>>>,
    session_id: String,
    rx: broadcast::Receiver<AgentEvent>,
) {
    let mut notes = session_notifications(rx, session_id.clone());
    let session_key = format!("session:{session_id}");
    while let Some(notification) = notes.next().await {
        let keep = subscriptions.read().await.contains(&session_key);
        if !keep {
            break;
        }
        if subscription_send(&sink, &notification).await == SubscriptionSend::Stop {
            break;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubscriptionSend {
    Sent,
    Stop,
}

async fn subscription_send(sink: &ResponseSink, notification: &Value) -> SubscriptionSend {
    match sink.send_value(notification).await {
        Ok(()) => SubscriptionSend::Sent,
        Err(err) => {
            tracing::warn!(
                target: "caliche_server::rpc::session",
                error = %err,
                "session subscription send failed; stopping so later events are not delivered without that frame"
            );
            SubscriptionSend::Stop
        }
    }
}

async fn sessions_unsubscribe(params: Value, ctx: ConnectionCtx) -> Result<Value, JsonRpcError> {
    ctx.ensure_ws_for_subscriptions()
        .map_err(|e| JsonRpcError::server_error(-32006, e.to_string(), None))?;
    let session_id = require_authorized_session(&ctx, &params).await?;
    let key = format!("session:{session_id}");
    let mut subs = ctx.subscriptions.write().await;
    let removed = subs.remove(&key);
    Ok(json!({"unsubscribed": removed, "session_id": session_id}))
}

#[cfg(test)]
mod tests {
    use super::{
        SubscriptionSend, run_session_forwarder, session_notifications, subscription_send,
    };
    use crate::rpc::connection::ResponseSink;
    use caliche::harness::{AgentEvent, BroadcastResync};
    use caliche_contracts::rpc::methods as rpc_methods;
    use futures_util::StreamExt;
    use serde_json::json;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::{RwLock, broadcast};

    fn info(name: &str) -> AgentEvent {
        AgentEvent::Info {
            message: name.to_owned(),
        }
    }

    #[tokio::test]
    async fn session_subscription_emits_resync_with_the_skipped_count() {
        let (tx, rx) = broadcast::channel(1);
        tx.send(info("m0")).expect("send");
        tx.send(info("m1")).expect("send");
        tx.send(info("m2")).expect("send");

        let mut notes = session_notifications(rx, "sess-1".to_owned());
        let resync = notes.next().await.expect("resync");
        assert_eq!(resync["method"], BroadcastResync::RPC_METHOD);
        assert_eq!(resync["params"]["session_id"], "sess-1");
        assert_eq!(resync["params"]["skipped"], 2);

        let kept = notes.next().await.expect("retained event");
        assert_eq!(kept["method"], rpc_methods::SESSIONS_EVENT);
        assert_eq!(kept["params"]["event"]["message"], "m2");
    }

    /// Session subscriptions run on the WebSocket sink. Both [`ResponseSink`]
    /// arms return an error once the peer is gone, and the forwarder must end
    /// on that error. A closed SSE sink takes the same stop path.
    #[tokio::test]
    async fn forwarder_stops_when_the_sink_is_closed() {
        let (tx, rx) = broadcast::channel(4);
        let (sink_tx, sink_rx) = tokio::sync::mpsc::channel(1);
        drop(sink_rx);
        let sink = ResponseSink::Sse(sink_tx);
        let subs = Arc::new(RwLock::new(HashSet::from(["session:sess-1".to_owned()])));
        tx.send(info("m0")).expect("send");
        let task = tokio::spawn(run_session_forwarder(sink, subs, "sess-1".to_owned(), rx));
        let finished = tokio::time::timeout(Duration::from_secs(10), task).await;
        assert!(finished.is_ok(), "forwarder must end after a closed sink");
        let _keep_sender = tx;
    }

    #[tokio::test]
    async fn subscription_send_stops_when_the_sink_is_closed() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let sink = ResponseSink::Sse(tx);
        let step = subscription_send(
            &sink,
            &json!({
                "method": rpc_methods::SESSIONS_RESYNC,
                "params": {"skipped": 2}
            }),
        )
        .await;
        assert_eq!(step, SubscriptionSend::Stop);
    }
}
