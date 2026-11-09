use super::RpcState;
use crate::middleware::OptionalUserContext;
use crate::rpc::connection::{ConnectionCtx, ResponseSink, SSE_CHANNEL_CAPACITY};
use crate::rpc::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use async_stream::stream;
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Json, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{error, warn};

pub async fn ws_rpc_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<RpcState>,
    OptionalUserContext(user_context): OptionalUserContext,
) -> Response {
    ws.on_upgrade(move |socket| handle_ws(socket, state, user_context))
}

async fn handle_ws(
    socket: WebSocket,
    state: RpcState,
    user_context: Option<crate::middleware::UserContext>,
) {
    let (sender, mut receiver) = socket.split();
    let sink = ResponseSink::WebSocket(Arc::new(Mutex::new(sender)));
    let subscriptions = Arc::new(RwLock::new(HashSet::new()));
    let cancel_token = CancellationToken::new();
    let ctx = ConnectionCtx {
        sink,
        app_state: state.app_state.clone(),
        agent_state: state.agent_state.clone(),
        sessions_state: state.sessions_state.clone(),
        subscriptions,
        user_context,
        cancel_token: cancel_token.clone(),
    };

    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let parsed: Result<Value, _> = serde_json::from_str(&text);
                let value = match parsed {
                    Ok(v) => v,
                    Err(e) => {
                        let err = JsonRpcResponse::error(
                            None,
                            JsonRpcError::parse_error(Some(json!({"error": e.to_string()}))),
                        );
                        let _ = ctx.sink.send_response(&err).await;
                        continue;
                    }
                };

                if let Some(arr) = value.as_array() {
                    let mut requests = Vec::new();
                    for item in arr {
                        match serde_json::from_value::<JsonRpcRequest>(item.clone()) {
                            Ok(req) => requests.push(req),
                            Err(e) => {
                                let err = JsonRpcResponse::error(
                                    None,
                                    JsonRpcError::invalid_request(Some(
                                        json!({"error": e.to_string()}),
                                    )),
                                );
                                let _ = ctx.sink.send_response(&err).await;
                            }
                        }
                    }
                    let responses = state
                        .method_router
                        .dispatch_batch(requests, ctx.clone())
                        .await;
                    if !responses.is_empty() {
                        let payload = json!(responses);
                        let _ = ctx.sink.send_value(&payload).await;
                    }
                } else {
                    match serde_json::from_value::<JsonRpcRequest>(value) {
                        Ok(req) => {
                            if let Some(resp) = state.method_router.dispatch(req, ctx.clone()).await
                            {
                                let _ = ctx.sink.send_response(&resp).await;
                            }
                        }
                        Err(e) => {
                            let err = JsonRpcResponse::error(
                                None,
                                JsonRpcError::invalid_request(Some(
                                    json!({"error": e.to_string()}),
                                )),
                            );
                            let _ = ctx.sink.send_response(&err).await;
                        }
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                if let ResponseSink::WebSocket(sender) = &ctx.sink {
                    let mut guard = sender.lock().await;
                    if guard.send(Message::Pong(data)).await.is_err() {
                        break;
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(_) => {}
            Err(e) => {
                warn!("rpc websocket error: {}", e);
                break;
            }
        }
    }
    // Notify any in-flight streaming method to stop producing events.
    cancel_token.cancel();
    state.subscription_manager.clear().await;
}

pub async fn http_rpc_handler(
    State(state): State<RpcState>,
    OptionalUserContext(user_context): OptionalUserContext,
    Json(payload): Json<Value>,
) -> Response {
    let subscriptions = Arc::new(RwLock::new(HashSet::new()));

    if let Some(batch) = payload.as_array() {
        let mut requests = Vec::new();
        for item in batch {
            match serde_json::from_value::<JsonRpcRequest>(item.clone()) {
                Ok(req) => requests.push(req),
                Err(e) => {
                    let response = JsonRpcResponse::error(
                        None,
                        JsonRpcError::invalid_request(Some(json!({"error": e.to_string()}))),
                    );
                    return (
                        StatusCode::BAD_REQUEST,
                        [(header::CONTENT_TYPE, "application/json")],
                        serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
                    )
                        .into_response();
                }
            }
        }

        let (tx, _rx) = mpsc::channel::<Value>(SSE_CHANNEL_CAPACITY);
        let cancel_token = CancellationToken::new();
        let ctx = ConnectionCtx {
            sink: ResponseSink::Sse(tx),
            app_state: state.app_state.clone(),
            agent_state: state.agent_state.clone(),
            sessions_state: state.sessions_state.clone(),
            subscriptions,
            user_context,
            cancel_token,
        };
        let responses = state.method_router.dispatch_batch(requests, ctx).await;
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&responses).unwrap_or_else(|_| "[]".to_string()),
        )
            .into_response();
    }

    let req = match serde_json::from_value::<JsonRpcRequest>(payload) {
        Ok(req) => req,
        Err(e) => {
            let response = JsonRpcResponse::error(
                None,
                JsonRpcError::invalid_request(Some(json!({"error": e.to_string()}))),
            );
            return (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
            )
                .into_response();
        }
    };

    let (tx, mut rx) = mpsc::channel::<Value>(SSE_CHANNEL_CAPACITY);
    let cancel_token = CancellationToken::new();
    let ctx = ConnectionCtx {
        sink: ResponseSink::Sse(tx),
        app_state: state.app_state.clone(),
        agent_state: state.agent_state.clone(),
        sessions_state: state.sessions_state.clone(),
        subscriptions,
        user_context,
        cancel_token: cancel_token.clone(),
    };

    let response = state.method_router.dispatch(req.clone(), ctx.clone()).await;

    let is_stream = matches!(
        state.method_router.kind_of(req.method.as_ref()),
        Some(super::dispatch::MethodKind::Streaming)
    );
    if is_stream {
        if let Some(resp) = response {
            if let Ok(v) = serde_json::to_value(resp) {
                let _ = ctx.sink.send_value(&v).await;
            }
        }
        let event_stream = stream! {
            loop {
                tokio::select! {
                    biased;
                    _ = cancel_token.cancelled() => {
                        break;
                    }
                    maybe = rx.recv() => {
                        match maybe {
                            Some(value) => {
                                let data = format!("data: {value}\n\n");
                                yield Ok::<_, std::io::Error>(axum::body::Bytes::from(data));
                            }
                            None => break,
                        }
                    }
                }
            }
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(Body::from_stream(event_stream))
            .unwrap_or_else(|e| {
                error!("failed to build SSE response: {}", e);
                Response::new(Body::from("internal error"))
            });
    }

    match response {
        Some(resp) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&resp).unwrap_or_else(|_| "{}".to_string()),
        )
            .into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}
