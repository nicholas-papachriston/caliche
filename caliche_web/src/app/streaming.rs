//! Chat transport helpers for native SSE and WASM WebSocket streaming.
//!
//! [`crate::state::streaming`] documents the epoch invalidation pattern; the UI
//! captures a [`StreamTicket`] after each bump and passes it into
//! [`consume_sse_stream`] and the WASM dispatch path.

#[cfg(any(test, target_arch = "wasm32"))]
use std::cell::RefCell;
#[cfg(any(test, target_arch = "wasm32"))]
use std::rc::Rc;

use dioxus::prelude::*;

use crate::state::chat::GENERATION_STOPPED_NOTICE;
use crate::state::streaming::StreamTicket;
use crate::state::{ChatState, ConfigState, ToolCall};

#[cfg(not(target_arch = "wasm32"))]
use futures_util::StreamExt;

#[cfg(not(target_arch = "wasm32"))]
use crate::api::{ApiClient, ChatRequest, StreamingEventType};

#[cfg(target_arch = "wasm32")]
use crate::api::{WsClient, WsClientMessage, WsServerMessage};

#[cfg(any(test, target_arch = "wasm32"))]
struct TurnTarget {
    ticket: StreamTicket,
    message_id: String,
}

/// Turn the long-lived websocket callback should paint.
///
/// The connect callback closes over one gate. A later send must
/// [`TurnGate::retarget`] that same gate. Allocating a replacement leaves the
/// callback on the previous message id, and the epoch bump for the new send
/// makes that old ticket look superseded, so the new turn never paints.
#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct TurnGate {
    inner: Rc<RefCell<TurnTarget>>,
}

#[cfg(any(test, target_arch = "wasm32"))]
impl TurnGate {
    fn new(ticket: StreamTicket, message_id: String) -> Self {
        Self {
            inner: Rc::new(RefCell::new(TurnTarget { ticket, message_id })),
        }
    }

    fn retarget(&self, ticket: StreamTicket, message_id: String) {
        *self.inner.borrow_mut() = TurnTarget { ticket, message_id };
    }

    pub(crate) fn message_id(&self) -> String {
        self.inner.borrow().message_id.clone()
    }

    pub(crate) fn is_superseded(&self) -> bool {
        self.inner.borrow().ticket.is_superseded()
    }

    #[cfg(test)]
    fn same_allocation(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Point the socket callback at this turn.
///
/// The first call stores the gate the callback will hold. Later calls update
/// that gate instead of replacing it.
#[cfg(any(test, target_arch = "wasm32"))]
pub(crate) fn bind_turn(
    slot: &mut Option<TurnGate>,
    ticket: StreamTicket,
    message_id: String,
) -> TurnGate {
    if let Some(gate) = slot.as_ref() {
        gate.retarget(ticket, message_id);
        return gate.clone();
    }
    let gate = TurnGate::new(ticket, message_id);
    *slot = Some(gate.clone());
    gate
}

/// Whether an inbound stream event should update the assistant row.
#[cfg(any(test, target_arch = "wasm32"))]
#[must_use]
pub(crate) fn accepts_stream_event(
    gate_message_id: &str,
    gate_superseded: bool,
    active_streaming_message_id: Option<&str>,
    session_lifecycle: bool,
) -> bool {
    if gate_superseded {
        return false;
    }
    if session_lifecycle {
        return true;
    }
    active_streaming_message_id == Some(gate_message_id)
}

/// Build the RPC chat-stream request for the desktop SSE path.
#[cfg(not(target_arch = "wasm32"))]
pub fn build_chat_stream_request(
    chat: &ChatState,
    config: &ConfigState,
    full_message: String,
) -> ChatRequest {
    let history = chat.get_history();
    let history_len = history.len();
    let history_for_request: Vec<_> = history
        .into_iter()
        .take(history_len.saturating_sub(1))
        .collect();

    ChatRequest {
        message: full_message,
        provider: config.provider.clone(),
        model: config.model.clone(),
        context_window: None,
        tools: if config.selected_tools.is_empty() {
            None
        } else {
            Some(config.selected_tools.clone())
        },
        messages: if history_for_request.is_empty() {
            None
        } else {
            Some(history_for_request)
        },
        mode: Some(chat.mode),
    }
}

/// Build WebSocket chat message params mirroring the native SSE request shape.
#[cfg(target_arch = "wasm32")]
fn build_ws_chat_message(
    chat: &ChatState,
    config: &ConfigState,
    full_message: String,
) -> WsClientMessage {
    let history = chat.get_history();
    let history_len = history.len();
    let history_for_request: Vec<_> = history
        .into_iter()
        .take(history_len.saturating_sub(1))
        .collect();

    WsClientMessage::Message {
        content: full_message,
        provider: config.provider.clone(),
        model: config.model.clone(),
        tools: if config.selected_tools.is_empty() {
            None
        } else {
            Some(config.selected_tools.clone())
        },
        messages: if history_for_request.is_empty() {
            None
        } else {
            Some(history_for_request)
        },
        mode: Some(chat.mode),
    }
}

/// Consume an SSE chat stream until completion, cancel, or epoch invalidation.
#[cfg(not(target_arch = "wasm32"))]
pub async fn consume_sse_stream(
    client: &ApiClient,
    request: ChatRequest,
    mut chat: Signal<ChatState>,
    cancel_signal: Signal<bool>,
    stream_ticket: StreamTicket,
    message_id: String,
) {
    match client.chat_stream(&request).await {
        Ok(stream) => {
            let mut stream = std::pin::pin!(stream);

            while let Some(event_result) = stream.next().await {
                if stream_ticket.is_superseded() {
                    tracing::debug!(
                        target: "caliche_web::stream",
                        message_id = %message_id,
                        captured = stream_ticket.captured(),
                        "stream epoch superseded; exiting reader"
                    );
                    break;
                }
                if *cancel_signal.read() {
                    chat.write()
                        .update_streaming_content(&message_id, GENERATION_STOPPED_NOTICE);
                    break;
                }

                match event_result {
                    Ok(event) => match event.event_type {
                        StreamingEventType::TextChunk
                        | StreamingEventType::ThinkingChunk
                        | StreamingEventType::Content => {
                            if let Some(ref content) = event.content {
                                chat.write().update_streaming_content(&message_id, content);
                            }
                        }
                        StreamingEventType::ToolCallStart => {
                            let name = event.name.unwrap_or_else(|| "unknown".to_string());
                            let tool_id = event
                                .action_id
                                .clone()
                                .or(event.id)
                                .unwrap_or_else(|| name.clone());
                            chat.write()
                                .add_tool_call(&message_id, ToolCall::new(tool_id, name));
                        }
                        StreamingEventType::ToolCallResult => {
                            if let Some(name) = event.name {
                                let result = event.result.unwrap_or_default();
                                let success = !result.starts_with("Error:");
                                let tool_key = event.action_id.clone().or(event.id).unwrap_or(name);
                                chat.write().complete_tool_call(
                                    &message_id,
                                    &tool_key,
                                    result,
                                    success,
                                );
                            }
                        }
                        StreamingEventType::Usage
                        | StreamingEventType::Metadata
                        | StreamingEventType::LimitReached => {
                            if let Some(usage) = event.usage {
                                chat.write().finalize_message(&message_id, Some(usage));
                            }
                        }
                        StreamingEventType::Error => {
                            let error_msg =
                                event.error.unwrap_or_else(|| "Unknown error".to_string());
                            tracing::error!("Stream error: {}", error_msg);
                            chat.write().mark_error(&message_id, &error_msg);
                        }
                        StreamingEventType::Done => {
                            chat.write().finalize_message(&message_id, None);
                        }
                        StreamingEventType::AudioChunk
                        | StreamingEventType::ImageGenerated
                        | StreamingEventType::AttachmentProcessed
                        | StreamingEventType::Compaction
                        | StreamingEventType::PhaseTransition
                        | StreamingEventType::Status
                        | StreamingEventType::AgentStart
                        | StreamingEventType::AgentEnd
                        | StreamingEventType::TurnStart
                        | StreamingEventType::TurnEnd => {}
                    },
                    Err(e) => {
                        tracing::error!("Stream event error: {}", e);
                        chat.write().mark_error(&message_id, &e.to_string());
                        break;
                    }
                }
            }

            if chat.read().is_streaming {
                chat.write().finalize_message(&message_id, None);
            }
        }
        Err(e) => {
            tracing::error!("Failed to create stream: {}", e);
            chat.write().mark_error(&message_id, &e.to_string());
        }
    }
}

#[cfg(target_arch = "wasm32")]
struct WsDispatchContext {
    cancel_signal: Signal<bool>,
    stream_gate: TurnGate,
}

#[cfg(target_arch = "wasm32")]
impl WsDispatchContext {
    fn is_stale(&self) -> bool {
        self.stream_gate.is_superseded()
    }

    fn message_id(&self) -> String {
        self.stream_gate.message_id()
    }
}

#[cfg(target_arch = "wasm32")]
fn dispatch_ws_server_message(
    msg: WsServerMessage,
    chat: &mut Signal<ChatState>,
    config: &mut Signal<ConfigState>,
    ctx: &WsDispatchContext,
) {
    if ctx.is_stale() {
        tracing::debug!(
            target: "caliche_web::stream",
            message_id = %ctx.message_id(),
            "ignoring stale WebSocket event after epoch bump"
        );
        return;
    }
    if *ctx.cancel_signal.read() {
        return;
    }

    let is_session_lifecycle = matches!(
        msg,
        WsServerMessage::SessionCreated { .. }
            | WsServerMessage::CurrentSettings { .. }
            | WsServerMessage::SettingsChanged { .. }
    );
    let active_id = chat.read().streaming_message_id.clone();
    if !accepts_stream_event(
        &ctx.message_id(),
        ctx.is_stale(),
        active_id.as_deref(),
        is_session_lifecycle,
    ) {
        return;
    }

    match msg {
        WsServerMessage::SessionCreated { session_id } => {
            chat.write().set_session(session_id);
        }
        WsServerMessage::CurrentSettings { settings } => {
            config.write().apply_from_settings(&settings);
        }
        WsServerMessage::SettingsChanged { settings } => {
            config.write().apply_from_settings(&settings);
        }
        WsServerMessage::TextChunk { content } => {
            let id = chat.read().streaming_message_id.clone();
            if let Some(ref id) = id.as_ref() {
                chat.write().update_streaming_content(id, &content);
            }
        }
        WsServerMessage::ToolQueued { tool_id, tool_name } => {
            let id = chat.read().streaming_message_id.clone();
            if let Some(ref id) = id.as_ref() {
                chat.write()
                    .add_tool_call(id, ToolCall::new(tool_id, tool_name));
            }
        }
        WsServerMessage::ToolComplete {
            tool_id,
            tool_name,
            result_preview,
            success,
            ..
        } => {
            let id = chat.read().streaming_message_id.clone();
            if let Some(ref id) = id.as_ref() {
                let result = result_preview.unwrap_or_default();
                let tool_key = if tool_id.is_empty() {
                    tool_name
                } else {
                    tool_id
                };
                chat.write()
                    .complete_tool_call(id, &tool_key, result, success);
            }
        }
        WsServerMessage::Done { usage, .. } => {
            let id = chat.read().streaming_message_id.clone();
            if let Some(ref id) = id.as_ref() {
                let usage_opt =
                    usage.and_then(|v| serde_json::from_value::<crate::api::TokenUsage>(v).ok());
                chat.write().finalize_message(id, usage_opt);
            }
        }
        WsServerMessage::Error { error, .. } => {
            let id = chat.read().streaming_message_id.clone();
            if let Some(ref id) = id.as_ref() {
                chat.write().mark_error(id, &error);
            } else {
                chat.write()
                    .add_system_message(format!("**Error:** {}", error));
            }
        }
        WsServerMessage::Cancelled => {
            let id = chat.read().streaming_message_id.clone();
            if let Some(ref id) = id.as_ref() {
                chat.write()
                    .update_streaming_content(id, GENERATION_STOPPED_NOTICE);
                chat.write().finalize_message(id, None);
            }
        }
        _ => {}
    }
}

#[cfg(target_arch = "wasm32")]
fn send_ws_chat_message(
    ws_client: &WsClient,
    chat: &mut Signal<ChatState>,
    config: &ConfigState,
    full_message: String,
    message_id: &str,
) -> bool {
    let chat_session_id = chat.read().session_id().map(str::to_string);
    match crate::api::ws_client::session_bind(chat_session_id.as_deref(), ws_client.has_session()) {
        crate::api::ws_client::SessionBind::Ready => {}
        crate::api::ws_client::SessionBind::Adopt(session_id) => {
            ws_client.adopt_session(&session_id);
        }
        crate::api::ws_client::SessionBind::CreateOnSocket => {
            let create_msg = WsClientMessage::CreateSession {
                workspace: None,
                max_turns: Some(20),
                system_prompt: None,
            };
            if let Err(e) = ws_client.send(&create_msg) {
                tracing::error!("WebSocket CreateSession: {}", e);
                chat.write().mark_error(message_id, &e.to_string());
                return false;
            }
        }
    }

    let chat_read = chat.read();
    let msg = build_ws_chat_message(&chat_read, config, full_message);
    drop(chat_read);

    if let Err(e) = ws_client.send(&msg) {
        tracing::error!("WebSocket send: {}", e);
        chat.write().mark_error(message_id, &e.to_string());
        return false;
    }
    true
}

/// Ensure a WebSocket client exists, then send the chat message (WASM only).
#[cfg(target_arch = "wasm32")]
pub fn send_via_websocket(
    mut chat: Signal<ChatState>,
    config: Signal<ConfigState>,
    chat_state: Signal<ChatState>,
    config_state: Signal<ConfigState>,
    mut ws_client_ref: Signal<Option<WsClient>>,
    base_url: String,
    full_message: String,
    message_id: String,
    cancel_signal: Signal<bool>,
    stream_ticket: StreamTicket,
    mut stream_gate_slot: Signal<Option<TurnGate>>,
) {
    let ws_client_ref_clone = ws_client_ref.clone();
    let mut opt_client = ws_client_ref.write();
    let mut ws_just_created = false;
    let stream_gate = {
        let mut slot = stream_gate_slot.write();
        bind_turn(&mut *slot, stream_ticket, message_id.clone())
    };

    if opt_client.is_none() {
        let ws_client = WsClient::new(&base_url);
        let mut chat_for_cb = chat_state.clone();
        let mut config_for_cb = config_state.clone();
        let ctx_for_cb = WsDispatchContext {
            cancel_signal,
            stream_gate: stream_gate.clone(),
        };
        match ws_client.connect_with_callback(move |result| match result {
            Ok(msg) => {
                dispatch_ws_server_message(msg, &mut chat_for_cb, &mut config_for_cb, &ctx_for_cb)
            }
            Err(e) => {
                if ctx_for_cb.is_stale() {
                    return;
                }
                let id = chat_for_cb.read().streaming_message_id.clone();
                if let Some(ref id) = id.as_ref() {
                    chat_for_cb.write().mark_error(id, &e.to_string());
                }
            }
        }) {
            Ok(open_rx) => {
                ws_just_created = true;
                let mut chat_for_open = chat_state.clone();
                let config_snap = config.read().clone();
                let full_message = full_message.clone();
                let message_id = message_id.clone();
                spawn(async move {
                    if open_rx.await.is_err() {
                        return;
                    }
                    if let Some(ref client) = ws_client_ref_clone.read().as_ref() {
                        let _ = send_ws_chat_message(
                            client,
                            &mut chat_for_open,
                            &config_snap,
                            full_message,
                            &message_id,
                        );
                    }
                });
                *opt_client = Some(ws_client);
            }
            Err(e) => {
                tracing::error!("WebSocket connect: {}", e);
                chat.write().mark_error(&message_id, &e.to_string());
            }
        }
    }

    if !ws_just_created {
        let config_read = config.read();
        if let Some(ref ws_client) = *opt_client {
            let _ = send_ws_chat_message(
                ws_client,
                &mut chat,
                &config_read,
                full_message,
                &message_id,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{accepts_stream_event, bind_turn};
    use crate::state::streaming::EpochCounter;

    #[test]
    fn second_send_retargets_the_socket_callback() {
        let epoch = EpochCounter::default();
        let mut slot = None;
        let first = bind_turn(&mut slot, epoch.capture(), "assistant-1".to_string());
        let callback = first.clone();

        epoch.bump();
        let second = bind_turn(&mut slot, epoch.capture(), "assistant-2".to_string());

        assert!(callback.same_allocation(&second));
        assert_eq!(callback.message_id(), "assistant-2");
        assert!(!callback.is_superseded());
        assert!(accepts_stream_event(
            &callback.message_id(),
            callback.is_superseded(),
            Some("assistant-2"),
            false,
        ));
    }

    #[test]
    fn a_replacement_gate_drops_the_second_turn() {
        let epoch = EpochCounter::default();
        let mut slot = None;
        let callback = bind_turn(&mut slot, epoch.capture(), "assistant-1".to_string());

        epoch.bump();
        let mut other_slot = None;
        let _replacement = bind_turn(&mut other_slot, epoch.capture(), "assistant-2".to_string());

        assert!(callback.is_superseded());
        assert_eq!(callback.message_id(), "assistant-1");
        assert!(!accepts_stream_event(
            &callback.message_id(),
            callback.is_superseded(),
            Some("assistant-2"),
            false,
        ));
    }
}
