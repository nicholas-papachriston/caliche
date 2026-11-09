//! WebSocket client for real-time JSON-RPC communication.
//!
//! Connects to /ws for JSON-RPC 2.0 transport.

#[cfg(target_arch = "wasm32")]
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use crate::api::types::AgentSettings;
#[cfg(target_arch = "wasm32")]
use caliche_contracts::stream::ACTION_ID_FIELD;

/// Client -> Server message types (must match server ClientMessage)
#[allow(
    dead_code,
    reason = "the client messages mirror the server protocol; some variants are not sent yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
)]
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    CreateSession {
        workspace: Option<String>,
        max_turns: Option<usize>,
        system_prompt: Option<String>,
    },
    Message {
        content: String,
        provider: Option<String>,
        model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tools: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        messages: Option<Vec<crate::api::types::ChatMessage>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mode: Option<crate::api::types::InteractionMode>,
    },
    UpdateSettings {
        settings: AgentSettings,
    },
    Command {
        command: String,
    },
    Cancel,
    Ping,
}

/// Server -> Client message types (must match server ServerMessage)
#[allow(
    dead_code,
    reason = "the server messages mirror the server protocol; some fields are not read yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
)]
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    SessionCreated {
        session_id: String,
    },
    TextChunk {
        content: String,
    },
    ToolQueued {
        tool_id: String,
        tool_name: String,
    },
    ToolExecuting {
        tool_id: String,
        tool_name: String,
    },
    ToolComplete {
        tool_id: String,
        tool_name: String,
        result_preview: Option<String>,
        success: bool,
        duration_ms: u64,
    },
    ToolError {
        tool_id: String,
        tool_name: String,
        error: String,
    },
    Thinking {
        content: String,
    },
    FileWrite {
        path: String,
        is_new: bool,
    },
    FileRead {
        path: String,
    },
    Done {
        usage: Option<serde_json::Value>,
        duration_ms: u64,
    },
    Error {
        error: String,
        recoverable: bool,
    },
    CommandResult {
        success: bool,
        data: serde_json::Value,
    },
    Cancelled,
    Pong,
    CurrentSettings {
        settings: AgentSettings,
    },
    SettingsChanged {
        settings: AgentSettings,
    },
}

/// Connection state for the WebSocket
#[allow(
    dead_code,
    reason = "only the wasm client reports connection state; lib.rs and main.rs both compile this module, so expect cannot hold in both"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsConnectionState {
    Disconnected,
    Connecting,
    Connected,
}

/// Whether a frame can go on the socket or must wait for `onopen`.
#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FramePlan {
    /// `readyState` is OPEN, or there is no socket yet and the caller should see the error.
    SendNow,
    /// The socket is still `CONNECTING`. `send` throws until `onopen`.
    QueueUntilOpen,
}

/// Browser `WebSocket.send` throws while `readyState` is `CONNECTING`.
///
/// Only `Connecting` queues. `Disconnected` still attempts a write so a send
/// with no socket in progress fails instead of sitting in a queue nobody flushes.
#[cfg(any(test, target_arch = "wasm32"))]
#[must_use]
pub fn frame_plan(state: WsConnectionState) -> FramePlan {
    match state {
        WsConnectionState::Connecting => FramePlan::QueueUntilOpen,
        WsConnectionState::Connected | WsConnectionState::Disconnected => FramePlan::SendNow,
    }
}

/// Drain frames queued before the socket opened, preserving order.
#[cfg(any(test, target_arch = "wasm32"))]
pub fn take_queued_frames(queue: &mut Vec<String>) -> Vec<String> {
    queue.drain(..).collect()
}

/// How the browser socket learns which session `chat.stream` belongs to.
///
/// `chat.stream` is sent only after [`WsClient`] has a session id. An HTTP
/// `sessions.create` that updates chat state but not the socket used to skip
/// the socket `sessions.create` and leave the chat frame queued forever.
#[cfg(any(test, target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SessionBind {
    /// The socket already has a session id.
    Ready,
    /// Chat state already has this session. Store it on the socket so the
    /// chat frame is not queued behind a `sessions.create` that will not run.
    Adopt(String),
    /// Neither side has a session. Open one on the socket; chat stays queued
    /// until that `sessions.create` returns.
    CreateOnSocket,
}

/// Decide how to bind the socket before sending `chat.stream`.
#[cfg(any(test, target_arch = "wasm32"))]
#[must_use]
pub(crate) fn session_bind(chat_session_id: Option<&str>, ws_has_session: bool) -> SessionBind {
    if ws_has_session {
        return SessionBind::Ready;
    }
    match chat_session_id {
        Some(id) if !id.is_empty() => SessionBind::Adopt(id.to_string()),
        _ => SessionBind::CreateOnSocket,
    }
}

/// Session id the socket will use after applying [`session_bind`].
///
/// `None` means `chat.stream` must wait for `sessions.create`.
#[cfg(test)]
#[must_use]
pub(crate) fn session_id_after_bind(
    ws_session_id: Option<&str>,
    chat_session_id: Option<&str>,
) -> Option<String> {
    match session_bind(chat_session_id, ws_session_id.is_some()) {
        SessionBind::Ready => ws_session_id.map(str::to_string),
        SessionBind::Adopt(id) => Some(id),
        SessionBind::CreateOnSocket => None,
    }
}

/// `chat.stream` is queued while the socket has no session id.
#[cfg(test)]
#[must_use]
pub(crate) fn chat_stream_waits_for_session(session_id: Option<&str>) -> bool {
    session_id.is_none()
}

/// Base reconnect delay (ms); doubles on each attempt up to [`WS_RECONNECT_MAX_MS`].
#[cfg(target_arch = "wasm32")]
pub const WS_RECONNECT_BASE_MS: u32 = 1_000;
#[cfg(target_arch = "wasm32")]
pub const WS_RECONNECT_MAX_MS: u32 = 30_000;

/// Build WebSocket URL from HTTP base URL (e.g. http://localhost:3000 -> ws://localhost:3000/ws)
#[cfg(target_arch = "wasm32")]
pub fn ws_url_from_base(base_url: &str) -> String {
    let normalized = base_url.trim().trim_end_matches('/');

    if normalized.starts_with("ws://") || normalized.starts_with("wss://") {
        return format!("{normalized}/ws");
    }

    if normalized.starts_with("http://") || normalized.starts_with("https://") {
        let ws_base = normalized
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        return format!("{ws_base}/ws");
    }

    if let Some(window) = web_sys::window() {
        let location = window.location();
        let protocol = location.protocol().ok().unwrap_or_default();
        let host = location
            .host()
            .ok()
            .filter(|h| !h.is_empty())
            .unwrap_or_else(|| "localhost:3000".to_string());
        let ws_scheme = if protocol == "https:" { "wss" } else { "ws" };
        let rel_base = if normalized.is_empty() {
            String::new()
        } else if normalized.starts_with('/') {
            normalized.to_string()
        } else {
            format!("/{normalized}")
        };
        return format!("{ws_scheme}://{host}{rel_base}/ws");
    }

    if normalized.is_empty() {
        "ws://localhost:3000/ws".to_string()
    } else if normalized.starts_with('/') {
        format!("ws://localhost:3000{normalized}/ws")
    } else {
        format!("ws://{normalized}/ws")
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_impl {
    use super::*;
    use futures_channel::oneshot;
    use std::cell::Cell;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use web_sys::WebSocket;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PendingMethod {
        CreateSession,
        Message,
        UpdateSettings,
        GetSettings,
        Command,
    }

    struct WsHandlerClosures {
        _onopen: Closure<dyn FnMut(web_sys::Event)>,
        _onmessage: Closure<dyn FnMut(web_sys::MessageEvent)>,
        _onclose: Closure<dyn FnMut(web_sys::CloseEvent)>,
        _onerror: Closure<dyn FnMut(web_sys::Event)>,
    }

    struct WsClientInner {
        base_url: String,
        ws: RefCell<Option<WebSocket>>,
        next_id: Cell<u64>,
        pending: RefCell<HashMap<u64, PendingMethod>>,
        session_id: RefCell<Option<String>>,
        handlers: RefCell<Option<WsHandlerClosures>>,
        pending_outbound: RefCell<Vec<ClientMessage>>,
        pre_open_frames: RefCell<Vec<String>>,
        stream_done_seen: Cell<bool>,
        reconnect_attempt: Cell<u32>,
        auto_reconnect: Cell<bool>,
        intentional_close: Cell<bool>,
        state: Cell<WsConnectionState>,
        message_handler: RefCell<Option<Box<dyn FnMut(Result<ServerMessage>)>>>,
    }

    /// WebSocket client for WASM (browser). Connects to JSON-RPC /ws.
    #[derive(Clone)]
    pub struct WsClient {
        inner: Rc<WsClientInner>,
    }

    impl WsClient {
        pub fn new(base_url: impl Into<String>) -> Self {
            Self {
                inner: Rc::new(WsClientInner {
                    base_url: base_url.into(),
                    ws: RefCell::new(None),
                    next_id: Cell::new(1),
                    pending: RefCell::new(HashMap::new()),
                    session_id: RefCell::new(None),
                    handlers: RefCell::new(None),
                    pending_outbound: RefCell::new(Vec::new()),
                    pre_open_frames: RefCell::new(Vec::new()),
                    stream_done_seen: Cell::new(false),
                    reconnect_attempt: Cell::new(0),
                    auto_reconnect: Cell::new(true),
                    intentional_close: Cell::new(false),
                    state: Cell::new(WsConnectionState::Disconnected),
                    message_handler: RefCell::new(None),
                }),
            }
        }

        fn allocate_id(&self, method: PendingMethod) -> u64 {
            let id = self.inner.next_id.get();
            self.inner.next_id.set(id.saturating_add(1));
            self.inner.pending.borrow_mut().insert(id, method);
            id
        }

        fn clear_pending_rpc(&self) {
            self.inner.pending.borrow_mut().clear();
        }

        fn write_frame(&self, json: &str) -> Result<()> {
            let guard = self
                .inner
                .ws
                .try_borrow()
                .map_err(|_| anyhow::anyhow!("WebSocket busy (reentrant call)"))?;
            let ws = guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("WebSocket not connected"))?;
            ws.send_with_str(json)
                .map_err(|e| anyhow::anyhow!("WebSocket::send: {:?}", e))?;
            Ok(())
        }

        fn send_rpc_value(&self, value: serde_json::Value) -> Result<()> {
            let json =
                serde_json::to_string(&value).map_err(|e| anyhow::anyhow!("serialize: {}", e))?;
            match frame_plan(self.inner.state.get()) {
                FramePlan::SendNow => self.write_frame(&json),
                FramePlan::QueueUntilOpen => {
                    self.inner.pre_open_frames.borrow_mut().push(json);
                    Ok(())
                }
            }
        }

        fn flush_pre_open_frames(&self) -> Result<()> {
            let frames = {
                let mut queue = self.inner.pre_open_frames.borrow_mut();
                take_queued_frames(&mut queue)
            };
            for frame in frames {
                self.write_frame(&frame)?;
            }
            Ok(())
        }

        fn send_notification(&self, method: &str, params: serde_json::Value) -> Result<()> {
            self.send_rpc_value(serde_json::json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "method": method,
                "params": params,
            }))
        }

        fn teardown_socket(&self) {
            *self.inner.handlers.borrow_mut() = None;
            if let Some(ws) = self.inner.ws.borrow_mut().take() {
                ws.set_onopen(None);
                ws.set_onmessage(None);
                ws.set_onclose(None);
                ws.set_onerror(None);
                ws.close().ok();
            }
            self.clear_pending_rpc();
            self.inner.pre_open_frames.borrow_mut().clear();
            self.inner.state.set(WsConnectionState::Disconnected);
        }

        fn reconnect_delay_ms(&self) -> u32 {
            let attempt = self.inner.reconnect_attempt.get();
            let shift = attempt.min(5);
            let delay = WS_RECONNECT_BASE_MS.saturating_mul(1u32 << shift);
            delay.min(WS_RECONNECT_MAX_MS)
        }

        fn schedule_reconnect(client: WsClient) {
            if !client.inner.auto_reconnect.get() || client.inner.intentional_close.get() {
                return;
            }
            let delay = client.reconnect_delay_ms();
            client
                .inner
                .reconnect_attempt
                .set(client.inner.reconnect_attempt.get().saturating_add(1));
            wasm_bindgen_futures::spawn_local(async move {
                gloo_timers::future::TimeoutFuture::new(delay).await;
                if client.inner.intentional_close.get() {
                    return;
                }
                if let Err(err) = client.reconnect() {
                    tracing::warn!(error = %err, "WebSocket reconnect failed");
                    Self::schedule_reconnect(client);
                }
            });
        }

        fn install_handlers(
            client: &WsClient,
            ws: &WebSocket,
            open_tx: Rc<RefCell<Option<oneshot::Sender<()>>>>,
        ) -> Result<()> {
            let open_tx_on_open = Rc::clone(&open_tx);
            let client_on_open = client.clone();
            let onopen = Closure::wrap(Box::new(move |_e: web_sys::Event| {
                client_on_open.inner.state.set(WsConnectionState::Connected);
                client_on_open.inner.reconnect_attempt.set(0);
                if let Err(err) = client_on_open.flush_pre_open_frames() {
                    tracing::warn!(error = %err, "failed to flush queued websocket frames");
                }
                if let Some(tx) = open_tx_on_open.borrow_mut().take() {
                    let _ = tx.send(());
                }
            }) as Box<dyn FnMut(_)>);
            ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));

            let client_msg = client.clone();
            let client_close = client.clone();
            let client_err = client.clone();
            let client_reconnect = client.clone();

            let onmessage = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                let text = e.data().as_string().unwrap_or_default();
                let parsed = serde_json::from_str::<serde_json::Value>(&text)
                    .map_err(|err| anyhow::anyhow!("Parse: {}", err));
                if let Ok(mut cb_guard) = client_msg.inner.message_handler.try_borrow_mut() {
                    let Some(cb) = cb_guard.as_mut() else {
                        return;
                    };
                    match parsed {
                        Err(e) => cb(Err(e)),
                        Ok(value) => {
                            if let Some(method) =
                                value.get("method").and_then(serde_json::Value::as_str)
                                && method == "chat.stream.event"
                            {
                                let event = value
                                    .get("params")
                                    .and_then(|p| p.get("event"))
                                    .cloned()
                                    .unwrap_or_default();
                                let body = event.get("payload").unwrap_or(&event);
                                let event_type = body
                                    .get("type")
                                    .and_then(serde_json::Value::as_str)
                                    .or_else(|| {
                                        event.get("type").and_then(serde_json::Value::as_str)
                                    })
                                    .unwrap_or_default();
                                match event_type {
                                    "text_chunk" | "content" => {
                                        let content = body
                                            .get("content")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or_default()
                                            .to_string();
                                        cb(Ok(ServerMessage::TextChunk { content }));
                                    }
                                    "tool_call_start" => {
                                        let tool_id = body
                                            .get(ACTION_ID_FIELD)
                                            .and_then(serde_json::Value::as_str)
                                            .or_else(|| {
                                                body.get("id").and_then(serde_json::Value::as_str)
                                            })
                                            .unwrap_or("tool")
                                            .to_string();
                                        let tool_name = body
                                            .get("name")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("tool")
                                            .to_string();
                                        cb(Ok(ServerMessage::ToolQueued { tool_id, tool_name }));
                                    }
                                    "tool_call_result" => {
                                        let tool_name = body
                                            .get("name")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("tool")
                                            .to_string();
                                        let tool_id = body
                                            .get(ACTION_ID_FIELD)
                                            .and_then(serde_json::Value::as_str)
                                            .or_else(|| {
                                                body.get("id").and_then(serde_json::Value::as_str)
                                            })
                                            .unwrap_or(tool_name.as_str())
                                            .to_string();
                                        let result_preview = body
                                            .get("result")
                                            .and_then(serde_json::Value::as_str)
                                            .map(ToString::to_string);
                                        cb(Ok(ServerMessage::ToolComplete {
                                            tool_id,
                                            tool_name,
                                            result_preview,
                                            success: true,
                                            duration_ms: 0,
                                        }));
                                    }
                                    "error" => {
                                        let error = body
                                            .get("error")
                                            .and_then(serde_json::Value::as_str)
                                            .unwrap_or("Stream error")
                                            .to_string();
                                        cb(Ok(ServerMessage::Error {
                                            error,
                                            recoverable: false,
                                        }));
                                    }
                                    "done" => {
                                        client_msg.inner.stream_done_seen.set(true);
                                        cb(Ok(ServerMessage::Done {
                                            usage: None,
                                            duration_ms: 0,
                                        }));
                                    }
                                    _ => {}
                                }
                                return;
                            }

                            if let Some(id) = value.get("id").and_then(serde_json::Value::as_u64) {
                                let pending = client_msg.inner.pending.borrow_mut().remove(&id);
                                if let Some(err) = value.get("error") {
                                    let message = err
                                        .get("message")
                                        .and_then(serde_json::Value::as_str)
                                        .unwrap_or("RPC error")
                                        .to_string();
                                    cb(Ok(ServerMessage::Error {
                                        error: message,
                                        recoverable: false,
                                    }));
                                    return;
                                }

                                if let Some(kind) = pending {
                                    let result = value.get("result").cloned().unwrap_or_default();
                                    match kind {
                                        PendingMethod::CreateSession => {
                                            if let Some(session_id) = result
                                                .get("session_id")
                                                .and_then(serde_json::Value::as_str)
                                            {
                                                *client_msg.inner.session_id.borrow_mut() =
                                                    Some(session_id.to_string());
                                                cb(Ok(ServerMessage::SessionCreated {
                                                    session_id: session_id.to_string(),
                                                }));
                                                if let Err(err) =
                                                    client_msg.flush_pending_outbound()
                                                {
                                                    tracing::warn!(
                                                        error = %err,
                                                        "failed to flush queued WebSocket messages"
                                                    );
                                                }
                                            }
                                        }
                                        PendingMethod::UpdateSettings => {
                                            if let Ok(settings) =
                                                serde_json::from_value::<AgentSettings>(result)
                                            {
                                                cb(Ok(ServerMessage::SettingsChanged { settings }));
                                            }
                                        }
                                        PendingMethod::GetSettings => {
                                            if let Ok(settings) =
                                                serde_json::from_value::<AgentSettings>(result)
                                            {
                                                cb(Ok(ServerMessage::CurrentSettings { settings }));
                                            }
                                        }
                                        PendingMethod::Message => {
                                            if !client_msg.inner.stream_done_seen.get() {
                                                cb(Ok(ServerMessage::Done {
                                                    usage: result.get("usage").cloned(),
                                                    duration_ms: 0,
                                                }));
                                            }
                                        }
                                        PendingMethod::Command => {
                                            cb(Ok(ServerMessage::CommandResult {
                                                success: true,
                                                data: result,
                                            }));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }) as Box<dyn FnMut(_)>);
            ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

            let onclose = Closure::wrap(Box::new(move |_e: web_sys::CloseEvent| {
                client_close.teardown_socket();
                if let Ok(mut cb_guard) = client_close.inner.message_handler.try_borrow_mut() {
                    if let Some(cb) = cb_guard.as_mut() {
                        cb(Err(anyhow::anyhow!("WebSocket closed")));
                    }
                }
                Self::schedule_reconnect(client_reconnect.clone());
            }) as Box<dyn FnMut(_)>);
            ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));

            let onerror = Closure::wrap(Box::new(move |_e: web_sys::Event| {
                if let Ok(mut cb_guard) = client_err.inner.message_handler.try_borrow_mut() {
                    if let Some(cb) = cb_guard.as_mut() {
                        cb(Err(anyhow::anyhow!("WebSocket error")));
                    }
                }
                client_err.inner.state.set(WsConnectionState::Disconnected);
            }) as Box<dyn FnMut(_)>);
            ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            *client.inner.handlers.borrow_mut() = Some(WsHandlerClosures {
                _onopen: onopen,
                _onmessage: onmessage,
                _onclose: onclose,
                _onerror: onerror,
            });
            Ok(())
        }

        fn connect_internal(
            &self,
            on_message: impl FnMut(Result<ServerMessage>) + 'static,
        ) -> Result<oneshot::Receiver<()>> {
            self.inner.intentional_close.set(false);
            self.teardown_socket();
            self.inner.state.set(WsConnectionState::Connecting);

            let url = ws_url_from_base(&self.inner.base_url);
            let ws =
                WebSocket::new(&url).map_err(|e| anyhow::anyhow!("WebSocket::new: {:?}", e))?;

            let (open_tx, open_rx) = oneshot::channel();
            let open_tx = Rc::new(RefCell::new(Some(open_tx)));

            *self.inner.message_handler.borrow_mut() = Some(Box::new(on_message));
            Self::install_handlers(self, &ws, open_tx)?;

            *self.inner.ws.borrow_mut() = Some(ws);

            let id = self.allocate_id(PendingMethod::GetSettings);
            self.send_rpc_value(serde_json::json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": id,
                "method": "settings.get",
                "params": {}
            }))?;
            Ok(open_rx)
        }

        /// Connect and invoke `on_message` from the WebSocket onmessage handler.
        pub fn connect_with_callback(
            &self,
            on_message: impl FnMut(Result<ServerMessage>) + 'static,
        ) -> Result<oneshot::Receiver<()>> {
            self.connect_internal(on_message)
        }

        /// Reconnect using the last registered message handler.
        pub fn reconnect(&self) -> Result<()> {
            let mut handler = self
                .inner
                .message_handler
                .borrow_mut()
                .take()
                .ok_or_else(|| anyhow::anyhow!("WebSocket reconnect requires a message handler"))?;
            let open_rx = self.connect_internal(move |result| {
                handler(result);
            })?;
            let _ = open_rx;
            Ok(())
        }

        /// Drain messages queued while waiting for [`ServerMessage::SessionCreated`].
        pub fn flush_pending_outbound(&self) -> Result<()> {
            let pending: Vec<ClientMessage> =
                self.inner.pending_outbound.borrow_mut().drain(..).collect();
            for msg in pending {
                self.send(&msg)?;
            }
            Ok(())
        }

        fn queue_until_session(&self, msg: ClientMessage) -> Result<()> {
            self.inner.pending_outbound.borrow_mut().push(msg);
            Ok(())
        }

        fn send_chat_stream(
            &self,
            content: &str,
            provider: &Option<String>,
            model: &Option<String>,
            tools: &Option<Vec<String>>,
            messages: &Option<Vec<crate::api::types::ChatMessage>>,
            mode: &Option<crate::api::types::InteractionMode>,
        ) -> Result<()> {
            if self.inner.session_id.borrow().is_none() {
                return self.queue_until_session(ClientMessage::Message {
                    content: content.to_string(),
                    provider: provider.clone(),
                    model: model.clone(),
                    tools: tools.clone(),
                    messages: messages.clone(),
                    mode: mode.clone(),
                });
            }

            self.inner.stream_done_seen.set(false);
            let id = self.allocate_id(PendingMethod::Message);
            let mut params = serde_json::json!({
                "message": content,
                "provider": provider,
                "model": model,
            });
            if let Some(tools) = tools {
                params["tools"] = serde_json::to_value(tools)?;
            }
            if let Some(messages) = messages {
                params["messages"] = serde_json::to_value(messages)?;
            }
            if let Some(mode) = mode {
                params["mode"] = serde_json::to_value(mode)?;
            }
            self.send_rpc_value(serde_json::json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": id,
                "method": "chat.stream",
                "params": params,
            }))
        }

        /// Send a client message. Must be called after connect(); connection is kept open.
        pub fn send(&self, msg: &ClientMessage) -> Result<()> {
            match msg {
                ClientMessage::CreateSession {
                    workspace,
                    max_turns,
                    system_prompt,
                } => {
                    let id = self.allocate_id(PendingMethod::CreateSession);
                    self.send_rpc_value(serde_json::json!({
                        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                        "id": id,
                        "method": "sessions.create",
                        "params": {
                            "workspace": workspace,
                            "context_files": [],
                            "max_turns": max_turns,
                            "system_prompt": system_prompt,
                        }
                    }))
                }
                ClientMessage::Message {
                    content,
                    provider,
                    model,
                    tools,
                    messages,
                    mode,
                } => self.send_chat_stream(content, provider, model, tools, messages, mode),
                ClientMessage::UpdateSettings { settings } => {
                    let id = self.allocate_id(PendingMethod::UpdateSettings);
                    self.send_rpc_value(serde_json::json!({
                        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                        "id": id,
                        "method": "settings.update",
                        "params": settings,
                    }))
                }
                ClientMessage::Cancel => {
                    self.send_notification("chat.cancel", serde_json::json!({}))
                }
                ClientMessage::Ping => {
                    // Browser WebSocket has no native ping frame; send a lightweight JSON-RPC notification.
                    self.send_notification("ping", serde_json::json!({}))
                }
                ClientMessage::Command { command } => {
                    let id = self.allocate_id(PendingMethod::Command);
                    self.send_rpc_value(serde_json::json!({
                        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                        "id": id,
                        "method": "chat.send",
                        "params": {
                            "message": command,
                        }
                    }))
                }
            }
        }

        /// Close the connection and suppress auto-reconnect.
        pub fn close(&self) {
            self.inner.intentional_close.set(true);
            self.inner.auto_reconnect.set(false);
            self.inner.pending_outbound.borrow_mut().clear();
            self.teardown_socket();
        }

        pub fn connection_state(&self) -> WsConnectionState {
            self.inner.state.get()
        }

        pub fn has_session(&self) -> bool {
            self.inner.session_id.borrow().is_some()
        }

        /// Remember a session created outside this socket (HTTP create or resume).
        ///
        /// Flushes chat frames that were waiting for `sessions.create`.
        pub fn adopt_session(&self, session_id: &str) {
            if session_id.is_empty() || self.has_session() {
                return;
            }
            *self.inner.session_id.borrow_mut() = Some(session_id.to_string());
            if let Err(err) = self.flush_pending_outbound() {
                tracing::warn!(
                    error = %err,
                    "failed to flush queued WebSocket messages after adopting session"
                );
            }
        }

        pub fn base_url(&self) -> &str {
            &self.inner.base_url
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_impl::WsClient;

#[cfg(not(target_arch = "wasm32"))]
pub struct WsClient {
    _base_url: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl WsClient {
    #[allow(
        dead_code,
        reason = "only the wasm client reports connection state; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn connection_state(&self) -> WsConnectionState {
        WsConnectionState::Disconnected
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FramePlan, SessionBind, WsConnectionState, chat_stream_waits_for_session, frame_plan,
        session_bind, session_id_after_bind, take_queued_frames,
    };

    #[test]
    fn frames_wait_until_the_socket_is_open() {
        assert_eq!(
            frame_plan(WsConnectionState::Connecting),
            FramePlan::QueueUntilOpen
        );
        assert_eq!(
            frame_plan(WsConnectionState::Disconnected),
            FramePlan::SendNow
        );
        assert_eq!(frame_plan(WsConnectionState::Connected), FramePlan::SendNow);

        let mut queued = Vec::new();
        if frame_plan(WsConnectionState::Connecting) == FramePlan::QueueUntilOpen {
            queued.push(r#"{"method":"settings.get"}"#.to_string());
            queued.push(r#"{"method":"chat.stream"}"#.to_string());
        }
        let flushed = take_queued_frames(&mut queued);
        assert_eq!(
            flushed,
            vec![
                r#"{"method":"settings.get"}"#.to_string(),
                r#"{"method":"chat.stream"}"#.to_string(),
            ]
        );
        assert!(queued.is_empty());
    }

    #[test]
    fn http_session_is_adopted_so_the_first_chat_frame_is_not_queued() {
        let bound = session_id_after_bind(None, Some("sess-http"));
        assert_eq!(
            session_bind(Some("sess-http"), false),
            SessionBind::Adopt("sess-http".to_string())
        );
        assert_eq!(bound.as_deref(), Some("sess-http"));
        assert!(!chat_stream_waits_for_session(bound.as_deref()));
    }

    #[test]
    fn missing_session_still_queues_until_the_socket_creates_one() {
        assert_eq!(session_bind(None, false), SessionBind::CreateOnSocket);
        assert!(session_id_after_bind(None, None).is_none());
        assert!(chat_stream_waits_for_session(None));
    }

    #[test]
    fn bound_socket_sends_chat_stream_immediately() {
        assert_eq!(session_bind(Some("sess-http"), true), SessionBind::Ready);
        let bound = session_id_after_bind(Some("sess-ws"), Some("sess-http"));
        assert_eq!(bound.as_deref(), Some("sess-ws"));
        assert!(!chat_stream_waits_for_session(bound.as_deref()));
    }
}
