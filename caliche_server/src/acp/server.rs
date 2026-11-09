//! ACP agent methods over JSON-RPC.
//!
//! Stable wire version is 1. `session/update` frames are written as stream
//! events arrive. `session/cancel` cancels the token passed into the in-flight
//! turn; the prompt result then reports `stopReason: cancelled`.
//!
//! `session/list` and `session/load` read the live session cache. Load writes
//! one `session/update` per replayed history entry, then the load result.
//!
//! A tool the engine will not run without approval emits
//! `session/request_permission` and waits for the client. Allow once runs the
//! tool. Any other result, a timeout, or a dropped stdin skips it.
//!
//! The prompt token is claimed before any await. A cancel that arrives after
//! the prompt is queued and before that claim is stored on the slot and applied
//! when the prompt claims, so a fresh token cannot outrun `session/cancel`.

use crate::acp::catalog::{SESSION_LIST_PAGE_SIZE, SessionListError, same_cwd, session_list_page};
use crate::acp::driver::{AcpEventSink, AcpTurnDriver};
use crate::acp::mapping::{
    PendingToolCalls, prompt_to_plain_text, updates_from_history, updates_from_stream_event,
    user_updates_from_prompt,
};
use crate::acp::mcp::validate_mcp_servers;
use crate::acp::outbound::AcpFrameSink;
use crate::acp::permission::{AcpPermissionGate, PermissionBroker, is_client_response};
use crate::adapter::StreamingTurnResult;
use crate::rpc::types::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use agent_client_protocol_schema::ProtocolVersion;
use agent_client_protocol_schema::v1::{
    AgentCapabilities, CancelNotification, Implementation, InitializeRequest, InitializeResponse,
    ListSessionsRequest, LoadSessionRequest, McpCapabilities, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, SessionCapabilities,
    SessionId, SessionListCapabilities, SessionNotification, SessionUpdate, StopReason,
};
use anyhow::Context;
use caliche_contracts::acp::methods;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const ACP_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion::V1;

struct EmitGate {
    pending: Mutex<PendingToolCalls>,
    error: Mutex<Option<String>>,
}

impl EmitGate {
    fn record_error(&self, message: String) {
        let mut slot = lock_mutex(&self.error);
        if slot.is_none() {
            *slot = Some(message);
        }
    }

    fn take_error(&self) -> Option<String> {
        lock_mutex(&self.error).take()
    }
}

impl Default for EmitGate {
    fn default() -> Self {
        Self {
            pending: Mutex::new(PendingToolCalls::default()),
            error: Mutex::new(None),
        }
    }
}

struct CancelSlot {
    token: CancellationToken,
    /// `session/prompt` was accepted and has not claimed a token yet.
    queued: bool,
    /// The prompt task holds `token` for this turn.
    claimed: bool,
    /// `session/cancel` arrived while `queued` and before `claimed`.
    pending_cancel: bool,
}

impl Default for CancelSlot {
    fn default() -> Self {
        Self {
            token: CancellationToken::new(),
            queued: false,
            claimed: false,
            pending_cancel: false,
        }
    }
}

struct PromptRelease<'a> {
    server: &'a AcpServer,
    session_id: String,
}

impl Drop for PromptRelease<'_> {
    fn drop(&mut self) {
        self.server.release_prompt(&self.session_id);
    }
}

pub struct AcpServer {
    driver: Arc<dyn AcpTurnDriver>,
    initialized: AtomicBool,
    agent_info: Implementation,
    cancels: Mutex<HashMap<String, CancelSlot>>,
    permissions: Arc<PermissionBroker>,
}

impl AcpServer {
    pub fn new(driver: Arc<dyn AcpTurnDriver>, agent_info: Implementation) -> Self {
        Self::with_permission_timeout(driver, agent_info, crate::acp::ACP_PERMISSION_TIMEOUT)
    }

    /// Same as [`Self::new`] with an explicit permission wait.
    ///
    /// The CLI uses [`Self::new`], which waits [`crate::acp::ACP_PERMISSION_TIMEOUT`].
    pub fn with_permission_timeout(
        driver: Arc<dyn AcpTurnDriver>,
        agent_info: Implementation,
        permission_timeout: Duration,
    ) -> Self {
        Self {
            driver,
            initialized: AtomicBool::new(false),
            agent_info,
            cancels: Mutex::new(HashMap::new()),
            permissions: Arc::new(PermissionBroker::new(permission_timeout)),
        }
    }

    /// Mark a prompt accepted before the reader handles the next stdin line.
    ///
    /// `session/cancel` in that gap sets [`CancelSlot::pending_cancel`] instead
    /// of only cancelling the placeholder token from `session/new`.
    pub(crate) fn queue_prompt(&self, session_id: &str) {
        let mut slots = lock_mutex(&self.cancels);
        let slot = slots.entry(session_id.to_string()).or_default();
        slot.queued = true;
    }

    /// Cancel every in-flight prompt. Used when stdin closes.
    ///
    /// Pending permission waits fail closed, and later asks are not written.
    pub fn cancel_all(&self) {
        let mut slots = lock_mutex(&self.cancels);
        for slot in slots.values_mut() {
            slot.token.cancel();
            if slot.queued && !slot.claimed {
                slot.pending_cancel = true;
            }
        }
        drop(slots);
        self.permissions.fail_closed();
    }

    pub async fn handle_json_value(
        &self,
        value: Value,
        frames: &Arc<dyn AcpFrameSink>,
    ) -> anyhow::Result<()> {
        if let Some(batch) = value.as_array() {
            for item in batch {
                Box::pin(self.dispatch_value(item.clone(), frames)).await?;
            }
            return Ok(());
        }

        self.dispatch_value(value, frames).await
    }

    async fn dispatch_value(
        &self,
        value: Value,
        frames: &Arc<dyn AcpFrameSink>,
    ) -> anyhow::Result<()> {
        if is_client_response(&value) {
            self.permissions.complete_inbound(&value);
            return Ok(());
        }

        match serde_json::from_value::<JsonRpcRequest>(value) {
            Ok(req) => Box::pin(self.handle_request(req, frames)).await,
            Err(err) => self.write_frame(
                frames,
                &json_rpc_error(
                    None,
                    JsonRpcError::invalid_request(Some(json!({"error": err.to_string()}))),
                ),
            ),
        }
    }

    async fn handle_request(
        &self,
        req: JsonRpcRequest,
        frames: &Arc<dyn AcpFrameSink>,
    ) -> anyhow::Result<()> {
        if req.jsonrpc.as_ref() != caliche_contracts::rpc::JSONRPC_VERSION {
            return write_rpc(
                frames,
                req.id,
                Err(JsonRpcError::invalid_request(Some(
                    json!({"reason": "jsonrpc must be 2.0"}),
                ))),
            );
        }

        match req.method.as_ref() {
            methods::INITIALIZE => {
                let result = self.handle_initialize(req.params);
                write_rpc(frames, req.id, result)
            }
            methods::SESSION_NEW => {
                if let Err(err) = self.ensure_initialized() {
                    return write_rpc(frames, req.id, Err(err));
                }
                let result = Box::pin(self.handle_session_new(req.params)).await;
                write_rpc(frames, req.id, result)
            }
            methods::SESSION_PROMPT => {
                if let Err(err) = self.ensure_initialized() {
                    return write_rpc(frames, req.id, Err(err));
                }
                Box::pin(self.handle_session_prompt(req.id, req.params, frames)).await
            }
            methods::SESSION_LIST => {
                if let Err(err) = self.ensure_initialized() {
                    return write_rpc(frames, req.id, Err(err));
                }
                let result = Box::pin(self.handle_session_list(req.params)).await;
                write_rpc(frames, req.id, result)
            }
            methods::SESSION_LOAD => {
                if let Err(err) = self.ensure_initialized() {
                    return write_rpc(frames, req.id, Err(err));
                }
                Box::pin(self.handle_session_load(req.id, req.params, frames)).await
            }
            methods::SESSION_CANCEL => {
                if let Err(err) = self.ensure_initialized() {
                    return write_rpc(frames, req.id, Err(err));
                }
                let result = self.handle_session_cancel(req.params);
                if req.id.is_some() {
                    write_rpc(frames, req.id, result)
                } else {
                    Ok(())
                }
            }
            _ => write_rpc(
                frames,
                req.id,
                Err(JsonRpcError::method_not_found(req.method.as_ref())),
            ),
        }
    }

    fn handle_initialize(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        parse_params::<InitializeRequest>(params)?;
        // v2 is a draft. Always answer with stable version 1.
        self.initialized.store(true, Ordering::Release);

        let result = InitializeResponse::new(ACP_PROTOCOL_VERSION)
            .agent_capabilities(agent_capabilities())
            .agent_info(self.agent_info.clone());
        serde_json::to_value(result)
            .map_err(|err| JsonRpcError::internal_error(Some(json!({"error": err.to_string()}))))
    }

    async fn handle_session_new(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let parsed = parse_session_request::<NewSessionRequest>(params)?;
        if !parsed.cwd.is_absolute() {
            return Err(JsonRpcError::invalid_params(Some(json!({
                "cwd": "must be an absolute path"
            }))));
        }
        let cwd = utf8_path(&parsed.cwd)?;

        validate_mcp_servers(&parsed.mcp_servers).map_err(|err| {
            JsonRpcError::invalid_params(Some(json!({
                "mcpServers": err.to_string(),
            })))
        })?;

        let session_id = self
            .driver
            .open_session(cwd.to_owned())
            .await
            .map_err(|err| JsonRpcError::internal_error(Some(json!({"error": err.to_string()}))))?;
        lock_mutex(&self.cancels).insert(session_id.clone(), CancelSlot::default());

        serde_json::to_value(NewSessionResponse::new(session_id))
            .map_err(|err| JsonRpcError::internal_error(Some(json!({"error": err.to_string()}))))
    }

    async fn handle_session_list(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let parsed = parse_params::<ListSessionsRequest>(params)?;
        let cwd = match parsed.cwd.as_deref() {
            Some(path) => Some(utf8_path(path)?),
            None => None,
        };
        let rows =
            self.driver.list_sessions().await.map_err(|err| {
                JsonRpcError::internal_error(Some(json!({"error": err.to_string()})))
            })?;
        let page = session_list_page(rows, cwd, parsed.cursor.as_deref(), SESSION_LIST_PAGE_SIZE)
            .map_err(list_error)?;
        serde_json::to_value(page)
            .map_err(|err| JsonRpcError::internal_error(Some(json!({"error": err.to_string()}))))
    }

    async fn handle_session_load(
        &self,
        id: Option<Value>,
        params: Option<Value>,
        frames: &Arc<dyn AcpFrameSink>,
    ) -> anyhow::Result<()> {
        let Some(id_value) = id else {
            return Ok(());
        };
        let parsed = match parse_session_request::<LoadSessionRequest>(params) {
            Ok(value) => value,
            Err(err) => return write_rpc(frames, Some(id_value), Err(err)),
        };
        if let Err(err) = validate_load_params(&parsed) {
            return write_rpc(frames, Some(id_value), Err(err));
        }
        let session_id = session_id_string(&parsed.session_id);
        let loaded = match self.driver.load_session(session_id.clone()).await {
            Ok(value) => value,
            Err(err) => {
                return write_rpc(
                    frames,
                    Some(id_value),
                    Err(JsonRpcError::internal_error(Some(
                        json!({"error": err.to_string()}),
                    ))),
                );
            }
        };
        let Some(loaded) = loaded else {
            return write_rpc(
                frames,
                Some(id_value),
                Err(JsonRpcError::invalid_params(Some(json!({
                    "sessionId": "unknown session"
                })))),
            );
        };
        let cwd = match utf8_path(&parsed.cwd) {
            Ok(cwd) => cwd,
            Err(err) => return write_rpc(frames, Some(id_value), Err(err)),
        };
        if !same_cwd(&loaded.listed.cwd, cwd) {
            return write_rpc(
                frames,
                Some(id_value),
                Err(JsonRpcError::invalid_params(Some(json!({
                    "cwd": "does not match the session workspace"
                })))),
            );
        }
        let updates = updates_from_history(&loaded.history);
        self.write_replay(frames, &session_id, updates)?;
        self.ensure_cancel_slot(&session_id);
        write_rpc(frames, Some(id_value), Ok(json!({})))
    }

    async fn handle_session_prompt(
        &self,
        id: Option<Value>,
        params: Option<Value>,
        frames: &Arc<dyn AcpFrameSink>,
    ) -> anyhow::Result<()> {
        let Some(id_value) = id else {
            self.release_queued(params.as_ref());
            return Ok(());
        };
        let queued_session = session_id_from_params(params.as_ref());
        let parsed = match parse_params::<PromptRequest>(params) {
            Ok(value) => value,
            Err(err) => {
                if let Some(session_id) = queued_session {
                    self.release_prompt(&session_id);
                }
                return write_rpc(frames, Some(id_value), Err(err));
            }
        };

        // Claim before has_session. That await is the window where a piped
        // session/cancel used to hit only the session/new placeholder.
        let session_id = session_id_string(&parsed.session_id);
        let cancel = self.claim_prompt(&session_id);
        let _release = PromptRelease {
            server: self,
            session_id: session_id.clone(),
        };
        if !self.driver.has_session(session_id.clone()).await {
            return write_rpc(
                frames,
                Some(id_value),
                Err(JsonRpcError::invalid_params(Some(json!({
                    "sessionId": "unknown session"
                })))),
            );
        }

        tracing::debug!(
            target: "caliche_server::acp",
            session_id = %session_id,
            "acp prompt started"
        );
        self.write_user_updates(frames, &parsed)?;

        let input = match prompt_to_plain_text(&parsed.prompt) {
            Ok(value) => value,
            Err(err) => {
                return write_rpc(
                    frames,
                    Some(id_value),
                    Err(JsonRpcError::invalid_params(Some(
                        json!({"prompt": err.to_string()}),
                    ))),
                );
            }
        };

        let (sink, gate) = self.event_sink(frames, &session_id);
        let permissions = self.permission_gate(frames, &session_id, cancel.clone());
        let turn = self
            .driver
            .run_turn(session_id, input, cancel.clone(), sink, permissions)
            .await;
        if let Some(message) = gate.take_error() {
            anyhow::bail!(message);
        }
        self.write_prompt_result(frames, id_value, turn, &cancel)
    }

    fn write_replay(
        &self,
        frames: &Arc<dyn AcpFrameSink>,
        session_id: &str,
        updates: Vec<SessionUpdate>,
    ) -> anyhow::Result<()> {
        for update in updates {
            let frame = session_update_notification(session_id, update)?;
            self.write_frame(frames, &frame)?;
        }
        Ok(())
    }

    fn ensure_cancel_slot(&self, session_id: &str) {
        lock_mutex(&self.cancels)
            .entry(session_id.to_string())
            .or_default();
    }

    fn write_user_updates(
        &self,
        frames: &Arc<dyn AcpFrameSink>,
        parsed: &PromptRequest,
    ) -> anyhow::Result<()> {
        for update in user_updates_from_prompt(&parsed.prompt) {
            let frame = session_update_notification(parsed.session_id.0.as_ref(), update)?;
            self.write_frame(frames, &frame)?;
        }
        Ok(())
    }

    fn event_sink(
        &self,
        frames: &Arc<dyn AcpFrameSink>,
        session_id: &str,
    ) -> (AcpEventSink, Arc<EmitGate>) {
        let gate = Arc::new(EmitGate::default());
        let frames = Arc::clone(frames);
        let session_id = session_id.to_string();
        let gate_for_emit = Arc::clone(&gate);
        let sink = AcpEventSink::new(move |event| {
            let updates = {
                let mut pending = lock_mutex(&gate_for_emit.pending);
                updates_from_stream_event(event, &mut pending)
            };
            for update in updates {
                match session_update_notification(&session_id, update) {
                    Ok(frame) => {
                        if let Err(err) = frames.write_frame(&frame) {
                            gate_for_emit.record_error(err.to_string());
                            return;
                        }
                    }
                    Err(err) => {
                        gate_for_emit.record_error(err.to_string());
                        return;
                    }
                }
            }
        });
        (sink, gate)
    }

    fn write_prompt_result(
        &self,
        frames: &Arc<dyn AcpFrameSink>,
        id_value: Value,
        turn: anyhow::Result<StreamingTurnResult>,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        match turn {
            Ok(result) => {
                let response = PromptResponse::new(stop_reason_for(&result, cancel));
                write_rpc(frames, Some(id_value), Ok(json!(response)))
            }
            Err(_) if cancel.is_cancelled() => {
                let response = PromptResponse::new(StopReason::Cancelled);
                write_rpc(frames, Some(id_value), Ok(json!(response)))
            }
            Err(err) => write_rpc(
                frames,
                Some(id_value),
                Err(JsonRpcError::internal_error(Some(
                    json!({"error": err.to_string()}),
                ))),
            ),
        }
    }

    fn handle_session_cancel(&self, params: Option<Value>) -> Result<Value, JsonRpcError> {
        let parsed = parse_params::<CancelNotification>(params)?;
        self.cancel_session(parsed.session_id.0.as_ref());
        Ok(json!({ "cancelled": true }))
    }

    fn claim_prompt(&self, session_id: &str) -> CancellationToken {
        let mut slots = lock_mutex(&self.cancels);
        let slot = slots.entry(session_id.to_string()).or_default();
        if slot.claimed {
            return slot.token.clone();
        }
        let token = CancellationToken::new();
        if slot.pending_cancel {
            token.cancel();
            slot.pending_cancel = false;
        }
        slot.token = token.clone();
        slot.claimed = true;
        slot.queued = false;
        token
    }

    fn cancel_session(&self, session_id: &str) {
        let mut slots = lock_mutex(&self.cancels);
        let Some(slot) = slots.get_mut(session_id) else {
            return;
        };
        slot.token.cancel();
        if slot.queued && !slot.claimed {
            slot.pending_cancel = true;
        }
    }

    fn release_prompt(&self, session_id: &str) {
        let mut slots = lock_mutex(&self.cancels);
        let Some(slot) = slots.get_mut(session_id) else {
            return;
        };
        slot.claimed = false;
        slot.queued = false;
        slot.pending_cancel = false;
    }

    fn release_queued(&self, params: Option<&Value>) {
        if let Some(session_id) = session_id_from_params(params) {
            self.release_prompt(&session_id);
        }
    }

    fn permission_gate(
        &self,
        frames: &Arc<dyn AcpFrameSink>,
        session_id: &str,
        cancel: CancellationToken,
    ) -> AcpPermissionGate {
        AcpPermissionGate::new(
            Arc::clone(&self.permissions),
            Arc::clone(frames),
            session_id.to_owned(),
            cancel,
        )
    }

    fn ensure_initialized(&self) -> Result<(), JsonRpcError> {
        if self.initialized.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(JsonRpcError::invalid_request(Some(
                json!({"reason": "initialize must be called before session methods"}),
            )))
        }
    }

    fn write_frame(&self, frames: &Arc<dyn AcpFrameSink>, frame: &Value) -> anyhow::Result<()> {
        frames.write_frame(frame).context("write ACP frame")
    }
}

fn validate_load_params(parsed: &LoadSessionRequest) -> Result<(), JsonRpcError> {
    if !parsed.cwd.is_absolute() {
        return Err(JsonRpcError::invalid_params(Some(json!({
            "cwd": "must be an absolute path"
        }))));
    }
    validate_mcp_servers(&parsed.mcp_servers).map_err(|err| {
        JsonRpcError::invalid_params(Some(json!({
            "mcpServers": err.to_string(),
        })))
    })
}

fn list_error(err: SessionListError) -> JsonRpcError {
    match err {
        SessionListError::RelativeCwd => JsonRpcError::invalid_params(Some(json!({
            "cwd": "must be an absolute path"
        }))),
        SessionListError::InvalidCursor => JsonRpcError::invalid_params(Some(json!({
            "cursor": "invalid"
        }))),
    }
}

fn stop_reason_for(result: &StreamingTurnResult, cancel: &CancellationToken) -> StopReason {
    if result.cancelled || cancel.is_cancelled() {
        StopReason::Cancelled
    } else {
        StopReason::EndTurn
    }
}

fn session_id_from_params(params: Option<&Value>) -> Option<String> {
    params?
        .get("sessionId")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn parse_params<T: DeserializeOwned>(params: Option<Value>) -> Result<T, JsonRpcError> {
    let payload = params.unwrap_or_else(|| json!({}));
    serde_json::from_value(payload).map_err(|err| {
        JsonRpcError::invalid_params(Some(json!({
            "error": err.to_string()
        })))
    })
}

fn agent_capabilities() -> AgentCapabilities {
    AgentCapabilities::new()
        .load_session(true)
        .prompt_capabilities(PromptCapabilities::new().embedded_context(true))
        .mcp_capabilities(McpCapabilities::new())
        .session_capabilities(SessionCapabilities::new().list(SessionListCapabilities::new()))
}

fn session_id_string(session_id: &SessionId) -> String {
    session_id.0.as_ref().to_owned()
}

fn utf8_path(path: &Path) -> Result<&str, JsonRpcError> {
    path.to_str().ok_or_else(|| {
        JsonRpcError::invalid_params(Some(json!({
            "cwd": "must be valid UTF-8"
        })))
    })
}

fn parse_session_request<T: DeserializeOwned>(params: Option<Value>) -> Result<T, JsonRpcError> {
    parse_params(Some(params.unwrap_or_else(|| json!({}))))
}

fn session_update_notification(session_id: &str, update: SessionUpdate) -> anyhow::Result<Value> {
    let params = SessionNotification::new(SessionId::new(session_id), update);
    Ok(json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "method": methods::SESSION_UPDATE,
        "params": params
    }))
}

fn write_rpc(
    frames: &Arc<dyn AcpFrameSink>,
    id: Option<Value>,
    result: Result<Value, JsonRpcError>,
) -> anyhow::Result<()> {
    let Some(id_value) = id else {
        return Ok(());
    };
    let frame = match result {
        Ok(value) => json_rpc_success(Some(id_value), value),
        Err(err) => json_rpc_error(Some(id_value), err),
    };
    frames.write_frame(&frame).context("write ACP response")
}

fn json_rpc_success(id: Option<Value>, result: Value) -> Value {
    serde_json::to_value(JsonRpcResponse::success(id.clone(), result)).unwrap_or_else(
        |_| json!({"jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION, "id": id, "result": {}}),
    )
}

fn json_rpc_error(id: Option<Value>, error: JsonRpcError) -> Value {
    serde_json::to_value(JsonRpcResponse::error(id.clone(), error)).unwrap_or_else(|_| {
        json!({"jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION, "id": id, "error": {"code": -32603, "message": "Internal error"}})
    })
}

pub(crate) fn parse_error_frame(err: &serde_json::Error) -> Value {
    json_rpc_error(
        None,
        JsonRpcError::parse_error(Some(json!({"error": err.to_string()}))),
    )
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}
