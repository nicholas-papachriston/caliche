//! ACP agent turn tests. The fake driver never reads provider credentials.

use super::driver::{AcpEventSink, AcpListedSession, AcpLoadedSession, AcpTurnDriver};
use super::outbound::{AcpFrameSink, StdFrameSink};
use super::permission::AcpPermissionGate;
use super::server::AcpServer;
use super::stdio::serve_acp;
use crate::adapter::StreamingTurnResult;
use agent_client_protocol_schema::v1::Implementation;
use async_trait::async_trait;
use caliche::agent::hooks::{
    ApprovalResolver, HookContext, HookMatcher, HookRegistry, HookResult, LifecycleHook,
};
use caliche::agent::tools::execution::{
    HookExecutionMiddleware, ToolExecutionContext, ToolExecutionMiddleware,
};
use caliche::messages::{Message, StreamEvent};
use chrono::TimeZone;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

mod catalog_tests;
mod permission_tests;

struct StoredSession {
    cwd: String,
    title: Option<String>,
    updated_at: chrono::DateTime<chrono::Utc>,
    history: Vec<Message>,
}

#[derive(Clone, Copy)]
enum Script {
    BlockUntilCancel,
    ToolPair,
    /// Block inside `has_session` so cancel lands before the prompt used to
    /// install its token.
    HoldHasSession,
    /// Return as soon as the turn token is already cancelled; otherwise wait.
    FinishWhenCancelled,
    /// Require approval through the real hook middleware, then run only on allow.
    AskPermission,
}

struct FakeDriver {
    sessions: Mutex<HashMap<String, StoredSession>>,
    next: AtomicU64,
    script: Script,
    captured: Mutex<Option<CancellationToken>>,
    entered_has_session: AtomicBool,
    release_has_session: Notify,
    tool_ran: AtomicBool,
    permission_call_id: String,
}

impl FakeDriver {
    fn block_until_cancel() -> Self {
        Self::new(Script::BlockUntilCancel)
    }

    fn tool_pair() -> Self {
        Self::new(Script::ToolPair)
    }

    fn hold_has_session() -> Self {
        Self::new(Script::HoldHasSession)
    }

    fn finish_when_cancelled() -> Self {
        Self::new(Script::FinishWhenCancelled)
    }

    fn ask_permission() -> Self {
        Self::new(Script::AskPermission)
    }

    fn new(script: Script) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
            script,
            captured: Mutex::new(None),
            entered_has_session: AtomicBool::new(false),
            release_has_session: Notify::new(),
            tool_ran: AtomicBool::new(false),
            permission_call_id: "call-7".to_owned(),
        }
    }

    fn with_permission_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.permission_call_id = call_id.into();
        self
    }

    fn tool_ran(&self) -> bool {
        self.tool_ran.load(Ordering::Acquire)
    }

    fn session_count(&self) -> usize {
        self.sessions.lock().expect("sessions").len()
    }

    fn seed_transcript(&self, session_id: &str, title: &str, history: Vec<Message>) {
        let mut sessions = self.sessions.lock().expect("sessions");
        let stored = sessions.get_mut(session_id).expect("session");
        stored.title = Some(title.to_string());
        stored.history = history;
        stored.updated_at = chrono::Utc
            .with_ymd_and_hms(2026, 1, 2, 3, 4, 5)
            .single()
            .expect("timestamp");
    }

    fn captured_token(&self) -> CancellationToken {
        self.captured
            .lock()
            .expect("captured")
            .clone()
            .expect("turn token")
    }
}

#[async_trait]
impl AcpTurnDriver for FakeDriver {
    async fn open_session(&self, workspace: String) -> anyhow::Result<String> {
        let id = format!("sess-{}", self.next.fetch_add(1, Ordering::Relaxed));
        self.sessions.lock().expect("sessions").insert(
            id.clone(),
            StoredSession {
                cwd: workspace,
                title: None,
                updated_at: chrono::Utc::now(),
                history: Vec::new(),
            },
        );
        Ok(id)
    }

    async fn has_session(&self, session_id: String) -> bool {
        if matches!(self.script, Script::HoldHasSession) {
            self.entered_has_session.store(true, Ordering::Release);
            self.release_has_session.notified().await;
        }
        self.sessions
            .lock()
            .expect("sessions")
            .contains_key(&session_id)
    }

    async fn list_sessions(&self) -> anyhow::Result<Vec<AcpListedSession>> {
        let sessions = self.sessions.lock().expect("sessions");
        let mut rows = Vec::with_capacity(sessions.len());
        for (id, stored) in sessions.iter() {
            rows.push(AcpListedSession {
                session_id: id.clone(),
                cwd: stored.cwd.clone(),
                title: stored.title.clone(),
                updated_at: stored.updated_at,
            });
        }
        Ok(rows)
    }

    async fn load_session(&self, session_id: String) -> anyhow::Result<Option<AcpLoadedSession>> {
        let sessions = self.sessions.lock().expect("sessions");
        let Some(stored) = sessions.get(&session_id) else {
            return Ok(None);
        };
        Ok(Some(AcpLoadedSession {
            listed: AcpListedSession {
                session_id,
                cwd: stored.cwd.clone(),
                title: stored.title.clone(),
                updated_at: stored.updated_at,
            },
            history: stored.history.clone(),
        }))
    }

    async fn run_turn(
        &self,
        _session_id: String,
        _input: String,
        cancel: CancellationToken,
        on_event: AcpEventSink,
        permissions: AcpPermissionGate,
    ) -> anyhow::Result<StreamingTurnResult> {
        *self.captured.lock().expect("captured") = Some(cancel.clone());
        match self.script {
            Script::BlockUntilCancel => {
                on_event.emit(StreamEvent::TextChunk {
                    content: "alpha".to_string(),
                });
                on_event.emit(StreamEvent::TextChunk {
                    content: "beta".to_string(),
                });
                cancel.cancelled().await;
                Ok(StreamingTurnResult { cancelled: true })
            }
            Script::HoldHasSession | Script::FinishWhenCancelled => {
                if !cancel.is_cancelled() {
                    cancel.cancelled().await;
                }
                Ok(StreamingTurnResult { cancelled: true })
            }
            Script::ToolPair => {
                on_event.emit(stream_event(json!({
                    "type": "tool_call_start",
                    "id": "call-7",
                    "name": "fs",
                    "arguments": {"op": "read"}
                })));
                on_event.emit(stream_event(json!({
                    "type": "tool_call_result",
                    "name": "fs",
                    "result": "file body"
                })));
                Ok(StreamingTurnResult { cancelled: false })
            }
            Script::AskPermission => {
                ask_then_maybe_run(
                    &self.tool_ran,
                    &cancel,
                    &on_event,
                    permissions,
                    &self.permission_call_id,
                )
                .await
            }
        }
    }
}

#[derive(Debug)]
struct RequireBashApproval;

#[async_trait]
impl LifecycleHook for RequireBashApproval {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::require_approval(
            "PermissionMode::Interactive requires approval for 'bash'",
            None,
        )
    }
}

async fn ask_then_maybe_run(
    tool_ran: &AtomicBool,
    cancel: &CancellationToken,
    on_event: &AcpEventSink,
    permissions: AcpPermissionGate,
    call_id: &str,
) -> anyhow::Result<StreamingTurnResult> {
    on_event.emit(stream_event(json!({
        "type": "tool_call_start",
        "id": call_id,
        "name": "bash",
        "arguments": {"command": "echo hi"}
    })));
    let mut hooks = HookRegistry::new();
    hooks.register(
        HookMatcher::tool("bash"),
        Arc::new(RequireBashApproval) as Arc<dyn LifecycleHook>,
    );
    let resolver: Arc<dyn ApprovalResolver> =
        Arc::new(super::permission::AcpApprovalResolver::new(permissions));
    let middleware = HookExecutionMiddleware::new(hooks).with_approval_resolver(Some(resolver));
    let args = json!({"command": "echo hi"});
    let ctx = ToolExecutionContext {
        tool_name: "bash",
        arguments: &args,
        call_id,
    };
    // Record Allow on its own. A later cancel flag must not hide an approval
    // that returned Allow: that is the bug these tests exist to catch.
    let allowed = middleware.before_execute(&ctx).await.is_ok();
    if allowed {
        tool_ran.store(true, Ordering::Release);
    }
    if allowed && !cancel.is_cancelled() {
        on_event.emit(stream_event(json!({
            "type": "tool_call_result",
            "name": "bash",
            "result": "tool-ran"
        })));
        return Ok(StreamingTurnResult { cancelled: false });
    }
    Ok(StreamingTurnResult {
        cancelled: cancel.is_cancelled(),
    })
}

fn stream_event(value: Value) -> StreamEvent {
    serde_json::from_value(value).expect("stream event")
}

fn assert_catalog_capabilities(result: &Value) {
    assert_eq!(result["authMethods"], json!([]));
    assert_eq!(result["agentCapabilities"]["loadSession"], true);
    assert_eq!(
        result["agentCapabilities"]["sessionCapabilities"]["list"],
        json!({})
    );
    assert!(
        result["agentCapabilities"]["sessionCapabilities"]
            .get("resume")
            .is_none()
    );
    assert!(
        result["agentCapabilities"]["sessionCapabilities"]
            .get("close")
            .is_none()
    );
}

fn agent_info() -> Implementation {
    Implementation::new("caliche", "0.1.0").title("Caliche")
}

fn server_for(fake: &Arc<FakeDriver>) -> AcpServer {
    let driver: Arc<dyn AcpTurnDriver> = Arc::clone(fake) as Arc<dyn AcpTurnDriver>;
    AcpServer::new(driver, agent_info())
}

fn server_for_timeout(fake: &Arc<FakeDriver>, timeout: Duration) -> AcpServer {
    let driver: Arc<dyn AcpTurnDriver> = Arc::clone(fake) as Arc<dyn AcpTurnDriver>;
    AcpServer::with_permission_timeout(driver, agent_info(), timeout)
}

#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBuf {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.0.lock().expect("stdout buf").extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn mem_sink() -> (Arc<Mutex<Vec<u8>>>, Arc<dyn AcpFrameSink>) {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let sink: Arc<dyn AcpFrameSink> = Arc::new(StdFrameSink::new(SharedBuf(Arc::clone(&buf))));
    (buf, sink)
}

fn snapshot(buf: &Arc<Mutex<Vec<u8>>>) -> Vec<u8> {
    buf.lock().expect("stdout buf").clone()
}

fn parse_frames(bytes: &[u8]) -> Vec<Value> {
    let text = std::str::from_utf8(bytes).expect("stdout utf-8");
    assert_json_lines(text);
    text.split('\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).expect("json frame"))
        .collect()
}

fn assert_json_lines(text: &str) {
    assert!(
        text.is_empty() || text.ends_with('\n'),
        "stdout must be JSON objects plus newlines, got {text:?}"
    );
    for line in text.split('\n') {
        if line.is_empty() {
            continue;
        }
        assert!(
            line.starts_with('{') && line.ends_with('}'),
            "log line on stdout: {line}"
        );
        let value: Value = serde_json::from_str(line).unwrap_or_else(|err| {
            panic!("stdout line is not JSON ({err}): {line}");
        });
        assert!(
            value.is_object(),
            "stdout line is not a JSON object: {line}"
        );
    }
}

fn initialize_request(version: u32) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": version}
    })
}

fn line(value: &Value) -> String {
    let mut text = value.to_string();
    text.push('\n');
    text
}

#[tokio::test]
async fn initialize_v1_and_v2_answer_version_one() {
    for version in [1_u32, 2] {
        let fake = Arc::new(FakeDriver::block_until_cancel());
        let server = server_for(&fake);
        let (buf, frames) = mem_sink();
        server
            .handle_json_value(initialize_request(version), &frames)
            .await
            .expect("initialize");
        let parsed = parse_frames(&snapshot(&buf));
        assert_eq!(parsed.len(), 1);
        let result = &parsed[0]["result"];
        assert_eq!(result["protocolVersion"], 1);
        assert_eq!(result["agentInfo"]["name"], "caliche");
        assert_catalog_capabilities(result);
    }
}

#[tokio::test]
async fn session_new_rejects_relative_cwd() {
    let fake = Arc::new(FakeDriver::block_until_cancel());
    let server = server_for(&fake);
    let (buf, frames) = mem_sink();
    server
        .handle_json_value(initialize_request(1), &frames)
        .await
        .expect("initialize");
    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 2,
                "method": "session/new",
                "params": {"cwd": "relative/workspace", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("session/new");
    let parsed = parse_frames(&snapshot(&buf));
    assert_eq!(parsed[1]["error"]["code"], -32602);
    assert_eq!(fake.session_count(), 0);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 3,
                "method": "session/new",
                "params": {"cwd": "/tmp/caliche-acp", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("absolute session/new");
    let parsed = parse_frames(&snapshot(&buf));
    let session_id = parsed[2]["result"]["sessionId"]
        .as_str()
        .expect("sessionId");
    assert!(session_id.starts_with("sess-"));
    assert_eq!(fake.session_count(), 1);
}

#[tokio::test]
async fn tool_call_update_uses_start_id() {
    let fake = Arc::new(FakeDriver::tool_pair());
    let server = Arc::new(server_for(&fake));
    let (buf, frames) = mem_sink();
    server
        .handle_json_value(initialize_request(1), &frames)
        .await
        .expect("initialize");
    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 2,
                "method": "session/new",
                "params": {"cwd": "/tmp/caliche-acp", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("session/new");
    let session_id = parse_frames(&snapshot(&buf))[1]["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string();
    server
        .handle_json_value(prompt_request(4, &session_id), &frames)
        .await
        .expect("prompt");
    let parsed = parse_frames(&snapshot(&buf));
    let ids = tool_call_ids(&parsed);
    assert_eq!(ids, vec!["call-7".to_string(), "call-7".to_string()]);
    assert!(
        parsed
            .iter()
            .all(|frame| frame["method"] != "session/request_permission"),
        "a turn that does not require approval must not ask the client"
    );
    assert!(
        parsed
            .iter()
            .any(|frame| { frame["id"] == 4 && frame["result"]["stopReason"] == "end_turn" })
    );
}

#[tokio::test]
async fn live_turn_writes_updates_before_prompt_and_cancel_stops_it() {
    let result = tokio::time::timeout(Duration::from_secs(5), live_turn_body()).await;
    assert!(result.is_ok(), "ACP live turn timed out");
}

async fn live_turn_body() {
    let fake = Arc::new(FakeDriver::block_until_cancel());
    let server = Arc::new(server_for(&fake));
    let (buf, frames) = mem_sink();
    let (mut client, server_read) = tokio::io::duplex(65_536);
    let reader = BufReader::new(server_read);
    let serve = tokio::spawn(async move { serve_acp(server, reader, frames).await });

    client
        .write_all(line(&initialize_request(1)).as_bytes())
        .await
        .expect("write initialize");
    let started = wait_for(&buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["result"]["protocolVersion"] == 1)
    })
    .await;
    assert_eq!(started[0]["result"]["agentInfo"]["name"], "caliche");
    assert_catalog_capabilities(&started[0]["result"]);

    client
        .write_all(
            line(&json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 2,
                "method": "session/new",
                "params": {"cwd": "/tmp/caliche-acp", "mcpServers": []}
            }))
            .as_bytes(),
        )
        .await
        .expect("write session/new");
    let created = wait_for(&buf, |frames| {
        frames.iter().any(|frame| {
            frame["result"]
                .get("sessionId")
                .and_then(Value::as_str)
                .is_some()
        })
    })
    .await;
    let session_id = created
        .iter()
        .find_map(|frame| frame["result"]["sessionId"].as_str().map(str::to_string))
        .expect("sessionId");

    client
        .write_all(line(&prompt_request(3, &session_id)).as_bytes())
        .await
        .expect("write prompt");
    let mid = wait_for(&buf, |frames| {
        chunk_text(frames, "alpha") && chunk_text(frames, "beta")
    })
    .await;
    assert!(
        mid.iter()
            .all(|frame| frame["result"].get("stopReason").is_none()),
        "prompt response arrived before cancel, while the fake model was still blocked"
    );

    client
        .write_all(
            line(&json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "method": "session/cancel",
                "params": {"sessionId": session_id}
            }))
            .as_bytes(),
        )
        .await
        .expect("write cancel");
    let done = wait_for(&buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "cancelled")
    })
    .await;
    assert!(fake.captured_token().is_cancelled());
    let update_at = done
        .iter()
        .position(|frame| is_agent_chunk(frame, "alpha"))
        .expect("alpha update");
    let response_at = done
        .iter()
        .position(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "cancelled")
        .expect("prompt response");
    assert!(update_at < response_at);

    drop(client);
    let joined = tokio::time::timeout(Duration::from_secs(10), serve)
        .await
        .expect("serve did not finish");
    joined.expect("serve join").expect("serve");
}

#[tokio::test(flavor = "current_thread")]
async fn early_cancel_during_has_session_stops_the_prompt() {
    let result = tokio::time::timeout(Duration::from_secs(5), early_cancel_body()).await;
    assert!(result.is_ok(), "early session/cancel timed out");
}

async fn early_cancel_body() {
    let fake = Arc::new(FakeDriver::hold_has_session());
    let server = Arc::new(server_for(&fake));
    let (buf, frames) = mem_sink();
    let (mut client, server_read) = tokio::io::duplex(65_536);
    let reader = BufReader::new(server_read);
    let serve = tokio::spawn(async move { serve_acp(server, reader, frames).await });

    client
        .write_all(line(&initialize_request(1)).as_bytes())
        .await
        .expect("write initialize");
    let started = wait_for(&buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["result"]["protocolVersion"] == 1)
    })
    .await;
    assert_eq!(started[0]["result"]["authMethods"], json!([]));

    client
        .write_all(
            line(&json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 2,
                "method": "session/new",
                "params": {"cwd": "/tmp/caliche-acp", "mcpServers": []}
            }))
            .as_bytes(),
        )
        .await
        .expect("write session/new");
    let created = wait_for(&buf, |frames| {
        frames.iter().any(|frame| {
            frame["result"]
                .get("sessionId")
                .and_then(Value::as_str)
                .is_some()
        })
    })
    .await;
    let session_id = created
        .iter()
        .find_map(|frame| frame["result"]["sessionId"].as_str().map(str::to_string))
        .expect("sessionId");

    client
        .write_all(line(&prompt_request(3, &session_id)).as_bytes())
        .await
        .expect("write prompt");
    let entered = tokio::time::Instant::now();
    while !fake.entered_has_session.load(Ordering::Acquire) {
        if entered.elapsed() > Duration::from_secs(10) {
            panic!("prompt never reached has_session");
        }
        tokio::task::yield_now().await;
    }
    let before_cancel = String::from_utf8(snapshot(&buf)).expect("stdout utf-8");
    assert!(
        !before_cancel.contains("session/update"),
        "cancel window is before the first session/update, saw {before_cancel}"
    );

    client
        .write_all(
            line(&json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "method": "session/cancel",
                "params": {"sessionId": session_id}
            }))
            .as_bytes(),
        )
        .await
        .expect("write cancel");
    // current_thread: let the reader apply cancel while has_session is still held.
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    fake.release_has_session.notify_one();

    let done = wait_for(&buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "cancelled")
    })
    .await;
    assert!(fake.captured_token().is_cancelled());
    assert!(
        done.iter()
            .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "cancelled")
    );
    assert_json_lines(&String::from_utf8(snapshot(&buf)).expect("stdout utf-8"));

    drop(client);
    let joined = tokio::time::timeout(Duration::from_secs(10), serve)
        .await
        .expect("serve did not finish");
    joined.expect("serve join").expect("serve");
}

#[tokio::test(flavor = "current_thread")]
async fn piped_cancel_before_claim_stops_the_prompt() {
    let result = tokio::time::timeout(Duration::from_secs(5), piped_cancel_body()).await;
    assert!(result.is_ok(), "piped session/cancel timed out");
}

async fn piped_cancel_body() {
    let fake = Arc::new(FakeDriver::finish_when_cancelled());
    let server = Arc::new(server_for(&fake));
    let (buf, frames) = mem_sink();
    let (mut client, server_read) = tokio::io::duplex(65_536);
    let reader = BufReader::new(server_read);
    let serve = tokio::spawn(async move { serve_acp(server, reader, frames).await });

    client
        .write_all(line(&initialize_request(1)).as_bytes())
        .await
        .expect("write initialize");
    wait_for(&buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["result"]["protocolVersion"] == 1)
    })
    .await;

    client
        .write_all(
            line(&json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 2,
                "method": "session/new",
                "params": {"cwd": "/tmp/caliche-acp", "mcpServers": []}
            }))
            .as_bytes(),
        )
        .await
        .expect("write session/new");
    let created = wait_for(&buf, |frames| {
        frames.iter().any(|frame| {
            frame["result"]
                .get("sessionId")
                .and_then(Value::as_str)
                .is_some()
        })
    })
    .await;
    let session_id = created
        .iter()
        .find_map(|frame| frame["result"]["sessionId"].as_str().map(str::to_string))
        .expect("sessionId");

    let mut piped = line(&prompt_request(3, &session_id));
    piped.push_str(&line(&json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "method": "session/cancel",
        "params": {"sessionId": session_id}
    })));
    client
        .write_all(piped.as_bytes())
        .await
        .expect("write prompt and cancel");

    wait_for(&buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "cancelled")
    })
    .await;
    assert!(fake.captured_token().is_cancelled());
    assert_json_lines(&String::from_utf8(snapshot(&buf)).expect("stdout utf-8"));

    drop(client);
    let joined = tokio::time::timeout(Duration::from_secs(10), serve)
        .await
        .expect("serve did not finish");
    joined.expect("serve join").expect("serve");
}

fn prompt_request(id: u64, session_id: &str) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "method": "session/prompt",
        "params": {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "hello"}]
        }
    })
}

fn chunk_text(frames: &[Value], text: &str) -> bool {
    frames.iter().any(|frame| is_agent_chunk(frame, text))
}

fn is_agent_chunk(frame: &Value, text: &str) -> bool {
    frame.get("method").and_then(Value::as_str) == Some("session/update")
        && frame
            .pointer("/params/update/content/text")
            .and_then(Value::as_str)
            == Some(text)
}

fn tool_call_ids(frames: &[Value]) -> Vec<String> {
    frames
        .iter()
        .filter_map(|frame| {
            let update = frame.pointer("/params/update")?;
            let kind = update.get("sessionUpdate")?.as_str()?;
            if kind != "tool_call" && kind != "tool_call_update" {
                return None;
            }
            Some(update.get("toolCallId")?.as_str()?.to_string())
        })
        .collect()
}

async fn wait_for(buf: &Arc<Mutex<Vec<u8>>>, pred: impl Fn(&[Value]) -> bool) -> Vec<Value> {
    let start = tokio::time::Instant::now();
    loop {
        let bytes = snapshot(buf);
        if !bytes.is_empty()
            && let Ok(text) = std::str::from_utf8(&bytes)
            && text.ends_with('\n')
        {
            let frames = parse_frames(&bytes);
            if pred(&frames) {
                return frames;
            }
        }
        if start.elapsed() > Duration::from_secs(3) {
            panic!(
                "timed out waiting for ACP stdout: {}",
                String::from_utf8_lossy(&snapshot(buf))
            );
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
