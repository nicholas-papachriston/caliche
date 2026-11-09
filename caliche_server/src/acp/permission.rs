//! ACP `session/request_permission` for tool calls the engine will not run
//! until someone approves them.
//!
//! The stdio agent sends one JSON-RPC request and waits
//! [`ACP_PERMISSION_TIMEOUT`] (60 seconds). The tool runs only when the client
//! selects `allow-once`. Reject, cancel, an unknown option, a JSON-RPC error,
//! a result that does not match the ACP outcome shape, a timeout, or a closed
//! stdin all skip the tool. `allow-always` is not offered: this path does not
//! persist permission rules.
//!
//! Modes that never return `RequireApproval` (`Default`, `AcceptAll`,
//! `DenyAll`) do not send this request. `AcceptEdits` still does for tools the
//! engine already refuses to run without approval.

use crate::acp::mapping::tool_kind_for_name;
use crate::acp::outbound::AcpFrameSink;
use agent_client_protocol_schema::v1::{
    ContentBlock, PermissionOption, PermissionOptionKind, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SessionId, TextContent, ToolCallContent,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use async_trait::async_trait;
use caliche::agent::hooks::{ApprovalOutcome, ApprovalResolver};
use caliche_contracts::acp::methods;
use futures_util::future::BoxFuture;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Production wait for `session/request_permission`.
///
/// Measured from when the request frame is written. A late result after this
/// elapses is ignored.
pub const ACP_PERMISSION_TIMEOUT: Duration = Duration::from_secs(60);

/// Option id the client must select for the tool to run.
pub const PERMISSION_ALLOW_ONCE: &str = "allow-once";

/// Option id that skips the tool.
pub const PERMISSION_REJECT_ONCE: &str = "reject-once";

const SUMMARY_CHARS: usize = 400;

/// One in-flight approval, already tied to a tool call id.
#[derive(Debug, Clone)]
pub struct PermissionPrompt {
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: Value,
    pub reason: String,
}

enum WaitReply {
    Payload(Value),
}

/// Correlates agent-originated permission request ids with client results.
pub(crate) struct PermissionBroker {
    next_id: AtomicU64,
    pending: Mutex<HashMap<String, oneshot::Sender<WaitReply>>>,
    timeout: Duration,
    closed: AtomicBool,
}

impl PermissionBroker {
    pub(crate) fn new(timeout: Duration) -> Self {
        Self {
            next_id: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            timeout,
            closed: AtomicBool::new(false),
        }
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    fn allocate_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("perm-{n}")
    }

    fn insert(&self, id: String, sender: oneshot::Sender<WaitReply>) {
        lock_mutex(&self.pending).insert(id, sender);
    }

    fn take(&self, id: &str) -> Option<oneshot::Sender<WaitReply>> {
        lock_mutex(&self.pending).remove(id)
    }

    /// Stdin closed. In-flight waits end, and later asks do not hit the wire.
    pub(crate) fn fail_closed(&self) {
        self.closed.store(true, Ordering::Release);
        let pending = std::mem::take(&mut *lock_mutex(&self.pending));
        drop(pending);
    }

    pub(crate) fn complete_inbound(&self, value: &Value) {
        let Some(id) = value.get("id").and_then(id_key) else {
            return;
        };
        let Some(sender) = self.take(&id) else {
            return;
        };
        let jsonrpc_ok = value.get("jsonrpc").and_then(Value::as_str)
            == Some(caliche_contracts::rpc::JSONRPC_VERSION);
        if !jsonrpc_ok || value.get("error").is_some() {
            drop(sender);
            return;
        }
        let Some(result) = value.get("result").cloned() else {
            drop(sender);
            return;
        };
        let _ = sender.send(WaitReply::Payload(result));
    }
}

/// Per-turn handle the approval resolver uses to ask the ACP client.
#[derive(Clone)]
pub struct AcpPermissionGate {
    broker: Arc<PermissionBroker>,
    frames: Arc<dyn AcpFrameSink>,
    session_id: String,
    cancel: CancellationToken,
}

impl AcpPermissionGate {
    pub(crate) fn new(
        broker: Arc<PermissionBroker>,
        frames: Arc<dyn AcpFrameSink>,
        session_id: String,
        cancel: CancellationToken,
    ) -> Self {
        Self {
            broker,
            frames,
            session_id,
            cancel,
        }
    }

    pub(crate) async fn decide(&self, prompt: PermissionPrompt) -> ApprovalOutcome {
        if prompt.tool_call_id.is_empty() {
            log_deny(&self.session_id, &prompt.tool_name, "empty-tool-call-id");
            return ApprovalOutcome::Deny;
        }
        if self.broker.is_closed() {
            log_deny(&self.session_id, &prompt.tool_name, "disconnected");
            return ApprovalOutcome::Deny;
        }

        let id = self.broker.allocate_id();
        let (sender, receiver) = oneshot::channel();
        self.broker.insert(id.clone(), sender);
        let Some(frame) = permission_frame(&id, &self.session_id, &prompt) else {
            self.broker.take(&id);
            log_deny(&self.session_id, &prompt.tool_name, "encode");
            return ApprovalOutcome::Deny;
        };
        if self.frames.write_frame(&frame).is_err() {
            self.broker.take(&id);
            log_deny(&self.session_id, &prompt.tool_name, "write");
            return ApprovalOutcome::Deny;
        }

        let timeout = self.broker.timeout;
        tokio::select! {
            biased;
            () = self.cancel.cancelled() => {
                self.broker.take(&id);
                log_deny(&self.session_id, &prompt.tool_name, "cancelled");
                ApprovalOutcome::Deny
            }
            reply = receiver => {
                match reply {
                    Ok(WaitReply::Payload(value)) => {
                        outcome_from_result(&value, &self.session_id, &prompt.tool_name)
                    }
                    Err(_) => {
                        log_deny(&self.session_id, &prompt.tool_name, "disconnected");
                        ApprovalOutcome::Deny
                    }
                }
            }
            () = tokio::time::sleep(timeout) => {
                self.broker.take(&id);
                log_deny(&self.session_id, &prompt.tool_name, "timeout");
                ApprovalOutcome::Deny
            }
        }
    }
}

/// Resolver installed on an ACP session for the duration of one turn.
pub(crate) struct AcpApprovalResolver {
    gate: AcpPermissionGate,
}

impl AcpApprovalResolver {
    pub(crate) fn new(gate: AcpPermissionGate) -> Self {
        Self { gate }
    }
}

#[async_trait]
impl ApprovalResolver for AcpApprovalResolver {
    fn resolve_approval(
        &self,
        _tool_name: &str,
        _tool_input: &Value,
        _reason: &str,
        _proposed_amendment: Option<&[String]>,
    ) -> BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
        // No stable tool-call id is available on this entry point.
        Box::pin(futures_util::future::ready(Ok(ApprovalOutcome::Deny)))
    }

    fn resolve_approval_for_call<'a>(
        &'a self,
        tool_name: &'a str,
        tool_input: &'a Value,
        reason: &'a str,
        proposed_amendment: Option<&'a [String]>,
        call_id: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<ApprovalOutcome>> {
        let _ = proposed_amendment;
        if call_id.is_empty() {
            return Box::pin(futures_util::future::ready(Ok(ApprovalOutcome::Deny)));
        }
        let gate = self.gate.clone();
        let prompt = PermissionPrompt {
            tool_call_id: call_id.to_owned(),
            tool_name: tool_name.to_owned(),
            arguments: tool_input.clone(),
            reason: reason.to_owned(),
        };
        Box::pin(async move { Ok(gate.decide(prompt).await) })
    }
}

pub(crate) fn is_client_response(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object.contains_key("method") {
        return false;
    }
    object.contains_key("id") && (object.contains_key("result") || object.contains_key("error"))
}

fn permission_frame(id: &str, session_id: &str, prompt: &PermissionPrompt) -> Option<Value> {
    let summary = permission_summary(&prompt.tool_name, &prompt.arguments, &prompt.reason);
    let params = RequestPermissionRequest::new(
        SessionId::new(session_id),
        ToolCallUpdate::new(
            prompt.tool_call_id.clone(),
            ToolCallUpdateFields::new()
                .title(prompt.tool_name.clone())
                .kind(tool_kind_for_name(&prompt.tool_name))
                .status(ToolCallStatus::Pending)
                .content(vec![ToolCallContent::from(ContentBlock::Text(
                    TextContent::new(summary),
                ))])
                .raw_input(prompt.arguments.clone()),
        ),
        vec![
            PermissionOption::new(
                PERMISSION_ALLOW_ONCE,
                "Allow once",
                PermissionOptionKind::AllowOnce,
            ),
            PermissionOption::new(
                PERMISSION_REJECT_ONCE,
                "Reject once",
                PermissionOptionKind::RejectOnce,
            ),
        ],
    );
    let params = serde_json::to_value(params).ok()?;
    Some(json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "method": methods::SESSION_REQUEST_PERMISSION,
        "params": params,
    }))
}

fn permission_summary(tool_name: &str, arguments: &Value, reason: &str) -> String {
    let args = truncate_chars(&arguments.to_string(), SUMMARY_CHARS);
    if reason.is_empty() {
        format!("{tool_name} {args}")
    } else {
        format!("{reason}\n{tool_name} {args}")
    }
}

fn outcome_from_result(value: &Value, session_id: &str, tool_name: &str) -> ApprovalOutcome {
    let Ok(parsed) = serde_json::from_value::<RequestPermissionResponse>(value.clone()) else {
        log_deny(session_id, tool_name, "malformed");
        return ApprovalOutcome::Deny;
    };
    match parsed.outcome {
        RequestPermissionOutcome::Selected(selected)
            if selected.option_id.0.as_ref() == PERMISSION_ALLOW_ONCE =>
        {
            ApprovalOutcome::Allow
        }
        RequestPermissionOutcome::Selected(_) | RequestPermissionOutcome::Cancelled => {
            log_deny(session_id, tool_name, "rejected");
            ApprovalOutcome::Deny
        }
        _ => {
            log_deny(session_id, tool_name, "rejected");
            ApprovalOutcome::Deny
        }
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut end = text.len();
    for (count, (index, _)) in text.char_indices().enumerate() {
        if count == max_chars {
            end = index;
            break;
        }
    }
    if end == text.len() {
        return text.to_owned();
    }
    let mut out = text[..end].to_owned();
    out.push_str("...");
    out
}

fn id_key(id: &Value) -> Option<String> {
    match id {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

fn log_deny(session_id: &str, tool_name: &str, why: &str) {
    tracing::debug!(
        target: "caliche_server::acp",
        session_id,
        tool = tool_name,
        reason = why,
        "acp permission denied; tool will not run"
    );
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|err| err.into_inner())
}

#[cfg(test)]
mod tests {
    use super::{
        AcpApprovalResolver, AcpPermissionGate, PERMISSION_ALLOW_ONCE, PermissionBroker,
        PermissionPrompt, WaitReply, outcome_from_result, permission_frame, truncate_chars,
    };
    use caliche::agent::hooks::ApprovalResolver;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tokio::sync::oneshot;
    use tokio_util::sync::CancellationToken;

    fn prompt() -> PermissionPrompt {
        PermissionPrompt {
            tool_call_id: "call-7".to_owned(),
            tool_name: "bash".to_owned(),
            arguments: json!({"command": "echo hi\nthere"}),
            reason: "PermissionMode::Interactive requires approval for 'bash'".to_owned(),
        }
    }

    #[test]
    fn frame_carries_call_id_args_and_only_once_options() {
        let frame = permission_frame("perm-1", "sess-1", &prompt()).expect("frame");
        assert_eq!(frame["method"], "session/request_permission");
        assert_eq!(frame["id"], "perm-1");
        assert_eq!(frame["params"]["sessionId"], "sess-1");
        assert_eq!(frame["params"]["toolCall"]["toolCallId"], "call-7");
        assert_eq!(frame["params"]["toolCall"]["title"], "bash");
        assert_eq!(
            frame["params"]["toolCall"]["rawInput"]["command"],
            "echo hi\nthere"
        );
        let options = frame["params"]["options"].as_array().expect("options");
        assert_eq!(options.len(), 2);
        assert_eq!(options[0]["optionId"], PERMISSION_ALLOW_ONCE);
        assert_eq!(options[0]["kind"], "allow_once");
        assert_eq!(options[1]["kind"], "reject_once");
        let bytes = serde_json::to_vec(&frame).expect("bytes");
        assert!(
            !bytes.contains(&b'\n'),
            "permission frame must stay one JSON line"
        );
    }

    #[test]
    fn only_allow_once_proceeds() {
        let allow = json!({"outcome": {"outcome": "selected", "optionId": "allow-once"}});
        assert!(matches!(
            outcome_from_result(&allow, "s", "bash"),
            caliche::agent::hooks::ApprovalOutcome::Allow
        ));
        for denied in [
            json!({"outcome": {"outcome": "selected", "optionId": "reject-once"}}),
            json!({"outcome": {"outcome": "selected", "optionId": "allow-always"}}),
            json!({"outcome": {"outcome": "cancelled"}}),
            json!({"outcome": "allow"}),
            json!({}),
        ] {
            assert!(
                matches!(
                    outcome_from_result(&denied, "s", "bash"),
                    caliche::agent::hooks::ApprovalOutcome::Deny
                ),
                "{denied}"
            );
        }
    }

    #[test]
    fn truncate_chars_keeps_a_char_boundary() {
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abcdef", 3), "abc...");
    }

    struct CaptureSink(Mutex<Vec<Value>>);

    impl crate::acp::outbound::AcpFrameSink for CaptureSink {
        fn write_frame(&self, frame: &Value) -> std::io::Result<()> {
            self.0.lock().expect("frames").push(frame.clone());
            Ok(())
        }
    }

    fn gate(timeout: Duration, cancel: CancellationToken) -> (AcpPermissionGate, Arc<CaptureSink>) {
        let sink = Arc::new(CaptureSink(Mutex::new(Vec::new())));
        let frames: Arc<dyn crate::acp::outbound::AcpFrameSink> = sink.clone();
        let gate = AcpPermissionGate::new(
            Arc::new(PermissionBroker::new(timeout)),
            frames,
            "sess-1".to_owned(),
            cancel,
        );
        (gate, sink)
    }

    #[tokio::test]
    async fn foreign_id_does_not_complete_another_wait() {
        let broker = PermissionBroker::new(Duration::from_secs(30));
        let (sender, mut receiver) = oneshot::channel();
        broker.insert("perm-1".to_owned(), sender);
        broker.complete_inbound(&json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "id": "perm-other-session",
            "result": {
                "sessionId": "sess-foreign",
                "outcome": {"outcome": "selected", "optionId": "allow-once"}
            }
        }));
        match receiver.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => {}
            Ok(WaitReply::Payload(value)) => {
                panic!("foreign id delivered a result to the pending wait: {value}");
            }
            Err(err) => panic!("foreign id dropped the pending wait: {err}"),
        }

        broker.complete_inbound(&json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "id": "perm-1",
            "result": {"outcome": {"outcome": "selected", "optionId": "reject-once"}}
        }));
        let WaitReply::Payload(value) = receiver.try_recv().expect("real id still pending");
        assert!(
            matches!(
                outcome_from_result(&value, "s", "bash"),
                caliche::agent::hooks::ApprovalOutcome::Deny
            ),
            "the earlier allow must not overwrite the real wait, got {value}"
        );
    }

    #[tokio::test]
    async fn empty_tool_call_id_denies_without_asking() {
        let (gate, sink) = gate(Duration::from_secs(30), CancellationToken::new());
        let outcome = gate
            .decide(PermissionPrompt {
                tool_call_id: String::new(),
                tool_name: "bash".to_owned(),
                arguments: json!({"command": "echo hi"}),
                reason: "needs approval".to_owned(),
            })
            .await;
        assert!(
            matches!(outcome, caliche::agent::hooks::ApprovalOutcome::Deny),
            "decide() must not Allow an empty tool call id"
        );
        assert!(
            sink.0.lock().expect("frames").is_empty(),
            "empty tool call id must not write session/request_permission"
        );

        let resolver = AcpApprovalResolver::new(gate);
        let outcome = resolver
            .resolve_approval_for_call(
                "bash",
                &json!({"command": "echo hi"}),
                "needs approval",
                None,
                "",
            )
            .await
            .expect("resolve");
        assert!(
            matches!(outcome, caliche::agent::hooks::ApprovalOutcome::Deny),
            "resolve_approval_for_call must not Allow an empty call id"
        );
    }

    #[tokio::test]
    async fn cancelled_wait_denies_instead_of_allow() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let (gate, _) = gate(Duration::from_secs(30), cancel);
        let outcome = gate.decide(prompt()).await;
        assert!(
            matches!(outcome, caliche::agent::hooks::ApprovalOutcome::Deny),
            "a cancelled permission wait must not Allow"
        );
    }
}
