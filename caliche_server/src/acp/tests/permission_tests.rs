//! Interactive approval over `session/request_permission`.
//!
//! The fake client answers on the stdio loop. The tool body runs only after
//! `allow-once`. Timeout is 80ms here; production is
//! [`crate::acp::ACP_PERMISSION_TIMEOUT`] (60 seconds).

use super::{
    FakeDriver, assert_catalog_capabilities, initialize_request, line, mem_sink, parse_frames,
    prompt_request, server_for_timeout, snapshot, wait_for,
};
use crate::acp::stdio::serve_acp;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::task::JoinHandle;

const PERMISSION_WAIT: Duration = Duration::from_millis(80);

fn allow_once(id: &Value) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "result": {"outcome": {"outcome": "selected", "optionId": "allow-once"}}
    })
}

fn selected(id: &Value, option_id: &str) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "result": {"outcome": {"outcome": "selected", "optionId": option_id}}
    })
}

fn cancelled(id: &Value) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "result": {"outcome": {"outcome": "cancelled"}}
    })
}

fn malformed(id: &Value) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "result": {"outcome": "allow"}
    })
}

fn rpc_error(id: &Value) -> Value {
    json!({
        "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
        "id": id,
        "error": {"code": -32603, "message": "no"}
    })
}

struct Running {
    buf: Arc<Mutex<Vec<u8>>>,
    client: tokio::io::DuplexStream,
    fake: Arc<FakeDriver>,
    serve: JoinHandle<anyhow::Result<()>>,
    session_id: String,
}

async fn start_turn(timeout: Duration) -> Running {
    start_permission_turn(timeout, "call-7").await
}

async fn start_permission_turn(timeout: Duration, call_id: &str) -> Running {
    let fake = Arc::new(FakeDriver::ask_permission().with_permission_call_id(call_id));
    let server = Arc::new(server_for_timeout(&fake, timeout));
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
    assert_catalog_capabilities(&started[0]["result"]);
    assert_eq!(started[0]["result"]["authMethods"], json!([]));
    assert_eq!(started[0]["result"]["protocolVersion"], 1);

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
        frames
            .iter()
            .any(|frame| frame["result"]["sessionId"].as_str().is_some())
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

    Running {
        buf,
        client,
        fake,
        serve,
        session_id,
    }
}

fn permission_frame(frames: &[Value]) -> &Value {
    frames
        .iter()
        .find(|frame| frame["method"] == "session/request_permission")
        .expect("permission request")
}

fn assert_permission_shape(frame: &Value, session_id: &str) {
    assert_eq!(frame["params"]["sessionId"], session_id);
    assert_eq!(frame["params"]["toolCall"]["toolCallId"], "call-7");
    assert_eq!(frame["params"]["toolCall"]["title"], "bash");
    assert_eq!(
        frame["params"]["toolCall"]["rawInput"]["command"],
        "echo hi"
    );
    let text = frame["params"]["toolCall"]["content"][0]["content"]["text"]
        .as_str()
        .expect("summary");
    assert!(
        text.contains("bash"),
        "summary should name the tool, got {text}"
    );
    let options = frame["params"]["options"].as_array().expect("options");
    assert_eq!(options.len(), 2);
    assert!(options.iter().any(|option| option["kind"] == "allow_once"));
    assert!(options.iter().any(|option| option["kind"] == "reject_once"));
    assert!(
        options
            .iter()
            .all(|option| option["kind"] != "allow_always" && option["kind"] != "reject_always")
    );
}

fn tool_ran_in(frames: &[Value]) -> bool {
    frames.iter().any(|frame| {
        frame.pointer("/params/update/content/0/content/text") == Some(&json!("tool-ran"))
    })
}

async fn finish(running: Running) {
    drop(running.client);
    let joined = tokio::time::timeout(Duration::from_secs(10), running.serve)
        .await
        .expect("serve did not finish");
    joined.expect("serve join").expect("serve");
}

/// Attach a permission result to the outstanding request.
///
/// A null id is filled with the real request id. An id that is already set
/// and does not match is left alone: stamping the real id onto a foreign
/// allow would hide a broker that completed the wrong wait.
fn bind_permission_response(request_id: &Value, mut response: Value) -> Value {
    let Some(object) = response.as_object_mut() else {
        return response;
    };
    let foreign = match object.get("id") {
        Some(Value::Null) | None => false,
        Some(id) => id != request_id,
    };
    if !foreign {
        object.insert("id".to_owned(), request_id.clone());
    }
    response
}

async fn reply_and_finish(mut running: Running, response: Value) {
    let asked = wait_for(&running.buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["method"] == "session/request_permission")
    })
    .await;
    let request = permission_frame(&asked);
    assert_permission_shape(request, &running.session_id);
    let same_id = asked.iter().any(|frame| {
        frame["method"] == "session/update"
            && frame
                .pointer("/params/update/toolCallId")
                .and_then(Value::as_str)
                == Some("call-7")
    });
    assert!(
        same_id,
        "permission toolCallId must match the tool_call update already on the wire"
    );
    let id = request["id"].clone();
    let response = bind_permission_response(&id, response);
    running
        .client
        .write_all(line(&response).as_bytes())
        .await
        .expect("write permission result");
    wait_for(&running.buf, |frames| {
        frames
            .iter()
            .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"].is_string())
    })
    .await;
    finish(running).await;
}

#[tokio::test]
async fn allow_once_runs_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        let buf = Arc::clone(&running.buf);
        reply_and_finish(running, allow_once(&Value::Null)).await;
        assert!(fake.tool_ran(), "allow must run the tool");
        let frames = parse_frames(&snapshot(&buf));
        assert!(tool_ran_in(&frames), "allow must emit the tool result");
    })
    .await;
    assert!(result.is_ok(), "allow permission test timed out");
}

#[tokio::test]
async fn reject_once_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        let buf = Arc::clone(&running.buf);
        reply_and_finish(running, selected(&Value::Null, "reject-once")).await;
        assert!(!fake.tool_ran());
        assert!(!tool_ran_in(&parse_frames(&snapshot(&buf))));
    })
    .await;
    assert!(result.is_ok(), "reject permission test timed out");
}

#[tokio::test]
async fn cancelled_outcome_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        reply_and_finish(running, cancelled(&Value::Null)).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "cancelled permission test timed out");
}

#[tokio::test]
async fn malformed_result_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        reply_and_finish(running, malformed(&Value::Null)).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "malformed permission test timed out");
}

#[tokio::test]
async fn unknown_option_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        reply_and_finish(running, selected(&Value::Null, "allow-always")).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "unknown option test timed out");
}

#[tokio::test]
async fn rpc_error_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        reply_and_finish(running, rpc_error(&Value::Null)).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "rpc error permission test timed out");
}

#[tokio::test]
async fn timeout_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(PERMISSION_WAIT).await;
        let fake = Arc::clone(&running.fake);
        let buf = Arc::clone(&running.buf);
        wait_for(&buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["method"] == "session/request_permission")
        })
        .await;
        wait_for(&buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "end_turn")
        })
        .await;
        assert!(!fake.tool_ran(), "timeout must fail closed");
        assert!(!tool_ran_in(&parse_frames(&snapshot(&buf))));
        finish(running).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "timeout permission test timed out");
}

#[tokio::test]
async fn late_allow_after_timeout_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let mut running = start_turn(PERMISSION_WAIT).await;
        let fake = Arc::clone(&running.fake);
        let asked = wait_for(&running.buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["method"] == "session/request_permission")
        })
        .await;
        let id = permission_frame(&asked)["id"].clone();
        wait_for(&running.buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "end_turn")
        })
        .await;
        assert!(!fake.tool_ran());
        running
            .client
            .write_all(line(&allow_once(&id)).as_bytes())
            .await
            .expect("late allow");
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            !fake.tool_ran(),
            "a result after the timeout must not run the tool"
        );
        finish(running).await;
    })
    .await;
    assert!(result.is_ok(), "late allow test timed out");
}

#[tokio::test]
async fn disconnect_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        wait_for(&running.buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["method"] == "session/request_permission")
        })
        .await;
        finish(running).await;
        assert!(!fake.tool_ran(), "disconnect must fail closed");
    })
    .await;
    assert!(result.is_ok(), "disconnect permission test timed out");
}

#[tokio::test]
async fn foreign_request_id_does_not_allow_the_pending_wait() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let mut running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        let buf = Arc::clone(&running.buf);
        let asked = wait_for(&running.buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["method"] == "session/request_permission")
        })
        .await;
        let real_id = permission_frame(&asked)["id"].clone();
        let foreign = bind_permission_response(
            &real_id,
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": "perm-other-session",
                "result": {
                    "sessionId": "sess-foreign",
                    "outcome": {"outcome": "selected", "optionId": "allow-once"}
                }
            }),
        );
        assert_eq!(
            foreign["id"], "perm-other-session",
            "a foreign permission id must not be rewritten onto the real wait"
        );
        running
            .client
            .write_all(line(&foreign).as_bytes())
            .await
            .expect("write foreign permission result");
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            !fake.tool_ran(),
            "a result for another request id or session must not run the tool"
        );
        assert!(
            parse_frames(&snapshot(&buf))
                .iter()
                .all(|frame| frame["id"] != 3 || frame["result"]["stopReason"].is_null()),
            "the real wait must still be pending after a mismatched allow"
        );

        let reject = bind_permission_response(&real_id, selected(&Value::Null, "reject-once"));
        assert_eq!(reject["id"], real_id);
        running
            .client
            .write_all(line(&reject).as_bytes())
            .await
            .expect("write real reject");
        wait_for(&buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"].is_string())
        })
        .await;
        assert!(
            !fake.tool_ran(),
            "the mismatched allow must not overwrite the real wait"
        );
        assert!(!tool_ran_in(&parse_frames(&snapshot(&buf))));
        finish(running).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "foreign permission id test timed out");
}

#[tokio::test]
async fn cancel_during_permission_wait_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let mut running = start_turn(Duration::from_secs(10)).await;
        let fake = Arc::clone(&running.fake);
        let buf = Arc::clone(&running.buf);
        let session_id = running.session_id.clone();
        let asked = wait_for(&running.buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["method"] == "session/request_permission")
        })
        .await;
        let real_id = permission_frame(&asked)["id"].clone();
        running
            .client
            .write_all(
                line(&json!({
                    "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                    "method": "session/cancel",
                    "params": {"sessionId": session_id}
                }))
                .as_bytes(),
            )
            .await
            .expect("write session/cancel");
        wait_for(&buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"] == "cancelled")
        })
        .await;
        assert!(
            !fake.tool_ran(),
            "session/cancel during the permission wait must not run the tool"
        );
        assert!(!tool_ran_in(&parse_frames(&snapshot(&buf))));

        // A late allow for the same request must not revive the cancelled wait.
        running
            .client
            .write_all(line(&allow_once(&real_id)).as_bytes())
            .await
            .expect("write late allow");
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        assert!(
            !fake.tool_ran(),
            "allow after session/cancel must not run the tool"
        );
        finish(running).await;
    })
    .await;
    assert!(result.is_ok(), "cancel during permission wait timed out");
}

#[tokio::test]
async fn empty_tool_call_id_does_not_run_the_tool() {
    let result = tokio::time::timeout(Duration::from_secs(15), async {
        let running = start_permission_turn(Duration::from_secs(10), "").await;
        let fake = Arc::clone(&running.fake);
        let buf = Arc::clone(&running.buf);
        wait_for(&buf, |frames| {
            frames
                .iter()
                .any(|frame| frame["id"] == 3 && frame["result"]["stopReason"].is_string())
        })
        .await;
        let frames = parse_frames(&snapshot(&buf));
        assert!(
            frames
                .iter()
                .all(|frame| frame["method"] != "session/request_permission"),
            "an empty tool call id must not ask the client"
        );
        assert!(
            !fake.tool_ran(),
            "an empty tool call id must not run the tool"
        );
        assert!(!tool_ran_in(&frames));
        finish(running).await;
        assert!(!fake.tool_ran());
    })
    .await;
    assert!(result.is_ok(), "empty tool call id test timed out");
}

#[tokio::test]
#[serial_test::serial]
async fn installing_the_resolver_keeps_the_permission_mode() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("OPENROUTER_API_KEY", "test-key-for-acp-permission");
    let outcome = resolver_does_not_force_interactive().await;
    outcome.expect("resolver install");
}

async fn resolver_does_not_force_interactive() -> anyhow::Result<()> {
    use crate::acp::driver::{install_acp_approval_resolver, restore_acp_approval_resolver};
    use crate::acp::permission::{AcpPermissionGate, PermissionBroker};
    use caliche::agent::hooks::PermissionMode;
    use caliche::harness::{HarnessConfig, SessionConfig};
    use tokio_util::sync::CancellationToken;

    let adapter = Arc::new(crate::adapter::ApiAdapter::with_session_ttl(
        HarnessConfig::default(),
        Duration::from_secs(60),
        None,
    ));
    let session_id = adapter
        .create_session(
            Some(SessionConfig {
                session_id: None,
                workspace: Some("/tmp/caliche-acp-perm".into()),
                context_files: Vec::new(),
                system_prompt_addition: None,
                max_turns: None,
                history: Vec::new(),
            }),
            None,
            None,
        )
        .await?;
    {
        let session = adapter.get_session(&session_id).await.expect("session");
        let state = session.write().await;
        let mut harness = state.harness.write().await;
        harness
            .service_mut()
            .agent_mut()
            .set_permission_mode(PermissionMode::AcceptAll);
    }
    let broker = Arc::new(PermissionBroker::new(Duration::from_millis(80)));
    let (_, frames) = mem_sink();
    let gate = AcpPermissionGate::new(broker, frames, session_id.clone(), CancellationToken::new());
    let previous = install_acp_approval_resolver(&adapter, &session_id, &gate).await?;
    assert!(previous.is_none(), "a fresh session has no resolver");
    {
        let session = adapter.get_session(&session_id).await.expect("session");
        let state = session.read().await;
        let harness = state.harness.read().await;
        let agent = harness.service().agent();
        assert!(agent.has_approval_resolver());
        assert_eq!(agent.permission_mode(), PermissionMode::AcceptAll);
    }
    restore_acp_approval_resolver(&adapter, &session_id, previous).await;
    let session = adapter.get_session(&session_id).await.expect("session");
    let state = session.read().await;
    let harness = state.harness.read().await;
    let agent = harness.service().agent();
    assert!(!agent.has_approval_resolver());
    assert_eq!(agent.permission_mode(), PermissionMode::AcceptAll);
    Ok(())
}
