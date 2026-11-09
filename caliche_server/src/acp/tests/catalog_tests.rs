//! ACP session/list and session/load against the live session cache.

use super::{
    FakeDriver, agent_info, assert_catalog_capabilities, initialize_request, mem_sink,
    parse_frames, server_for, snapshot,
};
use crate::acp::driver::AcpTurnDriver;
use crate::acp::outbound::AcpFrameSink;
use crate::acp::server::AcpServer;
use caliche::messages::Message;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn history_message(value: Value) -> Message {
    serde_json::from_value(value).expect("message")
}

fn replay_history() -> Vec<Message> {
    vec![
        history_message(json!({"role": "system", "content": "hidden system"})),
        history_message(json!({"role": "user", "content": "What's the capital of France?"})),
        history_message(json!({
            "role": "assistant",
            "content": "The capital of France is Paris.",
            "metadata": {"reasoning_content": "recall"}
        })),
        history_message(json!({
            "role": "assistant",
            "content": {
                "text": "",
                "tool_calls": [{
                    "id": "call-7",
                    "name": "fs",
                    "arguments": {"op": "read"}
                }]
            }
        })),
        history_message(json!({
            "role": "tool",
            "content": {"text": "file body", "tool_call_id": "call-7"}
        })),
    ]
}

#[tokio::test]
async fn session_list_and_load_replay_history_before_result() {
    let fake = Arc::new(FakeDriver::tool_pair());
    let server = server_for(&fake);
    let (buf, frames) = mem_sink();
    server
        .handle_json_value(initialize_request(1), &frames)
        .await
        .expect("initialize");
    assert_catalog_capabilities(&parse_frames(&snapshot(&buf))[0]["result"]);

    let first = open_session(&server, &buf, &frames, 2, "/tmp/caliche-acp-a").await;
    let second = open_session(&server, &buf, &frames, 3, "/tmp/caliche-acp-b").await;
    fake.seed_transcript(&first, "France", replay_history());

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 10,
                "method": "session/list",
                "params": {}
            }),
            &frames,
        )
        .await
        .expect("list");
    let listed = parse_frames(&snapshot(&buf))
        .into_iter()
        .find(|frame| frame["id"] == 10)
        .expect("list result");
    let sessions = listed["result"]["sessions"].as_array().expect("sessions");
    assert_eq!(sessions.len(), 2);
    assert_eq!(sessions[0]["sessionId"], first);
    assert_eq!(sessions[0]["cwd"], "/tmp/caliche-acp-a");
    assert_eq!(sessions[0]["title"], "France");
    assert_eq!(sessions[0]["updatedAt"], "2026-01-02T03:04:05Z");
    assert!(sessions[1].get("title").is_none());
    assert_eq!(sessions[1]["sessionId"], second);
    assert!(listed["result"].get("nextCursor").is_none());

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 11,
                "method": "session/list",
                "params": {"cwd": "/tmp/caliche-acp-a/"}
            }),
            &frames,
        )
        .await
        .expect("filtered list");
    let filtered = result_frame(&buf, 11);
    assert_eq!(
        filtered["result"]["sessions"]
            .as_array()
            .expect("rows")
            .len(),
        1
    );
    assert_eq!(filtered["result"]["sessions"][0]["sessionId"], first);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 12,
                "method": "session/list",
                "params": {"cwd": "relative"}
            }),
            &frames,
        )
        .await
        .expect("relative list");
    assert_eq!(result_frame(&buf, 12)["error"]["code"], -32602);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 13,
                "method": "session/list",
                "params": {"cursor": "not-a-cursor"}
            }),
            &frames,
        )
        .await
        .expect("bad cursor");
    assert_eq!(result_frame(&buf, 13)["error"]["code"], -32602);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 14,
                "method": "session/load",
                "params": {"sessionId": "missing", "cwd": "/tmp/caliche-acp-a", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("missing load");
    assert_eq!(result_frame(&buf, 14)["error"]["code"], -32602);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 15,
                "method": "session/load",
                "params": {"sessionId": first, "cwd": "relative", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("relative load");
    assert_eq!(result_frame(&buf, 15)["error"]["code"], -32602);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 16,
                "method": "session/load",
                "params": {"sessionId": first, "cwd": "/tmp/other", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("cwd mismatch");
    assert_eq!(result_frame(&buf, 16)["error"]["code"], -32602);

    let before = parse_frames(&snapshot(&buf)).len();
    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 17,
                "method": "session/load",
                "params": {"sessionId": first, "cwd": "/tmp/caliche-acp-a", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("load");
    let frames_now = parse_frames(&snapshot(&buf));
    let fresh = &frames_now[before..];
    let result_at = fresh
        .iter()
        .position(|frame| frame["id"] == 17)
        .expect("load result");
    assert!(
        result_at > 0,
        "history must replay before session/load returns"
    );
    assert!(fresh[..result_at].iter().all(|frame| {
        frame["method"] == "session/update" && frame["params"]["sessionId"] == first
    }));
    assert_eq!(fresh[result_at]["result"], json!({}));
    assert_eq!(fresh.len(), result_at + 1);
    let kinds = fresh[..result_at]
        .iter()
        .map(|frame| {
            frame["params"]["update"]["sessionUpdate"]
                .as_str()
                .unwrap_or("")
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        vec![
            "user_message_chunk",
            "agent_thought_chunk",
            "agent_message_chunk",
            "tool_call",
            "tool_call_update",
        ]
    );
    assert_eq!(
        fresh[0]["params"]["update"]["content"]["text"],
        "What's the capital of France?"
    );
    assert_eq!(fresh[3]["params"]["update"]["toolCallId"], "call-7");
    assert_eq!(fresh[4]["params"]["update"]["toolCallId"], "call-7");
    assert_eq!(fresh[4]["params"]["update"]["status"], "completed");
    let rendered = String::from_utf8(snapshot(&buf)).expect("stdout utf-8");
    assert!(
        !rendered.contains("hidden system"),
        "system text must not replay"
    );

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 18,
                "method": "session/resume",
                "params": {"sessionId": first, "cwd": "/tmp/caliche-acp-a", "mcpServers": []}
            }),
            &frames,
        )
        .await
        .expect("resume");
    assert_eq!(result_frame(&buf, 18)["error"]["code"], -32601);
}

async fn open_session(
    server: &AcpServer,
    buf: &Arc<Mutex<Vec<u8>>>,
    frames: &Arc<dyn AcpFrameSink>,
    id: u64,
    cwd: &str,
) -> String {
    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": id,
                "method": "session/new",
                "params": {"cwd": cwd, "mcpServers": []}
            }),
            frames,
        )
        .await
        .expect("session/new");
    result_frame(buf, id)["result"]["sessionId"]
        .as_str()
        .expect("sessionId")
        .to_string()
}

fn result_frame(buf: &Arc<Mutex<Vec<u8>>>, id: u64) -> Value {
    parse_frames(&snapshot(buf))
        .into_iter()
        .find(|frame| frame["id"] == id)
        .expect("result frame")
}

#[tokio::test]
#[serial_test::serial]
async fn session_list_load_reads_http_session_cache() {
    let previous = std::env::var("OPENROUTER_API_KEY").ok();
    // Harness construction checks for a provider key. This stub is never sent.
    unsafe {
        std::env::set_var("OPENROUTER_API_KEY", "test-key-for-acp-catalog");
    }
    let outcome = http_cache_catalog().await;
    match previous {
        Some(value) => unsafe {
            std::env::set_var("OPENROUTER_API_KEY", value);
        },
        None => unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        },
    }
    outcome.expect("http session cache");
}

async fn http_cache_catalog() -> anyhow::Result<()> {
    use caliche::harness::{HarnessConfig, SessionConfig};

    let adapter = Arc::new(crate::adapter::ApiAdapter::with_session_ttl(
        HarnessConfig::default(),
        Duration::from_secs(60),
        None,
    ));
    let workspace = "/tmp/caliche-acp-http";
    let session_id = adapter
        .create_session(
            Some(SessionConfig {
                session_id: None,
                workspace: Some(workspace.into()),
                context_files: Vec::new(),
                system_prompt_addition: None,
                max_turns: None,
                history: replay_history(),
            }),
            None,
            Some("France".to_string()),
        )
        .await?;

    let http_rows = adapter.list_process_sessions().await;
    assert_eq!(http_rows.len(), 1);
    assert_eq!(http_rows[0].session_id, session_id);
    assert_eq!(http_rows[0].title.as_deref(), Some("France"));
    let info = adapter
        .get_session_info(&session_id)
        .await
        .expect("GET /api/agent/sessions/{id}");
    assert_eq!(info.session_id, session_id);

    let driver: Arc<dyn AcpTurnDriver> = adapter.clone();
    let server = AcpServer::new(driver, agent_info());
    let (buf, frames) = mem_sink();
    server
        .handle_json_value(initialize_request(1), &frames)
        .await?;
    assert_catalog_capabilities(&parse_frames(&snapshot(&buf))[0]["result"]);

    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 2,
                "method": "session/list",
                "params": {"cwd": workspace}
            }),
            &frames,
        )
        .await?;
    let listed = result_frame(&buf, 2);
    let sessions = listed["result"]["sessions"].as_array().expect("sessions");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["sessionId"], session_id);
    assert_eq!(sessions[0]["cwd"], workspace);
    assert_eq!(sessions[0]["title"], "France");
    assert_eq!(
        sessions[0]["updatedAt"],
        info.last_activity
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
    );

    let before = parse_frames(&snapshot(&buf)).len();
    server
        .handle_json_value(
            json!({
                "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
                "id": 3,
                "method": "session/load",
                "params": {"sessionId": session_id, "cwd": workspace, "mcpServers": []}
            }),
            &frames,
        )
        .await?;
    let frames_now = parse_frames(&snapshot(&buf));
    let fresh = &frames_now[before..];
    let result_at = fresh
        .iter()
        .position(|frame| frame["id"] == 3)
        .expect("load result");
    assert!(result_at > 0);
    assert!(
        fresh[..result_at]
            .iter()
            .all(|frame| frame["method"] == "session/update")
    );
    assert_eq!(fresh[result_at]["result"], json!({}));
    assert_eq!(
        fresh[0]["params"]["update"]["content"]["text"],
        "What's the capital of France?"
    );
    Ok(())
}
