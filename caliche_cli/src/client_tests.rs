use super::*;
use tempfile::tempdir;

#[test]
fn lock_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let lock = lock_for_bind("127.0.0.1", 8787);
    write_lock(&path, &lock).unwrap();
    let loaded = read_lock(&path).unwrap().unwrap();
    assert_eq!(loaded.port, 8787);
    assert_eq!(loaded.url, "http://127.0.0.1:8787");
    assert_eq!(loaded.pid, process::id());
}

#[test]
fn lock_for_bind_sets_expected_fields() {
    let lock = lock_for_bind("127.0.0.1", 3000);
    assert_eq!(lock.host, "127.0.0.1");
    assert_eq!(lock.port, 3000);
    assert_eq!(lock.url, "http://127.0.0.1:3000");
    assert!(is_alive(lock.pid));
}

#[test]
fn lock_url_brackets_ipv6_unspecified_host() {
    let bare = lock_for_bind("::", 3046);
    assert_eq!(bare.host, "::");
    assert_eq!(bare.url, "http://[::]:3046");
    assert!(!bare.url.contains(":::"), "{}", bare.url);

    let bracketed = lock_for_bind("[::]", 3046);
    assert_eq!(bracketed.host, "[::]");
    assert_eq!(bracketed.url, "http://[::]:3046");
    assert!(!bracketed.url.contains(":::"), "{}", bracketed.url);
}

#[test]
fn clear_stale_server_lock_removes_a_dead_pid() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let mut lock = lock_for_bind("127.0.0.1", 9);
    lock.pid = 2_147_483_647;
    write_lock(&path, &lock).unwrap();
    assert!(clear_stale_server_lock(&path).unwrap());
    assert!(!path.exists());
}

#[test]
fn discover_bootstraps_without_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("missing.lock");
    let client = AgentClient::discover_at(path).unwrap();
    assert!(matches!(client.mode(), AgentClientMode::LocalBootstrap));
    assert!(client.base_url().is_none());
}

#[test]
fn register_local_server_writes_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let mut client = AgentClient::discover_at(path.clone()).unwrap();
    client.register_local_server("127.0.0.1", 9999).unwrap();
    assert_eq!(client.base_url(), Some("http://127.0.0.1:9999"));
    let lock = ServerLock::read_from(&path).unwrap().unwrap();
    assert_eq!(lock.port, 9999);
    client.clear_lock().unwrap();
    assert!(ServerLock::read_from(&path).unwrap().is_none());
}

#[test]
fn discover_clears_stale_lock_and_bootstraps() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let mut lock = lock_for_bind("127.0.0.1", 8787);
    // Extremely unlikely to be a live process; forces the stale-lock path.
    lock.pid = u32::MAX - 1;
    write_lock(&path, &lock).unwrap();
    assert!(path.exists());

    let client = AgentClient::discover_at(path.clone()).unwrap();
    assert!(matches!(client.mode(), AgentClientMode::LocalBootstrap));
    assert!(client.base_url().is_none());
    assert!(!path.exists());
}

#[test]
fn discover_prunes_dead_pid_registry_record_on_stale_lock() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let registry_dir = dir.path().join("servers");

    let dead_pid = u32::MAX - 1;
    let mut lock = lock_for_bind("127.0.0.1", 8787);
    lock.pid = dead_pid;
    write_lock(&path, &lock).unwrap();
    assert!(path.exists());

    let record = crate::server_registry::ServerRecord {
        version: 1,
        id: format!("foreground-8787-{dead_pid}-1"),
        pid: dead_pid,
        port: 8787,
        bind: "127.0.0.1".to_string(),
        build_id: "test".to_string(),
        started_at: 1_700_000_000,
        cwd: "/work".to_string(),
        exe: format!("/usr/local/bin/{}", crate::server_registry::SERVER_EXE_NAME),
        origin: crate::server_registry::ServerOrigin::Foreground,
        start_token: "token".to_string(),
    };
    let record_path = crate::server_registry::write_record(&registry_dir, &record).unwrap();
    assert!(record_path.exists());

    let client = AgentClient::discover_at(path.clone()).unwrap();
    assert!(matches!(client.mode(), AgentClientMode::LocalBootstrap));
    assert!(client.base_url().is_none());
    assert!(!path.exists());
    assert!(!record_path.exists());
    assert!(
        crate::server_registry::list_servers(&registry_dir)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn from_lock_rejects_dead_pid() {
    let mut lock = lock_for_bind("127.0.0.1", 8787);
    lock.pid = u32::MAX - 1;
    let err = AgentClient::from_lock(lock).expect_err("dead pid should fail");
    assert!(err.to_string().contains("not alive"));
}

#[test]
fn from_lock_accepts_live_pid() {
    let lock = lock_for_bind("127.0.0.1", 8787);
    let client = AgentClient::from_lock(lock).expect("live pid should succeed");
    assert_eq!(client.base_url(), Some("http://127.0.0.1:8787"));
    assert!(matches!(client.mode(), AgentClientMode::Remote { .. }));
}

#[test]
fn discover_uses_live_lock_as_remote() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let lock = lock_for_bind("127.0.0.1", 4545);
    write_lock(&path, &lock).unwrap();

    let client = AgentClient::discover_at(path.clone()).unwrap();
    assert_eq!(client.base_url(), Some("http://127.0.0.1:4545"));
    assert!(path.exists());
}

#[test]
fn remove_from_refuses_foreign_pid() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let mut lock = lock_for_bind("127.0.0.1", 8787);
    lock.pid = process::id().wrapping_add(1).max(1);
    // Ensure we are not accidentally using our own pid.
    if lock.pid == process::id() {
        lock.pid = process::id().wrapping_add(2).max(2);
    }
    write_lock(&path, &lock).unwrap();
    let err = ServerLock::remove_from(&path, false).expect_err("foreign pid should refuse");
    assert!(err.to_string().contains("refusing to remove"));
    assert!(path.exists());
}

#[test]
fn parse_sse_frame_reads_event_and_data() {
    let parsed = parse_sse_frame(
        "event: agent.thinking\ndata: {\"type\":\"thinking\",\"content\":\"hmm\"}\n",
    )
    .expect("frame");
    assert_eq!(parsed.0, "agent.thinking");
    assert_eq!(parsed.1["type"], "thinking");
    assert_eq!(parsed.1["content"], "hmm");
}

#[test]
fn parse_sse_frame_defaults_event_name_and_joins_data_lines() {
    let parsed = parse_sse_frame("data: {\"a\":1}\n").expect("frame");
    assert_eq!(parsed.0, "message");
    assert_eq!(parsed.1["a"], 1);

    let joined =
        parse_sse_frame("event: turn.complete\ndata: {\"ok\": true}\n").expect("joined frame");
    assert_eq!(joined.0, "turn.complete");
    assert_eq!(joined.1["ok"], true);
}

#[test]
fn sse_event_payload_unwraps_versioned_envelope() {
    let envelope = serde_json::json!({
        "version": "1",
        "type": "text_chunk",
        "payload": {"type": "text_chunk", "content": "hi"}
    });
    assert_eq!(sse_event_payload(&envelope)["content"], "hi");
}

#[test]
fn sse_event_payload_keeps_flat_event_bodies() {
    let flat = serde_json::json!({"response": "done", "duration_ms": 3});
    assert_eq!(sse_event_payload(&flat)["response"], "done");
}

#[test]
fn drain_complete_sse_frames_keeps_partial_tail() {
    let mut buffer =
        String::from("event: a\ndata: {\"n\":1}\n\nevent: b\ndata: {\"n\":2}\n\npartial");
    let frames = drain_complete_sse_frames(&mut buffer);
    assert_eq!(frames.len(), 2);
    assert_eq!(buffer, "partial");
    assert!(frames[0].contains("event: a"));
    assert!(frames[1].contains("event: b"));
}

#[tokio::test]
async fn run_turn_stream_invokes_callback_as_frames_arrive() {
    use axum::extract::Path;
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use caliche::messages::stream_contracts::sse_name;
    use futures_util::stream::Stream;
    use std::convert::Infallible;
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    async fn events(Path(_id): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (tx, rx) = mpsc::channel(8);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(Event::default()
                    .event(sse_name::TOOL_START)
                    .data(r#"{"tool_name":"bash"}"#)))
                .await;
            tokio::time::sleep(Duration::from_millis(80)).await;
            let _ = tx
                .send(Ok(Event::default()
                    .event(sse_name::TOOL_END)
                    .data(r#"{"tool_name":"bash","success":true}"#)))
                .await;
            tokio::time::sleep(Duration::from_millis(80)).await;
            let _ = tx
                .send(Ok(Event::default().event(sse_name::TURN_COMPLETE).data(
                    r#"{"type":"turn_complete","response":"done","duration_ms":1}"#,
                )))
                .await;
        });
        Sse::new(ReceiverStream::new(rx))
    }

    async fn turn(Path(_id): Path<String>) -> Json<serde_json::Value> {
        Json(serde_json::json!({"response": "done", "duration_ms": 1}))
    }

    let app = Router::new()
        .route("/api/agent/sessions/{id}/events", get(events))
        .route("/api/agent/sessions/{id}/turn", post(turn));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("listener addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve test app");
    });

    let client = AgentClient::new(format!("http://{addr}"));
    let mut names = Vec::new();
    let mut times = Vec::new();
    client
        .run_turn_stream(
            "sess-1",
            "hello",
            Some(("openrouter", "z-ai/glm-5.3")),
            |name, _data| {
                names.push(name.to_string());
                times.push(Instant::now());
            },
        )
        .await
        .expect("streamed turn");

    assert_eq!(
        names.first().map(String::as_str),
        Some(sse_name::TOOL_START)
    );
    assert!(names.iter().any(|name| name == sse_name::TOOL_END));
    assert_eq!(
        names.last().map(String::as_str),
        Some(sse_name::TURN_COMPLETE)
    );
    assert!(
        times.len() >= 2,
        "expected at least two streamed events, got {}",
        times.len()
    );
    let gap = times[1].duration_since(times[0]);
    assert!(
        gap >= Duration::from_millis(40),
        "events should arrive incrementally, gap was {gap:?}"
    );
}

#[test]
fn format_turn_failure_reads_the_provider_status() {
    let line = format_turn_failure(
        reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        "Stream error: anthropic API error (400 Bad Request): tool_result without tool_use",
    );
    assert_eq!(
        line,
        "Turn failed: anthropic 400: tool_result without tool_use"
    );
}

#[test]
fn clear_stale_server_lock_removes_a_dead_pid_only() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("server.lock");
    let mut lock = lock_for_bind("127.0.0.1", 9);
    lock.pid = u32::MAX - 1;
    write_lock(&path, &lock).unwrap();
    assert!(clear_stale_server_lock(&path).unwrap());
    assert!(!path.exists());

    let live = lock_for_bind("127.0.0.1", 9);
    write_lock(&path, &live).unwrap();
    assert!(!clear_stale_server_lock(&path).unwrap());
    assert!(path.exists());
}

async fn serve(app: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn server_500_returns_before_the_event_stream_closes() {
    use axum::Router;
    use axum::extract::Path;
    use axum::http::StatusCode;
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use futures_util::stream::Stream;
    use std::convert::Infallible;
    use std::time::Instant;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    async fn events(Path(_id): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (tx, rx) = mpsc::channel(2);
        tokio::spawn(async move {
            let _ = tx.send(Ok(Event::default().comment("keep-alive"))).await;
            std::future::pending::<()>().await;
        });
        Sse::new(ReceiverStream::new(rx))
    }

    async fn turn(Path(_id): Path<String>) -> (StatusCode, String) {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "anthropic API error (400 Bad Request): tool_result without tool_use".to_string(),
        )
    }

    let app = Router::new()
        .route("/api/agent/sessions/{id}/events", get(events))
        .route("/api/agent/sessions/{id}/turn", post(turn));
    let client = AgentClient::new(serve(app).await);
    let started = Instant::now();
    let error = client
        .run_turn_stream("sess", "hello", None, |_name, _data| {})
        .await
        .expect_err("500 should fail the turn");
    assert!(started.elapsed() < Duration::from_secs(10), "turn hung");
    assert_eq!(
        error.to_string(),
        "Turn failed: anthropic 400: tool_result without tool_use"
    );
}

#[tokio::test]
async fn turn_error_event_returns_without_turn_complete() {
    use axum::extract::Path;
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use futures_util::stream::Stream;
    use std::convert::Infallible;
    use std::time::Instant;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    async fn events(Path(_id): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (tx, rx) = mpsc::channel(2);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let _ = tx
                .send(Ok(Event::default().event(sse_name::TURN_ERROR).data(
                    r#"{"type":"turn_error","message":"anthropic 400: rejected"}"#,
                )))
                .await;
            std::future::pending::<()>().await;
        });
        Sse::new(ReceiverStream::new(rx))
    }

    async fn turn(Path(_id): Path<String>) -> Json<serde_json::Value> {
        tokio::time::sleep(Duration::from_secs(5)).await;
        Json(serde_json::json!({"response": "late"}))
    }

    let app = Router::new()
        .route("/api/agent/sessions/{id}/events", get(events))
        .route("/api/agent/sessions/{id}/turn", post(turn));
    let client = AgentClient::new(serve(app).await);
    let started = Instant::now();
    let error = client
        .run_turn_stream("sess", "hello", None, |_name, _data| {})
        .await
        .expect_err("turn.error should fail the turn");
    assert!(started.elapsed() < Duration::from_secs(10), "turn hung");
    assert_eq!(error.to_string(), "Turn failed: anthropic 400: rejected");
}

#[tokio::test]
async fn idle_timeout_fires_when_only_keepalives_arrive() {
    use axum::extract::Path;
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use futures_util::stream::Stream;
    use std::convert::Infallible;
    use std::time::Instant;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    async fn events(Path(_id): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (tx, rx) = mpsc::channel(2);
        tokio::spawn(async move {
            let _ = tx.send(Ok(Event::default().comment("keep-alive"))).await;
            std::future::pending::<()>().await;
        });
        Sse::new(ReceiverStream::new(rx))
    }

    async fn turn(Path(_id): Path<String>) -> Json<serde_json::Value> {
        tokio::time::sleep(Duration::from_secs(5)).await;
        Json(serde_json::json!({"response": "late"}))
    }

    let app = Router::new()
        .route("/api/agent/sessions/{id}/events", get(events))
        .route("/api/agent/sessions/{id}/turn", post(turn));
    let client =
        AgentClient::new(serve(app).await).with_turn_idle_timeout(Duration::from_millis(200));
    let started = Instant::now();
    let error = client
        .run_turn_stream("sess", "hello", None, |_name, _data| {})
        .await
        .expect_err("idle timeout");
    assert!(started.elapsed() < Duration::from_secs(10), "turn hung");
    assert!(error.to_string().contains("no session events"));
}

#[tokio::test]
async fn cancel_posts_the_session_cancel_endpoint() {
    use axum::extract::{Path, State};
    use axum::http::StatusCode;
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use futures_util::stream::Stream;
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    #[derive(Clone)]
    struct Hits(Arc<AtomicUsize>);

    async fn events(Path(_id): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (tx, rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let _ = tx.send(Ok(Event::default().comment("keep-alive"))).await;
            std::future::pending::<()>().await;
        });
        Sse::new(ReceiverStream::new(rx))
    }

    async fn turn(Path(_id): Path<String>) -> Json<serde_json::Value> {
        tokio::time::sleep(Duration::from_secs(5)).await;
        Json(serde_json::json!({"response": "late"}))
    }

    async fn cancel(State(hits): State<Hits>, Path(_id): Path<String>) -> StatusCode {
        hits.0.fetch_add(1, Ordering::SeqCst);
        StatusCode::OK
    }

    let hits = Hits(Arc::new(AtomicUsize::new(0)));
    let app = Router::new()
        .route("/api/agent/sessions/{id}/events", get(events))
        .route("/api/agent/sessions/{id}/turn", post(turn))
        .route("/api/agent/sessions/{id}/cancel", post(cancel))
        .with_state(hits.clone());
    let client = AgentClient::new(serve(app).await);
    let token = crate::cancellation::CancellationToken::new();
    let cancel_token = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        cancel_token.cancel();
    });
    let started = Instant::now();
    let error = client
        .run_turn_stream_with_cancel("sess", "hello", None, None, Some(&token), |_n, _d| {})
        .await
        .expect_err("cancel");
    assert!(started.elapsed() < Duration::from_secs(10), "turn hung");
    assert!(error.to_string().contains("cancelled by user"));
    assert_eq!(hits.0.load(Ordering::SeqCst), 1);
}

/// Pid 0 names this process group and `u32::MAX` wraps to -1 (every process
/// the user owns). Neither is a single process, so neither is alive.
#[cfg(unix)]
#[test]
fn is_alive_refuses_pids_that_name_more_than_one_process() {
    assert!(!super::is_alive(0));
    assert!(!super::is_alive(1));
    assert!(!super::is_alive(u32::MAX));
    assert!(super::is_alive(std::process::id()));
}
