use std::ffi::OsString;
use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;

use caliche_acp_client::{AcpClient, AcpClientError, AcpSpawn};
use caliche_contracts::acp::StopReason;

fn fixture_argv(mode: &str) -> Vec<OsString> {
    vec![
        OsString::from("/usr/bin/python3"),
        OsString::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixture_agent.py"
        )),
        OsString::from(mode),
    ]
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn spawn_mode(mode: &str) -> Result<AcpClient, AcpClientError> {
    AcpClient::spawn(AcpSpawn::new(fixture_argv(mode), workspace()))
}

async fn within<T>(future: impl Future<Output = T>) -> Result<T, tokio::time::error::Elapsed> {
    tokio::time::timeout(Duration::from_secs(5), future).await
}

async fn stderr_until(client: &AcpClient, needle: &str) -> String {
    for _ in 0..30 {
        let tail = client.stderr_tail();
        if tail.contains(needle) {
            return tail;
        }
        tokio::task::yield_now().await;
    }
    client.stderr_tail()
}

/// Hardcoded 30s. A paused runtime auto-advances to the next timer, so sleeping
/// the production const would still pass if that const became enormous.
const EXPECTED_HANDSHAKE_LIMIT: Duration = Duration::from_secs(30);

/// Let a paused-clock request reach its `timeout` await, then move time by
/// exactly [`EXPECTED_HANDSHAKE_LIMIT`].
///
/// Real time runs only long enough to arm the timer. [`tokio::time::advance`]
/// then adds 30s and does not jump to a later deadline.
async fn elapse_handshake_limit() {
    tokio::time::resume();
    tokio::time::sleep(Duration::from_millis(50)).await;
    tokio::time::pause();
    tokio::time::advance(EXPECTED_HANDSHAKE_LIMIT).await;
    tokio::task::yield_now().await;
}

#[tokio::test]
async fn initialize_times_out_when_the_peer_stays_silent() {
    let client = spawn_mode("silent").expect("spawn fixture");
    client.set_request_timeout(Some(Duration::from_millis(200)));
    let err = client.initialize().await.expect_err("silent peer");
    assert!(
        matches!(err, AcpClientError::TimedOut { ref method } if method == "initialize"),
        "{err:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn initialize_times_out_at_the_default_handshake_limit() {
    let client = spawn_mode("silent").expect("spawn fixture");
    let task = tokio::spawn(async move { client.initialize().await });
    elapse_handshake_limit().await;
    assert!(
        task.is_finished(),
        "initialize should time out at 30s without a caller override"
    );
    let err = task.await.expect("join").expect_err("silent peer");
    assert!(
        matches!(err, AcpClientError::TimedOut { ref method } if method == "initialize"),
        "{err:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn new_session_times_out_when_the_peer_stays_silent() {
    tokio::time::resume();
    let client = spawn_mode("silent-after-init").expect("spawn fixture");
    client.initialize().await.expect("initialize");
    tokio::time::pause();
    let task = tokio::spawn(async move { client.new_session(".").await });
    elapse_handshake_limit().await;
    assert!(
        task.is_finished(),
        "session/new should time out at 30s when the peer stays silent"
    );
    let err = task
        .await
        .expect("join")
        .expect_err("silent after initialize");
    assert!(
        matches!(err, AcpClientError::TimedOut { ref method } if method == "session/new"),
        "{err:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn prompt_has_no_default_timeout() {
    tokio::time::resume();
    let client = spawn_mode("silent-after-session").expect("spawn fixture");
    client.initialize().await.expect("initialize");
    let session = client.new_session(".").await.expect("session/new");
    let session_id = session.session_id.0.to_string();
    tokio::time::pause();
    let task = tokio::spawn(async move { client.prompt_text(&session_id, "hello").await });
    elapse_handshake_limit().await;
    assert!(
        !task.is_finished(),
        "session/prompt must stay open past 30s"
    );
    task.abort();
}

#[tokio::test]
async fn foreign_session_updates_replay_on_the_next_prompt() {
    let client = spawn_mode("cross-session").expect("spawn fixture");
    let (first, second) = within(async {
        client.initialize().await.expect("initialize");
        let session_a = client.new_session(".").await.expect("session a");
        let session_b = client.new_session(".").await.expect("session b");
        let first = client
            .prompt_text(session_a.session_id.0.as_ref(), "hello")
            .await
            .expect("prompt a");
        let second = client
            .prompt_text(session_b.session_id.0.as_ref(), "hello")
            .await
            .expect("prompt b");
        (first, second)
    })
    .await
    .expect("cross-session fixture timed out");

    assert_eq!(first.agent_text(), "local-turn");
    assert!(!first.agent_text().contains("parked-"));
    let replayed = second.agent_text();
    assert!(
        replayed.contains("parked-005"),
        "oldest kept note missing: {replayed}"
    );
    assert!(
        replayed.contains("parked-036"),
        "newest parked note missing: {replayed}"
    );
    assert!(
        !replayed.contains("parked-000"),
        "park queue kept notes past the cap: {replayed}"
    );
    assert!(replayed.contains("live-b"), "{replayed}");
}

#[tokio::test]
async fn initialize_new_prompt_updates_end() {
    let client = spawn_mode("happy").expect("spawn fixture");
    let outcome = within(async {
        let init = client.initialize().await.expect("initialize");
        assert_eq!(init.protocol_version.as_u16(), 1);
        assert!(init.auth_methods.is_empty());
        let session = client.new_session(".").await.expect("session/new");
        assert_eq!(session.session_id.0.as_ref(), "sess-fixture");
        client
            .prompt_text(session.session_id.0.as_ref(), "hello")
            .await
            .expect("prompt")
    })
    .await
    .expect("ACP fixture timed out");

    assert_eq!(outcome.stop_reason, StopReason::EndTurn);
    assert_eq!(outcome.agent_text(), "hello from fixture");
    assert!(outcome.updates.len() >= 2, "{:?}", outcome.updates);
    assert!(outcome.unrecognized_updates.is_empty());

    let tail = stderr_until(&client, "method session/prompt").await;
    assert!(tail.contains("initialize protocolVersion=1"), "{tail}");
    assert!(tail.contains("cwd /"), "{tail}");
    assert!(!tail.contains("RELATIVE_CWD"), "{tail}");
    assert!(!tail.contains("NON_JSON_STDIN"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
}

#[tokio::test]
async fn cancel_during_prompt_returns_cancelled() {
    let client = spawn_mode("hold-for-cancel").expect("spawn fixture");
    let init = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect("initialize");
    assert!(init.auth_methods.is_empty());
    let session = within(client.new_session(workspace()))
        .await
        .expect("ACP fixture timed out")
        .expect("session/new");
    let prompter = client.clone();
    let session_id = session.session_id.0.to_string();
    let prompt = tokio::spawn(async move { prompter.prompt_text(&session_id, "stop").await });
    within(client.cancel(session.session_id.0.as_ref()))
        .await
        .expect("ACP fixture timed out")
        .expect("cancel");
    let outcome = within(prompt)
        .await
        .expect("ACP fixture timed out")
        .expect("prompt task")
        .expect("prompt result");
    assert_eq!(outcome.stop_reason, StopReason::Cancelled);
    assert_eq!(outcome.agent_text(), "hello from fixture");
}

#[tokio::test]
async fn terminal_auth_proceeds_without_login() {
    let client = spawn_mode("auth-terminal").expect("spawn fixture");
    let outcome = within(async {
        let init = client.initialize().await.expect("initialize");
        assert_eq!(init.auth_methods.len(), 1);
        assert_eq!(init.auth_methods[0].id().0.as_ref(), "hermes-setup");
        let session = client.new_session(workspace()).await.expect("session/new");
        client
            .prompt_text(session.session_id.0.as_ref(), "hello")
            .await
            .expect("prompt")
    })
    .await
    .expect("ACP fixture timed out");
    assert_eq!(outcome.stop_reason, StopReason::EndTurn);
    let tail = stderr_until(&client, "method session/prompt").await;
    assert!(tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
    assert!(tail.contains("client-auth-terminal"), "{tail}");
    assert!(!tail.contains("client-auth-terminal True"), "{tail}");
    assert!(tail.contains("client-terminal-auth None"), "{tail}");
}

#[tokio::test]
async fn opencode_login_proceeds_without_authenticate() {
    let client = spawn_mode("auth-opencode").expect("spawn fixture");
    let outcome = within(async {
        let init = client.initialize().await.expect("initialize");
        assert_eq!(init.auth_methods[0].id().0.as_ref(), "opencode-login");
        let session = client.new_session(workspace()).await.expect("session/new");
        client
            .prompt_text(session.session_id.0.as_ref(), "hello")
            .await
            .expect("prompt")
    })
    .await
    .expect("ACP fixture timed out");
    assert_eq!(outcome.agent_text(), "hello from fixture");
    let tail = stderr_until(&client, "method session/new").await;
    assert!(!tail.contains("method authenticate"), "{tail}");
    assert!(tail.contains("client-terminal-auth None"), "{tail}");
}

#[tokio::test]
async fn auth_required_fails_closed_with_byo_guidance() {
    let client = spawn_mode("auth-required").expect("spawn fixture");
    let init = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect("initialize");
    assert_eq!(init.auth_methods[0].id().0.as_ref(), "opencode-login");
    let session = within(client.new_session(workspace()))
        .await
        .expect("ACP fixture timed out")
        .expect("session/new");
    let error = within(client.prompt_text(session.session_id.0.as_ref(), "hello"))
        .await
        .expect("ACP fixture timed out")
        .expect_err("auth_required");
    let text = error.to_string();
    assert!(text.contains("auth_required"), "{text}");
    assert!(text.contains("hermes config set"), "{text}");
    assert!(text.contains("OPENAI_API_KEY"), "{text}");
    assert!(text.contains("does not run"), "{text}");
    let tail = stderr_until(&client, "method session/prompt").await;
    assert!(tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
    let follow = client
        .new_session(workspace())
        .await
        .expect_err("shut down");
    assert!(follow.to_string().contains("shut down"), "{follow}");
}

#[tokio::test]
async fn vendor_account_login_is_refused() {
    let client = spawn_mode("auth-vendor").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("vendor auth must fail");
    let text = error.to_string();
    assert!(text.contains("cursor-account"), "{text}");
    assert!(text.contains("opencode-login"), "{text}");
    let tail = stderr_until(&client, "method initialize").await;
    assert!(!tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
}

#[tokio::test]
async fn generic_auth_does_not_call_authenticate() {
    let client = spawn_mode("auth-generic").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("custom-login must fail");
    let text = error.to_string();
    assert!(text.contains("custom-login"), "{text}");
    let tail = stderr_until(&client, "method initialize").await;
    assert!(!tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
}

#[tokio::test]
async fn custom_terminal_is_refused_before_session_new() {
    let client = spawn_mode("auth-custom-terminal").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("custom-terminal must fail");
    assert!(error.to_string().contains("custom-terminal"), "{error}");
    let tail = stderr_until(&client, "method initialize").await;
    assert!(!tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
}

#[tokio::test]
async fn openrouter_alone_is_refused_before_session_new() {
    let client = spawn_mode("auth-openrouter").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("openrouter must fail");
    assert!(error.to_string().contains("openrouter"), "{error}");
    let tail = stderr_until(&client, "method initialize").await;
    assert!(!tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
}

#[tokio::test]
async fn codex_with_hermes_setup_is_refused_before_session_new() {
    let client = spawn_mode("auth-codex").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("codex must fail");
    let text = error.to_string();
    assert!(text.contains("openai-codex"), "{text}");
    assert!(text.contains("hermes-setup"), "{text}");
    let tail = stderr_until(&client, "method initialize").await;
    assert!(!tail.contains("method session/new"), "{tail}");
    assert!(!tail.contains("method authenticate"), "{tail}");
}

#[tokio::test]
async fn protocol_version_other_than_one_disconnects() {
    let client = spawn_mode("version-2").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("version 2 must fail");
    let text = error.to_string();
    assert!(text.contains('2'), "{text}");
    assert!(text.contains('1'), "{text}");
}

#[tokio::test]
async fn non_json_stdout_is_rejected() {
    let client = spawn_mode("stdout-log").expect("spawn fixture");
    let error = within(client.initialize())
        .await
        .expect("ACP fixture timed out")
        .expect_err("log line on stdout must fail");
    let text = error.to_string();
    assert!(text.contains("not-json"), "{text}");
}

#[tokio::test]
async fn permission_request_fails_closed() {
    let client = spawn_mode("permission").expect("spawn fixture");
    let outcome = within(async {
        client.initialize().await.expect("initialize");
        let session = client.new_session(workspace()).await.expect("session");
        client
            .prompt_text(session.session_id.0.as_ref(), "read")
            .await
            .expect("prompt")
    })
    .await
    .expect("ACP fixture timed out");
    assert_eq!(outcome.stop_reason, StopReason::EndTurn);
    assert_eq!(outcome.agent_text(), "permission closed");
    let tail = stderr_until(&client, "permission-reply").await;
    assert!(tail.contains("cancelled"), "{tail}");
}
