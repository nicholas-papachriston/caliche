use std::path::PathBuf;
use std::time::Duration;

use caliche_acp_client::smoke::{
    PeerKind, PeerQuery, SmokeConfig, SmokeEvent, SmokeReport, SmokeStatus, run_smoke,
};

fn saw(report: &SmokeReport, event: SmokeEvent) -> bool {
    report.events.contains(&event)
}

fn scratch(name: &str) -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("caliche-acp-smoke-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn fixture_config(dir: &std::path::Path, mode: &str) -> SmokeConfig {
    SmokeConfig {
        peer: PeerQuery {
            peer: PeerKind::Fixture,
            opencode_bin: None,
            hermes_bin: None,
            fixture_script: PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixture_agent.py"
            )),
            fixture_mode: mode.to_owned(),
            opencode_print_logs: false,
        },
        cwd: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        prompt: "hello".to_owned(),
        transcript: dir.join(format!("{mode}.jsonl")),
        timeout: Duration::from_secs(5),
        inherit_env: None,
        env: Vec::new(),
    }
}

#[tokio::test]
async fn fixture_smoke_completes_and_records_frames() {
    let dir = scratch("happy").expect("scratch dir");
    let config = fixture_config(&dir, "happy");
    let transcript = config.transcript.clone();
    let report = run_smoke(config).await;
    assert_eq!(report.status, SmokeStatus::Completed, "{}", report.detail);
    assert_eq!(report.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(report.agent_text, "hello from fixture");
    assert_eq!(report.protocol_version, Some(1));
    assert!(
        report.auth_method_ids.is_empty(),
        "{:?}",
        report.auth_method_ids
    );
    assert!(saw(&report, SmokeEvent::SessionNew));
    assert!(saw(&report, SmokeEvent::SessionPrompt));
    assert!(!saw(&report, SmokeEvent::Authenticate));
    assert!(!saw(&report, SmokeEvent::PermissionRequest));
    let raw = std::fs::read_to_string(&transcript).expect("transcript");
    assert!(raw.contains("\"method\":\"initialize\""), "{raw}");
    assert!(raw.contains("\"method\":\"session/new\""), "{raw}");
    assert!(raw.contains("\"method\":\"session/prompt\""), "{raw}");
    assert!(raw.contains("end_turn"), "{raw}");
    assert!(raw.contains("hello from fixture"), "{raw}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fixture_terminal_auth_reaches_session_new() {
    let dir = scratch("terminal").expect("scratch dir");
    let report = run_smoke(fixture_config(&dir, "auth-terminal")).await;
    assert_eq!(report.status, SmokeStatus::Completed, "{}", report.detail);
    assert_eq!(report.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(report.auth_method_ids, vec!["hermes-setup".to_owned()]);
    assert!(saw(&report, SmokeEvent::SessionNew));
    assert!(saw(&report, SmokeEvent::SessionPrompt));
    assert!(!saw(&report, SmokeEvent::Authenticate));
    assert!(report.session_id.is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fixture_opencode_login_reaches_session_new() {
    let dir = scratch("opencode").expect("scratch dir");
    let report = run_smoke(fixture_config(&dir, "auth-opencode")).await;
    assert_eq!(report.status, SmokeStatus::Completed, "{}", report.detail);
    assert_eq!(report.auth_method_ids, vec!["opencode-login".to_owned()]);
    assert!(saw(&report, SmokeEvent::SessionNew));
    assert!(!saw(&report, SmokeEvent::Authenticate));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fixture_auth_required_fails_closed_after_session_new() {
    let dir = scratch("required").expect("scratch dir");
    let report = run_smoke(fixture_config(&dir, "auth-required")).await;
    assert_eq!(
        report.status,
        SmokeStatus::AuthRequired,
        "{}",
        report.detail
    );
    assert!(report.detail.contains("auth_required"), "{}", report.detail);
    assert!(
        report.detail.contains("hermes config set"),
        "{}",
        report.detail
    );
    assert!(
        report.detail.contains("OPENAI_API_KEY"),
        "{}",
        report.detail
    );
    assert_eq!(report.auth_method_ids, vec!["opencode-login".to_owned()]);
    assert!(saw(&report, SmokeEvent::SessionNew));
    assert!(saw(&report, SmokeEvent::SessionPrompt));
    assert!(!saw(&report, SmokeEvent::Authenticate));
    assert!(report.session_id.is_some());
    assert_eq!(report.status.code(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fixture_vendor_auth_is_refused_before_authenticate() {
    let dir = scratch("vendor").expect("scratch dir");
    let report = run_smoke(fixture_config(&dir, "auth-vendor")).await;
    assert_eq!(
        report.status,
        SmokeStatus::AuthRejected,
        "{}",
        report.detail
    );
    assert!(
        report.detail.contains("cursor-account"),
        "{}",
        report.detail
    );
    assert!(
        report.detail.contains("opencode-login"),
        "{}",
        report.detail
    );
    assert_eq!(report.auth_method_ids, vec!["cursor-account".to_owned()]);
    assert!(!saw(&report, SmokeEvent::Authenticate));
    assert!(!saw(&report, SmokeEvent::SessionNew));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fixture_custom_terminal_openrouter_and_custom_login_are_refused() {
    for mode in ["auth-custom-terminal", "auth-openrouter", "auth-generic"] {
        let dir = scratch(mode).expect("scratch dir");
        let report = run_smoke(fixture_config(&dir, mode)).await;
        assert_eq!(
            report.status,
            SmokeStatus::AuthRejected,
            "{mode}: {}",
            report.detail
        );
        assert!(!saw(&report, SmokeEvent::SessionNew), "{mode}");
        assert!(!saw(&report, SmokeEvent::Authenticate), "{mode}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[tokio::test]
async fn fixture_codex_is_refused_before_session_new() {
    let dir = scratch("codex").expect("scratch dir");
    let report = run_smoke(fixture_config(&dir, "auth-codex")).await;
    assert_eq!(
        report.status,
        SmokeStatus::AuthRejected,
        "{}",
        report.detail
    );
    assert!(report.detail.contains("openai-codex"), "{}", report.detail);
    assert!(report.auth_method_ids.contains(&"openai-codex".to_owned()));
    assert!(report.auth_method_ids.contains(&"hermes-setup".to_owned()));
    assert!(!saw(&report, SmokeEvent::Authenticate));
    assert!(!saw(&report, SmokeEvent::SessionNew));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn fixture_permission_reply_is_cancelled() {
    let dir = scratch("permission").expect("scratch dir");
    let report = run_smoke(fixture_config(&dir, "permission")).await;
    assert_eq!(report.status, SmokeStatus::Completed, "{}", report.detail);
    assert_eq!(report.stop_reason.as_deref(), Some("end_turn"));
    assert!(saw(&report, SmokeEvent::PermissionRequest));
    assert_eq!(report.permission_outcome.as_deref(), Some("cancelled"));
    assert!(!saw(&report, SmokeEvent::Authenticate));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn missing_peer_binary_is_reported() {
    let dir = scratch("missing").expect("scratch dir");
    let mut config = fixture_config(&dir, "happy");
    config.peer.peer = PeerKind::OpenCode;
    config.peer.opencode_bin = Some(dir.join("no-such-opencode"));
    let report = run_smoke(config).await;
    assert_eq!(report.status, SmokeStatus::PeerMissing, "{}", report.detail);
    assert!(report.detail.contains("opencode"), "{}", report.detail);
    assert!(
        report.detail.contains("command matrix"),
        "{}",
        report.detail
    );
    assert!(!saw(&report, SmokeEvent::Authenticate));
    let _ = std::fs::remove_dir_all(&dir);
}
