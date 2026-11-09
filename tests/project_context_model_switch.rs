//! A model switch sizes the project context for the new model's window.
//!
//! A stub on 127.0.0.1 plays LM Studio's model-state endpoint: `big-model`
//! has a 200K loaded window and `small-model` 32K. A host (the REPL) switches
//! the model by sending back a settings snapshot with its own copy of the
//! prompt, taken before the switch; the stale section in it is replaced.
//!
//! The switch builds the new provider from `LMSTUDIO_BASE_URL`. The test sets
//! it in the in-process credential store, so this file is its own test binary.

use caliche::llm::ProviderType;
use caliche::{AgentService, AgentServiceConfig, DEFAULT_PROJECT_CONTEXT_SHARE, ProjectContext};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const CUT_32K_END: &str = "rule 0937: keep this line\n\n[AGENTS.md is cut to fit this model's 32768-token context: these are its first 938 of 2000 lines. Read /repo/AGENTS.md for the rest.]";

/// Start the stub; returns its `/v1` base URL.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_lm_studio() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = [0_u8; 4096];
                let n = socket.read(&mut buf).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&buf[..n]);
                let asks = |id: &str| request.starts_with(&format!("GET /api/v0/models/{id} "));
                let (status, body) = if asks("big-model") {
                    (
                        "200 OK",
                        r#"{"id":"big-model","state":"loaded","loaded_context_length":200000}"#,
                    )
                } else if asks("small-model") {
                    (
                        "200 OK",
                        r#"{"id":"small-model","state":"loaded","loaded_context_length":32768}"#,
                    )
                } else {
                    ("404 Not Found", r#"{"error":"unexpected route"}"#)
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(reply.as_bytes()).await;
            });
        }
    });
    format!("http://{addr}/v1")
}

/// An AGENTS.md of 2000 lines (52,000 characters, about 17K tokens).
fn large_agents_md() -> ProjectContext {
    ProjectContext {
        path: std::path::PathBuf::from("/repo/AGENTS.md"),
        source_filename: "AGENTS.md".to_string(),
        content: (0..2000)
            .map(|i| format!("rule {i:04}: keep this line\n"))
            .collect(),
        share_percent: DEFAULT_PROJECT_CONTEXT_SHARE,
    }
}

#[tokio::test]
async fn a_switch_through_settings_sizes_the_project_context_for_the_new_model() {
    let base_url = stub_lm_studio().await;
    caliche_contracts::credential_env::set("LMSTUDIO_BASE_URL", base_url.clone());
    let mut service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(Some(ProviderType::LMStudio), Some("big-model".to_string()))
            .with_base_url(Some(base_url))
            .with_tool_names(Vec::new()),
    )
    .await
    .expect("service");
    let whole = service.budget_project_context(large_agents_md());
    assert!(whole.ends_with("rule 1999: keep this line"), "{whole}");
    let host_prompt = format!("base\n\n{whole}");
    service.agent_mut().set_system_prompt(host_prompt.clone());

    // To 32K: the host sends the prompt it had before the switch.
    let mut settings = service.snapshot_settings();
    settings.model = "small-model".to_string();
    settings.system_prompt = Some(host_prompt);
    service.apply_settings(settings).await.expect("to small");

    let prompt = service.agent().system_prompt().to_string();
    assert!(prompt.starts_with("base\n\n# Project Context (from AGENTS.md)\n\nrule 0000"));
    assert!(prompt.ends_with(CUT_32K_END), "{prompt}");

    // Back to 200K: the whole file again.
    let mut settings = service.snapshot_settings();
    settings.model = "big-model".to_string();
    service.apply_settings(settings).await.expect("to big");

    assert_eq!(service.agent().system_prompt(), format!("base\n\n{whole}"));
}
