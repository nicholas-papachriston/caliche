//! PUT /api/settings writes provider, model, and temperature beside mcp.json.
//!
//! The system prompt stays out of that file. An empty MCP save does not wipe
//! an existing mcp.json unless the caller sets `allow_empty`.

#[path = "harness.rs"]
mod harness;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use caliche::llm::ProviderType;
use caliche::traits::credentials::{Credentials, StaticCredentialProvider};
use caliche::{AgentService, AgentServiceConfig};
use caliche_server::RouterOptions;
use caliche_server::{create_router, durable_settings};
use serde_json::json;
use serial_test::serial;
use tempfile::tempdir;
use tokio::sync::RwLock;
use tower::ServiceExt;

async fn hosted_app(dir: &std::path::Path) -> (axum::Router, Arc<RwLock<AgentService>>) {
    let _config = harness::hold();
    let mcp_path = dir.join("mcp.json");
    tokio::fs::write(
        &mcp_path,
        r#"{"mcpServers":{"demo":{"command":"npx","args":["-y","demo"],"enabled":true}}}"#,
    )
    .await
    .expect("write mcp.json");

    let mut service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(
                Some(ProviderType::Anthropic),
                Some("claude-3-5-sonnet-20241022".to_string()),
            )
            .with_credential_provider(Arc::new(StaticCredentialProvider::new(
                Credentials::ApiKey("test-key".to_string()),
            )))
            .with_defer_missing_credentials(true),
    )
    .await
    .expect("service");
    service
        .agent_mut()
        .set_system_prompt("BASE-PROMPT-DO-NOT-FREEZE".to_string());
    let shared = Arc::new(RwLock::new(service));
    let app = create_router(
        ProviderType::Anthropic,
        "claude-3-5-sonnet-20241022".to_string(),
        Some(RouterOptions {
            shared_service: Some(Arc::clone(&shared)),
            mcp_config_path: Some(mcp_path),
            ..RouterOptions::default()
        }),
    )
    .await
    .expect("router");
    (app, shared)
}

#[tokio::test]
#[serial]
async fn save_persists_model_and_temperature_without_prompt_or_mcp_wipe() {
    let dir = tempdir().expect("tempdir");
    let (app, shared) = hosted_app(dir.path()).await;
    let settings_path = durable_settings::path_beside_mcp(&dir.path().join("mcp.json"));

    let put = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/settings")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "provider": "anthropic",
                        "model": "saved-model",
                        "temperature": 0.5,
                        "max_tokens": 128,
                        "system_prompt": "MARKER-SHOULD-NOT-PERSIST",
                        "tools": []
                    })
                    .to_string(),
                ))
                .expect("settings request"),
        )
        .await
        .expect("settings response");
    let status = put.status();
    let body = axum::body::to_bytes(put.into_body(), 64 * 1024)
        .await
        .expect("settings body");
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let raw = tokio::fs::read_to_string(&settings_path)
        .await
        .expect("settings.json");
    assert!(raw.contains("saved-model"), "{raw}");
    assert!(raw.contains("0.5"), "{raw}");
    assert!(raw.contains("128"), "{raw}");
    assert!(!raw.contains("system_prompt"), "{raw}");
    assert!(!raw.contains("MARKER-SHOULD-NOT-PERSIST"), "{raw}");
    assert!(!raw.contains("BASE-PROMPT"), "{raw}");

    let loaded = durable_settings::load(&settings_path)
        .await
        .expect("load settings");
    assert_eq!(loaded.provider, "anthropic");
    assert_eq!(loaded.model, "saved-model");
    assert_eq!(loaded.temperature, Some(0.5));
    assert_eq!(loaded.max_tokens, Some(128));

    let prompt = shared.read().await.agent().system_prompt().to_string();
    assert_eq!(prompt, "BASE-PROMPT-DO-NOT-FREEZE");
    assert_eq!(shared.read().await.agent().temperature(), Some(0.5));

    let mcp_before = tokio::fs::read_to_string(dir.path().join("mcp.json"))
        .await
        .expect("mcp");
    let empty_put = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/mcp/config")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(json!({ "servers": [] }).to_string()))
                .expect("mcp request"),
        )
        .await
        .expect("mcp response");
    assert_eq!(empty_put.status(), StatusCode::OK);
    let mcp_after = tokio::fs::read_to_string(dir.path().join("mcp.json"))
        .await
        .expect("mcp after");
    assert_eq!(mcp_after, mcp_before);
    assert!(mcp_after.contains("demo"), "{mcp_after}");

    let clear = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/api/mcp/config")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({ "servers": [], "allow_empty": true }).to_string(),
                ))
                .expect("clear request"),
        )
        .await
        .expect("clear response");
    assert_eq!(clear.status(), StatusCode::OK);
    let cleared = tokio::fs::read_to_string(dir.path().join("mcp.json"))
        .await
        .expect("cleared mcp");
    assert!(!cleared.contains("demo"), "{cleared}");
}
