//! API server startup and configuration.

mod acp;
mod cluster;
mod mcp;
pub mod routes;
mod rpc;
mod shutdown;

use anyhow::{Context, Result};
use axum::serve;
use caliche::AgentService;
use caliche::agent::core::ToolExecutionDriver;
use caliche::harness::config::HarnessConfig;
use caliche_server::security::{
    cors_permissive_from_env, listen_socket_addr, warn_if_open_unspecified_bind,
};
use caliche_server::{RouterOptions, create_router};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use crate::server::routes::add_cli_parity_routes;
use crate::storage::database::Database;

pub use acp::handle_acp_stdio;
pub use mcp::handle_mcp_stdio;

/// Build router options from the CLI shared service and args so the server uses the same
/// provider, model, tools, workspace, permission mode, and config as the CLI.
///
/// # Errors
///
/// Returns an error when `permission_mode` is not a known mode. Missing and
/// empty values select the default. The token table is
/// `caliche::agent::hooks::parse_permission_mode_str`.
pub async fn build_harness_config_from_cli(
    shared_service: Arc<RwLock<AgentService>>,
    args: &crate::commands::Args,
    cli_config: Option<&crate::config::Config>,
) -> Result<HarnessConfig> {
    let guard = shared_service.read().await;
    let tool_names: Vec<String> = guard
        .list_tools()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();

    let mut harness = HarnessConfig::default()
        .with_provider(guard.provider())
        .with_model(guard.model().to_string())
        .with_tools(tool_names);
    drop(guard);

    if let Ok(cwd) = std::env::current_dir() {
        harness.workspace_root = Some(cwd);
    }

    harness.permission_mode =
        caliche::agent::hooks::parse_permission_mode_str(args.permission_mode.as_deref())
            .map_err(anyhow::Error::msg)?;

    if let Some(ref cfg) = cli_config {
        if let Some(ref prompt) = cfg.system_prompt {
            harness.system_prompt = Some(prompt.clone());
        }
        if let Some(ref driver_str) = cfg.tool_execution_driver {
            let s = driver_str.trim().to_lowercase();
            if s != "auto" {
                if let Some(driver) = ToolExecutionDriver::parse(driver_str) {
                    harness.tool_execution_driver = Some(driver);
                }
            }
        }
        if let Some(ref cmd) = cfg.notify_command {
            harness.notify_command = Some(cmd.clone());
        }
        if !cfg.formatter.is_empty() {
            harness.formatter_rules = crate::bootstrap::formatter_rules_from_config(cfg);
        }
    }

    Ok(harness)
}

/// Keys written to the CLI settings database after a successful PUT /api/settings.
#[must_use]
pub(crate) fn persisted_settings_updates(
    settings: &caliche::AgentSettings,
) -> Vec<(String, String)> {
    let mut updates = vec![
        ("last_provider".to_string(), settings.provider.to_string()),
        ("last_model".to_string(), settings.model.clone()),
        (
            "show_sources".to_string(),
            if settings.show_sources.unwrap_or(true) {
                "true".to_string()
            } else {
                "false".to_string()
            },
        ),
    ];
    if let Some(provider) = settings
        .embedding_provider
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        updates.push(("embedding_provider".to_string(), provider.to_string()));
    }
    if let Some(model) = settings
        .embedding_model
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        updates.push(("embedding_model".to_string(), model.to_string()));
    }
    updates
}

/// Build router options for the embedded CLI server.
///
/// # Errors
///
/// Returns an error when the CLI permission mode is not a known mode.
pub async fn build_router_options(
    shared_service: Arc<RwLock<AgentService>>,
    args: &crate::commands::Args,
    cli_config: Option<&crate::config::Config>,
    database: Option<Database>,
) -> Result<RouterOptions> {
    let harness = build_harness_config_from_cli(shared_service.clone(), args, cli_config).await?;

    let share_config = cli_config
        .and_then(|cfg| cfg.share.as_ref())
        .and_then(|share| match share.to_runtime() {
            Ok(config) => Some(config),
            Err(error) => {
                tracing::warn!("Ignoring invalid [share] config: {error}");
                None
            }
        });

    let on_settings_applied = database.map(|db| {
        std::sync::Arc::new(move |settings: &caliche::AgentSettings| {
            let db = db.clone();
            let updates = persisted_settings_updates(settings);
            tokio::spawn(async move {
                for (key, value) in updates {
                    if let Err(error) = db.save_setting(&key, &value).await {
                        tracing::warn!(error = %error, key, "failed to persist setting");
                    }
                }
            });
        }) as std::sync::Arc<dyn Fn(&caliche::AgentSettings) + Send + Sync>
    });

    Ok(RouterOptions {
        harness_config: Some(harness),
        shared_service: Some(shared_service),
        context_window: args
            .context_window
            .or_else(|| cli_config.and_then(|cfg| cfg.context_window)),
        endpoint_manager: None,
        rpc_method_registrars: vec![Arc::new(rpc::register_cli_rpc_methods)],
        share_config,
        on_settings_applied,
        mcp_config_path: None,
    })
}
async fn build_app(
    shared_service: Arc<RwLock<AgentService>>,
    mcp_transport: Option<String>,
    router_options: Option<RouterOptions>,
) -> Result<axum::Router> {
    let guard = shared_service.read().await;
    let tool_registry = guard.tool_registry().clone();
    let default_provider = guard.provider();
    let default_model = guard.model().to_string();
    drop(guard);
    let mut router = create_router(default_provider, default_model, router_options).await?;
    router = add_cli_parity_routes(router);

    if let Some(ref transport_str) = mcp_transport {
        router =
            mcp::attach_mcp_transport_routes(router, Arc::clone(&tool_registry), transport_str)
                .await;
    }

    if cors_permissive_from_env()? {
        router = router.layer(CorsLayer::permissive());
    }
    Ok(router.layer(axum::middleware::from_fn(
        routes::require_loopback_for_credential_routes,
    )))
}

pub async fn start_server(
    shared_service: Arc<RwLock<AgentService>>,
    bind: String,
    port: u16,
    mcp_transport: Option<String>,
    router_options: Option<RouterOptions>,
) -> Result<()> {
    // `--bind` defaults to 127.0.0.1 and does not read API_HOST.
    warn_if_open_unspecified_bind(&bind);
    let addr = listen_socket_addr(&bind, port)?;

    let app = build_app(shared_service, mcp_transport, router_options).await?;

    println!("\nStarting HTTP server on http://{}", addr);
    #[cfg(feature = "embedded-ui")]
    println!(
        "Embedded UI at http://{addr}/ on the same origin as /api and /health. An open bind with auth off serves this UI on every interface. A bearer pasted in the UI is an operator credential."
    );
    #[cfg(not(feature = "embedded-ui"))]
    println!(
        "No embedded UI in this binary. GET / returns 404. From the repo, `./caliche rust install` builds the wasm UI and cargo-installs this CLI with --features embedded-ui. GET /health is liveness. A 404 on / means the UI feature or assets are absent."
    );

    let listener = TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind to {}: {}", addr, e))?;

    let registry_dir = crate::server_registry::default_registry_dir();
    if let Err(err) = crate::server_registry::prune_dead_records(&registry_dir) {
        tracing::warn!(
            error = %err,
            path = %registry_dir.display(),
            "could not prune dead server records"
        );
    }
    let lock = crate::client::ServerLock::new(bind.clone(), port);
    let lock_path = crate::client::ServerLock::default_path();
    if let Err(err) = crate::client::clear_stale_server_lock(&lock_path) {
        tracing::warn!(
            error = %err,
            path = %lock_path.display(),
            "could not remove stale server.lock"
        );
    }
    if let Err(err) = lock.write_to(&lock_path) {
        tracing::warn!(error = %err, path = %lock_path.display(), "failed to write server.lock");
    } else {
        tracing::info!(path = %lock_path.display(), url = %lock.url, "wrote server.lock");
    }
    if let Err(err) = register_bound_server(
        &registry_dir,
        crate::server_registry::ServerOrigin::from_process_env(),
        &bind,
        port,
    ) {
        let _ = crate::client::ServerLock::remove_from(&lock_path, false);
        return Err(err);
    }

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let signal_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        let signal = caliche_server::shutdown::recv_process_signal().await;
        tracing::info!(
            signal = signal.as_str(),
            "shutdown signal received; draining"
        );
        let _ = signal_tx.send(true);
    });
    let server = serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(caliche_server::shutdown::wait_for_shutdown(
        shutdown_rx.clone(),
    ))
    .into_future();
    let serve_result = shutdown::select_server(server, shutdown_rx).await;

    if let Err(err) = crate::server_registry::unregister_current(&registry_dir) {
        tracing::warn!(
            error = %err,
            path = %registry_dir.display(),
            "failed to remove server record"
        );
    }
    let _ = crate::client::ServerLock::remove_from(&lock_path, false);
    serve_result
}

/// Register this process, or return the error so the caller can rely on a record id.
fn register_bound_server(
    registry_dir: &Path,
    origin: crate::server_registry::ServerOrigin,
    bind: &str,
    port: u16,
) -> Result<crate::server_registry::ServerRecord> {
    crate::server_registry::register_current(registry_dir, origin, bind, port)
        .context("failed to register server record")
}

/// Bind an ephemeral loopback port and serve the API in a background task.
///
/// Used by the thin-client REPL when no remote server.lock is present. The
/// caller is responsible for writing `server.lock` (via
/// [`crate::client::AgentClient::register_local_server`]) and aborting the task
/// on shutdown.
pub async fn start_embedded_server_background(
    shared_service: Arc<RwLock<AgentService>>,
    router_options: Option<RouterOptions>,
) -> Result<(String, u16, tokio::task::JoinHandle<Result<()>>)> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind embedded server: {}", e))?;
    let addr = listener
        .local_addr()
        .map_err(|e| anyhow::anyhow!("Failed to read embedded bind address: {}", e))?;
    let host = addr.ip().to_string();
    let port = addr.port();

    let app = build_app(shared_service, None, router_options).await?;
    let task = tokio::spawn(async move {
        serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Embedded server error: {}", e))
    });

    tracing::info!(%host, port, "started embedded agent server for thin-client REPL");
    Ok((host, port, task))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::Args;
    use async_trait::async_trait;
    use caliche::AgentSettings;
    use caliche::harness::config::PermissionModeConfig;
    use caliche::llm::ProviderType;
    use caliche::traits::credentials::{CredentialProvider, Credentials};
    use clap::Parser;

    #[derive(Debug)]
    struct MissingKey;

    #[async_trait]
    impl CredentialProvider for MissingKey {
        async fn get_credentials(&self, provider: &str) -> anyhow::Result<Credentials> {
            anyhow::bail!(
                "No credentials found for provider '{provider}'. Set {provider}_API_KEY or API_KEY environment variable."
            )
        }

        fn supports_provider(&self, _provider: &str) -> bool {
            true
        }

        fn source_name(&self) -> &str {
            "missing-key-test"
        }
    }

    fn settings() -> AgentSettings {
        AgentSettings {
            provider: ProviderType::OpenAI,
            model: "gpt-4o".to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: None,
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
        }
    }

    #[test]
    fn persist_hook_omits_empty_embedding_when_snapshot_has_none() {
        let updates = persisted_settings_updates(&settings());
        assert!(updates.contains(&("last_provider".to_string(), "openai".to_string())));
        assert!(updates.contains(&("last_model".to_string(), "gpt-4o".to_string())));
        assert!(!updates.iter().any(|(key, _)| key == "embedding_provider"));
        assert!(!updates.iter().any(|(key, _)| key == "embedding_model"));
        assert!(updates.contains(&("show_sources".to_string(), "true".to_string())));
    }

    #[test]
    fn persist_hook_saves_embedding_and_hidden_sources() {
        let mut payload = settings();
        payload.embedding_provider = Some("openai".to_string());
        payload.embedding_model = Some("text-embedding-3-small".to_string());
        payload.show_sources = Some(false);

        let updates = persisted_settings_updates(&payload);
        assert!(updates.contains(&("embedding_provider".to_string(), "openai".to_string())));
        assert!(updates.contains(&(
            "embedding_model".to_string(),
            "text-embedding-3-small".to_string()
        )));
        assert!(updates.contains(&("show_sources".to_string(), "false".to_string())));
    }

    /// `--server` listen proof: session bookkeeping does not need a model key.
    /// A turn on the selected provider fails with that provider's credential error.
    #[tokio::test]
    async fn listen_create_list_cancel_without_model_key_turn_fails() {
        let service = deferred_openrouter_without_key().await;
        let shared = Arc::new(RwLock::new(service));
        let options = RouterOptions {
            harness_config: None,
            shared_service: Some(Arc::clone(&shared)),
            endpoint_manager: None,
            context_window: None,
            rpc_method_registrars: Vec::new(),
            share_config: None,
            on_settings_applied: None,
            mcp_config_path: None,
        };
        let (host, port, server) = start_embedded_server_background(shared, Some(options))
            .await
            .expect("loopback listen");
        let base = format!("http://{host}:{port}");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("http client");

        let (create_status, create_body) = exchange(
            &client,
            reqwest::Method::POST,
            &format!("{base}/api/agent/sessions"),
            Some("{}"),
        )
        .await;
        assert_eq!(create_status, 200, "{create_body}");
        let session_id = create_body
            .split("\"session_id\":\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("session id")
            .to_owned();

        let (list_status, list_body) = exchange(
            &client,
            reqwest::Method::GET,
            &format!("{base}/api/agent/sessions"),
            None,
        )
        .await;
        assert_eq!(list_status, 200, "{list_body}");
        assert!(list_body.contains(&session_id), "{list_body}");

        let (turn_status, turn_body) = exchange(
            &client,
            reqwest::Method::POST,
            &format!("{base}/api/agent/sessions/{session_id}/turn"),
            Some(r#"{"input":"hello"}"#),
        )
        .await;
        assert_eq!(turn_status, 500, "{turn_body}");
        assert!(turn_body.contains("No credentials found"), "{turn_body}");
        assert!(turn_body.contains("provider 'openrouter'"), "{turn_body}");
        let turn_lower = turn_body.to_ascii_lowercase();
        assert!(!turn_lower.contains("fallback"), "{turn_body}");

        let (cancel_status, cancel_body) = exchange(
            &client,
            reqwest::Method::POST,
            &format!("{base}/api/agent/sessions/{session_id}/cancel"),
            None,
        )
        .await;
        assert_eq!(cancel_status, 200, "{cancel_body}");
        assert!(cancel_body.contains("\"cancelled\":true"), "{cancel_body}");

        server.abort();
    }

    /// A selected provider whose base URL egress rejects still lets the
    /// process listen. The URL is link-local, which stays blocked. Bookkeeping
    /// stays 200. The turn reports that egress error for the selected provider.
    #[tokio::test]
    async fn listen_create_list_cancel_with_blocked_ollama_url_turn_fails() {
        let service = deferred_blocked_ollama().await;
        let shared = Arc::new(RwLock::new(service));
        let options = RouterOptions {
            harness_config: None,
            shared_service: Some(Arc::clone(&shared)),
            endpoint_manager: None,
            context_window: None,
            rpc_method_registrars: Vec::new(),
            share_config: None,
            on_settings_applied: None,
            mcp_config_path: None,
        };
        let (host, port, server) = start_embedded_server_background(shared, Some(options))
            .await
            .expect("loopback listen");
        let base = format!("http://{host}:{port}");
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .expect("http client");

        let (create_status, create_body) = exchange(
            &client,
            reqwest::Method::POST,
            &format!("{base}/api/agent/sessions"),
            Some("{}"),
        )
        .await;
        assert_eq!(create_status, 200, "{create_body}");
        let session_id = create_body
            .split("\"session_id\":\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .expect("session id")
            .to_owned();

        let (list_status, list_body) = exchange(
            &client,
            reqwest::Method::GET,
            &format!("{base}/api/agent/sessions"),
            None,
        )
        .await;
        assert_eq!(list_status, 200, "{list_body}");
        assert!(list_body.contains(&session_id), "{list_body}");

        let (turn_status, turn_body) = exchange(
            &client,
            reqwest::Method::POST,
            &format!("{base}/api/agent/sessions/{session_id}/turn"),
            Some(r#"{"input":"hello"}"#),
        )
        .await;
        assert_eq!(turn_status, 500, "{turn_body}");
        assert!(turn_body.contains("provider 'ollama'"), "{turn_body}");
        let turn_lower = turn_body.to_ascii_lowercase();
        assert!(
            turn_lower.contains("egress provider url rejected")
                || turn_lower.contains("plugin http url targets a blocked host")
                || turn_lower.contains("egress denied for provider"),
            "{turn_body}"
        );
        assert!(!turn_body.contains("No credentials found"), "{turn_body}");
        assert!(!turn_lower.contains("openrouter"), "{turn_body}");

        let (cancel_status, cancel_body) = exchange(
            &client,
            reqwest::Method::POST,
            &format!("{base}/api/agent/sessions/{session_id}/cancel"),
            None,
        )
        .await;
        assert_eq!(cancel_status, 200, "{cancel_body}");
        assert!(cancel_body.contains("\"cancelled\":true"), "{cancel_body}");

        server.abort();
    }

    async fn deferred_blocked_ollama() -> AgentService {
        caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(
                    Some(caliche::llm::ProviderType::Ollama),
                    Some("llama3.2".to_string()),
                )
                .with_base_url(Some("http://169.254.169.254/v1".to_string()))
                .with_tool_names(Vec::new())
                .with_credential_provider(Arc::new(
                    caliche::traits::credentials::EnvironmentCredentialProvider::new(),
                ))
                .with_defer_missing_credentials(true),
        )
        .await
        .expect("listen with an egress-blocked provider url")
    }

    async fn deferred_openrouter_without_key() -> AgentService {
        caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(
                    Some(caliche::llm::ProviderType::OpenRouter),
                    Some("test-model".to_string()),
                )
                .with_tool_names(Vec::new())
                .with_credential_provider(Arc::new(MissingKey))
                .with_defer_missing_credentials(true),
        )
        .await
        .expect("listen without a model key")
    }

    async fn exchange(
        client: &reqwest::Client,
        method: reqwest::Method,
        url: &str,
        body: Option<&str>,
    ) -> (u16, String) {
        let mut request = client.request(method, url);
        if let Some(body) = body {
            request = request
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_owned());
        }
        let response = request.send().await.expect("http response");
        let status = response.status().as_u16();
        let text = response.text().await.expect("http body");
        (status, text)
    }

    #[test]
    fn register_bound_server_returns_a_record_id() {
        let dir = tempfile::tempdir().expect("temp dir");
        let record = register_bound_server(
            dir.path(),
            crate::server_registry::ServerOrigin::Foreground,
            "127.0.0.1",
            9,
        )
        .expect("register");
        assert_eq!(record.pid, std::process::id());
        assert_eq!(record.port, 9);
        assert_eq!(
            record.id,
            format!("foreground-9-{}-{}", std::process::id(), record.started_at)
        );
        assert!(dir.path().join(format!("{}.json", record.id)).exists());
    }

    #[test]
    fn register_bound_server_returns_directory_errors() {
        let dir = tempfile::tempdir().expect("temp dir");
        let blocker = dir.path().join("blocked");
        std::fs::write(&blocker, "not a directory").expect("blocker");
        let err = register_bound_server(
            &blocker,
            crate::server_registry::ServerOrigin::Foreground,
            "127.0.0.1",
            9,
        )
        .expect_err("registration failure must reach the caller");
        let text = format!("{err:#}");
        assert!(text.contains("failed to register server record"), "{text}");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn unwritable_registry_aborts_start_server_and_removes_the_lock() {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set(caliche_server::paths::CONFIG_DIR_ENV, dir.path());
        std::fs::write(dir.path().join("servers"), "not a directory").expect("blocker");
        let shared = Arc::new(RwLock::new(deferred_openrouter_without_key().await));
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            start_server(shared, "127.0.0.1".to_string(), 0, None, None),
        )
        .await;
        let err = match result {
            Ok(Err(err)) => err,
            Ok(Ok(())) => panic!("start_server kept serving after registration failed"),
            Err(_) => panic!("start_server did not return when the registry was not writable"),
        };
        let text = format!("{err:#}");
        assert!(text.contains("failed to register server record"), "{text}");
        assert!(
            !crate::client::ServerLock::default_path().exists(),
            "server.lock must be removed when registration fails"
        );
    }

    #[tokio::test]
    async fn cli_server_rejects_unknown_permission_mode() {
        let shared = Arc::new(RwLock::new(deferred_openrouter_without_key().await));
        let unknown = Args::parse_from(["caliche", "--server", "--permission-mode", "nope"]);
        let error = build_harness_config_from_cli(Arc::clone(&shared), &unknown, None)
            .await
            .expect_err("unknown mode");
        let text = format!("{error:#}");
        assert!(text.contains("nope"), "{text}");
        assert!(text.contains("default"), "{text}");
        assert!(text.contains("accept-edits"), "{text}");
        assert!(text.contains("accept-all"), "{text}");
        assert!(text.contains("deny-all"), "{text}");
        assert!(text.contains("interactive"), "{text}");

        let known = Args::parse_from(["caliche", "--permission-mode", "ACCEPT-ALL"]);
        let harness = build_harness_config_from_cli(Arc::clone(&shared), &known, None)
            .await
            .expect("known mode");
        assert_eq!(harness.permission_mode, PermissionModeConfig::AcceptAll);

        let missing = Args::parse_from(["caliche"]);
        let harness = build_harness_config_from_cli(Arc::clone(&shared), &missing, None)
            .await
            .expect("missing mode");
        assert_eq!(harness.permission_mode, PermissionModeConfig::Default);

        let empty = Args::parse_from(["caliche", "--permission-mode", ""]);
        let harness = build_harness_config_from_cli(shared, &empty, None)
            .await
            .expect("empty mode");
        assert_eq!(harness.permission_mode, PermissionModeConfig::Default);
    }
}
