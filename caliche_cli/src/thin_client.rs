//! Bootstrap helpers for the transport-agnostic thin-client REPL (ROADMAP 2.5).
//!
//! Discovers a live HTTP server via [`crate::client::AgentClient`], or starts an
//! embedded loopback server and registers it in `server.lock` so message turns
//! go through HTTP + SSE. The local [`AgentService`] stays in-process for slash
//! commands / completions; agent turns execute on the server.

use crate::client::{AgentClient, AgentClientMode};
use crate::commands::Args;
use crate::server::{self, build_router_options};
use crate::storage::database::Database;
use anyhow::{Context, Result};
use caliche::{AgentService, AgentServiceConfig};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

/// Live thin-client handle used by the REPL for remote turns.
pub struct ThinClientSession {
    pub client: AgentClient,
    pub session_id: String,
    /// Background embedded server task (when we bootstrapped locally).
    pub server_task: Option<JoinHandle<Result<()>>>,
    /// Live embedded-server `AgentService` so deferred workspace RAG can attach.
    pub shared_service: Option<Arc<RwLock<AgentService>>>,
}

impl ThinClientSession {
    /// Tear down an embedded server and clear our lock file when we own it.
    pub async fn shutdown(self) {
        if let Some(task) = self.server_task {
            task.abort();
            let _ = task.await;
        }
        let _ = crate::server_registry::unregister_current(
            &crate::server_registry::default_registry_dir(),
        );
        let _ = self.client.clear_lock();
    }
}

/// Discover a remote server or start an embedded one, then create an API session.
///
/// `service` is snapshotted into router harness config for the embedded server and
/// returned unchanged for local REPL state (slash commands, completions).
pub async fn bootstrap_thin_client(
    service: AgentService,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
    database: Option<Database>,
) -> Result<(AgentService, ThinClientSession)> {
    let mut client = AgentClient::discover().context("discover agent server")?;

    let (server_task, shared_service) = match client.mode() {
        AgentClientMode::Remote { .. } => {
            if client.health().await.unwrap_or(false) {
                (None, None)
            } else {
                let _ = client.clear_lock();
                client = AgentClient::discover_at(client.lock_path().to_path_buf())?;
                let (task, shared) =
                    spawn_embedded(&service, args, cli_config, database.clone(), &mut client)
                        .await?;
                (Some(task), Some(shared))
            }
        }
        AgentClientMode::LocalBootstrap => {
            let (task, shared) =
                spawn_embedded(&service, args, cli_config, database, &mut client).await?;
            (Some(task), Some(shared))
        }
    };

    let mut healthy = false;
    for _ in 0..80 {
        if client.health().await.unwrap_or(false) {
            healthy = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    if !healthy {
        anyhow::bail!("agent server failed health check after thin-client bootstrap");
    }

    let session_id = client
        .create_session()
        .await
        .context("create remote agent session")?;

    Ok((
        service,
        ThinClientSession {
            client,
            session_id,
            server_task,
            shared_service,
        },
    ))
}

async fn spawn_embedded(
    service: &AgentService,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
    database: Option<Database>,
    client: &mut AgentClient,
) -> Result<(JoinHandle<Result<()>>, Arc<RwLock<AgentService>>)> {
    let snapshot = snapshot_service_for_server(service).await?;
    let shared = Arc::new(RwLock::new(snapshot));
    let router_options = build_router_options(shared.clone(), args, cli_config, database).await?;
    let (host, port, task) =
        server::start_embedded_server_background(shared.clone(), Some(router_options)).await?;
    if let Err(err) = register_embedded_server(
        client,
        &host,
        port,
        &crate::server_registry::default_registry_dir(),
    ) {
        task.abort();
        return Err(err);
    }
    Ok((task, shared))
}

/// Error returned when thin-client bootstrap fails.
///
/// The alternate format keeps the cause chain. `{err}` would drop it.
#[must_use]
pub fn format_bootstrap_failure(err: &anyhow::Error) -> anyhow::Error {
    anyhow::anyhow!("thin-client bootstrap failed: {err:#}")
}

/// Write the loopback lock and the registry record.
///
/// A registry failure removes the lock and returns the error. Callers abort
/// the embedded server task when this fails.
fn register_embedded_server(
    client: &mut AgentClient,
    host: &str,
    port: u16,
    registry_dir: &Path,
) -> Result<crate::server_registry::ServerRecord> {
    client
        .register_local_server(host, port)
        .context("register embedded server.lock")?;
    match crate::server_registry::register_current(
        registry_dir,
        crate::server_registry::ServerOrigin::Embedded,
        host,
        port,
    ) {
        Ok(record) => Ok(record),
        Err(err) => {
            let _ = client.clear_lock();
            Err(err).context("could not register embedded server")
        }
    }
}

async fn snapshot_service_for_server(service: &AgentService) -> Result<AgentService> {
    let provider = service.provider();
    let model = service.model().to_string();
    let tool_names: Vec<String> = service
        .list_tools()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();
    // Like a standalone server, it listens without a model key: completions
    // report the missing key, and the REPL stays up to fix it.
    let mut snapshot = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(Some(provider), Some(model))
            .with_tool_names_or_defaults(Some(tool_names))
            .with_defer_missing_credentials(true),
    )
    .await
    .context("create embedded-server AgentService snapshot")?;
    if let Some(embedding) = service.embedding_provider() {
        snapshot.set_embedding_selection(
            Some(embedding),
            service.embedding_provider_id().map(str::to_owned),
            service.embedding_model_id().map(str::to_owned),
        );
    }
    if let Some(window) = service.context_window_override() {
        snapshot.set_context_window(window)?;
    }
    let prompt = service.agent().system_prompt();
    if !prompt.is_empty() {
        snapshot.agent_mut().set_system_prompt(prompt.to_string());
    }
    snapshot.inherit_project_context(service);
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::snapshot_service_for_server;
    use caliche::llm::ProviderType;
    use caliche::{AgentService, AgentServiceConfig};
    use clap::Parser;

    #[tokio::test]
    #[serial_test::serial]
    async fn embedded_snapshot_copies_the_cli_system_prompt() {
        let mut source = AgentService::from_config(
            AgentServiceConfig::new()
                .with_provider(Some(ProviderType::OpenAI), Some("gpt-4o".to_string()))
                .with_defer_missing_credentials(true),
        )
        .await
        .expect("source service");
        source
            .agent_mut()
            .set_system_prompt("You are Caliche (caliche).");

        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("OPENAI_API_KEY", "test-key-for-snapshot");
        let snapshot = snapshot_service_for_server(&source)
            .await
            .expect("snapshot");
        drop(env);
        assert!(
            snapshot.agent().system_prompt().contains("caliche"),
            "embedded snapshot dropped the CLI prompt: {}",
            snapshot.agent().system_prompt()
        );
    }

    #[test]
    fn embedded_registration_failure_clears_the_lock_and_returns() {
        let dir = tempfile::tempdir().expect("temp dir");
        let lock_path = dir.path().join("server.lock");
        let mut client =
            crate::client::AgentClient::discover_at(lock_path.clone()).expect("client");
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, "x").expect("blocker");
        let err = super::register_embedded_server(&mut client, "127.0.0.1", 9, &blocker)
            .expect_err("registration failure must reach the caller");
        let text = format!("{err:#}");
        assert!(
            text.contains("could not register embedded server"),
            "{text}"
        );
        assert!(
            crate::client::ServerLock::read_from(&lock_path)
                .expect("read lock")
                .is_none()
        );
    }

    #[test]
    fn bootstrap_failure_keeps_the_cause_chain() {
        let err = anyhow::anyhow!("registry directory was not writable")
            .context("could not register embedded server");
        let text = super::format_bootstrap_failure(&err).to_string();
        assert!(text.contains("thin-client bootstrap failed"), "{text}");
        assert!(
            text.contains("could not register embedded server"),
            "{text}"
        );
        assert!(
            text.contains("registry directory was not writable"),
            "{text}"
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn unwritable_registry_aborts_the_embedded_server_and_removes_the_lock() {
        let dir = tempfile::tempdir().expect("temp dir");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set(caliche_server::paths::CONFIG_DIR_ENV, dir.path());
        std::fs::write(dir.path().join("servers"), "not a directory").expect("blocker");
        env.set("OPENAI_API_KEY", "test-key-for-embedded-registry");
        let service = AgentService::from_config(
            AgentServiceConfig::new()
                .with_provider(Some(ProviderType::OpenAI), Some("gpt-4o".to_string()))
                .with_defer_missing_credentials(true),
        )
        .await
        .expect("service");
        let args = crate::commands::Args::parse_from(["caliche"]);
        let lock_path = crate::client::ServerLock::default_path();
        let mut client =
            crate::client::AgentClient::discover_at(lock_path.clone()).expect("client");
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        let before = observed_listen_ports();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            super::spawn_embedded(&service, &args, None, None, &mut client),
        )
        .await;
        let err = match result {
            Ok(Err(err)) => err,
            Ok(Ok((task, _))) => {
                task.abort();
                panic!("spawn_embedded kept serving after registration failed");
            }
            Err(_) => panic!("spawn_embedded did not return when the registry was not writable"),
        };
        let text = format!("{err:#}");
        assert!(
            text.contains("could not register embedded server"),
            "{text}"
        );
        assert!(
            !lock_path.exists(),
            "server.lock must be removed when registration fails"
        );
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        assert_no_new_listeners(&before).await;
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    async fn assert_no_new_listeners(before: &std::collections::HashSet<u16>) {
        let mut still_listening = observed_listen_ports()
            .difference(before)
            .copied()
            .collect::<Vec<_>>();
        for _ in 0..20 {
            if still_listening.is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            still_listening = observed_listen_ports()
                .difference(before)
                .copied()
                .collect();
        }
        assert!(
            still_listening.is_empty(),
            "embedded server still listening on {still_listening:?}"
        );
    }

    /// 127.0.0.1 listeners. Ports already open cancel out of the diff.
    #[cfg(target_os = "linux")]
    fn observed_listen_ports() -> std::collections::HashSet<u16> {
        let text = std::fs::read_to_string("/proc/net/tcp").expect("/proc/net/tcp");
        let mut ports = std::collections::HashSet::new();
        for line in text.lines().skip(1) {
            let mut columns = line.split_whitespace();
            let Some(local) = columns.nth(1) else {
                continue;
            };
            let Some(state) = columns.nth(1) else {
                continue;
            };
            if state != "0A" {
                continue;
            }
            let Some((ip, port_hex)) = local.split_once(':') else {
                continue;
            };
            if !ip.eq_ignore_ascii_case("0100007F") {
                continue;
            }
            if let Ok(port) = u16::from_str_radix(port_hex, 16) {
                ports.insert(port);
            }
        }
        ports
    }

    /// Every listening TCP port. Ports already open cancel out of the diff.
    #[cfg(target_os = "windows")]
    fn observed_listen_ports() -> std::collections::HashSet<u16> {
        crate::process_liveness::current_listen_ports()
            .expect("tcp listen table")
            .into_iter()
            .collect()
    }
}
