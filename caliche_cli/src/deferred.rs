//! Startup work that must not block the REPL prompt.
//!
//! Embedding-provider probes take seconds (each probe is bounded by a ~5s
//! timeout), and building the workspace store lists the repositories under the
//! root. Interactive hosts spawn all of that here; polling only installs a
//! finished result, so it never waits on the network or the file tree.

use caliche::AgentService;
use caliche::RagTool;
use caliche::llm::{EmbeddingProvider, EmbeddingStopped};
use caliche::workspace::WorkspaceIndexerConfig;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

/// The one line shown when the embedding provider stopped at startup.
/// The session goes on without workspace search.
fn embedding_stopped_line(stopped: &EmbeddingStopped) -> String {
    // The provider's reason may end with its own period.
    let reason = stopped.reason.trim_end().trim_end_matches('.');
    format!(
        "Workspace search is off for this session: {reason}. Load or download the embedding model, or start with --no-workspace-embed."
    )
}

/// Workspace RAG tools built off the input path, ready to install.
struct Prepared {
    provider: Arc<dyn EmbeddingProvider>,
    config: WorkspaceIndexerConfig,
    local: Arc<RagTool>,
    server: Option<Arc<RagTool>>,
}

/// Background workspace-RAG attachment for the interactive REPL.
pub struct DeferredWorkspaceRag {
    discovery:
        Option<JoinHandle<Option<caliche::discovery::embedding::EmbeddingProviderDiscovery>>>,
    build: Option<JoinHandle<anyhow::Result<Prepared>>>,
    config: Option<WorkspaceIndexerConfig>,
    server_service: Option<Arc<RwLock<AgentService>>>,
    /// Discover a provider when the service has none at the first poll.
    discover: bool,
}

impl DeferredWorkspaceRag {
    /// Attach workspace RAG without waiting. The first poll uses the
    /// service's embedding provider; only when it has none does embedding
    /// discovery start, so a provider is never probed twice.
    #[must_use]
    pub fn spawn(config: WorkspaceIndexerConfig) -> Self {
        Self {
            discovery: None,
            build: None,
            config: Some(config),
            server_service: None,
            discover: true,
        }
    }

    /// Attach workspace RAG with a provider that is already on the service.
    /// Does not probe the network for a replacement provider.
    #[must_use]
    pub fn spawn_without_discovery(config: WorkspaceIndexerConfig) -> Self {
        Self {
            discovery: None,
            build: None,
            config: Some(config),
            server_service: None,
            discover: false,
        }
    }

    /// Also attach RAG to the thin-client server snapshot when discovery finishes.
    pub fn set_server_service(&mut self, service: Arc<RwLock<AgentService>>) {
        self.server_service = Some(service);
    }

    /// Advance the attachment without waiting: start building the workspace
    /// tool once a provider is known (the one on the service, else the
    /// discovered one), and install it once built.
    ///
    /// Returns a short status line when work completed on this poll.
    pub async fn poll_apply(&mut self, service: &mut AgentService) -> Option<String> {
        if self.build.is_some() {
            return self.poll_build(service).await;
        }

        if let Some(existing) = service.embedding_provider() {
            let config = self.config.take()?;
            if let Some(handle) = self.discovery.take() {
                handle.abort();
            }
            self.start_build(config, existing);
            return None;
        }

        if self.discover {
            self.discover = false;
            self.config.as_ref()?;
            self.discovery = Some(tokio::spawn(async {
                Box::pin(caliche::discovery::embedding::discover_best_embedding_provider()).await
            }));
            return None;
        }

        if self.discovery.is_none() {
            self.config.take()?;
            return Some(
                "Workspace search is waiting. Saved embedding provider was not restored. Use /embedding to set one."
                    .to_string(),
            );
        }

        let handle = self.discovery.as_ref()?;
        if !handle.is_finished() {
            return None;
        }
        let handle = self.discovery.take()?;
        let discovery = match handle.await {
            Ok(found) => found,
            Err(error) => {
                tracing::warn!(error = %error, "deferred embedding discovery task failed");
                self.config.take();
                return Some(
                    "Workspace search unavailable (embedding discovery failed).".to_string(),
                );
            }
        };
        let config = self.config.take()?;
        let Some(discovery) = discovery else {
            tracing::debug!("Deferred embedding discovery found no provider");
            return Some(
                "Workspace search unavailable (no embedding provider). Use /embedding to set one."
                    .to_string(),
            );
        };
        self.start_build(config, Arc::clone(&discovery.provider));
        None
    }

    fn start_build(
        &mut self,
        config: WorkspaceIndexerConfig,
        provider: Arc<dyn EmbeddingProvider>,
    ) {
        let with_server = self.server_service.is_some();
        self.build = Some(tokio::spawn(async move {
            let indexed =
                AgentService::workspace_embedding_config(config.clone(), provider.as_ref());
            let local =
                AgentService::build_workspace_rag_tool(&indexed, Arc::clone(&provider)).await?;
            let server = if with_server {
                AgentService::build_workspace_rag_tool(&indexed, Arc::clone(&provider))
                    .await
                    .inspect_err(|error| {
                        tracing::warn!(
                            error = %error,
                            "failed to build deferred workspace RAG for the embedded server"
                        );
                    })
                    .ok()
            } else {
                None
            };
            Ok(Prepared {
                provider,
                config,
                local,
                server,
            })
        }));
    }

    async fn poll_build(&mut self, service: &mut AgentService) -> Option<String> {
        if !self.build.as_ref()?.is_finished() {
            return None;
        }
        let prepared = match self.build.take()?.await {
            Ok(Ok(prepared)) => prepared,
            Ok(Err(error)) => {
                tracing::warn!(error = %error, "failed to build deferred workspace RAG");
                if let Some(stopped) = error.downcast_ref::<EmbeddingStopped>() {
                    return Some(embedding_stopped_line(stopped));
                }
                return Some(format!("Workspace search failed to start: {error}"));
            }
            Err(error) => {
                tracing::warn!(error = %error, "deferred workspace RAG task failed");
                return Some("Workspace search failed to start.".to_string());
            }
        };
        // A provider chosen while the tool was being built wins.
        if let Some(current) = service.embedding_provider()
            && !Arc::ptr_eq(&current, &prepared.provider)
        {
            self.start_build(prepared.config, current);
            return None;
        }
        Some(self.install(service, prepared).await)
    }

    async fn install(&mut self, service: &mut AgentService, prepared: Prepared) -> String {
        let config =
            AgentService::workspace_embedding_config(prepared.config, prepared.provider.as_ref());
        if let Err(error) = service
            .attach_workspace_rag(
                config.clone(),
                Arc::clone(&prepared.provider),
                prepared.local,
            )
            .await
        {
            tracing::warn!(error = %error, "failed to attach deferred workspace RAG");
            return format!("Workspace search failed to start: {error}");
        }

        if let (Some(shared), Some(tool)) = (self.server_service.take(), prepared.server) {
            let mut server = shared.write().await;
            if let Err(error) = server
                .attach_workspace_rag(config, prepared.provider, tool)
                .await
            {
                tracing::warn!(
                    error = %error,
                    "failed to attach deferred workspace RAG on the embedded server"
                );
            }
        }

        "Indexing workspace in background.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::DeferredWorkspaceRag;
    use caliche::workspace::{PersistenceConfig, WorkspaceIndexerConfig};
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn spawn_returns_without_waiting_for_discovery() {
        let dir = tempfile::tempdir().expect("workspace");
        let started = Instant::now();
        let deferred = DeferredWorkspaceRag::spawn(
            WorkspaceIndexerConfig::new(dir.path().to_path_buf())
                .persistence(PersistenceConfig::InMemory),
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "spawning deferred discovery must not block on provider probes"
        );
        drop(deferred);
    }

    #[test]
    fn spawn_without_discovery_does_not_start_a_probe() {
        let dir = tempfile::tempdir().expect("workspace");
        let deferred = DeferredWorkspaceRag::spawn_without_discovery(
            WorkspaceIndexerConfig::new(dir.path().to_path_buf())
                .persistence(PersistenceConfig::InMemory),
        );
        assert!(
            deferred.discovery.is_none(),
            "a saved embedding provider must not start network discovery"
        );
        drop(deferred);
    }

    /// Answers after `delay`, like a local model that is still loading.
    #[derive(Debug)]
    struct SlowEmbedding {
        delay: Duration,
    }

    #[async_trait::async_trait]
    impl caliche::llm::EmbeddingProvider for SlowEmbedding {
        async fn embed(
            &self,
            request: caliche::llm::EmbeddingRequest,
        ) -> anyhow::Result<caliche::llm::EmbeddingResponse> {
            tokio::time::sleep(self.delay).await;
            Ok(caliche::llm::EmbeddingResponse {
                embeddings: request.texts.iter().map(|_| vec![0.5, 0.5, 0.5]).collect(),
                usage: None,
                model: "slow".to_string(),
            })
        }

        fn name(&self) -> &str {
            "slow"
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[serial_test::serial(env)]
    async fn polling_never_waits_on_the_embedding_provider() {
        let home = tempfile::tempdir().expect("home");
        let workspace = tempfile::tempdir().expect("workspace");
        std::fs::write(workspace.path().join("notes.md"), "# Notes\n").expect("file");
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("HOME", home.path());
        env.set(
            caliche_server::paths::CONFIG_DIR_ENV,
            home.path().join("config"),
        );
        let mut service = caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(
                    Some(caliche::llm::ProviderType::Ollama),
                    Some("gpt-oss:20b".to_string()),
                )
                .with_base_url(Some("http://127.0.0.1:9/v1".to_string()))
                .with_tool_names(Vec::new()),
        )
        .await
        .expect("service");
        service.set_embedding_selection(
            Some(std::sync::Arc::new(SlowEmbedding {
                delay: Duration::from_secs(2),
            })),
            Some("slow".to_string()),
            None,
        );
        // Disk persistence, the default: building the store probes the provider.
        let mut deferred = DeferredWorkspaceRag::spawn_without_discovery(
            WorkspaceIndexerConfig::new(workspace.path().to_path_buf()),
        );

        let deadline = Instant::now() + Duration::from_secs(20);
        let mut polls = Vec::new();
        let status = loop {
            let started = Instant::now();
            let status = deferred.poll_apply(&mut service).await;
            polls.push(started.elapsed());
            if let Some(status) = status {
                break status;
            }
            assert!(Instant::now() < deadline, "workspace RAG never attached");
            tokio::time::sleep(Duration::from_millis(20)).await;
        };

        let slowest = polls.iter().max().copied().unwrap_or_default();
        assert!(
            slowest < Duration::from_secs(1),
            "a poll waited {slowest:?} on the embedding provider"
        );
        assert!(
            polls.len() > 1,
            "the first poll attached: it waited for the probe"
        );
        assert_eq!(status, "Indexing workspace in background.");
        assert!(service.has_workspace_rag());
    }
}
