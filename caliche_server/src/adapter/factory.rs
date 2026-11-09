//! Harness construction for API sessions.

use anyhow::Result;
use caliche::harness::{AgentHarness, HarnessConfig, SessionConfig};
use std::sync::Arc;

/// Factory for creating harnesses
pub struct HarnessFactory {
    /// Default configuration
    default_config: HarnessConfig,
    /// When set (CLI hosting), sessions are derived from this service via clone_for_request + from_service.
    shared_service: Option<Arc<tokio::sync::RwLock<caliche::AgentService>>>,
}

impl HarnessFactory {
    async fn prepare_service_for_config(
        &self,
        config: &HarnessConfig,
    ) -> Result<caliche::AgentService> {
        let shared = self
            .shared_service
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("shared service is not configured"))?;
        let guard = shared.read().await;
        let mut svc = guard.clone_for_request();
        drop(guard);

        if config.provider.is_some() || config.model.is_some() {
            let requested_provider = config.provider.or_else(|| Some(svc.provider()));
            let requested_model = config
                .model
                .clone()
                .or_else(|| Some(svc.model().to_string()));
            if requested_provider != Some(svc.provider())
                || requested_model.as_deref() != Some(svc.model())
            {
                svc.change_model(requested_provider, requested_model)
                    .await?;
            }
        }

        if let Some(ref tools) = config.tools {
            svc.agent_mut().set_allowed_tools(Some(tools.clone()));
        }

        Ok(svc)
    }

    async fn apply_session_config(
        harness: &mut AgentHarness,
        session_config: SessionConfig,
    ) -> Result<()> {
        if let Some(id) = session_config.session_id {
            harness.set_session_id(id);
        }
        if let Some(workspace) = session_config.workspace {
            harness
                .context_manager()
                .write()
                .await
                .set_workspace(workspace);
        }
        for file in session_config.context_files {
            let _ = harness
                .context_manager()
                .write()
                .await
                .track_file(&file)
                .await;
        }
        if let Some(max_turns) = session_config.max_turns {
            harness.set_max_turns(max_turns);
        }
        harness.set_history(session_config.history);
        if let Some(addition) = session_config.system_prompt_addition {
            let service = harness.service_mut();
            let current = service.agent().system_prompt().to_string();
            service
                .agent_mut()
                .set_system_prompt(format!("{}\n\n{}", current, addition));
        }
        Ok(())
    }

    /// Create a new factory with default config
    pub fn new(default_config: HarnessConfig) -> Self {
        Self {
            default_config,
            shared_service: None,
        }
    }

    /// Create a factory that derives harnesses from a shared AgentService (CLI-hosted mode).
    pub fn new_with_shared(
        default_config: HarnessConfig,
        shared_service: Option<Arc<tokio::sync::RwLock<caliche::AgentService>>>,
    ) -> Self {
        Self {
            default_config,
            shared_service,
        }
    }

    /// Create a harness with default config
    pub async fn create(&self) -> Result<AgentHarness> {
        if self.shared_service.is_some() {
            let svc = self
                .prepare_service_for_config(&self.default_config)
                .await?;
            AgentHarness::from_service(svc, self.default_config.clone())
        } else {
            AgentHarness::new(self.default_config.clone()).await
        }
    }

    /// Create a harness with session config
    pub async fn create_with_session(&self, session_config: SessionConfig) -> Result<AgentHarness> {
        if self.shared_service.is_some() {
            let svc = self
                .prepare_service_for_config(&self.default_config)
                .await?;
            let mut harness = AgentHarness::from_service(svc, self.default_config.clone())?;
            Self::apply_session_config(&mut harness, session_config).await?;
            Ok(harness)
        } else {
            Box::pin(AgentHarness::with_session(
                self.default_config.clone(),
                session_config,
            ))
            .await
        }
    }

    /// Create a harness with custom config
    pub async fn create_with_config(&self, config: HarnessConfig) -> Result<AgentHarness> {
        if self.shared_service.is_some() {
            let svc = self.prepare_service_for_config(&config).await?;
            AgentHarness::from_service(svc, config)
        } else {
            AgentHarness::new(config).await
        }
    }

    /// Create a harness with custom harness config plus session overrides.
    pub async fn create_with_config_and_session(
        &self,
        config: HarnessConfig,
        session_config: SessionConfig,
    ) -> Result<AgentHarness> {
        if self.shared_service.is_some() {
            let mut harness = self.create_with_config(config).await?;
            Self::apply_session_config(&mut harness, session_config).await?;
            Ok(harness)
        } else {
            Box::pin(AgentHarness::with_session(config, session_config)).await
        }
    }
}
