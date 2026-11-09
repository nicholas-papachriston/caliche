//! A sandbox instance created when a sandboxed tool first uses it.
//!
//! Creating an instance runs the backend's checks of the workspace: every
//! backend reads git indexes, and the Linux and Docker backends also scan the
//! writable roots for nested `.git` entries. [`DeferredInstance`] lets a
//! caller start without paying for them: they run once, on first use. The
//! first call creates the real instance and keeps the result. When creation
//! fails, that call and every later one return the refusal, so sandboxed
//! tools fail closed and nothing runs outside the sandbox.

use crate::config::SandboxConfig;
use crate::traits::{ExecResult, SandboxBackend, SandboxInstance};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;

/// Start of the error every call returns when the instance was refused.
pub const SANDBOX_UNAVAILABLE: &str = "sandboxed tools are unavailable in this workspace";

/// A [`SandboxInstance`] that creates the backend's instance on first use.
#[derive(Debug)]
pub struct DeferredInstance {
    id: String,
    backend: Arc<dyn SandboxBackend>,
    config: SandboxConfig,
    workspace: PathBuf,
    created: OnceCell<Result<Box<dyn SandboxInstance>, String>>,
}

impl DeferredInstance {
    /// An instance of `backend` for `workspace`, created on first use.
    #[must_use]
    pub fn new(
        backend: Arc<dyn SandboxBackend>,
        config: SandboxConfig,
        workspace: PathBuf,
    ) -> Self {
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            backend,
            config,
            workspace,
            created: OnceCell::new(),
        }
    }

    /// The created instance, or the refusal as an error.
    async fn instance(&self) -> Result<&dyn SandboxInstance> {
        let created = self
            .created
            .get_or_init(|| async {
                self.backend
                    .create(&self.config, &self.workspace)
                    .await
                    .map_err(|err| format!("{err:#}"))
            })
            .await;
        match created {
            Ok(instance) => Ok(instance.as_ref()),
            Err(reason) => Err(anyhow::anyhow!("{SANDBOX_UNAVAILABLE}: {reason}")),
        }
    }
}

#[async_trait]
impl SandboxInstance for DeferredInstance {
    fn id(&self) -> &str {
        &self.id
    }

    async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult> {
        self.instance()
            .await?
            .exec(command, working_dir, env, timeout)
            .await
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        self.instance().await?.read_file(path).await
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        self.instance().await?.write_file(path, content).await
    }

    async fn is_alive(&self) -> bool {
        match self.created.get() {
            None => true,
            Some(Ok(instance)) => instance.is_alive().await,
            Some(Err(_)) => false,
        }
    }

    async fn destroy(&self) -> Result<()> {
        match self.created.get() {
            Some(Ok(instance)) => instance.destroy().await,
            None | Some(Err(_)) => Ok(()),
        }
    }

    async fn reset(&self) -> Result<()> {
        self.instance().await?.reset().await
    }
}
