//! AgentFS-backed sandbox: filesystem isolation via AgentFS virtual filesystem.
//!
//! read_file/write_file delegate to AgentFS. Exec is refused: AgentFS does not
//! confine host processes (that would require a FUSE/NFS mount plus a native
//! backend). Use Seatbelt, Landlock, or Docker when commands must run.

use crate::config::SandboxConfig;
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn to_agentfs_path(path: &str) -> String {
    let path = path.trim_start_matches('/');
    if path.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", path)
    }
}

pub struct AgentFsSandboxBackend {
    agentfs: Arc<agentfs_sdk::AgentFS>,
}

impl std::fmt::Debug for AgentFsSandboxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentFsSandboxBackend")
            .finish_non_exhaustive()
    }
}

impl AgentFsSandboxBackend {
    pub fn new(agentfs: Arc<agentfs_sdk::AgentFS>) -> Self {
        Self { agentfs }
    }
}

#[async_trait]
impl SandboxBackend for AgentFsSandboxBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: false,
            filesystem_isolation: true,
            tcp_isolation: false,
            resource_limits: false,
            host_mounts: false,
            pooling: false,
            backend_name: "agentfs",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        Ok(Box::new(AgentFsSandboxInstance {
            id: uuid::Uuid::now_v7().to_string(),
            agentfs: Arc::clone(&self.agentfs),
            workspace: workspace.to_path_buf(),
        }))
    }
}

struct AgentFsSandboxInstance {
    id: String,
    agentfs: Arc<agentfs_sdk::AgentFS>,
    workspace: std::path::PathBuf,
}

impl std::fmt::Debug for AgentFsSandboxInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentFsSandboxInstance")
            .field("id", &self.id)
            .field("workspace", &self.workspace)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl SandboxInstance for AgentFsSandboxInstance {
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
        let _ = (command, working_dir, env, timeout);
        anyhow::bail!(
            "AgentFS sandbox does not isolate process execution; use a native Seatbelt, Landlock, or Docker backend"
        )
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let af_path = to_agentfs_path(path);
        self.agentfs
            .fs
            .read_file(&af_path)
            .await?
            .ok_or_else(|| anyhow::anyhow!("File not found: {}", path))
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        let af_path = to_agentfs_path(path);
        if let Some(parent) = Path::new(&af_path).parent()
            && parent != Path::new("/")
        {
            let parent_str = parent.to_string_lossy();
            let _ = self.agentfs.fs.mkdir(&parent_str, 0, 0).await;
        }
        self.agentfs.fs.pwrite(&af_path, 0, content).await?;
        Ok(())
    }

    async fn is_alive(&self) -> bool {
        true
    }

    async fn destroy(&self) -> Result<()> {
        Ok(())
    }
}
