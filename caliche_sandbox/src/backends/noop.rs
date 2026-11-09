//! No-op sandbox backend (runs on host, for testing).
//!
//! Exec is unsandboxed. File reads and writes still go through path
//! confinement unless the policy is `danger-full-access`.

use crate::config::SandboxConfig;
use crate::paths::{read_validated_file, write_validated_file};
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct NoopBackend;

#[async_trait]
impl SandboxBackend for NoopBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: true,
            filesystem_isolation: false,
            tcp_isolation: false,
            resource_limits: false,
            host_mounts: false,
            pooling: false,
            backend_name: "noop",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        Ok(Box::new(NoopInstance {
            id: uuid::Uuid::now_v7().to_string(),
            workspace: workspace.to_path_buf(),
            config: config.clone(),
        }))
    }
}

#[derive(Debug)]
struct NoopInstance {
    id: String,
    workspace: PathBuf,
    config: SandboxConfig,
}

impl NoopInstance {
    fn unrestricted(&self) -> bool {
        self.config.policy_name.as_deref() == Some("danger-full-access")
    }
}

#[async_trait]
impl SandboxInstance for NoopInstance {
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
        let cwd = working_dir.map_or_else(|| self.workspace.clone(), std::path::PathBuf::from);
        let mut cmd = std::process::Command::new(command.first().copied().unwrap_or(""));
        cmd.args(command.get(1..).unwrap_or(&[]))
            .current_dir(&cwd)
            .env_clear()
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(env) = env {
            cmd.envs(env);
        }
        // Its own process group, so a timeout kills the background processes too.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut cmd, 0);

        tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let child = cmd.spawn().context("spawn")?;
            #[cfg(unix)]
            let kill_tree = super::child_output::kill_process_group;
            #[cfg(not(unix))]
            let kill_tree = |_: &std::process::Child| {};
            super::child_output::collect(child, started, timeout, kill_tree)
        })
        .await
        .context("spawn_blocking join")?
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        if self.unrestricted() {
            let p = if path.starts_with('/') {
                PathBuf::from(path)
            } else {
                self.workspace.join(path)
            };
            return Ok(std::fs::read(&p)?);
        }
        let workspace = self.workspace.clone();
        let config = self.config.clone();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || read_validated_file(&workspace, &config, &path))
            .await
            .context("spawn_blocking join")?
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        if self.unrestricted() {
            let p = if path.starts_with('/') {
                PathBuf::from(path)
            } else {
                self.workspace.join(path)
            };
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            return Ok(std::fs::write(&p, content)?);
        }
        let content = content.to_vec();
        let workspace = self.workspace.clone();
        let config = self.config.clone();
        let path = path.to_string();
        tokio::task::spawn_blocking(move || {
            write_validated_file(&workspace, &config, &path, &content)
        })
        .await
        .context("spawn_blocking join")?
    }

    async fn is_alive(&self) -> bool {
        true
    }

    async fn destroy(&self) -> Result<()> {
        Ok(())
    }
}
