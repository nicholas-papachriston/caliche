//! Type-erased sandbox handle.

use crate::config::{SandboxMode, ToolPolicy};
use crate::ops::SandboxHandleOps;
use crate::policy;
use crate::traits::{ExecResult, SandboxCapabilities, SandboxInstance};
use anyhow::{Result, bail};
use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

/// Type-erased sandbox handle with policy check.
pub struct SandboxHandle {
    instance: Box<dyn SandboxInstance>,
    policy: ToolPolicy,
    mode: SandboxMode,
    /// When mode is [`SandboxMode::NonMain`], sandboxing applies only when this is `false`.
    is_main_agent: bool,
    /// Whether the backend reported filesystem isolation (see [`Self::with_capabilities`]).
    confines_filesystem: bool,
}

impl SandboxHandle {
    #[must_use]
    pub fn new(instance: Box<dyn SandboxInstance>, policy: ToolPolicy) -> Self {
        Self {
            instance,
            policy,
            mode: SandboxMode::All,
            is_main_agent: true,
            confines_filesystem: false,
        }
    }

    /// Record what the backend that created the instance can isolate.
    ///
    /// Without this call the handle reports no filesystem confinement.
    #[must_use]
    pub const fn with_capabilities(mut self, capabilities: &SandboxCapabilities) -> Self {
        self.confines_filesystem = capabilities.filesystem_isolation;
        self
    }

    /// Whether the backend confines file access to the sandbox policy.
    #[must_use]
    pub const fn confines_filesystem(&self) -> bool {
        self.confines_filesystem
    }

    #[must_use]
    pub const fn with_mode(mut self, mode: SandboxMode) -> Self {
        self.mode = mode;
        self
    }

    /// Mark whether this handle serves the primary (main) agent.
    ///
    /// With [`SandboxMode::NonMain`], only non-main agents are sandboxed.
    #[must_use]
    pub const fn with_main_agent(mut self, is_main: bool) -> Self {
        self.is_main_agent = is_main;
        self
    }

    #[must_use]
    pub fn is_tool_allowed(&self, tool_name: &str) -> bool {
        policy::evaluate_tool_policy(&self.policy, tool_name)
    }

    #[must_use]
    pub const fn sandbox_active(&self) -> bool {
        match self.mode {
            SandboxMode::Off => false,
            SandboxMode::All => true,
            SandboxMode::NonMain => !self.is_main_agent,
        }
    }

    /// Execute a command in the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when sandboxing is inactive or command execution fails.
    pub async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult> {
        if !self.sandbox_active() {
            bail!("sandbox mode is Off; refusing sandbox exec");
        }
        self.instance.exec(command, working_dir, env, timeout).await
    }

    /// Read a file from the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when sandboxing is inactive or the file cannot be read.
    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        if !self.sandbox_active() {
            bail!("sandbox mode is Off; refusing sandbox read_file");
        }
        self.instance.read_file(path).await
    }

    /// Write a file in the sandbox.
    ///
    /// # Errors
    ///
    /// Returns an error when sandboxing is inactive or the file cannot be written.
    pub async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        if !self.sandbox_active() {
            bail!("sandbox mode is Off; refusing sandbox write_file");
        }
        self.instance.write_file(path, content).await
    }

    /// Destroy the sandbox instance.
    ///
    /// # Errors
    ///
    /// Returns an error when backend cleanup fails.
    pub async fn destroy(self) -> Result<()> {
        self.instance.destroy().await
    }
}

#[async_trait]
impl SandboxHandleOps for SandboxHandle {
    async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult> {
        Self::exec(self, command, working_dir, env, timeout).await
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        Self::read_file(self, path).await
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        Self::write_file(self, path, content).await
    }

    fn sandbox_active(&self) -> bool {
        Self::sandbox_active(self)
    }

    fn confines_filesystem(&self) -> bool {
        Self::confines_filesystem(self)
    }

    fn is_tool_allowed(&self, tool_name: &str) -> bool {
        Self::is_tool_allowed(self, tool_name)
    }
}
