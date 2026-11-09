//! Sandbox backend and instance traits.

use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Most bytes of stdout, and separately of stderr, that one `exec` keeps.
/// Output past the cap is read and discarded, so the command never blocks on a
/// full pipe, and the stream's `*_truncated` flag is set.
pub const EXEC_OUTPUT_LIMIT: usize = 4 * 1024 * 1024;

/// Result of executing a command inside a sandbox.
#[derive(Debug, Clone)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    /// True when stdout passed [`EXEC_OUTPUT_LIMIT`] and the rest was dropped.
    pub stdout_truncated: bool,
    /// True when stderr passed [`EXEC_OUTPUT_LIMIT`] and the rest was dropped.
    pub stderr_truncated: bool,
    pub exit_code: i64,
    pub duration: Duration,
}

/// Error from `exec` when the command, or a process holding its output pipes,
/// is still running at the timeout. Carries the output read before the kill.
#[derive(Debug, Clone)]
pub struct ExecTimeout {
    pub timeout: Duration,
    pub stdout: String,
    pub stderr: String,
    /// True when stdout passed [`EXEC_OUTPUT_LIMIT`] and the rest was dropped.
    pub stdout_truncated: bool,
    /// True when stderr passed [`EXEC_OUTPUT_LIMIT`] and the rest was dropped.
    pub stderr_truncated: bool,
}

impl std::fmt::Display for ExecTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "command timed out after {:?}", self.timeout)
    }
}

impl std::error::Error for ExecTimeout {}

/// Capabilities that a sandbox backend supports.
#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each boolean is an independent field in the public backend-capability ABI"
)]
pub struct SandboxCapabilities {
    pub exec: bool,
    pub filesystem_isolation: bool,
    /// TCP bind and connect are confined.
    ///
    /// This does not claim UDP, raw sockets, or sockets inherited from the parent.
    pub tcp_isolation: bool,
    pub resource_limits: bool,
    pub host_mounts: bool,
    pub pooling: bool,
    pub backend_name: &'static str,
}

/// Core trait for sandbox backends.
#[async_trait]
pub trait SandboxBackend: Send + Sync + std::fmt::Debug {
    fn capabilities(&self) -> SandboxCapabilities;

    async fn create(
        &self,
        config: &crate::config::SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>>;
}

/// A live sandbox instance.
#[async_trait]
pub trait SandboxInstance: Send + Sync + std::fmt::Debug {
    fn id(&self) -> &str;

    async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult>;

    async fn read_file(&self, path: &str) -> Result<Vec<u8>>;

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()>;

    async fn is_alive(&self) -> bool;

    async fn destroy(&self) -> Result<()>;

    async fn reset(&self) -> Result<()> {
        Err(anyhow::anyhow!("Reset not supported by this backend"))
    }
}
