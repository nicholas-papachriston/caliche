//! Trait for sandbox handle operations (used by `caliche` when sandbox feature is enabled).

use crate::traits::ExecResult;
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

/// Erased sandbox handle interface so `caliche` can call
/// `exec`/`read_file`/`write_file` without depending on the concrete `SandboxHandle` type.
#[async_trait]
pub trait SandboxHandleOps: Send + Sync {
    async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult>;

    async fn read_file(&self, path: &str) -> Result<Vec<u8>>;

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()>;

    /// Whether the sandbox is actively enforcing isolation for tool execution.
    fn sandbox_active(&self) -> bool {
        true
    }

    /// Whether the backend confines file access to the sandbox policy.
    ///
    /// `false` unless the backend reports filesystem isolation, so a handle that
    /// cannot prove confinement is treated as unconfined.
    fn confines_filesystem(&self) -> bool {
        false
    }

    /// Whether a tool is allowed by the configured tool policy.
    fn is_tool_allowed(&self, tool_name: &str) -> bool {
        let _ = tool_name;
        true
    }
}
