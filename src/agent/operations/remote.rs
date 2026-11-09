//! Optional remote [`OperationManager`] that ships serialized specs to a worker.
//!
//! The coordinator keeps provider / MCP credentials. The worker (typically a
//! sandbox process) receives only versioned [`Operation`] JSON.

use super::codec::{DecodedOperation, decode_operation, sandbox_allows};
use super::store::OperationStore;
use super::{
    Operation, OperationExecution, OperationExecutionMode, OperationId, OperationManager,
    OperationStatus,
};
use anyhow::{Result, bail};
use async_trait::async_trait;
use std::sync::Arc;

/// Transport that delivers one committed operation spec and returns its result.
#[async_trait]
pub trait RemoteOperationTransport: Send + Sync + std::fmt::Debug {
    /// Execute `spec` on the remote side. Must not require coordinator secrets.
    async fn dispatch(&self, spec: &Operation) -> Result<OperationExecution>;

    /// True when this transport enforces OS isolation (sandbox process).
    fn isolates_in_sandbox(&self) -> bool {
        false
    }

    /// True when this isolating transport is the agent's configured sandbox.
    #[cfg(feature = "sandbox")]
    fn binds_configured_sandbox(
        &self,
        configured: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> bool {
        let _ = configured;
        false
    }
}

/// In-memory transport used by tests and as a recording fake.
#[derive(Debug, Default)]
pub struct RecordingTransport {
    /// Specs received, in order.
    pub received: std::sync::Mutex<Vec<Operation>>,
    /// Optional canned output. When `None`, echoes the spec JSON.
    pub canned: Option<String>,
}

#[async_trait]
impl RemoteOperationTransport for RecordingTransport {
    async fn dispatch(&self, spec: &Operation) -> Result<OperationExecution> {
        let id = spec
            .id
            .clone()
            .unwrap_or_else(|| OperationId::new("uncommitted"));
        if let Ok(mut guard) = self.received.lock() {
            guard.push(spec.clone());
        }
        Ok(OperationExecution::completed(
            id,
            self.canned.clone().unwrap_or_else(|| spec.spec.to_string()),
        ))
    }
}

/// Manager that commits locally and executes via [`RemoteOperationTransport`].
#[derive(Debug)]
pub struct RemoteOperationManager {
    transport: Arc<dyn RemoteOperationTransport>,
    store: OperationStore,
}

impl RemoteOperationManager {
    /// Wrap `transport`.
    #[must_use]
    pub fn new(transport: Arc<dyn RemoteOperationTransport>) -> Self {
        Self {
            transport,
            store: OperationStore::new("rop"),
        }
    }
}

#[async_trait]
impl OperationManager for RemoteOperationManager {
    async fn commit(&self, ops: &[Operation]) -> Result<Vec<OperationId>> {
        self.store.commit(ops)
    }

    async fn execute(&self, ids: &[OperationId]) -> Result<Vec<OperationExecution>> {
        self.store
            .run_batch(ids, |operation| {
                let transport = Arc::clone(&self.transport);
                async move { transport.dispatch(&operation).await }
            })
            .await
    }

    async fn status(&self, id: &OperationId) -> Result<OperationStatus> {
        self.store.status(id)
    }

    async fn cancel(&self, id: &OperationId) -> Result<OperationStatus> {
        self.store.cancel(id)
    }

    fn execution_mode(&self) -> OperationExecutionMode {
        OperationExecutionMode::Remote
    }

    fn isolates_in_sandbox(&self) -> bool {
        self.transport.isolates_in_sandbox()
    }

    #[cfg(feature = "sandbox")]
    fn binds_configured_sandbox(
        &self,
        configured: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> bool {
        self.transport.binds_configured_sandbox(configured)
    }
}

/// Maps serializable file/shell specs onto [`caliche_sandbox::SandboxHandleOps`].
///
/// Provider keys never enter the sandbox — only the operation spec is sent.
#[cfg(feature = "sandbox")]
pub struct SandboxOperationTransport {
    handle: Arc<dyn caliche_sandbox::SandboxHandleOps>,
}

#[cfg(feature = "sandbox")]
impl std::fmt::Debug for SandboxOperationTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxOperationTransport").finish()
    }
}

#[cfg(feature = "sandbox")]
impl SandboxOperationTransport {
    /// Bind to an existing sandbox handle.
    #[must_use]
    pub fn new(handle: Arc<dyn caliche_sandbox::SandboxHandleOps>) -> Self {
        Self { handle }
    }
}

#[cfg(feature = "sandbox")]
#[async_trait]
impl RemoteOperationTransport for SandboxOperationTransport {
    fn isolates_in_sandbox(&self) -> bool {
        true
    }

    fn binds_configured_sandbox(
        &self,
        configured: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> bool {
        configured.is_some_and(|configured| std::ptr::addr_eq(self.handle.as_ref(), configured))
    }

    async fn dispatch(&self, spec: &Operation) -> Result<OperationExecution> {
        let id = spec
            .id
            .clone()
            .unwrap_or_else(|| OperationId::new("uncommitted"));
        if !sandbox_allows(spec.kind) {
            bail!(
                "operation kind {:?} is not on the sandbox allowlist; \
                 keep coordinator-side tools (MCP, providers) on LocalOperationManager",
                spec.kind
            );
        }
        match decode_operation(spec)? {
            DecodedOperation::Shell {
                command,
                timeout_secs,
                working_dir,
            } => {
                let result = self
                    .handle
                    .exec(
                        &["sh", "-c", command.as_str()],
                        working_dir.as_deref(),
                        None,
                        std::time::Duration::from_secs(timeout_secs),
                    )
                    .await
                    .map_err(|err| anyhow::anyhow!("sandbox exec: {err:#}"))?;
                let note = crate::agent::tools::common::sandbox_exec::truncation_note(&result);
                if result.exit_code == 0 {
                    Ok(OperationExecution::completed(id, result.stdout + &note))
                } else {
                    Ok(OperationExecution::failed(
                        id,
                        format!(
                            "exit {}: {}{note}",
                            result.exit_code,
                            if result.stderr.is_empty() {
                                result.stdout
                            } else {
                                result.stderr
                            }
                        ),
                    ))
                }
            }
            DecodedOperation::FileRead { path } => {
                let bytes = self
                    .handle
                    .read_file(&path)
                    .await
                    .map_err(|err| anyhow::anyhow!("sandbox read_file: {err:#}"))?;
                let text = String::from_utf8_lossy(&bytes).into_owned();
                Ok(OperationExecution::completed(id, text))
            }
            DecodedOperation::FileWrite { path, content } => {
                self.handle
                    .write_file(&path, content.as_bytes())
                    .await
                    .map_err(|err| anyhow::anyhow!("sandbox write_file: {err:#}"))?;
                Ok(OperationExecution::completed(id, String::new()))
            }
            other => bail!(
                "sandbox allowlist accepted {:?} but decode produced {other:?}",
                spec.kind
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::operations::OperationKind;

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn sandbox_transport_runs_shell_without_coordinator_secrets() {
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        #[derive(Debug)]
        struct FakeSandbox;

        #[async_trait]
        impl caliche_sandbox::SandboxHandleOps for FakeSandbox {
            async fn exec(
                &self,
                command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> Result<ExecResult> {
                Ok(ExecResult {
                    stdout: format!("ran:{}", command.join(" ")),
                    stderr: String::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
                Ok(format!("contents of {path}").into_bytes())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> Result<()> {
                Ok(())
            }
        }

        let transport = SandboxOperationTransport::new(Arc::new(FakeSandbox));
        let manager = RemoteOperationManager::new(Arc::new(transport));
        let op = Operation::new(
            OperationKind::Shell,
            "bash",
            serde_json::json!({"command": "echo hi"}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        let executions = manager.execute(&ids).await.expect("execute");
        assert_eq!(executions[0].status, OperationStatus::Completed);
        assert!(executions[0].output.contains("echo hi"));
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn sandbox_transport_names_stdout_cut_by_the_sandbox() {
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        struct CutSandbox;

        #[async_trait]
        impl caliche_sandbox::SandboxHandleOps for CutSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> Result<ExecResult> {
                Ok(ExecResult {
                    stdout: "out".to_string(),
                    stderr: String::new(),
                    stdout_truncated: true,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, _path: &str) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> Result<()> {
                Ok(())
            }
        }

        let manager = RemoteOperationManager::new(Arc::new(SandboxOperationTransport::new(
            Arc::new(CutSandbox),
        )));
        let op = Operation::new(
            OperationKind::Shell,
            "bash",
            serde_json::json!({"command": "produce"}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        let executions = manager.execute(&ids).await.expect("execute");
        assert_eq!(executions[0].output, "out\n[stdout truncated at 4 MiB]");
    }

    #[tokio::test]
    async fn remote_manager_records_serialized_specs() {
        let transport = Arc::new(RecordingTransport {
            received: std::sync::Mutex::new(Vec::new()),
            canned: Some("remote-ok".to_string()),
        });
        let manager = RemoteOperationManager::new(transport.clone());
        let op = Operation::new(
            OperationKind::Shell,
            "bash",
            serde_json::json!({"command": "echo hi"}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        let executions = manager.execute(&ids).await.expect("execute");
        assert_eq!(executions[0].status, OperationStatus::Completed);
        assert_eq!(executions[0].output, "remote-ok");
        let received = transport.received.lock().expect("lock");
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].kind, OperationKind::Shell);
        assert!(received[0].id.is_some());
        // Spec JSON is the only payload — no coordinator credentials.
        assert_eq!(received[0].spec["command"].as_str(), Some("echo hi"));
    }

    #[tokio::test]
    async fn transport_error_finishes_as_failed_not_awaiting() {
        #[derive(Debug)]
        struct BoomTransport;

        #[async_trait]
        impl RemoteOperationTransport for BoomTransport {
            async fn dispatch(&self, _spec: &Operation) -> Result<OperationExecution> {
                bail!("transport exploded");
            }
        }

        let manager = RemoteOperationManager::new(Arc::new(BoomTransport));
        let first = Operation::new(
            crate::agent::operations::OperationKind::Shell,
            "bash",
            serde_json::json!({"command": "one"}),
        );
        let second = Operation::new(
            crate::agent::operations::OperationKind::Shell,
            "bash",
            serde_json::json!({"command": "two"}),
        );
        let ids = manager.commit(&[first, second]).await.expect("commit");
        let executions = manager.execute(&ids).await.expect("finish-all");
        assert_eq!(executions.len(), 2);
        assert_eq!(executions[0].status, OperationStatus::Failed);
        assert_eq!(executions[1].status, OperationStatus::Failed);
        assert_eq!(
            manager.status(&ids[0]).await.expect("status"),
            OperationStatus::Failed
        );
        assert_eq!(
            manager.status(&ids[1]).await.expect("status"),
            OperationStatus::Failed
        );
        assert!(
            executions[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("exploded")
        );
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn sandbox_file_write_fails_closed_on_non_string_content() {
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        #[derive(Debug)]
        struct FakeSandbox;

        #[async_trait]
        impl caliche_sandbox::SandboxHandleOps for FakeSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> Result<ExecResult> {
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, _path: &str) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> Result<()> {
                Ok(())
            }
        }

        let transport = SandboxOperationTransport::new(Arc::new(FakeSandbox));
        let manager = RemoteOperationManager::new(Arc::new(transport));
        let op = Operation::new(
            crate::agent::operations::OperationKind::FileWrite,
            "fs",
            serde_json::json!({"path": "a.txt", "content": 1}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        let executions = manager.execute(&ids).await.expect("mapped");
        assert_eq!(executions[0].status, OperationStatus::Failed);
        assert!(
            executions[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("string"),
            "fail-closed: {:?}",
            executions[0].error
        );
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn sandbox_transport_binds_only_the_configured_handle() {
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        #[derive(Debug)]
        struct FakeSandbox;

        #[async_trait]
        impl caliche_sandbox::SandboxHandleOps for FakeSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> Result<ExecResult> {
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, _path: &str) -> Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> Result<()> {
                Ok(())
            }
        }

        let bound: Arc<dyn caliche_sandbox::SandboxHandleOps> = Arc::new(FakeSandbox);
        let other = FakeSandbox;
        let transport = SandboxOperationTransport::new(Arc::clone(&bound));
        assert!(transport.binds_configured_sandbox(Some(bound.as_ref())));
        assert!(!transport.binds_configured_sandbox(Some(&other)));
        assert!(!transport.binds_configured_sandbox(None));
        assert!(!RecordingTransport::default().binds_configured_sandbox(Some(bound.as_ref())));
    }
}
