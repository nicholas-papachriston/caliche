//! Asynchronous operation manager.

use super::{Operation, OperationExecution, OperationId, OperationStatus};
use anyhow::Result;
use async_trait::async_trait;

/// Where committed operations actually run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationExecutionMode {
    /// Translate + commit, then run through [`crate::agent::tools::execution::ToolExecutionEngine`].
    ///
    /// This keeps sandbox, cancel, workspace root, metrics, AgentFS, hook
    /// deny lists, and parallel batching identical to the default tool path.
    InProcessEngine,
    /// Translate + commit, then [`OperationManager::execute`] on a worker.
    Remote,
}

/// Executes committed operations asynchronously.
///
/// Local and remote implementations are swappable. The coordinator holds
/// provider credentials; a remote manager ships only serializable specs.
#[async_trait]
pub trait OperationManager: Send + Sync + std::fmt::Debug {
    /// Persist `ops` and assign ids. Returns the committed ids in order.
    async fn commit(&self, ops: &[Operation]) -> Result<Vec<OperationId>>;

    /// Execute previously committed operations.
    async fn execute(&self, ids: &[OperationId]) -> Result<Vec<OperationExecution>>;

    /// Current status of a committed operation.
    async fn status(&self, id: &OperationId) -> Result<OperationStatus>;

    /// Request cancellation. May return `Canceling` until the worker acks.
    async fn cancel(&self, id: &OperationId) -> Result<OperationStatus>;

    /// How this manager should be driven from the tool pipeline.
    fn execution_mode(&self) -> OperationExecutionMode {
        OperationExecutionMode::InProcessEngine
    }

    /// True when a remote worker itself enforces OS isolation (sandbox transport).
    fn isolates_in_sandbox(&self) -> bool {
        false
    }

    /// True when an isolating transport is bound to the agent's sandbox handle.
    ///
    /// Default is false (fail closed). [`crate::agent::operations::SandboxOperationTransport`]
    /// compares the configured handle by address.
    #[cfg(feature = "sandbox")]
    fn binds_configured_sandbox(
        &self,
        configured: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> bool {
        let _ = configured;
        false
    }

    /// Record engine-backed results for [`OperationExecutionMode::InProcessEngine`].
    ///
    /// Local managers persist these so committed specs leave `Ready`. Remote
    /// managers ignore this (they finish inside [`Self::execute`]).
    fn record_engine_results(&self, results: &[(OperationId, OperationExecution)]) {
        let _ = results;
    }
}
