//! In-process [`OperationManager`]: audit-only commit table for the engine path.

use super::store::OperationStore;
use super::{
    Operation, OperationExecution, OperationExecutionMode, OperationId, OperationManager,
    OperationStatus,
};
use anyhow::Result;
use async_trait::async_trait;

/// Default manager: commit versioned specs for audit. Live execution is
/// [`crate::agent::tools::execution::ToolExecutionEngine`] on the
/// [`OperationExecutionMode::InProcessEngine`] path (sandbox, cancel,
/// workspace root, metrics, AgentFS, hook deny, parallel batching).
///
/// [`Self::execute`] is quarantined: it only resolves canceled / already
/// finished records. Ready operations stay Ready until
/// [`OperationManager::record_engine_results`].
#[derive(Debug)]
pub struct LocalOperationManager {
    store: OperationStore,
}

impl LocalOperationManager {
    /// Create an audit-only local manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: OperationStore::new("op"),
        }
    }

    /// Ids currently in the audit table (test / introspection).
    #[must_use]
    pub fn committed_ids(&self) -> Vec<OperationId> {
        self.store.committed_ids()
    }
}

impl Default for LocalOperationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OperationManager for LocalOperationManager {
    async fn commit(&self, ops: &[Operation]) -> Result<Vec<OperationId>> {
        self.store.commit(ops)
    }

    async fn execute(&self, ids: &[OperationId]) -> Result<Vec<OperationExecution>> {
        self.store.audit_resolve(ids)
    }

    async fn status(&self, id: &OperationId) -> Result<OperationStatus> {
        self.store.status(id)
    }

    async fn cancel(&self, id: &OperationId) -> Result<OperationStatus> {
        self.store.cancel(id)
    }

    fn execution_mode(&self) -> OperationExecutionMode {
        OperationExecutionMode::InProcessEngine
    }

    fn record_engine_results(&self, results: &[(OperationId, OperationExecution)]) {
        for (id, execution) in results {
            let _ = self.store.finish(id, execution.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::operations::OperationKind;

    #[tokio::test]
    async fn execute_on_ready_is_audit_only() {
        let manager = LocalOperationManager::new();
        let op = Operation::new(
            OperationKind::ToolInvoke,
            "echo",
            serde_json::json!({"name": "echo", "arguments": {"msg": "hi"}}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        assert_eq!(
            manager.status(&ids[0]).await.expect("status"),
            OperationStatus::Ready
        );
        let err = manager
            .execute(&ids)
            .await
            .expect_err("Ready ops must not run via Local::execute");
        assert!(
            err.to_string().contains("audit-only"),
            "contract error: {err}"
        );
        assert_eq!(
            manager.status(&ids[0]).await.expect("still ready"),
            OperationStatus::Ready
        );
    }

    #[tokio::test]
    async fn cancel_ready_operation_resolves_without_running() {
        let manager = LocalOperationManager::new();
        let op = Operation::new(
            OperationKind::ToolInvoke,
            "echo",
            serde_json::json!({"name": "echo", "arguments": {}}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        assert_eq!(
            manager.cancel(&ids[0]).await.expect("cancel"),
            OperationStatus::Canceled
        );
        let executions = manager.execute(&ids).await.expect("audit resolve");
        assert_eq!(executions[0].status, OperationStatus::Canceled);
    }

    #[tokio::test]
    async fn record_engine_results_finishes_ready_ops() {
        let manager = LocalOperationManager::new();
        let op = Operation::new(
            OperationKind::ToolInvoke,
            "echo",
            serde_json::json!({"name": "echo", "arguments": {}}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        manager.record_engine_results(&[(
            ids[0].clone(),
            OperationExecution::completed(ids[0].clone(), "hi"),
        )]);
        assert_eq!(
            manager.status(&ids[0]).await.expect("status"),
            OperationStatus::Completed
        );
    }

    #[tokio::test]
    async fn record_engine_results_does_not_clobber_canceled() {
        let manager = LocalOperationManager::new();
        let op = Operation::new(
            OperationKind::ToolInvoke,
            "echo",
            serde_json::json!({"name": "echo", "arguments": {}}),
        );
        let ids = manager.commit(&[op]).await.expect("commit");
        assert_eq!(
            manager.cancel(&ids[0]).await.expect("cancel"),
            OperationStatus::Canceled
        );
        manager.record_engine_results(&[(
            ids[0].clone(),
            OperationExecution::completed(ids[0].clone(), "late-ok"),
        )]);
        assert_eq!(
            manager.status(&ids[0]).await.expect("status"),
            OperationStatus::Canceled
        );
    }
}
