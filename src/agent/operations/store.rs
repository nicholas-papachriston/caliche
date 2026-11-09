//! Shared committed-operation table used by local and remote managers.

use super::{Operation, OperationExecution, OperationId, OperationStatus};
use anyhow::{Context, Result, bail};
use dashmap::DashMap;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub(super) struct OperationRecord {
    pub operation: Operation,
    pub status: OperationStatus,
    pub execution: Option<OperationExecution>,
}

#[derive(Debug)]
pub(super) struct OperationStore {
    records: DashMap<OperationId, OperationRecord>,
    next_id: AtomicU64,
    prefix: &'static str,
}

#[derive(Debug)]
pub(super) enum BeginExecute {
    Run(Operation),
    AlreadyCanceled(OperationExecution),
    AlreadyFinished(OperationExecution),
}

fn canceled_execution(id: &OperationId) -> OperationExecution {
    OperationExecution {
        id: id.clone(),
        status: OperationStatus::Canceled,
        output: String::new(),
        error: Some("canceled".to_string()),
        typed_payload: None,
    }
}

impl OperationStore {
    pub(super) fn new(prefix: &'static str) -> Self {
        Self {
            records: DashMap::new(),
            next_id: AtomicU64::new(1),
            prefix,
        }
    }

    fn allocate_id(&self) -> OperationId {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        OperationId::new(format!("{}-{n}", self.prefix))
    }

    pub(super) fn commit(&self, ops: &[Operation]) -> Result<Vec<OperationId>> {
        let mut planned = Vec::with_capacity(ops.len());
        let mut reserved = std::collections::HashSet::new();
        for op in ops {
            if op.version == 0 {
                bail!("operation version must be non-zero");
            }
            let id = op.id.clone().unwrap_or_else(|| self.allocate_id());
            if !reserved.insert(id.clone()) {
                bail!("duplicate operation id {id}");
            }
            planned.push((id, op));
        }
        let mut ids = Vec::with_capacity(planned.len());
        for (id, op) in planned {
            match self.records.entry(id.clone()) {
                dashmap::mapref::entry::Entry::Occupied(_) => {
                    for rolled in &ids {
                        self.records.remove(rolled);
                    }
                    bail!("duplicate operation id {id}");
                }
                dashmap::mapref::entry::Entry::Vacant(slot) => {
                    let mut stored = op.clone();
                    stored.id = Some(id.clone());
                    slot.insert(OperationRecord {
                        operation: stored,
                        status: OperationStatus::Ready,
                        execution: None,
                    });
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    pub(super) fn begin_execute(&self, id: &OperationId) -> Result<BeginExecute> {
        let mut record = self
            .records
            .get_mut(id)
            .with_context(|| format!("unknown operation {id}"))?;
        match record.status {
            OperationStatus::Canceled | OperationStatus::Canceling => {
                record.status = OperationStatus::Canceled;
                let execution = record
                    .execution
                    .clone()
                    .unwrap_or_else(|| canceled_execution(id));
                record.execution = Some(execution.clone());
                Ok(BeginExecute::AlreadyCanceled(execution))
            }
            OperationStatus::Completed | OperationStatus::Failed => {
                let execution = record
                    .execution
                    .clone()
                    .with_context(|| format!("missing execution for terminal operation {id}"))?;
                Ok(BeginExecute::AlreadyFinished(execution))
            }
            OperationStatus::Awaiting => {
                bail!("operation {id} is already in flight");
            }
            OperationStatus::Ready => {
                record.status = OperationStatus::Awaiting;
                Ok(BeginExecute::Run(record.operation.clone()))
            }
        }
    }

    /// Persist a terminal execution unless cancel already won.
    pub(super) fn finish(
        &self,
        id: &OperationId,
        execution: OperationExecution,
    ) -> Result<OperationExecution> {
        let mut record = self
            .records
            .get_mut(id)
            .with_context(|| format!("unknown operation {id}"))?;
        match record.status {
            OperationStatus::Canceled | OperationStatus::Canceling => {
                record.status = OperationStatus::Canceled;
                let kept = record
                    .execution
                    .clone()
                    .unwrap_or_else(|| canceled_execution(id));
                record.execution = Some(kept.clone());
                Ok(kept)
            }
            _ => {
                record.status = execution.status;
                record.execution = Some(execution.clone());
                Ok(execution)
            }
        }
    }

    fn rollback_awaiting(&self, ids: &[OperationId]) {
        for id in ids {
            if let Some(mut record) = self.records.get_mut(id)
                && record.status == OperationStatus::Awaiting
            {
                record.status = OperationStatus::Ready;
            }
        }
    }

    pub(super) fn status(&self, id: &OperationId) -> Result<OperationStatus> {
        self.records
            .get(id)
            .map(|record| record.status)
            .with_context(|| format!("unknown operation {id}"))
    }

    pub(super) fn cancel(&self, id: &OperationId) -> Result<OperationStatus> {
        let mut record = self
            .records
            .get_mut(id)
            .with_context(|| format!("unknown operation {id}"))?;
        record.status = match record.status {
            OperationStatus::Completed | OperationStatus::Failed | OperationStatus::Canceled => {
                record.status
            }
            OperationStatus::Awaiting | OperationStatus::Canceling => OperationStatus::Canceling,
            OperationStatus::Ready => OperationStatus::Canceled,
        };
        Ok(record.status)
    }

    pub(super) fn committed_ids(&self) -> Vec<OperationId> {
        self.records
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Resolve canceled / already-finished records. Ready/Awaiting is an error
    /// so InProcessEngine cannot run a second execute path.
    pub(super) fn audit_resolve(&self, ids: &[OperationId]) -> Result<Vec<OperationExecution>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let record = self
                .records
                .get(id)
                .with_context(|| format!("unknown operation {id}"))?;
            match record.status {
                OperationStatus::Canceled | OperationStatus::Canceling => {
                    out.push(record.execution.clone().unwrap_or_else(|| {
                        let mut execution = OperationExecution::failed(id.clone(), "canceled");
                        execution.status = OperationStatus::Canceled;
                        execution
                    }));
                }
                OperationStatus::Completed | OperationStatus::Failed => {
                    out.push(
                        record
                            .execution
                            .clone()
                            .with_context(|| format!("missing execution for {id}"))?,
                    );
                }
                OperationStatus::Ready | OperationStatus::Awaiting => {
                    bail!(
                        "LocalOperationManager is audit-only; ToolExecutionEngine owns \
                         execution (id {id} still {:?})",
                        record.status
                    );
                }
            }
        }
        Ok(out)
    }

    /// Begin every id, run pending work in parallel, map transport errors to
    /// failed executions, finish all, then return results in `ids` order.
    pub(super) async fn run_batch<F, Fut>(
        &self,
        ids: &[OperationId],
        run: F,
    ) -> Result<Vec<OperationExecution>>
    where
        F: Fn(Operation) -> Fut,
        Fut: Future<Output = Result<OperationExecution>> + Send,
    {
        let mut finished = std::collections::HashMap::new();
        let mut pending = Vec::new();
        let mut started = Vec::new();
        for id in ids {
            match self.begin_execute(id) {
                Ok(BeginExecute::AlreadyCanceled(execution))
                | Ok(BeginExecute::AlreadyFinished(execution)) => {
                    finished.insert(id.clone(), execution);
                }
                Ok(BeginExecute::Run(operation)) => {
                    started.push(id.clone());
                    pending.push((id.clone(), operation));
                }
                Err(err) => {
                    self.rollback_awaiting(&started);
                    return Err(err);
                }
            }
        }
        let dispatched =
            futures_util::future::join_all(pending.into_iter().map(|(id, operation)| {
                let fut = run(operation);
                async move {
                    let execution = match fut.await {
                        Ok(execution) => execution,
                        Err(err) => OperationExecution::failed(id.clone(), format!("{err:#}")),
                    };
                    (id, execution)
                }
            }))
            .await;
        for (id, execution) in dispatched {
            let kept = self.finish(&id, execution)?;
            finished.insert(id, kept);
        }
        ids.iter()
            .map(|id| {
                finished
                    .remove(id)
                    .ok_or_else(|| anyhow::anyhow!("operation store missing execution for {id}"))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::operations::OperationKind;

    fn shell(command: &str) -> Operation {
        Operation::new(
            OperationKind::Shell,
            "bash",
            serde_json::json!({"command": command}),
        )
    }

    fn shell_with_id(id: &str, command: &str) -> Operation {
        let mut op = shell(command);
        op.id = Some(OperationId::new(id));
        op
    }

    #[tokio::test]
    async fn begin_execute_refuses_terminal_and_in_flight() {
        let store = OperationStore::new("t");
        let completed_ids = store.commit(&[shell("done")]).expect("commit");
        store
            .finish(
                &completed_ids[0],
                OperationExecution::completed(completed_ids[0].clone(), "ok"),
            )
            .expect("finish");
        match store.begin_execute(&completed_ids[0]).expect("begin") {
            BeginExecute::AlreadyFinished(execution) => {
                assert_eq!(execution.status, OperationStatus::Completed);
                assert_eq!(execution.output, "ok");
            }
            other => panic!("expected AlreadyFinished, got {other:?}"),
        }
        assert_eq!(
            store.status(&completed_ids[0]).expect("status"),
            OperationStatus::Completed
        );

        let failed_ids = store.commit(&[shell("fail")]).expect("commit");
        store
            .finish(
                &failed_ids[0],
                OperationExecution::failed(failed_ids[0].clone(), "boom"),
            )
            .expect("finish");
        match store.begin_execute(&failed_ids[0]).expect("begin") {
            BeginExecute::AlreadyFinished(execution) => {
                assert_eq!(execution.status, OperationStatus::Failed);
            }
            other => panic!("expected AlreadyFinished, got {other:?}"),
        }

        let ready_ids = store.commit(&[shell("ready")]).expect("commit");
        match store.begin_execute(&ready_ids[0]).expect("first begin") {
            BeginExecute::Run(_) => {}
            other => panic!("expected Run, got {other:?}"),
        }
        let err = store
            .begin_execute(&ready_ids[0])
            .expect_err("in-flight must refuse");
        assert!(
            err.to_string().contains("in flight"),
            "in-flight reason: {err}"
        );
        assert_eq!(
            store.status(&ready_ids[0]).expect("status"),
            OperationStatus::Awaiting
        );
    }

    #[tokio::test]
    async fn run_batch_rolls_back_awaiting_on_begin_failure() {
        let store = OperationStore::new("t");
        let ids = store.commit(&[shell("one"), shell("two")]).expect("commit");
        let unknown = OperationId::new("missing");
        let err = store
            .run_batch(&[ids[0].clone(), unknown], |_| async {
                panic!("begin failure must not dispatch");
            })
            .await
            .expect_err("unknown id");
        assert!(
            err.to_string().contains("unknown operation"),
            "begin error: {err}"
        );
        assert_eq!(
            store.status(&ids[0]).expect("rolled back"),
            OperationStatus::Ready
        );
        assert_eq!(
            store.status(&ids[1]).expect("untouched"),
            OperationStatus::Ready
        );
    }

    #[tokio::test]
    async fn commit_rejects_duplicate_ids() {
        let store = OperationStore::new("t");
        let first = shell_with_id("dup", "one");
        store.commit(&[first.clone()]).expect("first commit");
        let err = store
            .commit(&[first])
            .expect_err("second commit of same id");
        assert!(
            err.to_string().contains("duplicate operation id"),
            "dup reason: {err}"
        );
        assert_eq!(
            store.status(&OperationId::new("dup")).expect("kept"),
            OperationStatus::Ready
        );

        let a = shell_with_id("batch-dup", "a");
        let b = shell_with_id("batch-dup", "b");
        let err = store.commit(&[a, b]).expect_err("intra-batch duplicate");
        assert!(
            err.to_string().contains("duplicate operation id"),
            "batch dup: {err}"
        );
        assert!(
            store.status(&OperationId::new("batch-dup")).is_err(),
            "intra-batch reject must not insert"
        );
    }

    #[tokio::test]
    async fn finish_refuses_to_clobber_canceled() {
        let store = OperationStore::new("t");
        let ready_ids = store.commit(&[shell("ready")]).expect("commit");
        assert_eq!(
            store.cancel(&ready_ids[0]).expect("cancel"),
            OperationStatus::Canceled
        );
        let kept = store
            .finish(
                &ready_ids[0],
                OperationExecution::completed(ready_ids[0].clone(), "should-not-stick"),
            )
            .expect("refuse finish");
        assert_eq!(kept.status, OperationStatus::Canceled);
        assert_eq!(
            store.status(&ready_ids[0]).expect("status"),
            OperationStatus::Canceled
        );

        let inflight_ids = store.commit(&[shell("inflight")]).expect("commit");
        match store.begin_execute(&inflight_ids[0]).expect("begin") {
            BeginExecute::Run(_) => {}
            other => panic!("expected Run, got {other:?}"),
        }
        assert_eq!(
            store.cancel(&inflight_ids[0]).expect("cancel"),
            OperationStatus::Canceling
        );
        let kept = store
            .finish(
                &inflight_ids[0],
                OperationExecution::completed(inflight_ids[0].clone(), "late-ok"),
            )
            .expect("refuse finish over canceling");
        assert_eq!(kept.status, OperationStatus::Canceled);
        assert_eq!(
            store.status(&inflight_ids[0]).expect("status"),
            OperationStatus::Canceled
        );
    }

    #[tokio::test]
    async fn run_batch_does_not_redispatch_finished_ids() {
        let store = OperationStore::new("t");
        let ids = store.commit(&[shell("once")]).expect("commit");
        let first = store
            .run_batch(&ids, |op| async move {
                Ok(OperationExecution::completed(
                    op.id.clone().expect("id"),
                    "first",
                ))
            })
            .await
            .expect("first batch");
        assert_eq!(first[0].output, "first");
        let second = store
            .run_batch(&ids, |_| async {
                panic!("finished id must not re-dispatch");
            })
            .await
            .expect("second batch");
        assert_eq!(second[0].status, OperationStatus::Completed);
        assert_eq!(second[0].output, "first");
        assert_eq!(
            store.status(&ids[0]).expect("status"),
            OperationStatus::Completed
        );
    }

    #[tokio::test]
    async fn run_batch_keeps_cancel_when_transport_succeeds() {
        let store = std::sync::Arc::new(OperationStore::new("t"));
        let ids = store.commit(&[shell("one")]).expect("commit");
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let started_tx = std::sync::Mutex::new(Some(started_tx));
        let release_rx = std::sync::Mutex::new(Some(release_rx));
        let store_for_batch = std::sync::Arc::clone(&store);
        let batch_ids = ids.clone();
        let batch = tokio::spawn(async move {
            store_for_batch
                .run_batch(&batch_ids, move |_op| {
                    let started_tx = started_tx.lock().expect("started lock").take();
                    let release_rx = release_rx.lock().expect("release lock").take();
                    async move {
                        if let Some(tx) = started_tx {
                            let _ = tx.send(());
                        }
                        if let Some(rx) = release_rx {
                            let _ = rx.await;
                        }
                        Ok(OperationExecution::completed(
                            OperationId::new("unused"),
                            "should-not-stick",
                        ))
                    }
                })
                .await
        });
        started_rx.await.expect("started");
        assert_eq!(
            store.cancel(&ids[0]).expect("cancel"),
            OperationStatus::Canceling
        );
        release_tx.send(()).expect("release");
        let executions = batch.await.expect("join").expect("batch");
        assert_eq!(executions[0].status, OperationStatus::Canceled);
        assert_eq!(
            store.status(&ids[0]).expect("status"),
            OperationStatus::Canceled
        );
    }

    #[tokio::test]
    async fn run_batch_maps_transport_errors_and_finishes_all() {
        let store = OperationStore::new("t");
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
        let ids = store.commit(&[first, second]).expect("commit");
        let executions = store
            .run_batch(&ids, |op| async move {
                if op.spec["command"] == "one" {
                    anyhow::bail!("transport exploded");
                }
                Ok(OperationExecution::completed(
                    op.id.clone().expect("id"),
                    "ok",
                ))
            })
            .await
            .expect("batch");
        assert_eq!(executions.len(), 2);
        assert_eq!(executions[0].status, OperationStatus::Failed);
        assert!(
            executions[0]
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("exploded")
        );
        assert_eq!(executions[1].status, OperationStatus::Completed);
        assert_eq!(
            store.status(&ids[0]).expect("status"),
            OperationStatus::Failed
        );
        assert_eq!(
            store.status(&ids[1]).expect("status"),
            OperationStatus::Completed
        );
    }
}
