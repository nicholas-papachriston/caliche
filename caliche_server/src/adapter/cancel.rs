//! Process-wide session cancel and the live session list.
//!
//! Cancel handles sit beside the session `RwLock`. An in-flight turn holds that
//! lock, so stop must be able to signal without acquiring it.
//!
//! The list is every live session in this process (the in-memory Moka cache
//! registered with [`caliche::harness::AgentRuntime`]). It is not filtered
//! by JWT `sub` or roles. With auth off, any caller who can reach a session id
//! can stop it.

use super::{ApiAdapter, ApiAdapterError, SessionInfo};
use anyhow::Result;
use caliche::harness::{ControlMessage, ControlMode, InboxInput};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio_util::sync::CancellationToken;

/// Cooperative cancel flag plus a token that wakes a blocked turn.
#[derive(Clone, Debug)]
pub struct TurnCancel {
    pub token: CancellationToken,
    pub flag: Arc<AtomicBool>,
}

impl TurnCancel {
    pub(super) fn new() -> Self {
        Self {
            token: CancellationToken::new(),
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(super) fn signal(&self) {
        self.flag.store(true, Ordering::Release);
        self.token.cancel();
    }
}

pub(super) async fn await_turn<T>(
    session_id: &str,
    cancel: &TurnCancel,
    fut: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::pin!(fut);
    tokio::select! {
        biased;
        () = cancel.token.cancelled() => {
            Err(ApiAdapterError::TurnCancelled(session_id.to_string()).into())
        }
        result = fut.as_mut() => result,
    }
}

pub(super) async fn await_streaming_turn<T>(
    session_id: &str,
    session_cancel: &TurnCancel,
    caller_cancel: &CancellationToken,
    fut: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    tokio::pin!(fut);
    tokio::select! {
        biased;
        () = session_cancel.token.cancelled() => {
            Err(ApiAdapterError::TurnCancelled(session_id.to_string()).into())
        }
        () = caller_cancel.cancelled() => {
            Err(anyhow::anyhow!("caller turn cancelled"))
        }
        result = fut.as_mut() => result,
    }
}

fn is_turn_cancelled(err: &anyhow::Error) -> bool {
    err.downcast_ref::<ApiAdapterError>()
        .is_some_and(|inner| matches!(inner, ApiAdapterError::TurnCancelled(_)))
        || err.to_string().contains("session turn cancelled")
        || err.to_string().contains("Stream cancelled by user")
        || err.to_string().contains("caller turn cancelled")
}

impl ApiAdapter {
    pub(super) fn turn_cancel_handle(&self, session_id: &str) -> Option<TurnCancel> {
        self.turn_cancels
            .get(session_id)
            .map(|entry| entry.value().clone())
    }

    pub(super) fn signal_turn_cancel(&self, session_id: &str) {
        if let Some(cancel) = self.turn_cancels.get(session_id) {
            cancel.signal();
        }
    }

    /// Cancel an in-flight turn and remove the session from this process.
    ///
    /// Returns `false` when the id is not in the live cache. A following turn
    /// on that id fails with [`ApiAdapterError::SessionNotFound`]. Idle
    /// time-to-idle eviction is not this path.
    ///
    /// Hard-stop goes through the harness inbox (`ControlMode::Hard`), including
    /// the unlocked side channel when a turn already holds the harness lock.
    pub async fn cancel_session(&self, session_id: &str) -> bool {
        if self.get_session(session_id).await.is_none() {
            self.turn_cancels.remove(session_id);
            return false;
        }
        self.signal_turn_cancel(session_id);
        self.signal_hard_stop(session_id).await;
        self.delete_session(session_id).await
    }

    async fn signal_hard_stop(&self, session_id: &str) {
        let input = InboxInput::control(
            None::<String>,
            ControlMessage {
                mode: ControlMode::Hard,
                reason: "http session cancel".to_string(),
                settings: None,
            },
        );
        if let Err(err) = self.runtime.submit_input(session_id, input).await {
            tracing::debug!(
                target: "caliche_server::adapter",
                session_id,
                error = %err,
                "hard-stop during http cancel"
            );
        }
    }

    /// Live sessions in this process, sorted by id.
    ///
    /// Reads the in-memory session cache only. Tracker rows for sessions that
    /// are not in the cache are omitted. Rows are not filtered by owner,
    /// JWT `sub`, or roles.
    pub async fn list_process_sessions(&self) -> Vec<SessionInfo> {
        let mut ids = self.list_sessions();
        ids.sort();
        let mut sessions = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(info) = self.process_session_info(id).await {
                sessions.push(info);
            }
        }
        sessions
    }

    async fn process_session_info(&self, session_id: &str) -> Option<SessionInfo> {
        let session = self.get_session(session_id).await?;
        let name_slug = self.cached_name_slug(session_id).await;
        let state = session.read().await;
        let harness = state.harness.read().await;
        Some(SessionInfo {
            session_id: session_id.to_string(),
            name_slug,
            title: state.title.clone(),
            created_at: state.created_at,
            last_activity: state.last_activity,
            turn_count: harness.turn_count(),
            total_tokens: harness
                .total_usage()
                .total_tokens
                .map(|tokens| tokens as u64),
        })
    }

    async fn cached_name_slug(&self, session_id: &str) -> Option<String> {
        let tracker = self.tracker.as_ref()?;
        let id = uuid::Uuid::parse_str(session_id).ok()?;
        let tracked = tracker.get(id).await.ok().flatten()?;
        tracked.name_slug
    }

    pub(super) fn turn_was_cancelled(err: &anyhow::Error) -> bool {
        is_turn_cancelled(err)
    }
}
