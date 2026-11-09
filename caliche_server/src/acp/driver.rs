//! Turn driver used by the ACP agent.
//!
//! The stdio server needs session create plus a streaming turn. Production
//! uses [`ApiAdapter`]. Tests supply a fake that never talks to a provider.

use crate::acp::permission::{AcpApprovalResolver, AcpPermissionGate};
use crate::adapter::{ApiAdapter, StreamingTurnResult};
use anyhow::Result;
use async_trait::async_trait;
use caliche::agent::hooks::ApprovalResolver;
use caliche::harness::SessionConfig;
use caliche::messages::{Message, StreamEvent};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// One row from the live session cache (`GET /api/agent/sessions`).
#[derive(Debug, Clone)]
pub struct AcpListedSession {
    pub session_id: String,
    pub cwd: String,
    pub title: Option<String>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Live session plus the harness history `session/load` replays.
#[derive(Debug, Clone)]
pub struct AcpLoadedSession {
    pub listed: AcpListedSession,
    pub history: Vec<Message>,
}

/// Receives harness stream events while a prompt is still running.
#[derive(Clone)]
pub struct AcpEventSink {
    emit: Arc<dyn Fn(StreamEvent) + Send + Sync>,
}

impl AcpEventSink {
    pub fn new(emit: impl Fn(StreamEvent) + Send + Sync + 'static) -> Self {
        Self {
            emit: Arc::new(emit),
        }
    }

    pub fn emit(&self, event: StreamEvent) {
        (self.emit)(event);
    }
}

#[async_trait]
pub trait AcpTurnDriver: Send + Sync {
    async fn open_session(&self, workspace: String) -> Result<String>;

    async fn has_session(&self, session_id: String) -> bool;

    /// Live sessions, the same cache as `GET /api/agent/sessions`.
    async fn list_sessions(&self) -> Result<Vec<AcpListedSession>>;

    /// One live session, the same cache as `GET /api/agent/sessions/{id}`.
    ///
    /// `Ok(None)` when the id is not in that cache. Tracker-only rows are not
    /// loaded: they have no harness history to replay.
    async fn load_session(&self, session_id: String) -> Result<Option<AcpLoadedSession>>;

    /// Run one prompt.
    ///
    /// `permissions` asks the ACP client when the engine returns
    /// `RequireApproval`. Drivers that never hit that path leave the gate
    /// unused. The production adapter installs it as the session's approval
    /// resolver for this turn only, then restores whatever resolver was there.
    async fn run_turn(
        &self,
        session_id: String,
        input: String,
        cancel: CancellationToken,
        on_event: AcpEventSink,
        permissions: AcpPermissionGate,
    ) -> Result<StreamingTurnResult>;
}

#[async_trait]
impl AcpTurnDriver for ApiAdapter {
    async fn open_session(&self, workspace: String) -> Result<String> {
        let session_config = SessionConfig {
            session_id: None,
            workspace: Some(workspace.into()),
            context_files: Vec::new(),
            system_prompt_addition: None,
            max_turns: None,
            history: Vec::new(),
        };
        self.create_session(Some(session_config), None, None).await
    }

    async fn has_session(&self, session_id: String) -> bool {
        self.get_session(&session_id).await.is_some()
    }

    async fn list_sessions(&self) -> Result<Vec<AcpListedSession>> {
        let mut ids = ApiAdapter::list_sessions(self);
        ids.sort();
        let mut rows = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(row) = read_listed(self, id).await {
                rows.push(row);
            }
        }
        Ok(rows)
    }

    async fn load_session(&self, session_id: String) -> Result<Option<AcpLoadedSession>> {
        Ok(read_loaded(self, &session_id).await)
    }

    async fn run_turn(
        &self,
        session_id: String,
        input: String,
        cancel: CancellationToken,
        on_event: AcpEventSink,
        permissions: AcpPermissionGate,
    ) -> Result<StreamingTurnResult> {
        let previous = install_acp_approval_resolver(self, &session_id, &permissions).await?;
        let result = self
            .execute_turn_streaming(&session_id, &input, cancel, |event| {
                on_event.emit(event);
            })
            .await;
        restore_acp_approval_resolver(self, &session_id, previous).await;
        result
    }
}

pub(crate) async fn install_acp_approval_resolver(
    adapter: &ApiAdapter,
    session_id: &str,
    permissions: &AcpPermissionGate,
) -> Result<Option<Arc<dyn ApprovalResolver>>> {
    let session = adapter
        .get_session(session_id)
        .await
        .ok_or_else(|| anyhow::anyhow!("session not found"))?;
    let state = session.write().await;
    let mut harness = state.harness.write().await;
    let resolver: Arc<dyn ApprovalResolver> =
        Arc::new(AcpApprovalResolver::new(permissions.clone()));
    Ok(harness
        .service_mut()
        .agent_mut()
        .replace_approval_resolver(Some(resolver)))
}

pub(crate) async fn restore_acp_approval_resolver(
    adapter: &ApiAdapter,
    session_id: &str,
    previous: Option<Arc<dyn ApprovalResolver>>,
) {
    let Some(session) = adapter.get_session(session_id).await else {
        return;
    };
    let state = session.write().await;
    let mut harness = state.harness.write().await;
    let _ = harness
        .service_mut()
        .agent_mut()
        .replace_approval_resolver(previous);
}

async fn read_listed(adapter: &ApiAdapter, session_id: &str) -> Option<AcpListedSession> {
    let session = adapter.get_session(session_id).await?;
    let state = session.read().await;
    let harness = state.harness.read().await;
    let cwd = cwd_of(&harness).await;
    Some(AcpListedSession {
        session_id: session_id.to_string(),
        cwd,
        title: state.title.clone(),
        updated_at: state.last_activity,
    })
}

async fn read_loaded(adapter: &ApiAdapter, session_id: &str) -> Option<AcpLoadedSession> {
    let session = adapter.get_session(session_id).await?;
    let state = session.read().await;
    let harness = state.harness.read().await;
    let cwd = cwd_of(&harness).await;
    let history = harness.history().to_vec();
    Some(AcpLoadedSession {
        listed: AcpListedSession {
            session_id: session_id.to_string(),
            cwd,
            title: state.title.clone(),
            updated_at: state.last_activity,
        },
        history,
    })
}

async fn cwd_of(harness: &caliche::harness::AgentHarness) -> String {
    let contexts = harness.context_manager();
    let context = contexts.read().await;
    absolute_cwd_string(context.workspace_root())
}

fn absolute_cwd_string(path: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match std::env::current_dir() {
            Ok(dir) => dir.join(path),
            Err(_) => PathBuf::from(path),
        }
    };
    match absolute.to_str() {
        Some(text) => text.to_owned(),
        None => absolute.to_string_lossy().into_owned(),
    }
}
