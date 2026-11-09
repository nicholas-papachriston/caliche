//! Multi-session runtime facade over [`AgentHarness`].
//!
//! CLI and API hosts can keep one process-local map of sessions without
//! re-implementing create/lookup/turn routing. This intentionally wraps the
//! existing harness rather than replacing it.
//!
//! Storage uses `DashMap` + `Arc<RwLock<_>>` so hosts (e.g. Axum `ApiAdapter`)
//! can share the runtime behind `&self` / `Arc<AgentRuntime>`.

use crate::agent::service::AgentService;
use crate::harness::config::HarnessConfig;
use crate::harness::core::{AgentHarness, HarnessTurnResult};
use crate::harness::suspend::{SuspendStore, SuspendedSession};
use crate::harness::{MessageDiff, SessionDiffSummary};
use anyhow::{Context, Result, bail};
use dashmap::DashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Shared handle to a live harness session.
pub type SharedHarness = Arc<RwLock<AgentHarness>>;

/// Process-local multi-session facade over [`AgentHarness`].
#[derive(Debug)]
pub struct AgentRuntime {
    sessions: DashMap<String, SharedHarness>,
    sides:
        DashMap<String, std::sync::Arc<parking_lot::Mutex<crate::harness::core::SessionSideState>>>,
    default_config: HarnessConfig,
    persist_dir: Option<PathBuf>,
}

impl AgentRuntime {
    /// Create an empty runtime with the given default harness config.
    #[must_use]
    pub fn new(default_config: HarnessConfig) -> Self {
        Self {
            sessions: DashMap::new(),
            sides: DashMap::new(),
            default_config,
            persist_dir: std::env::var("CALICHE_SUSPEND_DIR")
                .ok()
                .map(|raw| raw.trim().to_string())
                .filter(|raw| !raw.is_empty())
                .map(PathBuf::from),
        }
    }

    /// Persist suspend snapshots under `dir` so resume survives process restart.
    #[must_use]
    pub fn with_persist_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.persist_dir = Some(dir.into());
        self
    }

    /// Directory used by [`Self::suspend`] / [`Self::resume_with`].
    #[must_use]
    pub fn persist_dir(&self) -> Option<&Path> {
        self.persist_dir.as_deref()
    }
}

impl Default for AgentRuntime {
    fn default() -> Self {
        Self::new(HarnessConfig::default())
    }
}

impl AgentRuntime {
    /// Default harness config used when creating sessions without an override.
    #[must_use]
    pub fn default_config(&self) -> &HarnessConfig {
        &self.default_config
    }

    /// Number of live sessions.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Snapshot of current session ids (unordered).
    #[must_use]
    pub fn session_ids(&self) -> Vec<String> {
        self.sessions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Whether a session id is currently tracked.
    #[must_use]
    pub fn contains(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    /// Insert an already-built harness, keyed by its session id.
    pub fn insert_harness(&self, harness: AgentHarness) -> SharedHarness {
        let session_id = harness.session_id().to_string();
        self.sides
            .insert(session_id.clone(), Arc::clone(&harness.side));
        let shared = Arc::new(RwLock::new(harness));
        self.sessions.insert(session_id, Arc::clone(&shared));
        shared
    }

    /// Insert a shared harness handle under an explicit session id.
    pub fn insert_shared(&self, session_id: impl Into<String>, harness: SharedHarness) {
        let session_id = session_id.into();
        if let Ok(guard) = harness.try_read() {
            self.sides
                .insert(session_id.clone(), Arc::clone(&guard.side));
        }
        self.sessions.insert(session_id, harness);
    }

    /// Create a new harness session from the default config.
    pub async fn create_session(&self) -> Result<String> {
        self.create_session_with_config(self.default_config.clone())
            .await
    }

    /// Create a new harness session from an explicit config.
    pub async fn create_session_with_config(&self, config: HarnessConfig) -> Result<String> {
        let harness = AgentHarness::new(config)
            .await
            .context("AgentRuntime: create harness session")?;
        let session_id = harness.session_id().to_string();
        self.insert_harness(harness);
        Ok(session_id)
    }

    /// Create a session from an existing [`AgentService`].
    pub fn create_session_from_service(
        &self,
        service: AgentService,
        config: Option<HarnessConfig>,
    ) -> Result<String> {
        let config = config.unwrap_or_else(|| self.default_config.clone());
        let harness = AgentHarness::from_service(service, config)
            .context("AgentRuntime: create harness from service")?;
        let session_id = harness.session_id().to_string();
        self.insert_harness(harness);
        Ok(session_id)
    }

    /// Borrow a shared harness handle.
    #[must_use]
    pub fn get(&self, session_id: &str) -> Option<SharedHarness> {
        self.sessions
            .get(session_id)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// Remove and return a shared harness handle.
    pub fn remove(&self, session_id: &str) -> Option<SharedHarness> {
        self.sides.remove(session_id);
        self.sessions.remove(session_id).map(|(_, harness)| harness)
    }

    /// Run one turn on an existing session.
    ///
    /// The prompt is submitted through the session inbox with a generated
    /// input id. Use [`Self::submit_input`] to supply a stable id.
    pub async fn run_turn(&self, session_id: &str, input: &str) -> Result<HarnessTurnResult> {
        let outcome = self
            .submit_input(
                session_id,
                crate::harness::inbox::InboxInput::external(None::<String>, input),
            )
            .await?;
        outcome.turn.ok_or_else(|| {
            anyhow::anyhow!("AgentRuntime: external input produced no turn for '{session_id}'")
        })
    }

    /// Submit an inbox item (external / control / crash) with caller-supplied or generated id.
    ///
    /// Redelivery of the same id is a no-op (`ack.duplicate == true`).
    pub async fn submit_input(
        &self,
        session_id: &str,
        input: crate::harness::inbox::InboxInput,
    ) -> Result<crate::harness::inbox::InboxTurnOutcome> {
        let harness = self
            .get(session_id)
            .with_context(|| format!("AgentRuntime: unknown session '{session_id}'"))?;
        if matches!(input.kind, crate::harness::inbox::InputKind::Control)
            && harness.try_write().is_err()
        {
            return self.submit_control_unlocked(session_id, input);
        }
        let mut guard = harness.write().await;
        if guard.is_suspending() {
            bail!("AgentRuntime: session '{session_id}' is suspending");
        }
        if guard.is_hard_stopped()
            && matches!(
                input.kind,
                crate::harness::inbox::InputKind::External
                    | crate::harness::inbox::InputKind::Crash
            )
        {
            bail!("session is hard-stopped; refusing new turns");
        }
        let ack = guard
            .submit_inbox(input)
            .with_context(|| format!("AgentRuntime: inbox submit failed for '{session_id}'"))?;
        if ack.duplicate {
            return Ok(crate::harness::InboxTurnOutcome {
                ack,
                turn: None,
                control: None,
            });
        }
        guard
            .apply_inbox(&ack.input_id)
            .await
            .with_context(|| format!("AgentRuntime: inbox apply failed for '{session_id}'"))
    }

    fn submit_control_unlocked(
        &self,
        session_id: &str,
        input: crate::harness::inbox::InboxInput,
    ) -> Result<crate::harness::inbox::InboxTurnOutcome> {
        let side = self
            .sides
            .get(session_id)
            .with_context(|| format!("AgentRuntime: unknown session '{session_id}'"))?;
        let mut side = side.lock();
        if side.suspending {
            bail!("AgentRuntime: session '{session_id}' is suspending");
        }
        let ack = side.inbox.submit(input.clone())?;
        if ack.duplicate {
            return Ok(crate::harness::InboxTurnOutcome {
                ack,
                turn: None,
                control: None,
            });
        }
        let control = input
            .control
            .ok_or_else(|| anyhow::anyhow!("control input missing payload"))?;
        let outcome = crate::harness::core::session_apply_control_flags(&mut side, control, false)?;
        side.inbox
            .mark_applied(&ack.input_id, Some(format!("{outcome:?}")))?;
        Ok(crate::harness::InboxTurnOutcome {
            ack: crate::harness::inbox::InboxAck {
                input_id: ack.input_id,
                duplicate: false,
                kind: crate::harness::inbox::InputKind::Control,
                applied: true,
            },
            turn: None,
            control: Some(outcome),
        })
    }

    /// Undo the last message-level change set for a session.
    pub async fn undo_last(&self, session_id: &str) -> Result<Option<MessageDiff>> {
        let harness = self.require(session_id)?;
        let mut guard = harness.write().await;
        guard
            .undo_last()
            .await
            .with_context(|| format!("AgentRuntime: undo failed for session '{session_id}'"))
    }

    /// Undo one message-level change set and every newer set for a session.
    pub async fn undo_to_message(
        &self,
        session_id: &str,
        message_id: uuid::Uuid,
    ) -> Result<Vec<MessageDiff>> {
        let harness = self.require(session_id)?;
        let mut guard = harness.write().await;
        guard.undo_to_message(message_id).await.with_context(|| {
            format!("AgentRuntime: undo-to-message failed for session '{session_id}'")
        })
    }

    /// Redo the most recently reverted message-level change set for a session.
    pub async fn redo_last(&self, session_id: &str) -> Result<Option<MessageDiff>> {
        let harness = self.require(session_id)?;
        let mut guard = harness.write().await;
        guard
            .redo_last()
            .await
            .with_context(|| format!("AgentRuntime: redo failed for session '{session_id}'"))
    }

    /// Summarize recorded message-level diffs for a session.
    pub async fn session_diff_summary(
        &self,
        session_id: &str,
        message_id: Option<uuid::Uuid>,
    ) -> Result<SessionDiffSummary> {
        let harness = self.require(session_id)?;
        let guard = harness.read().await;
        Ok(guard
            .diff_tracker()
            .read()
            .await
            .session_summary(message_id))
    }

    /// Require a shared harness or return a structured error.
    pub fn require(&self, session_id: &str) -> Result<SharedHarness> {
        self.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("AgentRuntime: unknown session '{session_id}'"))
    }

    /// Require a shared harness or bail with a runtime error.
    pub fn require_mut(&self, session_id: &str) -> Result<SharedHarness> {
        match self.get(session_id) {
            Some(harness) => Ok(harness),
            None => bail!("AgentRuntime: unknown session '{session_id}'"),
        }
    }

    fn suspend_store(&self) -> Result<SuspendStore> {
        let dir = self.persist_dir.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "AgentRuntime: persist dir is required for suspend/resume \
                 (AgentRuntime::with_persist_dir or CALICHE_SUSPEND_DIR)"
            )
        })?;
        Ok(SuspendStore::new(dir.clone()))
    }

    /// Persist session state and drop the in-memory harness.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is unknown, persist dir is unset, or
    /// the snapshot cannot be written.
    pub async fn suspend(&self, session_id: &str) -> Result<SuspendedSession> {
        let harness = self.require(session_id)?;
        let snapshot = {
            let guard = harness.write().await;
            if guard.is_suspending() {
                bail!("AgentRuntime: session '{session_id}' is already suspending");
            }
            guard.set_suspending(true);
            let snapshot = guard.suspend_snapshot();
            if let Err(err) = self.suspend_store()?.save(&snapshot) {
                guard.set_suspending(false);
                return Err(err);
            }
            snapshot
        };
        self.remove(session_id);
        Ok(snapshot)
    }

    /// Restore a previously suspended session into `harness` after a process boundary.
    ///
    /// `harness` is a freshly constructed actor (typically
    /// [`AgentHarness::from_service`]); conversation, phase, and checkpoint
    /// identity are overwritten from disk.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot is missing or the session is already live.
    pub async fn resume_with(&self, session_id: &str, mut harness: AgentHarness) -> Result<String> {
        if self.contains(session_id) {
            bail!("AgentRuntime: session '{session_id}' is already live");
        }
        let snapshot = self.suspend_store()?.load(session_id)?;
        if snapshot.session_id != session_id {
            bail!(
                "AgentRuntime: suspend snapshot session '{}' does not match '{session_id}'",
                snapshot.session_id
            );
        }
        harness
            .apply_suspended(&snapshot)
            .await
            .context("AgentRuntime: apply suspend snapshot")?;
        let restored = harness.session_id().to_string();
        self.insert_harness(harness);
        Ok(restored)
    }

    /// Resume a snapshot using the default harness config and `service`.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot cannot be applied.
    pub async fn resume_from_service(
        &self,
        session_id: &str,
        service: AgentService,
        config: Option<HarnessConfig>,
    ) -> Result<String> {
        let config = config.unwrap_or_else(|| self.default_config.clone());
        let harness = AgentHarness::from_service(service, config)
            .context("AgentRuntime: create harness for resume")?;
        self.resume_with(session_id, harness).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::ToolRegistry;
    use crate::llm::ProviderType;
    use crate::messages::Message;
    use crate::testing::MockProvider;
    use std::sync::Arc;

    fn mock_service() -> AgentService {
        let provider = Arc::new(MockProvider::new(vec!["first".into(), "second".into()]));
        let tools = Arc::new(ToolRegistry::new());
        let agent = crate::agent::Agent::new(provider, tools);
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service")
    }

    #[tokio::test]
    async fn create_lookup_remove_session_roundtrip() {
        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        assert_eq!(runtime.session_count(), 1);
        assert!(runtime.contains(&session_id));
        assert!(runtime.session_ids().contains(&session_id));
        assert_eq!(
            runtime
                .get(&session_id)
                .expect("session")
                .read()
                .await
                .session_id(),
            session_id
        );

        let removed = runtime.remove(&session_id).expect("remove session");
        assert_eq!(removed.read().await.session_id(), session_id);
        assert_eq!(runtime.session_count(), 0);
        assert!(!runtime.contains(&session_id));
    }

    #[tokio::test]
    async fn run_turn_routes_to_session_and_unknown_errors() {
        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        let result = runtime
            .run_turn(&session_id, "hello")
            .await
            .expect("turn should succeed");
        assert_eq!(result.response, "first");
        assert_eq!(result.discipline.tool_calls, 0);

        let err = runtime
            .run_turn("missing-session", "hello")
            .await
            .expect_err("unknown session should fail");
        assert!(
            err.to_string().contains("unknown session"),
            "error should name missing session, got: {err}"
        );
    }

    #[tokio::test]
    async fn require_mut_exposes_harness_history_helpers() {
        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        {
            let harness = runtime.require_mut(&session_id).expect("session");
            let mut guard = harness.write().await;
            guard.set_history(vec![Message::user("seed")]);
            assert_eq!(guard.history().len(), 1);
        }

        let err = runtime
            .require_mut("nope")
            .expect_err("missing session should fail");
        assert!(err.to_string().contains("unknown session"));
    }

    #[tokio::test]
    async fn insert_harness_registers_shared_handle() {
        let runtime = AgentRuntime::new(HarnessConfig::default());
        let service = mock_service();
        let harness =
            AgentHarness::from_service(service, HarnessConfig::default()).expect("harness");
        let session_id = harness.session_id().to_string();
        let shared = runtime.insert_harness(harness);
        assert!(runtime.contains(&session_id));
        assert_eq!(shared.read().await.session_id(), session_id);
    }

    #[tokio::test]
    async fn suspend_survives_process_boundary_and_resumes() {
        use crate::harness::suspend::SessionPhase;
        use crate::messages::Message;

        let dir = tempfile::TempDir::new().expect("temp dir");
        let runtime = AgentRuntime::new(HarnessConfig::default()).with_persist_dir(dir.path());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        {
            let harness = runtime.require(&session_id).expect("session");
            let mut guard = harness.write().await;
            guard.set_history(vec![Message::user("first"), Message::assistant("ack")]);
            guard.set_phase(SessionPhase::InTurn);
            let checkpoint_id = guard.checkpoint("before-suspend").await;
            assert!(!checkpoint_id.is_empty());
        }

        let snapshot = runtime.suspend(&session_id).await.expect("suspend");
        assert!(!runtime.contains(&session_id));
        assert_eq!(snapshot.phase, SessionPhase::InTurn);
        assert_eq!(snapshot.turn_count, 1);
        assert!(snapshot.last_checkpoint_id.is_some());

        drop(runtime);

        let runtime = AgentRuntime::new(HarnessConfig::default()).with_persist_dir(dir.path());
        let resumed = runtime
            .resume_from_service(&session_id, mock_service(), None)
            .await
            .expect("resume after drop");
        assert_eq!(resumed, session_id);
        let harness = runtime.require(&session_id).expect("restored");
        let guard = harness.read().await;
        assert_eq!(guard.session_id(), session_id);
        assert_eq!(guard.history().len(), 2);
        assert_eq!(guard.phase(), SessionPhase::InTurn);
        assert_eq!(
            guard.last_checkpoint_id(),
            snapshot.last_checkpoint_id.as_deref()
        );
        assert_eq!(
            guard.state_manager().read().await.list_checkpoints().len(),
            1
        );
    }

    #[tokio::test]
    async fn inbox_redelivery_does_not_double_apply() {
        use crate::harness::inbox::InboxInput;

        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        let first = runtime
            .submit_input(&session_id, InboxInput::external(Some("stable-1"), "hello"))
            .await
            .expect("first apply");
        assert!(!first.ack.duplicate);
        assert_eq!(
            first.turn.as_ref().map(|t| t.response.as_str()),
            Some("first")
        );

        let second = runtime
            .submit_input(
                &session_id,
                InboxInput::external(Some("stable-1"), "hello again"),
            )
            .await
            .expect("redeliver");
        assert!(second.ack.duplicate);
        assert!(second.turn.is_none());

        let harness = runtime.require(&session_id).expect("session");
        assert_eq!(harness.read().await.history().len(), 2);
    }

    #[tokio::test]
    async fn hard_stop_control_rejects_further_turns() {
        use crate::harness::inbox::{ControlMessage, ControlMode, InboxInput};

        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        let stop = runtime
            .submit_input(
                &session_id,
                InboxInput::control(
                    Some("stop-1"),
                    ControlMessage {
                        mode: ControlMode::Hard,
                        reason: "operator stop".into(),
                        settings: None,
                    },
                ),
            )
            .await
            .expect("hard stop");
        assert!(matches!(
            stop.control,
            Some(crate::harness::inbox::ControlOutcome::HardStopped { .. })
        ));

        let err = runtime
            .submit_input(
                &session_id,
                InboxInput::external(Some("after-stop"), "hello"),
            )
            .await
            .expect_err("hard-stopped session rejects turns");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("hard-stopped"),
            "expected hard-stop cause in error chain, got: {rendered}"
        );
    }

    #[tokio::test]
    async fn settings_control_applies_reasoning_effort() {
        use crate::harness::inbox::{ControlMessage, ControlMode, InboxInput, InboxSettings};
        use crate::messages::ReasoningEffort;

        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        let applied = runtime
            .submit_input(
                &session_id,
                InboxInput::control(
                    Some("set-1"),
                    ControlMessage {
                        mode: ControlMode::Settings,
                        reason: String::new(),
                        settings: Some(InboxSettings {
                            reasoning_effort: Some("high".into()),
                        }),
                    },
                ),
            )
            .await
            .expect("settings");
        assert!(matches!(
            applied.control,
            Some(crate::harness::inbox::ControlOutcome::SettingsApplied { .. })
        ));

        let harness = runtime.require(&session_id).expect("session");
        let guard = harness.read().await;
        assert_eq!(
            guard.service().agent().current_reasoning_effort(),
            Some(ReasoningEffort::High)
        );
    }

    #[tokio::test]
    async fn when_idle_defers_until_turn_finishes() {
        use crate::harness::inbox::{ControlMessage, ControlMode, InboxInput};

        let runtime = AgentRuntime::new(HarnessConfig::default());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");
        {
            let harness = runtime.require(&session_id).expect("session");
            let mut guard = harness.write().await;
            guard.set_phase(crate::harness::suspend::SessionPhase::InTurn);
        }

        let deferred = runtime
            .submit_input(
                &session_id,
                InboxInput::control(
                    Some("idle-1"),
                    ControlMessage {
                        mode: ControlMode::WhenIdle,
                        reason: "wrap up".into(),
                        settings: None,
                    },
                ),
            )
            .await
            .expect("when-idle");
        assert!(matches!(
            deferred.control,
            Some(crate::harness::inbox::ControlOutcome::WhenIdle {
                applied_immediately: false,
                ..
            })
        ));

        {
            let harness = runtime.require(&session_id).expect("session");
            let mut guard = harness.write().await;
            assert!(!guard.is_hard_stopped());
            guard.finish_turn_side_effects_after_error();
            assert!(guard.is_hard_stopped());
        }
    }

    #[tokio::test]
    async fn inbox_and_compaction_report_survive_suspend() {
        use crate::harness::inbox::{ControlMessage, ControlMode, InboxInput};
        use crate::messages::{CompactionChangeKind, CompactionReport};

        let dir = tempfile::TempDir::new().expect("temp dir");
        let runtime = AgentRuntime::new(HarnessConfig::default()).with_persist_dir(dir.path());
        let session_id = runtime
            .create_session_from_service(mock_service(), None)
            .expect("create session");

        runtime
            .submit_input(&session_id, InboxInput::external(Some("ext-1"), "hello"))
            .await
            .expect("external");
        let hb = runtime
            .submit_input(
                &session_id,
                InboxInput::control(
                    Some("hb-1"),
                    ControlMessage {
                        mode: ControlMode::Heartbeat,
                        reason: "ping".into(),
                        settings: None,
                    },
                ),
            )
            .await
            .expect("heartbeat");
        assert!(matches!(
            hb.control,
            Some(crate::harness::inbox::ControlOutcome::Heartbeat { .. })
        ));

        {
            let harness = runtime.require(&session_id).expect("session");
            let mut guard = harness.write().await;
            let mut report = CompactionReport::new();
            report.record(CompactionChangeKind::Compacted, "jev", "unit-test");
            guard.compaction_reports.push(report);
        }

        runtime.suspend(&session_id).await.expect("suspend");
        drop(runtime);

        let runtime = AgentRuntime::new(HarnessConfig::default()).with_persist_dir(dir.path());
        runtime
            .resume_from_service(&session_id, mock_service(), None)
            .await
            .expect("resume");

        let redeliver = runtime
            .submit_input(&session_id, InboxInput::external(Some("ext-1"), "hello"))
            .await
            .expect("redeliver after resume");
        assert!(redeliver.ack.duplicate);

        let harness = runtime.require(&session_id).expect("restored");
        let guard = harness.read().await;
        assert!(guard.inbox().contains("ext-1"));
        assert!(guard.inbox().contains("hb-1"));
        assert_eq!(guard.compaction_reports().len(), 1);
        assert_eq!(guard.compaction_reports()[0].changes[0].source, "jev");
    }
}
