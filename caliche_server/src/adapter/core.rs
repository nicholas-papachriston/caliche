//! Core session and gateway operations for [`super::ApiAdapter`].

use super::broadcast_resume::{ResumedSessionItem, live_broadcast_resume};
use super::{
    ApiAdapter, ApiAdapterError, LlmSlugGenerator, bindings::ApiSessionBindingStore,
    session::SessionState,
};
use super::{SessionId, SessionInfo, SessionOwner, StreamingTurnResult};
use anyhow::Result;
use caliche::agent::session::{
    ShareConfig, SharePayload, ShareVisibility, TranscriptRedactor, default_share_store,
};
use caliche::gateway::{
    BindingKey, GatewayAdapterRegistry, GatewayRuntime, GatewayRuntimeConfig,
    GatewayRuntimeMetricsSnapshot, InMemoryDedupeStore, MemoryDeadLetterStore, SessionBinding,
    SessionBindingStore,
};
use caliche::harness::{
    AgentEvent, AgentRuntime, CodingTaskRequest, CodingTaskResult, HarnessConfig,
    HarnessTurnResult, MessageDiff, SessionConfig, TaskResult, run_coding_task_with_harness,
};
use caliche::llm::LlmClient;
use caliche::messages::StreamEvent;
use moka::future::Cache;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

impl ApiAdapter {
    /// Create a new API adapter
    pub fn new(default_config: HarnessConfig) -> Self {
        Self::with_session_ttl(default_config, Duration::from_secs(3600), None)
    }

    /// Create a new adapter and initialize session tracking from environment.
    /// When shared_service is Some (CLI hosting), session harnesses are derived from it.
    pub async fn new_with_env(
        default_config: HarnessConfig,
        shared_service: Option<Arc<tokio::sync::RwLock<caliche::AgentService>>>,
    ) -> Self {
        Self::with_session_ttl_from_env(default_config, Duration::from_secs(3600), shared_service)
            .await
    }

    /// Create with custom session TTL and optional shared service (CLI-hosted mode).
    pub fn with_session_ttl(
        default_config: HarnessConfig,
        session_ttl: Duration,
        shared_service: Option<Arc<tokio::sync::RwLock<caliche::AgentService>>>,
    ) -> Self {
        let agent_runtime = Arc::new(AgentRuntime::new(default_config.clone()));
        let runtime_for_eviction = Arc::clone(&agent_runtime);
        let turn_cancels: Arc<dashmap::DashMap<SessionId, super::cancel::TurnCancel>> =
            Arc::new(dashmap::DashMap::new());
        let cancels_for_eviction = Arc::clone(&turn_cancels);
        let sessions = Cache::builder()
            .max_capacity(1000)
            .time_to_idle(session_ttl)
            .eviction_listener(move |key: Arc<String>, _value, _cause| {
                runtime_for_eviction.remove(key.as_str());
                if let Some((_, cancel)) = cancels_for_eviction.remove(key.as_str()) {
                    cancel.signal();
                }
            })
            .build();
        let dedupe = Arc::new(InMemoryDedupeStore::new());
        let gateway_runtime = Arc::new(GatewayRuntime::new(
            GatewayRuntimeConfig::default(),
            Some(dedupe),
        ));

        Self {
            security: Arc::new(crate::security::SecurityPolicy {
                jwt_enabled: false,
                require_auth: false,
                jwt_state: None,
                trusted_proxy: false,
                trusted_proxy_cidrs: Vec::new(),
                metrics_token: None,
                allow_unauth_orchestration: false,
                public_chat: false,
            }),
            factory: super::factory::HarnessFactory::new_with_shared(
                default_config,
                shared_service,
            ),
            runtime: agent_runtime,
            turn_cancels,
            sessions,
            session_ttl,
            tracker: None,
            slug_client: None,
            gateway_bindings: Arc::new(ApiSessionBindingStore::new(None)),
            gateway_adapters: Arc::new(GatewayAdapterRegistry::with_default_adapters()),
            gateway_dlq: Arc::new(MemoryDeadLetterStore::new(512)),
            gateway_runtime,
            shares: default_share_store().unwrap_or_else(|error| {
                tracing::warn!("Share directory store unavailable ({error}); using memory");
                Arc::new(caliche::agent::session::MemoryShareStore::new())
            }),
            share_config: ShareConfig::default(),
        }
    }

    /// Override share policy / redaction configuration.
    #[must_use]
    pub fn with_share_config(mut self, share_config: ShareConfig) -> Self {
        self.share_config = share_config;
        self
    }

    /// Process-local SDK session runtime backing this adapter.
    pub fn runtime(&self) -> &AgentRuntime {
        self.runtime.as_ref()
    }

    async fn store_session_state(&self, session_id: SessionId, session_state: SessionState) {
        self.turn_cancels
            .insert(session_id.clone(), super::cancel::TurnCancel::new());
        self.runtime
            .insert_shared(session_id.clone(), session_state.shared_harness());
        let state = Arc::new(tokio::sync::RwLock::new(session_state));
        self.sessions.insert(session_id, state).await;
    }

    /// Attach the process [`SecurityPolicy`] resolved at router startup.
    #[must_use]
    pub fn with_security(mut self, security: Arc<crate::security::SecurityPolicy>) -> Self {
        self.security = security;
        self
    }

    pub fn security(&self) -> &crate::security::SecurityPolicy {
        self.security.as_ref()
    }

    /// Create with custom session TTL and optional persistent store configured by env vars.
    pub async fn with_session_ttl_from_env(
        default_config: HarnessConfig,
        session_ttl: Duration,
        shared_service: Option<Arc<tokio::sync::RwLock<caliche::AgentService>>>,
    ) -> Self {
        let mut adapter = Self::with_session_ttl(default_config, session_ttl, shared_service);

        adapter.tracker = match super::build_tracker_from_env().await {
            Ok(tracker) => tracker,
            Err(err) => {
                tracing::warn!("Session tracking disabled: {err}");
                None
            }
        };
        adapter
            .gateway_bindings
            .set_tracker(adapter.tracker.clone())
            .await;

        adapter.slug_client = match super::slug_provider_model_from_env() {
            Ok((provider, model)) => LlmClient::builder()
                .provider(provider)
                .model(model)
                .build()
                .await
                .ok(),
            Err(err) => {
                tracing::debug!("Slug LLM disabled: {err}");
                None
            }
        };

        adapter
    }

    /// Gateway session-binding store backed by this adapter.
    pub fn gateway_binding_store(&self) -> Arc<dyn SessionBindingStore> {
        self.gateway_bindings.clone()
    }

    pub fn gateway_adapter_registry(&self) -> Arc<GatewayAdapterRegistry> {
        Arc::clone(&self.gateway_adapters)
    }

    pub fn gateway_dead_letter_store(&self) -> Arc<MemoryDeadLetterStore> {
        Arc::clone(&self.gateway_dlq)
    }

    pub fn gateway_runtime(&self) -> Arc<GatewayRuntime> {
        Arc::clone(&self.gateway_runtime)
    }

    pub fn gateway_runtime_metrics(&self) -> GatewayRuntimeMetricsSnapshot {
        self.gateway_runtime.metrics.snapshot()
    }

    pub fn gateway_set_adapter_enabled(&self, adapter_id: &str, enabled: bool) -> bool {
        self.gateway_adapters.set_enabled(adapter_id, enabled)
    }

    pub async fn gateway_list_bindings(&self, session_id: &str) -> Result<Vec<SessionBinding>> {
        self.gateway_bindings.list_for_session(session_id).await
    }

    pub async fn gateway_override_binding(
        &self,
        key: BindingKey,
        session_id: String,
    ) -> Result<SessionBinding> {
        let now = chrono::Utc::now();
        let binding = SessionBinding {
            key,
            session_id,
            created_at: now,
            updated_at: now,
            metadata: std::collections::HashMap::new(),
        };
        self.gateway_bindings.put(binding.clone()).await?;
        Ok(binding)
    }

    /// Resolve or create a session id for a gateway binding key.
    ///
    /// If `preferred_session_id` is provided, it is persisted as the binding target.
    /// Otherwise this will reuse an existing bound session or create a new one.
    pub async fn resolve_gateway_session(
        &self,
        key: BindingKey,
        preferred_session_id: Option<String>,
    ) -> Result<String> {
        if let Some(existing) = self.gateway_bindings.get(&key).await? {
            return Ok(existing.session_id);
        }

        let session_id = if let Some(id) = preferred_session_id {
            id
        } else {
            Box::pin(self.create_session(None, None, None)).await?
        };

        let now = chrono::Utc::now();
        self.gateway_bindings
            .put(SessionBinding {
                key,
                session_id: session_id.clone(),
                created_at: now,
                updated_at: now,
                metadata: std::collections::HashMap::new(),
            })
            .await?;
        Ok(session_id)
    }

    /// Create a new session, optionally binding an authenticated owner and title.
    ///
    /// `title` is the already-normalized human title. `None` leaves the field
    /// off the live session and off list JSON.
    pub async fn create_session(
        &self,
        config: Option<SessionConfig>,
        owner: Option<SessionOwner>,
        title: Option<String>,
    ) -> Result<SessionId> {
        let harness = if let Some(session_config) = config {
            Box::pin(self.factory.create_with_session(session_config)).await?
        } else {
            Box::pin(self.factory.create()).await?
        };

        let session_id = harness.session_id().to_string();
        let mut session_state = SessionState::new(harness);
        let owner = owner.unwrap_or_default();
        session_state.owner = owner.clone();
        session_state.title = title;
        self.store_session_state(session_id.clone(), session_state)
            .await;

        if let Some(tracker) = &self.tracker
            && let Ok(id) = Uuid::parse_str(&session_id)
        {
            let mut tracked = match tracker.get(id).await? {
                Some(existing) => existing,
                None => {
                    tracker
                        .create_with_id(id, "api", owner.user_id.as_deref())
                        .await?
                }
            };
            tracked.user_id = owner.user_id.clone();
            if let Some(tenant_id) = &owner.tenant_id {
                if !tracked.metadata.is_object() {
                    tracked.metadata = serde_json::json!({});
                }
                if let Some(root) = tracked.metadata.as_object_mut() {
                    root.insert(
                        "tenant_id".to_string(),
                        serde_json::Value::String(tenant_id.clone()),
                    );
                }
            }
            tracked.conversation_id = Some(session_id.clone());
            tracker.registry().update(&tracked).await?;
        }

        Ok(session_id)
    }

    /// Resolve the authenticated owner for a session, if recorded.
    pub async fn session_owner(&self, session_id: &str) -> Option<SessionOwner> {
        if let Some(session) = self.get_session(session_id).await {
            let state = session.read().await;
            if state.owner.user_id.is_some() {
                return Some(state.owner.clone());
            }
        }

        let tracker = self.tracker.as_ref()?;
        let id = Uuid::parse_str(session_id).ok()?;
        let tracked = tracker.get(id).await.ok()??;
        Some(SessionOwner {
            user_id: tracked.user_id,
            tenant_id: tracked
                .metadata
                .get("tenant_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
        })
    }

    /// Create a new session with explicit harness config and optional session config.
    pub async fn create_session_with_harness_config(
        &self,
        harness_config: HarnessConfig,
        session_config: Option<SessionConfig>,
    ) -> Result<SessionId> {
        let harness = if let Some(session_config) = session_config {
            Box::pin(
                self.factory
                    .create_with_config_and_session(harness_config, session_config),
            )
            .await?
        } else {
            Box::pin(self.factory.create_with_config(harness_config)).await?
        };

        let session_id = harness.session_id().to_string();
        self.store_session_state(session_id.clone(), SessionState::new(harness))
            .await;

        if let Some(tracker) = &self.tracker
            && let Ok(id) = Uuid::parse_str(&session_id)
        {
            let mut tracked = match tracker.get(id).await? {
                Some(existing) => existing,
                None => tracker.create_with_id(id, "api", None).await?,
            };
            tracked.conversation_id = Some(session_id.clone());
            tracker.registry().update(&tracked).await?;
        }

        Ok(session_id)
    }

    /// Get a session by ID
    pub async fn get_session(
        &self,
        session_id: &str,
    ) -> Option<Arc<tokio::sync::RwLock<SessionState>>> {
        self.sessions.get(session_id).await
    }

    /// Remove gateway bindings that reference sessions evicted from the in-memory session cache.
    pub async fn sweep_orphan_gateway_bindings(&self) {
        use caliche::gateway::SessionBindingStore;
        let bindings = self.gateway_bindings.list_all_bindings();
        for binding in bindings {
            if self.get_session(&binding.session_id).await.is_some() {
                continue;
            }
            if let Err(err) =
                SessionBindingStore::delete(&*self.gateway_bindings, &binding.key).await
            {
                tracing::debug!(
                    target: "caliche_server::adapter",
                    error = %err,
                    session_id = %binding.session_id,
                    "gateway binding sweep delete failed"
                );
            }
        }
    }

    /// Write turn counts onto the durable session row when a tracker is set.
    async fn record_turn_activity(&self, session_id: &str, message_count: u32) -> Result<()> {
        let Some(tracker) = &self.tracker else {
            return Ok(());
        };
        let Ok(id) = Uuid::parse_str(session_id) else {
            return Ok(());
        };
        let now = chrono::Utc::now();
        let mut tracked = match tracker.get(id).await? {
            Some(existing) => existing,
            None => tracker.create_with_id(id, "api", None).await?,
        };
        tracked.conversation_id = Some(session_id.to_string());
        tracked.message_count = message_count;
        tracked.updated_at = now;
        tracked.last_accessed = now;
        tracker.registry().update(&tracked).await?;
        Ok(())
    }

    /// Execute a turn in a session, keeping the session's current provider and model.
    pub async fn execute_turn(&self, session_id: &str, input: &str) -> Result<HarnessTurnResult> {
        self.execute_turn_with_selection(session_id, input, None, None, None)
            .await
    }

    /// Execute a turn after aligning the live session with an optional provider and model.
    ///
    /// `provider` and `model` come from the thin-client turn body. When either
    /// differs from the harness, the same session is retargeted and its history
    /// stays in place. Omitted fields keep the current identity.
    pub async fn execute_turn_with_selection(
        &self,
        session_id: &str,
        input: &str,
        provider: Option<&str>,
        model: Option<&str>,
        context_window: Option<usize>,
    ) -> Result<HarnessTurnResult> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;
        let Some(cancel) = self.turn_cancel_handle(session_id) else {
            return Err(ApiAdapterError::SessionNotFound(session_id.to_string()).into());
        };

        let mut state = session.write().await;
        state.touch();
        let result = {
            let mut harness = state.harness.write().await;
            // Retarget before the turn future is polled. A cancel that is already
            // signalled skips the provider call and still leaves the new model.
            super::model::apply_turn_model(&mut harness, provider, model).await?;
            super::model::apply_turn_context_window(&mut harness, context_window)?;
            let run = harness.run_turn_stream_with_cancel(
                input,
                Some(std::sync::Arc::clone(&cancel.flag)),
                |_| {},
            );
            super::cancel::await_turn(session_id, &cancel, run).await?
        };

        let message_count = state.harness.read().await.history().len() as u32;
        self.record_turn_activity(session_id, message_count).await?;

        Ok(result)
    }

    /// Execute a turn and forward stream events to a callback.
    ///
    /// The caller token and the session cancel token both abort the turn.
    /// Session cancel also removes the id via [`Self::cancel_session`].
    pub async fn execute_turn_streaming<F>(
        &self,
        session_id: &str,
        input: &str,
        cancel_token: CancellationToken,
        mut on_event: F,
    ) -> Result<StreamingTurnResult>
    where
        F: FnMut(StreamEvent),
    {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;
        let Some(session_cancel) = self.turn_cancel_handle(session_id) else {
            return Err(ApiAdapterError::SessionNotFound(session_id.to_string()).into());
        };

        if cancel_token.is_cancelled() || session_cancel.token.is_cancelled() {
            return Ok(StreamingTurnResult { cancelled: true });
        }

        let mut state = session.write().await;
        state.touch();

        let mut cancelled = false;
        let mut harness = state.harness.write().await;
        let flag = std::sync::Arc::clone(&session_cancel.flag);
        let run = harness.run_turn_stream_with_cancel(
            input,
            Some(std::sync::Arc::clone(&flag)),
            |event_result| {
                if cancel_token.is_cancelled() || flag.load(std::sync::atomic::Ordering::Relaxed) {
                    cancelled = true;
                    return;
                }
                if let Ok(event) = event_result {
                    on_event(event);
                }
            },
        );

        let run_result =
            super::cancel::await_streaming_turn(session_id, &session_cancel, &cancel_token, run)
                .await;
        if cancel_token.is_cancelled() || session_cancel.token.is_cancelled() {
            cancelled = true;
        }

        match run_result {
            Ok(_result) => {
                self.record_turn_activity(session_id, harness.history().len() as u32)
                    .await?;
                Ok(StreamingTurnResult { cancelled })
            }
            Err(err) if cancelled || Self::turn_was_cancelled(&err) => {
                Ok(StreamingTurnResult { cancelled: true })
            }
            Err(err) => Err(err),
        }
    }

    /// Execute a complete task in a session
    pub async fn execute_task(&self, session_id: &str, task: &str) -> Result<TaskResult> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;
        let Some(cancel) = self.turn_cancel_handle(session_id) else {
            return Err(ApiAdapterError::SessionNotFound(session_id.to_string()).into());
        };

        let mut state = session.write().await;
        state.touch();
        let result = {
            let mut harness = state.harness.write().await;
            let run = harness.run_task(task);
            super::cancel::await_turn(session_id, &cancel, run).await?
        };

        let message_count = state.harness.read().await.history().len() as u32;
        self.record_turn_activity(session_id, message_count).await?;

        Ok(result)
    }

    /// Execute a shared coding task. If `session_id` is provided the existing
    /// session harness is reused; otherwise a new harness is created from the
    /// request's provider/model/workspace/tools options.
    pub async fn execute_coding_task(
        &self,
        session_id: Option<&str>,
        request: CodingTaskRequest,
    ) -> Result<CodingTaskResult> {
        if let Some(session_id) = session_id {
            let session = self
                .get_session(session_id)
                .await
                .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;
            let mut state = session.write().await;
            state.touch();
            return run_coding_task_with_harness(&mut *state.harness.write().await, &request).await;
        }

        let mut harness = self
            .factory
            .create_with_config(request.to_harness_config())
            .await?;
        let result = run_coding_task_with_harness(&mut harness, &request).await?;
        let session_id = harness.session_id().to_string();
        self.store_session_state(session_id.clone(), SessionState::new(harness))
            .await;

        if let Some(tracker) = &self.tracker
            && let Ok(id) = Uuid::parse_str(&session_id)
        {
            let mut tracked = match tracker.get(id).await? {
                Some(existing) => existing,
                None => tracker.create_with_id(id, "coding", None).await?,
            };
            tracked.conversation_id = Some(session_id);
            tracked.updated_at = chrono::Utc::now();
            tracked.last_accessed = tracked.updated_at;
            tracker.registry().update(&tracked).await?;
        }

        Ok(result)
    }

    /// Subscribe to events from a session
    pub async fn subscribe_events(
        &self,
        session_id: &str,
    ) -> Result<broadcast::Receiver<AgentEvent>> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let state = session.read().await;
        Ok(state.subscribe())
    }

    /// Subscribe to events with optional resume from a durable stream (`after_id`).
    ///
    /// A durable tail yields [`ResumedSessionItem::Event`] with the stored id.
    /// A live broadcast yields the same variant with an empty id, and
    /// [`ResumedSessionItem::Resync`] when this subscriber lagged.
    pub(crate) async fn subscribe_events_with_resume(
        &self,
        session_id: &str,
        after_id: Option<String>,
    ) -> Result<
        std::pin::Pin<
            Box<
                dyn futures_util::Stream<Item = anyhow::Result<ResumedSessionItem>>
                    + Send
                    + 'static,
            >,
        >,
    > {
        use caliche::streams::STREAM_START;
        use futures_util::StreamExt;

        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let event_stream = {
            let state = session.read().await;
            state.harness.read().await.event_stream().clone()
        };

        let from_id = after_id.unwrap_or_else(|| STREAM_START.to_string());
        if let Some(stream) =
            event_stream.subscribe_durable(from_id, Some(Duration::from_secs(3600)))
        {
            return Ok(Box::pin(stream.map(|item| {
                item.map(|(id, event)| ResumedSessionItem::Event { id, event })
            })));
        }

        let rx = {
            let state = session.read().await;
            state.subscribe()
        };
        Ok(Box::pin(live_broadcast_resume(rx)))
    }

    /// Rollback a session to a checkpoint
    pub async fn rollback(
        &self,
        session_id: &str,
        checkpoint_id: &str,
    ) -> Result<Vec<std::path::PathBuf>> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let mut state = session.write().await;
        state.touch();
        state.harness.write().await.rollback(checkpoint_id).await
    }

    /// Create a checkpoint in a session
    pub async fn checkpoint(&self, session_id: &str, label: &str) -> Result<String> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let mut state = session.write().await;
        state.touch();
        Ok(state.harness.write().await.checkpoint(label).await)
    }

    /// Undo the last message's file changes in a session (per-message undo).
    pub async fn undo_last(&self, session_id: &str) -> Result<Option<MessageDiff>> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let mut state = session.write().await;
        state.touch();
        drop(state);
        self.runtime.undo_last(session_id).await
    }

    /// Undo a specific message and every newer message after it.
    pub async fn undo_to_message(
        &self,
        session_id: &str,
        message_id: Uuid,
    ) -> Result<Vec<MessageDiff>> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let mut state = session.write().await;
        state.touch();
        drop(state);
        self.runtime.undo_to_message(session_id, message_id).await
    }

    /// Redo the last reverted message's file changes.
    pub async fn redo_last(&self, session_id: &str) -> Result<Option<MessageDiff>> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let mut state = session.write().await;
        state.touch();
        drop(state);
        self.runtime.redo_last(session_id).await
    }

    /// Get session diff list (optionally filter by message ID).
    pub async fn session_diffs(
        &self,
        session_id: &str,
        message_id: Option<Uuid>,
    ) -> Result<caliche::harness::SessionDiffSummary> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let state = session.read().await;
        drop(state);
        self.runtime
            .session_diff_summary(session_id, message_id)
            .await
    }

    /// Clear history in a session
    pub async fn clear_history(&self, session_id: &str) -> Result<()> {
        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let mut state = session.write().await;
        state.touch();
        state.harness.write().await.clear_history();
        Ok(())
    }

    /// Create or replace the public share for a session.
    pub async fn share_session(&self, session_id: &str) -> Result<SharePayload> {
        self.share_session_with_visibility(session_id, None).await
    }

    /// Create or replace a share, optionally overriding default visibility.
    pub async fn share_session_with_visibility(
        &self,
        session_id: &str,
        visibility: Option<ShareVisibility>,
    ) -> Result<SharePayload> {
        self.share_config.ensure_create_allowed()?;

        let session = self
            .get_session(session_id)
            .await
            .ok_or_else(|| ApiAdapterError::SessionNotFound(session_id.to_string()))?;

        let harness = {
            let mut state = session.write().await;
            state.touch();
            state.shared_harness()
        };
        let history = harness.read().await.history().to_vec();

        self.shares.delete_session(session_id);
        // UUIDv7's leading hex encodes the timestamp; take the trailing
        // characters so rapid recreates do not collide within the same ms.
        let mut slug = String::new();
        for _ in 0..8 {
            let candidate = Uuid::now_v7().simple().to_string();
            let candidate = candidate[candidate.len() - 12..].to_string();
            if self.shares.get(&candidate).is_none() {
                slug = candidate;
                break;
            }
        }
        if slug.is_empty() {
            slug = Uuid::now_v7().simple().to_string();
        }
        let redactor = TranscriptRedactor::from_config(&self.share_config)?;
        let transcript = redactor.redact_messages(&history);
        let record = SharePayload {
            slug,
            session_id: session_id.to_string(),
            created_at: chrono::Utc::now(),
            messages: transcript.messages,
            redaction_count: transcript.redaction_count,
            visibility: self.share_config.resolve_visibility(visibility),
        };
        self.shares.put(record)
    }

    /// Public HTML/JSON path for a share slug (`/s/{slug}`).
    pub fn public_share_path(slug: &str) -> String {
        format!("/s/{slug}")
    }

    /// Absolute or relative public share URL using `ShareConfig.backend_url` when set.
    pub fn public_share_url(&self, slug: &str) -> String {
        let path = Self::public_share_path(slug);
        match self.share_config.backend_url.as_deref() {
            Some(base) if !base.trim().is_empty() => {
                format!("{}{}", base.trim_end_matches('/'), path)
            }
            _ => path,
        }
    }

    /// Resolve a public share slug to its stored redacted transcript.
    pub fn get_share(&self, slug: &str) -> Option<SharePayload> {
        self.shares.get(slug)
    }

    /// Remove all share records for a session.
    pub fn delete_session_share(&self, session_id: &str) -> bool {
        self.shares.delete_session(session_id)
    }

    /// Remove a share record by slug.
    pub fn delete_share_slug(&self, slug: &str) -> bool {
        self.shares.delete_slug(slug)
    }

    /// Delete a session.
    ///
    /// Also signals turn cancel so an in-flight turn unwinds. Idle time-to-idle
    /// eviction is a separate cache policy and is not this teardown.
    pub async fn delete_session(&self, session_id: &str) -> bool {
        self.signal_turn_cancel(session_id);
        let removed = self.sessions.remove(session_id).await.is_some();
        self.turn_cancels.remove(session_id);
        let _ = self.runtime.remove(session_id);
        if removed {
            self.delete_session_share(session_id);
        }
        if removed && let Err(err) = self.gateway_bindings.remove_session(session_id) {
            tracing::debug!("Failed to remove gateway bindings for {session_id}: {err}");
        }
        if removed
            && let Some(tracker) = &self.tracker
            && let Ok(id) = Uuid::parse_str(session_id)
            && let Err(err) = tracker.delete(id).await
        {
            tracing::debug!("Failed to delete tracked session {session_id}: {err}");
        }
        removed
    }

    /// Get session info
    pub async fn get_session_info(&self, session_id: &str) -> Option<SessionInfo> {
        if let Some(session) = self.get_session(session_id).await {
            let state = session.read().await;

            let mut name_slug: Option<String> = None;
            if let Some(tracker) = &self.tracker
                && let Ok(id) = Uuid::parse_str(session_id)
            {
                match tracker.get(id).await {
                    Ok(Some(tracked)) => {
                        name_slug = tracked.name_slug;
                    }
                    Ok(None) => {}
                    Err(err) => {
                        tracing::debug!("Failed to read tracked session {session_id}: {err}")
                    }
                }

                if name_slug.is_none()
                    && let Some(client) = &self.slug_client
                {
                    let slug_generator = LlmSlugGenerator {
                        provider: client.provider().as_ref(),
                    };
                    let history = state.harness.read().await.history().to_vec();
                    match tracker.ensure_slug(id, &history, &slug_generator).await {
                        Ok(slug) => name_slug = Some(slug),
                        Err(err) => tracing::debug!(
                            "Failed to generate slug for session {session_id}: {err}"
                        ),
                    }
                }
            }

            let harness = state.harness.read().await;
            return Some(SessionInfo {
                session_id: session_id.to_string(),
                name_slug,
                title: state.title.clone(),
                created_at: state.created_at,
                last_activity: state.last_activity,
                turn_count: harness.turn_count(),
                total_tokens: harness.total_usage().total_tokens.map(|t| t as u64),
            });
        }

        // Fallback to persisted tracking info even if the in-memory session isn't active.
        let tracker = self.tracker.as_ref()?;
        let id = Uuid::parse_str(session_id).ok()?;
        let tracked = tracker.get(id).await.ok()??;

        Some(SessionInfo {
            session_id: session_id.to_string(),
            name_slug: tracked.name_slug,
            title: None,
            created_at: tracked.created_at,
            last_activity: tracked.last_accessed,
            turn_count: 0,
            total_tokens: None,
        })
    }

    /// Session ids currently held in the in-memory TTL cache.
    ///
    /// Process-wide. Prefer [`Self::list_process_sessions`] for the HTTP collection.
    pub fn list_sessions(&self) -> Vec<String> {
        self.sessions
            .iter()
            .map(|(id, _)| id.as_ref().clone())
            .collect()
    }
}
