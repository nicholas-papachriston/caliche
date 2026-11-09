//! API Adapter connecting HTTP handlers to the Agent Harness.
//!
//! This module provides session management and SSE event streaming
//! for the API layer.

mod bindings;
mod broadcast_resume;
mod cancel;
mod core;
mod factory;
mod model;
mod resume;
mod session;

pub use bindings::ApiSessionBindingStore;
pub(crate) use broadcast_resume::ResumedSessionItem;
use cancel::TurnCancel;
pub use factory::HarnessFactory;
pub use session::{SessionId, SessionInfo, SessionOwner, SessionState, StreamingTurnResult};

use anyhow::Result;
use async_trait::async_trait;
use caliche::agent::session::{ShareConfig, ShareStore};
use caliche::gateway::{
    GatewayAdapterRegistry, GatewayRuntime, MemoryDeadLetterStore, SessionFactory,
};
use caliche::harness::{AgentRuntime, HarnessConfig};
use caliche::llm::{LlmClient, Provider, ProviderType};
use caliche::messages::{CompletionRequest, Message};
use caliche::persistence::{SessionTracker, SlugGenerator};
use dashmap::DashMap;
use moka::future::Cache;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
pub enum ApiAdapterError {
    SessionNotFound(String),
    /// In-flight turn observed HTTP cancel/stop. The session is removed; a later turn is 404.
    TurnCancelled(String),
}

impl std::fmt::Display for ApiAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionNotFound(session_id) => write!(f, "Session not found: {session_id}"),
            Self::TurnCancelled(session_id) => write!(f, "Session turn cancelled: {session_id}"),
        }
    }
}

impl std::error::Error for ApiAdapterError {}

/// Parse provider and model strings used for gateway session slug defaults (testable without env).
fn parse_slug_provider_model(
    provider_raw: &str,
    model: &str,
) -> anyhow::Result<(ProviderType, String)> {
    let provider = provider_raw
        .trim()
        .parse::<ProviderType>()
        .map_err(|_| anyhow::anyhow!("invalid CALICHE_PROVIDER value: {}", provider_raw))?;
    Ok((provider, model.to_string()))
}

fn slug_provider_model_from_env() -> Result<(ProviderType, String)> {
    let provider_raw = std::env::var("CALICHE_PROVIDER")
        .map_err(|_| anyhow::anyhow!("CALICHE_PROVIDER is not set"))?;
    let model =
        std::env::var("CALICHE_MODEL").map_err(|_| anyhow::anyhow!("CALICHE_MODEL is not set"))?;
    parse_slug_provider_model(&provider_raw, &model)
}

fn build_slug_prompt(messages: &[Message]) -> String {
    let mut lines: Vec<String> = Vec::new();

    for msg in messages.iter().take(6) {
        let role = format!("{:?}", msg.role).to_lowercase();
        let text = msg.content.as_text().unwrap_or("");
        if text.trim().is_empty() {
            continue;
        }
        let snippet = truncate(text.trim(), 280);
        lines.push(format!("{role}: {snippet}"));
    }

    if lines.is_empty() {
        return "No messages available. Create a generic slug.".to_string();
    }

    format!(
        "Create a slug that describes what this session is about.\n\n{}",
        lines.join("\n")
    )
}

fn truncate(value: &str, max_len: usize) -> String {
    if value.len() <= max_len {
        return value.to_string();
    }
    let mut out = caliche_contracts::text::truncate_utf8(value, max_len).to_string();
    out.push_str("...");
    out
}

fn slugify(input: &str) -> String {
    let s = input.trim().to_lowercase();

    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;

    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            prev_dash = false;
            continue;
        }

        if ch == '-' || ch.is_ascii_whitespace() || ch == '_' {
            if !prev_dash && !out.is_empty() {
                out.push('-');
                prev_dash = true;
            }
        }
    }

    while out.ends_with('-') {
        out.pop();
    }

    if out.len() > 64 {
        out.truncate(64);
    }

    out.trim_matches('-').to_string()
}

struct LlmSlugGenerator<'a> {
    provider: &'a dyn Provider,
}

#[async_trait]
impl SlugGenerator for LlmSlugGenerator<'_> {
    async fn generate_slug(&self, messages: &[Message]) -> Result<String> {
        let prompt = build_slug_prompt(messages);
        let req_messages = vec![
            Message::system(
                "You create short, descriptive session slugs.\n\
Return ONLY a kebab-case slug (lowercase ASCII, digits, hyphens).\n\
Max 6 words.\n\
No punctuation besides hyphens.\n\
No surrounding quotes.",
            ),
            Message::user(prompt),
        ];

        let mut request = CompletionRequest::new(Arc::new(req_messages));
        request.max_tokens = Some(32);
        request.temperature = Some(0.2);

        let response = self.provider.complete(request).await?;
        let raw = response.content.as_text();
        let slug = slugify(raw);
        anyhow::ensure!(!slug.is_empty(), "LLM returned an empty slug");
        Ok(slug)
    }
}

/// API Adapter for managing sessions and executing agent tasks
pub struct ApiAdapter {
    /// Resolved startup security policy (session IDOR, webhook flags).
    pub(super) security: Arc<crate::security::SecurityPolicy>,
    /// Harness factory
    pub(super) factory: HarnessFactory,
    /// Canonical multi-session harness map (SDK facade).
    pub(super) runtime: Arc<AgentRuntime>,
    /// Cancel handles for live sessions. Kept beside the session lock so an
    /// in-flight turn (which holds that lock) can still be signalled.
    ///
    /// Process-wide: any caller who can reach a session id can cancel it when
    /// session auth is off.
    pub(super) turn_cancels: Arc<DashMap<SessionId, TurnCancel>>,
    /// Active sessions cache with TTL (metadata + shared harness handle)
    pub(super) sessions: Cache<SessionId, Arc<tokio::sync::RwLock<SessionState>>>,
    /// Session TTL
    pub(super) session_ttl: Duration,
    /// Optional persistent session tracker
    pub(super) tracker: Option<SessionTracker>,
    /// Optional LLM client for lazy slug generation
    pub(super) slug_client: Option<LlmClient>,
    /// Session bindings for multi-channel gateway routing.
    pub(super) gateway_bindings: Arc<ApiSessionBindingStore>,
    /// Adapter registry for channel integrations.
    pub(super) gateway_adapters: Arc<GatewayAdapterRegistry>,
    /// Dead-letter store for failed gateway deliveries.
    pub(super) gateway_dlq: Arc<MemoryDeadLetterStore>,
    /// Runtime controls for fairness, backpressure, and dedupe.
    pub(super) gateway_runtime: Arc<GatewayRuntime>,
    /// Public redacted transcript shares keyed by slug.
    pub(super) shares: Arc<dyn ShareStore>,
    /// Share redaction / backend configuration.
    pub(super) share_config: ShareConfig,
}

async fn build_tracker_from_env() -> Result<Option<SessionTracker>> {
    let backend = std::env::var("SESSION_STORE_BACKEND")
        .ok()
        .unwrap_or_else(|| "redis".to_string())
        .to_lowercase();
    let url = std::env::var("SESSION_STORE_URL").ok();
    let Some(url) = url else {
        return Ok(None);
    };
    let prefix = std::env::var("SESSION_STORE_PREFIX").ok();

    match backend.as_str() {
        #[cfg(feature = "redis")]
        "redis" => Ok(Some(SessionTracker::with_redis(&url, prefix)?)),
        #[cfg(not(feature = "redis"))]
        "redis" => {
            let _ = prefix;
            anyhow::bail!(
                "SESSION_STORE_BACKEND=redis needs a server built with the `redis` feature"
            )
        }
        "sqlite" => Ok(Some(SessionTracker::with_sqlite(&url).await?)),
        #[cfg(feature = "postgres")]
        "postgres" | "postgresql" => Ok(Some(SessionTracker::with_postgres(&url).await?)),
        #[cfg(not(feature = "postgres"))]
        "postgres" | "postgresql" => anyhow::bail!(
            "SESSION_STORE_BACKEND={backend} needs a server built with the `postgres` feature"
        ),
        other => anyhow::bail!("unknown SESSION_STORE_BACKEND: {other}"),
    }
}

impl Default for ApiAdapter {
    fn default() -> Self {
        Self::new(HarnessConfig::default())
    }
}

#[async_trait]
impl SessionFactory for ApiAdapter {
    async fn create_session(
        &self,
        _tenant_id: Option<&str>,
        _system_prompt_hint: Option<&str>,
    ) -> Result<String> {
        Box::pin(ApiAdapter::create_session(self, None, None, None)).await
    }
}

impl std::fmt::Debug for ApiAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiAdapter")
            .field("session_ttl", &self.session_ttl)
            .finish()
    }
}

#[cfg(test)]
mod slug_parse_tests {
    use super::parse_slug_provider_model;
    use caliche::llm::ProviderType;

    #[test]
    fn trims_provider_whitespace() {
        let parsed = parse_slug_provider_model("  openai  ", "gpt-4o");
        let (p, m) = parsed.unwrap_or_else(|e| panic!("parse should succeed: {e}"));
        assert_eq!(p, ProviderType::OpenAI);
        assert_eq!(m, "gpt-4o");
    }

    #[test]
    fn invalid_provider_errors() {
        let Err(err) = parse_slug_provider_model("not-a-real-provider", "m") else {
            panic!("expected invalid provider to error");
        };
        let s = format!("{err:#}");
        assert!(s.contains("invalid CALICHE_PROVIDER"));
    }
}

#[cfg(test)]
mod runtime_registration_tests {
    use super::*;
    use caliche::harness::HarnessConfig;
    use std::time::Duration;

    #[tokio::test]
    #[serial_test::serial]
    async fn create_and_delete_session_keeps_runtime_in_sync() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-runtime");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        assert_eq!(adapter.runtime().session_count(), 0);

        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session should be created");
        assert!(adapter.runtime().contains(&session_id));
        assert_eq!(adapter.runtime().session_count(), 1);
        assert!(adapter.get_session(&session_id).await.is_some());

        assert!(adapter.delete_session(&session_id).await);
        assert!(!adapter.runtime().contains(&session_id));
        assert_eq!(adapter.runtime().session_count(), 0);

        match previous {
            Some(value) => unsafe {
                std::env::set_var("OPENROUTER_API_KEY", value);
            },
            None => unsafe {
                std::env::remove_var("OPENROUTER_API_KEY");
            },
        }
    }
}

#[cfg(test)]
mod share_tests {
    use super::*;
    use caliche::harness::HarnessConfig;
    use std::time::Duration;

    #[tokio::test]
    #[serial_test::serial]
    async fn share_get_returns_redacted_transcript_payload() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-share");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session should be created");

        let record = adapter
            .share_session(&session_id)
            .await
            .expect("share should succeed");
        assert_eq!(record.session_id, session_id);
        assert_eq!(record.slug.len(), 12);

        let resolved = adapter
            .get_share(&record.slug)
            .expect("slug should resolve");
        assert_eq!(resolved.session_id, session_id);
        assert_eq!(resolved.slug, record.slug);
        // Resolve returns a copied redacted transcript payload, not metadata-only.
        assert_eq!(resolved.messages.len(), record.messages.len());
        assert_eq!(resolved.redaction_count, record.redaction_count);

        assert!(adapter.delete_session_share(&session_id));
        assert!(adapter.get_share(&record.slug).is_none());

        match previous {
            Some(value) => unsafe {
                std::env::set_var("OPENROUTER_API_KEY", value);
            },
            None => unsafe {
                std::env::remove_var("OPENROUTER_API_KEY");
            },
        }
    }

    #[tokio::test]
    async fn share_unknown_session_errors() {
        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let err = adapter
            .share_session("missing-session")
            .await
            .expect_err("missing session should fail");
        let message = format!("{err:#}");
        assert!(message.contains("not found") || message.contains("Session"));
    }

    #[tokio::test]
    async fn share_disabled_policy_rejects_create() {
        use caliche::agent::session::{ShareConfig, SharePolicy};

        let mut share_config = ShareConfig::default();
        share_config.policy = SharePolicy::Disabled;
        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None)
                .with_share_config(share_config);
        let err = adapter
            .share_session("any-session")
            .await
            .expect_err("disabled policy should fail");
        assert!(err.to_string().contains("sharing is disabled"));
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn list_sessions_includes_active_cache_entries() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-list-sessions");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        assert!(adapter.list_sessions().is_empty());

        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session should be created");
        let listed = adapter.list_sessions();
        assert!(
            listed.iter().any(|id| id == &session_id),
            "expected {session_id} in {listed:?}"
        );

        match previous {
            Some(value) => unsafe {
                std::env::set_var("OPENROUTER_API_KEY", value);
            },
            None => unsafe {
                std::env::remove_var("OPENROUTER_API_KEY");
            },
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn share_session_replaces_previous_slug() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-share-replace");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session should be created");

        let first = adapter
            .share_session(&session_id)
            .await
            .expect("first share");
        let second = adapter
            .share_session(&session_id)
            .await
            .expect("second share");
        assert_ne!(first.slug, second.slug);
        assert!(adapter.get_share(&first.slug).is_none());
        assert!(adapter.get_share(&second.slug).is_some());

        match previous {
            Some(value) => unsafe {
                std::env::set_var("OPENROUTER_API_KEY", value);
            },
            None => unsafe {
                std::env::remove_var("OPENROUTER_API_KEY");
            },
        }
    }

    #[tokio::test]
    async fn delete_session_share_returns_false_when_absent() {
        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        assert!(!adapter.delete_session_share("no-such-session"));
    }

    #[tokio::test]
    async fn get_share_unknown_slug_is_none() {
        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        assert!(adapter.get_share("missing-slug").is_none());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn delete_session_removes_share_records() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-share-delete");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session should be created");
        let record = adapter
            .share_session(&session_id)
            .await
            .expect("share should succeed");
        assert!(
            record.slug.chars().all(|c| c.is_ascii_hexdigit()) && record.slug.len() == 12,
            "slug should be 12 lowercase/upper hex chars, got {}",
            record.slug
        );

        assert!(adapter.delete_session(&session_id).await);
        assert!(adapter.get_share(&record.slug).is_none());

        match previous {
            Some(value) => unsafe {
                std::env::set_var("OPENROUTER_API_KEY", value);
            },
            None => unsafe {
                std::env::remove_var("OPENROUTER_API_KEY");
            },
        }
    }
}

#[cfg(test)]
mod process_session_tests {
    use super::{ApiAdapter, ApiAdapterError, SessionOwner};
    use caliche::harness::HarnessConfig;
    use std::time::Duration;

    fn restore_openrouter_key(previous: Option<String>) {
        match previous {
            Some(value) => unsafe {
                std::env::set_var("OPENROUTER_API_KEY", value);
            },
            None => unsafe {
                std::env::remove_var("OPENROUTER_API_KEY");
            },
        }
    }

    #[tokio::test]
    async fn list_process_sessions_is_empty_array_source() {
        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        assert!(adapter.list_process_sessions().await.is_empty());
    }

    #[tokio::test]
    #[serial_test::serial]
    #[expect(
        unsafe_code,
        reason = "tests set OPENROUTER_API_KEY for harness construction"
    )]
    async fn list_process_sessions_includes_every_live_owner() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-process-list");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let bot_a = adapter
            .create_session(
                None,
                Some(SessionOwner {
                    user_id: Some("bot-a".to_string()),
                    tenant_id: None,
                }),
                None,
            )
            .await
            .expect("bot-a session");
        let bot_b = adapter
            .create_session(
                None,
                Some(SessionOwner {
                    user_id: Some("bot-b".to_string()),
                    tenant_id: None,
                }),
                Some("bot b".to_string()),
            )
            .await
            .expect("bot-b session");

        let listed = adapter.list_process_sessions().await;
        let ids: Vec<&str> = listed.iter().map(|info| info.session_id.as_str()).collect();
        assert!(
            ids.contains(&bot_a.as_str()),
            "process list dropped {bot_a}"
        );
        assert!(
            ids.contains(&bot_b.as_str()),
            "process list dropped {bot_b}"
        );
        assert_eq!(listed.len(), 2);
        let titled = listed
            .iter()
            .find(|info| info.session_id == bot_b)
            .expect("bot-b row");
        assert_eq!(titled.title.as_deref(), Some("bot b"));
        let untitled = listed
            .iter()
            .find(|info| info.session_id == bot_a)
            .expect("bot-a row");
        assert!(untitled.title.is_none());

        restore_openrouter_key(previous);
    }

    #[tokio::test]
    #[serial_test::serial]
    #[expect(
        unsafe_code,
        reason = "tests set OPENROUTER_API_KEY for harness construction"
    )]
    async fn cancel_while_turn_lock_is_held_then_turn_is_missing() {
        let previous = std::env::var("OPENROUTER_API_KEY").ok();
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", "test-key-for-process-cancel");
        }

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session should be created");
        let handle = adapter
            .turn_cancel_handle(&session_id)
            .expect("cancel handle");
        let token = handle.token.clone();
        let session = adapter
            .get_session(&session_id)
            .await
            .expect("live session");
        let holder = tokio::spawn(async move {
            let state = session.write().await;
            let _harness = state.harness.write().await;
            token.cancelled().await;
        });
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(30)).await;

        let cancelled =
            tokio::time::timeout(Duration::from_secs(2), adapter.cancel_session(&session_id))
                .await
                .expect("cancel must not block on the in-flight session lock");
        assert!(cancelled);
        tokio::time::timeout(Duration::from_secs(2), holder)
            .await
            .expect("holder should observe cancel")
            .expect("holder task");

        let err = adapter
            .execute_turn(&session_id, "next")
            .await
            .expect_err("dead session");
        assert!(
            matches!(
                err.downcast_ref::<ApiAdapterError>(),
                Some(ApiAdapterError::SessionNotFound(_))
            ),
            "expected missing session, got {err:#}"
        );
        assert!(
            adapter
                .list_process_sessions()
                .await
                .iter()
                .all(|info| info.session_id != session_id)
        );
        assert!(!adapter.cancel_session(&session_id).await);

        restore_openrouter_key(previous);
    }
}
