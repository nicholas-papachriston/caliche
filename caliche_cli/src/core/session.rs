//! Session management for interactive modes.
//!
//! Provides unified session persistence using SQLite-backed storage.

use crate::core::state::InteractiveState;
use crate::session::{CliSession, SessionSummary};
use crate::storage::sessions::SessionStore;
use anyhow::Result;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Session manager for interactive modes
pub struct SessionManager {
    store: SessionStore,
    autosave_interval: Duration,
    last_autosave: Instant,
}

impl SessionManager {
    /// Create a new session manager (opens its own database connection).
    pub async fn new() -> Result<Self> {
        Ok(Self {
            store: SessionStore::new().await?,
            autosave_interval: Duration::from_secs(30),
            last_autosave: Instant::now(),
        })
    }

    /// Create a session manager from an existing store (reuses connection; no extra DB open).
    pub fn from_store(store: SessionStore) -> Self {
        Self {
            store,
            autosave_interval: Duration::from_secs(30),
            last_autosave: Instant::now(),
        }
    }

    /// Create or load a session
    pub async fn get_or_create_session(&self, session_id: Option<String>) -> Result<CliSession> {
        let session_id = session_id.unwrap_or_else(|| uuid::Uuid::now_v7().to_string());
        match self.store.load_async(&session_id).await {
            Ok(session) => Ok(session),
            Err(_) => {
                let wd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                self.store.create(session_id, wd).await
            }
        }
    }

    /// Save a session
    pub async fn save(&self, state: &InteractiveState) -> Result<()> {
        let session = self.build_session(state);
        self.store.save(&session).await
    }

    /// Autosave if enough time has passed
    pub async fn maybe_autosave(&mut self, state: &InteractiveState) -> Result<()> {
        if !state.autosave_enabled {
            return Ok(());
        }

        if self.last_autosave.elapsed() >= self.autosave_interval {
            let session = self.build_session(state);
            self.store.autosave_async(&session).await?;
            self.last_autosave = Instant::now();
        }

        Ok(())
    }

    /// List all sessions with optional search filter (matches id, name, working_dir, provider, model)
    pub async fn list_sessions(&self, search: Option<&str>) -> Result<Vec<SessionSummary>> {
        let mut summaries = self.store.list_detailed_async().await?;
        if let Some(query) = search {
            let q = query.to_lowercase();
            summaries.retain(|s| {
                s.id.to_lowercase().contains(&q)
                    || s.name
                        .as_ref()
                        .map(|n| n.to_lowercase().contains(&q))
                        .unwrap_or(false)
                    || s.working_directory
                        .to_string_lossy()
                        .to_lowercase()
                        .contains(&q)
                    || s.provider
                        .as_ref()
                        .map(|p| p.to_lowercase().contains(&q))
                        .unwrap_or(false)
                    || s.model
                        .as_ref()
                        .map(|m| m.to_lowercase().contains(&q))
                        .unwrap_or(false)
            });
        }
        Ok(summaries)
    }

    /// Resolve session ID (supports prefix matching)
    pub async fn resolve_session_id(&self, session_id: &str) -> Result<String> {
        if self.store.exists_async(session_id).await? {
            return Ok(session_id.to_string());
        }
        let sessions = self.store.list_async().await?;
        let matches: Vec<_> = sessions
            .iter()
            .filter(|s| s.starts_with(session_id))
            .collect();
        if matches.is_empty() {
            anyhow::bail!("Session '{}' not found", session_id);
        }
        if matches.len() == 1 {
            return Ok(matches[0].clone());
        }
        anyhow::bail!(
            "Multiple sessions match '{}': {}",
            session_id,
            matches
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    /// Load a session and convert to InteractiveState
    pub async fn load(
        &self,
        session_id: &str,
        provider: &str,
        model: &str,
        system_prompt: String,
    ) -> Result<InteractiveState> {
        let session = self.store.load_async(session_id).await?;
        let effective_provider = session.provider.as_deref().unwrap_or(provider);
        let effective_model = session.model.as_deref().unwrap_or(model);
        let mut state = InteractiveState::new(
            session.clone(),
            effective_provider,
            effective_model,
            system_prompt,
        );

        // Restore tracked files from session
        state.messages = session.conversation_history.clone();
        state.tracked_files = session.context_files.clone();
        state.total_tokens = session.total_tokens;
        state.context_tokens = Some(session.total_tokens);

        Ok(state)
    }

    /// Build a CliSession from InteractiveState
    fn build_session(&self, state: &InteractiveState) -> CliSession {
        // Convert messages to conversation history
        let conversation_history = state.messages.clone();

        CliSession {
            id: state.session_id.clone(),
            name: state.session.name.clone(),
            working_directory: state.working_directory.clone(),
            conversation_history,
            context_files: state.tracked_files.clone(),
            permissions_granted: state.session.permissions_granted.clone(),
            created_at: state.session.created_at,
            updated_at: chrono::Utc::now(),
            total_tokens: state.total_tokens,
            provider: Some(state.token_budget.provider.clone()),
            model: Some(state.token_budget.model.clone()),
            file_content_cache: state.session.file_content_cache.clone(),
            files_created: state.session.files_created.clone(),
        }
    }
}
