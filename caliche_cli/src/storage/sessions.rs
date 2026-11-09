//! SQLite-backed session storage using caliche core library

use anyhow::{Context, Result};
use caliche::persistence::{
    ConversationRecord, SessionRegistry, SessionTracker, SqliteConversationStore,
    SqliteSessionRegistry, backend::ConversationStoreBackend,
};
use chrono::{Duration, Utc};
use serde_json::json;
use sqlx::sqlite::SqlitePool;
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use crate::session::{AggregateStats, CliSession, SessionSummary};

/// Session store backed by SQLite using caliche core
pub struct SessionStore {
    store: Arc<SqliteConversationStore>,
    tracker: SessionTracker,
}

impl SessionStore {
    /// Create a new session store (async)
    pub async fn new() -> Result<Self> {
        let db_path = Self::default_db_path()?;

        crate::storage::schema::ensure_schema(&db_path).await?;

        let store = SqliteConversationStore::from_path(&db_path).await?;
        let registry = SqliteSessionRegistry::from_path(&db_path).await?;
        let tracker = SessionTracker::new(Arc::new(registry));

        Ok(Self {
            store: Arc::new(store),
            tracker,
        })
    }

    /// Create a session store from an existing pool (reuses connection; caller must ensure schema exists).
    pub async fn from_pool(pool: &SqlitePool) -> Result<Self> {
        let store = SqliteConversationStore::from_pool(pool.clone());
        let registry = SqliteSessionRegistry::from_pool(pool.clone());
        registry.init().await?;
        let tracker = SessionTracker::new(Arc::new(registry));
        Ok(Self {
            store: Arc::new(store),
            tracker,
        })
    }

    /// Get the default database path
    fn default_db_path() -> Result<PathBuf> {
        let home = crate::storage::cli_home_dir()?;
        Ok(home.join(".caliche").join("caliche.db"))
    }

    /// Convert CliSession to ConversationRecord
    fn session_to_record(session: &CliSession, is_autosave: bool) -> ConversationRecord {
        let metadata = json!({
            "name": session.name,
            "working_directory": session.working_directory.to_string_lossy(),
            "context_files": session.context_files.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
            "permissions": session.permissions_granted,
            "total_tokens": session.total_tokens,
            "provider": session.provider,
            "model": session.model,
            "is_autosave": is_autosave,
        });

        let id = if is_autosave {
            format!("{}::autosave", session.id)
        } else {
            session.id.clone()
        };

        ConversationRecord {
            id,
            agent_id: "cli".to_string(),
            messages: session.conversation_history.clone(),
            metadata,
            created_at: session.created_at,
            updated_at: session.updated_at,
            expires_at: None,
        }
    }

    /// Convert ConversationRecord to CliSession
    fn record_to_session(record: ConversationRecord) -> Result<CliSession> {
        let metadata = &record.metadata;
        let name = metadata
            .get("name")
            .and_then(|v| v.as_str())
            .map(String::from);
        let working_directory = metadata
            .get("working_directory")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let context_files: Vec<PathBuf> = metadata
            .get("context_files")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(PathBuf::from))
                    .collect()
            })
            .unwrap_or_default();
        let permissions_granted: Vec<String> = metadata
            .get("permissions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let total_tokens = metadata
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let provider = metadata
            .get("provider")
            .and_then(|v| v.as_str())
            .map(String::from);
        let model = metadata
            .get("model")
            .and_then(|v| v.as_str())
            .map(String::from);

        let id = record
            .id
            .strip_suffix("::autosave")
            .map_or_else(|| record.id.clone(), str::to_owned);

        Ok(CliSession {
            id,
            name,
            working_directory,
            conversation_history: record.messages,
            context_files,
            permissions_granted,
            created_at: record.created_at,
            updated_at: record.updated_at,
            total_tokens,
            provider,
            model,
            file_content_cache: std::collections::HashMap::new(),
            files_created: Vec::new(),
        })
    }

    /// Create a new session (sync wrapper)
    pub fn create_sync(&self, id: String, working_directory: PathBuf) -> Result<CliSession> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.create(id, working_directory))
    }

    /// Create a new session (async)
    pub async fn create(&self, id: String, working_directory: PathBuf) -> Result<CliSession> {
        let now = Utc::now();
        let session = CliSession {
            id,
            name: None,
            working_directory,
            conversation_history: Vec::new(),
            context_files: Vec::new(),
            permissions_granted: Vec::new(),
            created_at: now,
            updated_at: now,
            total_tokens: 0,
            provider: None,
            model: None,
            file_content_cache: std::collections::HashMap::new(),
            files_created: Vec::new(),
        };

        self.save(&session).await?;
        Ok(session)
    }

    /// Save session to database (sync wrapper)
    pub fn save_sync(&self, session: &CliSession) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.save(session))
    }

    /// Save session to database (async)
    pub async fn save(&self, session: &CliSession) -> Result<()> {
        let record = Self::session_to_record(session, false);
        self.store.save(&record).await?;

        // Also save as autosave for crash recovery
        let autosave_record = Self::session_to_record(session, true);
        self.store.save(&autosave_record).await?;

        // Update last session pointer
        self.set_last_session_id(&session.id)?;

        // Update tracking record (best-effort, but fail if id is invalid).
        let session_uuid =
            Uuid::parse_str(&session.id).context("Session id must be a valid UUIDv7")?;
        let now = Utc::now();
        let mut tracked = match self.tracker.get(session_uuid).await? {
            Some(existing) => existing,
            None => {
                let mut created = self
                    .tracker
                    .create_with_id(session_uuid, "cli", None)
                    .await?;
                created.conversation_id = Some(session.id.clone());
                created
            }
        };

        tracked.message_count = session.conversation_history.len() as u32;
        tracked.updated_at = now;
        tracked.last_accessed = now;
        tracked.conversation_id = Some(session.id.clone());

        self.tracker.registry().update(&tracked).await?;

        Ok(())
    }

    /// Autosave session (sync wrapper)
    pub fn autosave(&self, session: &CliSession) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.autosave_async(session))
    }

    /// Autosave session (for crash recovery, async)
    pub async fn autosave_async(&self, session: &CliSession) -> Result<()> {
        let record = Self::session_to_record(session, true);
        self.store.save(&record).await?;
        Ok(())
    }

    /// Load session from database (sync wrapper)
    pub fn load(&self, session_id: &str) -> Result<CliSession> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.load_async(session_id))
    }

    /// Load session from database (async)
    pub async fn load_async(&self, session_id: &str) -> Result<CliSession> {
        let record = self
            .store
            .fetch(session_id)
            .await?
            .context("Session not found")?;

        // Check if this is an autosave
        let is_autosave = record
            .metadata
            .get("is_autosave")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if is_autosave {
            anyhow::bail!("Session is an autosave, use load_autosave instead");
        }

        Self::record_to_session(record)
    }

    /// Try to load a session, distinguishing "not found" from other errors.
    ///
    /// Returns `Ok(None)` when the session id is unknown, `Ok(Some(session))`
    /// on success, and `Err(_)` for storage/decode failures so callers do not
    /// mistake a transient DB error for a missing session.
    pub async fn try_load_async(&self, session_id: &str) -> Result<Option<CliSession>> {
        let Some(record) = self.store.fetch(session_id).await? else {
            return Ok(None);
        };

        let is_autosave = record
            .metadata
            .get("is_autosave")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if is_autosave {
            anyhow::bail!("Session is an autosave, use load_autosave instead");
        }

        Self::record_to_session(record).map(Some)
    }

    /// Check if a session exists (sync wrapper)
    pub fn exists(&self, session_id: &str) -> Result<bool> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.exists_async(session_id))
    }

    /// Check if a session exists (async)
    pub async fn exists_async(&self, session_id: &str) -> Result<bool> {
        Ok(self.store.fetch(session_id).await?.is_some())
    }

    /// List all session IDs (sync wrapper)
    pub fn list(&self) -> Result<Vec<String>> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.list_async())
    }

    /// List all session IDs (async)
    pub async fn list_async(&self) -> Result<Vec<String>> {
        let conversations = self.store.list_by_agent("cli", 1000, 0).await?;
        let mut session_ids = Vec::new();
        for conv in conversations {
            let is_autosave = conv
                .metadata
                .get("is_autosave")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !is_autosave {
                session_ids.push(conv.id);
            }
        }
        Ok(session_ids)
    }

    /// List all sessions with detailed summaries (sync wrapper)
    pub fn list_detailed(&self) -> Result<Vec<SessionSummary>> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.list_detailed_async())
    }

    /// List all sessions with detailed summaries (async)
    pub async fn list_detailed_async(&self) -> Result<Vec<SessionSummary>> {
        let conversations = self.store.list_by_agent("cli", 1000, 0).await?;
        let mut summaries = Vec::new();

        for conv in conversations {
            let is_autosave = conv
                .metadata
                .get("is_autosave")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_autosave {
                continue;
            }

            let metadata = &conv.metadata;
            let mut name = metadata
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from);
            let working_directory = metadata
                .get("working_directory")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            let total_tokens = metadata
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let provider = metadata
                .get("provider")
                .and_then(|v| v.as_str())
                .map(String::from);
            let model = metadata
                .get("model")
                .and_then(|v| v.as_str())
                .map(String::from);

            // If no explicit name is set, try to use the tracked session slug.
            if name.is_none() {
                if let Ok(session_uuid) = Uuid::parse_str(&conv.id) {
                    if let Some(tracked) = self.tracker.get(session_uuid).await? {
                        name = tracked.name_slug;
                    }
                }
            }

            summaries.push(SessionSummary {
                id: conv.id,
                name,
                working_directory,
                message_count: conv.messages.len(),
                created_at: conv.created_at,
                updated_at: conv.updated_at,
                total_tokens,
                provider,
                model,
            });
        }

        summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(summaries)
    }

    /// Aggregate statistics across all sessions (sync wrapper)
    pub fn aggregate_stats(
        &self,
        provider_filter: Option<&str>,
        model_filter: Option<&str>,
        days_filter: Option<u32>,
    ) -> Result<AggregateStats> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.aggregate_stats_async(provider_filter, model_filter, days_filter))
    }

    /// Aggregate statistics across all sessions (async)
    pub async fn aggregate_stats_async(
        &self,
        provider_filter: Option<&str>,
        model_filter: Option<&str>,
        days_filter: Option<u32>,
    ) -> Result<AggregateStats> {
        let summaries = self.list_detailed_async().await?;
        let mut stats = AggregateStats::default();

        let cutoff_date = days_filter.map(|days| Utc::now() - Duration::days(days as i64));

        for summary in summaries {
            if let Some(cutoff) = cutoff_date
                && summary.updated_at < cutoff
            {
                continue;
            }

            if let Some(filter) = provider_filter {
                match &summary.provider {
                    Some(p) if p.to_lowercase().contains(&filter.to_lowercase()) => {}
                    _ => continue,
                }
            }

            if let Some(filter) = model_filter {
                match &summary.model {
                    Some(m) if m.to_lowercase().contains(&filter.to_lowercase()) => {}
                    _ => continue,
                }
            }

            stats.total_sessions += 1;
            stats.total_tokens += summary.total_tokens;
            stats.total_messages += summary.message_count;

            if let Some(ref provider) = summary.provider {
                let entry = stats
                    .provider_usage
                    .entry(provider.clone())
                    .or_insert((0, 0));
                entry.0 += 1;
                entry.1 += summary.total_tokens;
            }

            if let Some(ref model) = summary.model {
                let entry = stats.model_usage.entry(model.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += summary.total_tokens;
            }

            match stats.earliest_session {
                Some(earliest) if summary.created_at < earliest => {
                    stats.earliest_session = Some(summary.created_at);
                }
                None => {
                    stats.earliest_session = Some(summary.created_at);
                }
                _ => {}
            }

            match stats.latest_session {
                Some(latest) if summary.updated_at > latest => {
                    stats.latest_session = Some(summary.updated_at);
                }
                None => {
                    stats.latest_session = Some(summary.updated_at);
                }
                _ => {}
            }
        }

        Ok(stats)
    }

    /// Delete a session (sync wrapper)
    pub fn delete(&self, session_id: &str) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.delete_async(session_id))
    }

    /// Delete a session (async)
    pub async fn delete_async(&self, session_id: &str) -> Result<()> {
        // Delete both regular and autosave (suffixed key)
        self.store.delete(session_id).await?;
        let autosave_key = format!("{session_id}::autosave");
        if let Err(e) = self.store.delete(&autosave_key).await {
            tracing::warn!("Failed to remove autosave entry for {}: {}", session_id, e);
        }
        Ok(())
    }

    /// Rename a session (sync wrapper)
    pub fn rename(&self, session_id: &str, new_name: &str) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.rename_async(session_id, new_name))
    }

    /// Rename a session (async)
    pub async fn rename_async(&self, session_id: &str, new_name: &str) -> Result<()> {
        let mut session = self.load_async(session_id).await?;
        session.name = Some(new_name.to_string());
        self.save(&session).await?;
        Ok(())
    }

    /// Set the last session ID
    fn set_last_session_id(&self, session_id: &str) -> Result<()> {
        let home = crate::storage::cli_home_dir()?;
        let session_dir = home.join(".caliche").join("sessions");
        std::fs::create_dir_all(&session_dir)?;

        let path = session_dir.join("last_session.json");
        let data = serde_json::json!({ "last_session_id": session_id });
        std::fs::write(&path, serde_json::to_string(&data)?)?;

        Ok(())
    }

    /// Get the last session ID
    pub fn get_last_session_id(&self) -> Result<Option<String>> {
        let home = crate::storage::cli_home_dir()?;
        let path = home
            .join(".caliche")
            .join("sessions")
            .join("last_session.json");

        if !path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;

        Ok(data
            .get("last_session_id")
            .and_then(|v| v.as_str())
            .map(String::from))
    }

    /// Load the most recent session (sync wrapper)
    pub fn load_last(&self) -> Result<Option<CliSession>> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.load_last_async())
    }

    /// Load the most recent session (async)
    pub async fn load_last_async(&self) -> Result<Option<CliSession>> {
        match self.get_last_session_id()? {
            Some(id) => {
                if self.exists_async(&id).await? {
                    Ok(Some(self.load_async(&id).await?))
                } else {
                    let summaries = self.list_detailed_async().await?;
                    if let Some(first) = summaries.first() {
                        Ok(Some(self.load_async(&first.id).await?))
                    } else {
                        Ok(None)
                    }
                }
            }
            None => {
                let summaries = self.list_detailed_async().await?;
                if let Some(first) = summaries.first() {
                    Ok(Some(self.load_async(&first.id).await?))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Load an autosaved session by ID (sync wrapper)
    pub fn load_autosave(&self, session_id: &str) -> Result<Option<CliSession>> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.load_autosave_async(session_id))
    }

    /// Load an autosaved session by ID (async)
    async fn load_autosave_async(&self, session_id: &str) -> Result<Option<CliSession>> {
        let autosave_key = format!("{session_id}::autosave");
        let record = self.store.fetch(&autosave_key).await?;

        if let Some(record) = record {
            let is_autosave = record
                .metadata
                .get("is_autosave")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_autosave {
                Ok(Some(Self::record_to_session(record)?))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Get available autosave (sync wrapper)
    pub fn get_available_autosave(&self) -> Result<Option<CliSession>> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.get_available_autosave_async())
    }

    /// Get available autosave (async)
    pub async fn get_available_autosave_async(&self) -> Result<Option<CliSession>> {
        let conversations = self.store.list_by_agent("cli", 1000, 0).await?;

        // Find the most recent autosave
        let mut best_autosave: Option<ConversationRecord> = None;
        for conv in conversations {
            let is_autosave = conv
                .metadata
                .get("is_autosave")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            if is_autosave {
                match &best_autosave {
                    Some(best) if conv.updated_at > best.updated_at => {
                        best_autosave = Some(conv);
                    }
                    None => {
                        best_autosave = Some(conv);
                    }
                    _ => {}
                }
            }
        }

        if let Some(record) = best_autosave {
            Ok(Some(Self::record_to_session(record)?))
        } else {
            Ok(None)
        }
    }

    /// Clear autosave for a session (sync wrapper)
    pub fn clear_autosave(&self, session_id: &str) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.clear_autosave_async(session_id))
    }

    /// Clear autosave for a session (async)
    pub async fn clear_autosave_async(&self, session_id: &str) -> Result<()> {
        let autosave_key = format!("{session_id}::autosave");
        if let Err(e) = self.store.delete(&autosave_key).await {
            tracing::warn!("Failed to clear autosave for {}: {}", session_id, e);
        }
        Ok(())
    }

    /// Promote autosave to regular save (sync wrapper)
    pub fn promote_autosave(&self, session: &CliSession) -> Result<()> {
        let handle = tokio::runtime::Handle::try_current().context("No tokio runtime available")?;
        handle.block_on(self.promote_autosave_async(session))
    }

    /// Promote autosave to regular save (async)
    pub async fn promote_autosave_async(&self, session: &CliSession) -> Result<()> {
        self.save(session).await?;
        Ok(())
    }

    /// Get a reference to the underlying store
    pub fn store(&self) -> &SqliteConversationStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::SessionStore;
    use caliche::messages::Message;
    use caliche::persistence::SqliteConversationStore;
    use std::path::PathBuf;

    #[tokio::test]
    async fn autosave_round_trip_keeps_reasoning_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("caliche.db");
        let conversation = SqliteConversationStore::from_path(&db_path)
            .await
            .expect("conversation store");
        let pool = conversation.pool().clone();
        let store = SessionStore::from_pool(&pool).await.expect("session store");

        let mut session = crate::session::CliSession::new(
            "01936b8e-7c3a-7000-8000-0000000000aa".to_string(),
            PathBuf::from("/tmp/work"),
        );
        session
            .conversation_history
            .push(Message::assistant_with_reasoning(
                "the answer",
                "hidden chain of thought",
            ));

        store.autosave_async(&session).await.expect("autosave");
        let loaded = store
            .load_autosave_async(&session.id)
            .await
            .expect("load")
            .expect("autosave row");

        assert_eq!(loaded.conversation_history.len(), 1);
        assert_eq!(loaded.conversation_history[0].extract_text(), "the answer");
        assert_eq!(
            loaded.conversation_history[0].reasoning_content(),
            Some("hidden chain of thought")
        );
    }
}
