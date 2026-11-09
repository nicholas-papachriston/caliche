//! Gateway session-binding store backed by in-memory state with optional persistence.

use anyhow::Result;
use async_trait::async_trait;
use caliche::gateway::{BindingKey, SessionBinding, SessionBindingStore};
use caliche::persistence::{SessionQuery, SessionStatus, SessionTracker};
use dashmap::DashMap;
use serde_json::json;
use uuid::Uuid;

/// Gateway binding store backed by memory with durable persistence via SessionTracker metadata.
#[derive(Debug, Default)]
pub struct ApiSessionBindingStore {
    bindings: DashMap<BindingKey, SessionBinding>,
    tracker: tokio::sync::RwLock<Option<SessionTracker>>,
}

impl ApiSessionBindingStore {
    const METADATA_BINDINGS_KEY: &'static str = "gateway_bindings";

    pub fn new(tracker: Option<SessionTracker>) -> Self {
        Self {
            bindings: DashMap::new(),
            tracker: tokio::sync::RwLock::new(tracker),
        }
    }

    /// In-memory snapshot of all bindings (for periodic maintenance).
    pub(crate) fn list_all_bindings(&self) -> Vec<SessionBinding> {
        self.bindings
            .iter()
            .map(|row| row.value().clone())
            .collect()
    }

    pub async fn set_tracker(&self, tracker: Option<SessionTracker>) {
        let mut guard = self.tracker.write().await;
        *guard = tracker;
    }

    pub fn remove_session(&self, session_id: &str) -> Result<()> {
        let keys: Vec<BindingKey> = self
            .bindings
            .iter()
            .filter(|row| row.value().session_id == session_id)
            .map(|row| row.key().clone())
            .collect();
        for key in keys {
            self.bindings.remove(&key);
        }
        Ok(())
    }

    fn key_string(key: &BindingKey) -> Option<String> {
        serde_json::to_string(key).ok()
    }

    fn bindings_from_metadata(metadata: &serde_json::Value) -> Vec<SessionBinding> {
        metadata
            .get(Self::METADATA_BINDINGS_KEY)
            .and_then(serde_json::Value::as_object)
            .map(|obj| {
                obj.values()
                    .filter_map(|v| serde_json::from_value::<SessionBinding>(v.clone()).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    async fn persist_binding_metadata(&self, binding: &SessionBinding) -> Result<()> {
        let tracker = self.tracker.read().await.clone();
        let Some(tracker) = tracker else {
            return Ok(());
        };
        let Ok(id) = Uuid::parse_str(&binding.session_id) else {
            return Ok(());
        };
        let Some(mut tracked) = tracker.get(id).await? else {
            return Ok(());
        };

        let key_str = match Self::key_string(&binding.key) {
            Some(v) => v,
            None => return Ok(()),
        };

        if !tracked.metadata.is_object() {
            tracked.metadata = json!({});
        }
        if let Some(root) = tracked.metadata.as_object_mut() {
            if !root
                .get(Self::METADATA_BINDINGS_KEY)
                .and_then(serde_json::Value::as_object)
                .is_some()
            {
                root.insert(Self::METADATA_BINDINGS_KEY.to_string(), json!({}));
            }
            if let Some(bindings_obj) = root
                .get_mut(Self::METADATA_BINDINGS_KEY)
                .and_then(serde_json::Value::as_object_mut)
            {
                bindings_obj.insert(key_str, serde_json::to_value(binding)?);
            }
        }
        tracked.updated_at = chrono::Utc::now();
        tracked.last_accessed = chrono::Utc::now();
        tracker.registry().update(&tracked).await?;
        Ok(())
    }

    async fn remove_binding_metadata(&self, binding: &SessionBinding) -> Result<()> {
        let tracker = self.tracker.read().await.clone();
        let Some(tracker) = tracker else {
            return Ok(());
        };
        let Ok(id) = Uuid::parse_str(&binding.session_id) else {
            return Ok(());
        };
        let Some(mut tracked) = tracker.get(id).await? else {
            return Ok(());
        };

        let Some(key_str) = Self::key_string(&binding.key) else {
            return Ok(());
        };
        if let Some(bindings_obj) = tracked
            .metadata
            .get_mut(Self::METADATA_BINDINGS_KEY)
            .and_then(serde_json::Value::as_object_mut)
        {
            bindings_obj.remove(&key_str);
            tracked.updated_at = chrono::Utc::now();
            tracked.last_accessed = chrono::Utc::now();
            tracker.registry().update(&tracked).await?;
        }
        Ok(())
    }

    async fn hydrate_from_tracker_for_key(
        &self,
        key: &BindingKey,
    ) -> Result<Option<SessionBinding>> {
        let tracker = self.tracker.read().await.clone();
        let Some(tracker) = tracker else {
            return Ok(None);
        };
        let needle = Self::key_string(key);
        let mut offset = 0;
        let page_size = 200;
        let max_scan = 2_000;

        while offset < max_scan {
            let mut query = SessionQuery::new(page_size, offset);
            query.status = Some(SessionStatus::Active);
            let sessions = tracker.list(&query).await?;
            if sessions.is_empty() {
                break;
            }
            for tracked in sessions {
                for binding in Self::bindings_from_metadata(&tracked.metadata) {
                    self.bindings.insert(binding.key.clone(), binding.clone());
                    if Some(
                        Self::key_string(&binding.key)
                            .unwrap_or_else(|| "__invalid_binding_key__".to_string()),
                    ) == needle
                    {
                        return Ok(Some(binding));
                    }
                }
            }
            offset += page_size;
        }
        Ok(None)
    }
}

#[async_trait]
impl SessionBindingStore for ApiSessionBindingStore {
    async fn get(&self, key: &BindingKey) -> Result<Option<SessionBinding>> {
        if let Some(existing) = self.bindings.get(key) {
            return Ok(Some(existing.clone()));
        }
        self.hydrate_from_tracker_for_key(key).await
    }

    async fn put(&self, binding: SessionBinding) -> Result<()> {
        self.bindings.insert(binding.key.clone(), binding.clone());
        self.persist_binding_metadata(&binding).await
    }

    async fn get_or_insert(&self, candidate: SessionBinding) -> Result<(SessionBinding, bool)> {
        use dashmap::mapref::entry::Entry;

        match self.bindings.entry(candidate.key.clone()) {
            Entry::Occupied(entry) => Ok((entry.get().clone(), false)),
            Entry::Vacant(entry) => {
                self.persist_binding_metadata(&candidate).await?;
                entry.insert(candidate.clone());
                Ok((candidate, true))
            }
        }
    }

    async fn delete(&self, key: &BindingKey) -> Result<()> {
        if let Some((_, existing)) = self.bindings.remove(key) {
            self.remove_binding_metadata(&existing).await?;
        }
        Ok(())
    }

    async fn list_for_session(&self, session_id: &str) -> Result<Vec<SessionBinding>> {
        let mut out: Vec<SessionBinding> = self
            .bindings
            .iter()
            .filter(|row| row.value().session_id == session_id)
            .map(|row| row.value().clone())
            .collect();

        if out.is_empty() {
            let tracker = self.tracker.read().await.clone();
            if let Some(tracker) = tracker
                && let Ok(id) = Uuid::parse_str(session_id)
                && let Some(tracked) = tracker.get(id).await?
            {
                for binding in Self::bindings_from_metadata(&tracked.metadata) {
                    self.bindings.insert(binding.key.clone(), binding.clone());
                    out.push(binding);
                }
            }
        }

        Ok(out)
    }
}
