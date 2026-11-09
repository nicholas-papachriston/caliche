//! Persistence traits and platform-specific implementations for UI state.
//!
//! The UI keeps two long-lived state bags on the client: the list of
//! [`SavedConversation`]s and the user's [`ConfigState`]. Both need to survive
//! reloads; how they survive depends on where the UI is running:
//!
//! - **WASM (browser)**: backed by `localStorage` via [`web_sys`].
//! - **Native (desktop shell)**: backed by a JSON file inside the platform's
//!   config dir (e.g. `~/Library/Application Support/caliche_web/`).
//!
//! Call sites just depend on [`HistoryStore`] / [`ConfigStore`] traits so the
//! same `app.rs` wiring works in either target without `#[cfg]` ladders.

use anyhow::Result;

use super::{ConfigState, SavedConversation};

#[cfg(target_arch = "wasm32")]
const HISTORY_KEY: &str = "caliche_web.history.v1";
#[cfg(target_arch = "wasm32")]
const CONFIG_KEY: &str = "caliche_web.config.v1";

/// Persistence surface for the conversation history tab.
pub trait HistoryStore: Send + Sync + 'static {
    fn load(&self) -> Result<Vec<SavedConversation>>;
    fn save(&self, entries: &[SavedConversation]) -> Result<()>;
}

/// Persistence surface for the operator's [`ConfigState`].
pub trait ConfigStore: Send + Sync + 'static {
    fn load(&self) -> Result<Option<ConfigState>>;
    fn save(&self, config: &ConfigState) -> Result<()>;
}

/// Build the default [`HistoryStore`] for the current target.
pub fn default_history_store() -> Box<dyn HistoryStore> {
    #[cfg(target_arch = "wasm32")]
    {
        Box::new(wasm::WebStorageHistoryStore)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Box::new(native::FileHistoryStore::default())
    }
}

/// Build the default [`ConfigStore`] for the current target.
pub fn default_config_store() -> Box<dyn ConfigStore> {
    #[cfg(target_arch = "wasm32")]
    {
        Box::new(wasm::WebStorageConfigStore)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Box::new(native::FileConfigStore::default())
    }
}

// ── wasm backends ────────────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use super::{CONFIG_KEY, ConfigStore, HISTORY_KEY, HistoryStore};
    use super::{ConfigState, SavedConversation};
    use anyhow::{Result, anyhow};

    /// Browser `localStorage`-backed history store.
    pub struct WebStorageHistoryStore;

    impl HistoryStore for WebStorageHistoryStore {
        fn load(&self) -> Result<Vec<SavedConversation>> {
            let storage = match local_storage() {
                Some(s) => s,
                None => return Ok(Vec::new()),
            };
            match storage
                .get_item(HISTORY_KEY)
                .map_err(|e| anyhow!("localStorage get_item failed: {:?}", e))?
            {
                Some(raw) if !raw.is_empty() => serde_json::from_str(&raw)
                    .map_err(|e| anyhow!("failed to decode saved history: {e}")),
                _ => Ok(Vec::new()),
            }
        }

        fn save(&self, entries: &[SavedConversation]) -> Result<()> {
            let storage = match local_storage() {
                Some(s) => s,
                None => return Ok(()),
            };
            let raw = serde_json::to_string(entries)?;
            storage
                .set_item(HISTORY_KEY, &raw)
                .map_err(|e| anyhow!("localStorage set_item failed: {:?}", e))?;
            Ok(())
        }
    }

    /// Browser `localStorage`-backed config store.
    pub struct WebStorageConfigStore;

    impl ConfigStore for WebStorageConfigStore {
        fn load(&self) -> Result<Option<ConfigState>> {
            let storage = match local_storage() {
                Some(s) => s,
                None => return Ok(None),
            };
            match storage
                .get_item(CONFIG_KEY)
                .map_err(|e| anyhow!("localStorage get_item failed: {:?}", e))?
            {
                Some(raw) if !raw.is_empty() => Ok(Some(
                    serde_json::from_str::<ConfigState>(&raw)
                        .map_err(|e| anyhow!("failed to decode saved config: {e}"))?,
                )),
                _ => Ok(None),
            }
        }

        fn save(&self, config: &ConfigState) -> Result<()> {
            let storage = match local_storage() {
                Some(s) => s,
                None => return Ok(()),
            };
            let raw = serde_json::to_string(config)?;
            storage
                .set_item(CONFIG_KEY, &raw)
                .map_err(|e| anyhow!("localStorage set_item failed: {:?}", e))?;
            Ok(())
        }
    }

    fn local_storage() -> Option<web_sys::Storage> {
        web_sys::window().and_then(|w| w.local_storage().ok().flatten())
    }
}

// ── native backends ──────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use super::{ConfigState, ConfigStore, HistoryStore, SavedConversation};
    use anyhow::{Context, Result};
    use std::path::{Path, PathBuf};

    /// Filesystem-backed history store keyed by JSON under the user's config dir.
    pub struct FileHistoryStore {
        path: PathBuf,
    }

    impl Default for FileHistoryStore {
        fn default() -> Self {
            Self {
                path: default_state_path("history.json"),
            }
        }
    }

    impl FileHistoryStore {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            allow(
                dead_code,
                reason = "the non-wasm builds do not call it; lib.rs and main.rs both compile this module, so expect cannot hold in both"
            )
        )]
        pub fn new(path: impl Into<PathBuf>) -> Self {
            Self { path: path.into() }
        }
    }

    impl HistoryStore for FileHistoryStore {
        fn load(&self) -> Result<Vec<SavedConversation>> {
            read_json_vec::<SavedConversation>(&self.path)
        }

        fn save(&self, entries: &[SavedConversation]) -> Result<()> {
            write_json(&self.path, entries)
        }
    }

    /// Filesystem-backed config store.
    pub struct FileConfigStore {
        path: PathBuf,
    }

    impl Default for FileConfigStore {
        fn default() -> Self {
            Self {
                path: default_state_path("config.json"),
            }
        }
    }

    impl FileConfigStore {
        #[cfg_attr(
            not(target_arch = "wasm32"),
            allow(
                dead_code,
                reason = "the non-wasm builds do not call it; lib.rs and main.rs both compile this module, so expect cannot hold in both"
            )
        )]
        pub fn new(path: impl Into<PathBuf>) -> Self {
            Self { path: path.into() }
        }
    }

    impl ConfigStore for FileConfigStore {
        fn load(&self) -> Result<Option<ConfigState>> {
            if !self.path.exists() {
                return Ok(None);
            }
            let raw = std::fs::read_to_string(&self.path)
                .with_context(|| format!("reading {}", self.path.display()))?;
            if raw.trim().is_empty() {
                return Ok(None);
            }
            Ok(Some(serde_json::from_str(&raw).with_context(|| {
                format!("decoding config from {}", self.path.display())
            })?))
        }

        fn save(&self, config: &ConfigState) -> Result<()> {
            write_json(&self.path, config)
        }
    }

    fn default_state_path(filename: &str) -> PathBuf {
        let base = dirs::config_dir()
            .unwrap_or_else(|| std::env::temp_dir())
            .join("caliche_web");
        base.join(filename)
    }

    fn read_json_vec<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        if raw.trim().is_empty() {
            return Ok(Vec::new());
        }
        serde_json::from_str(&raw).with_context(|| format!("decoding vec from {}", path.display()))
    }

    fn write_json<T: serde::Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("mkdir {}", parent.display()))?;
        }
        let tmp = path.with_extension("json.tmp");
        let raw = serde_json::to_string_pretty(value)?;
        std::fs::write(&tmp, raw).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::state::{ConfigState, SavedConversation};

        fn tmp_path(name: &str) -> PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "caliche_web_test_{}_{}",
                std::process::id(),
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&dir).expect("mkdir tmp");
            dir.join(name)
        }

        #[test]
        fn history_store_roundtrip() {
            let path = tmp_path("history.json");
            let store = FileHistoryStore::new(path.clone());

            let loaded = store.load().expect("empty load");
            assert!(loaded.is_empty(), "empty file should yield empty list");

            let entries = vec![SavedConversation {
                id: "conv-1".into(),
                title: "Saved".into(),
                saved_at: chrono::Utc::now(),
                session_id: Some("sess-1".into()),
                messages: Vec::new(),
                total_tokens: 42,
            }];
            store.save(&entries).expect("save");

            let reloaded = store.load().expect("load");
            assert_eq!(reloaded.len(), 1);
            assert_eq!(reloaded[0].id, "conv-1");
            assert_eq!(reloaded[0].title, "Saved");

            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }

        #[test]
        fn config_store_roundtrip() {
            let path = tmp_path("config.json");
            let store = FileConfigStore::new(path.clone());

            assert!(store.load().expect("missing ok").is_none());

            let mut cfg = ConfigState::default();
            cfg.api_base_url = "http://testing.local:4242".into();
            store.save(&cfg).expect("save");

            let loaded = store.load().expect("load").expect("some");
            assert_eq!(loaded.api_base_url, "http://testing.local:4242");

            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }

        #[test]
        fn config_store_ignores_empty_file() {
            let path = tmp_path("config.json");
            std::fs::write(&path, "").expect("touch");
            let store = FileConfigStore::new(path.clone());
            assert!(store.load().expect("empty ok").is_none());
            let _ = std::fs::remove_dir_all(path.parent().unwrap());
        }
    }
}
