//! Unified session persistence: snapshot types and store trait.

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::codec::{decode_session_snapshot_v1, encode_session_snapshot_v1};
use crate::agent::memory::MemorySnapshot;
use crate::harness::SessionDiffState;
use crate::messages::Message;

/// Full snapshot of a session for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Session identifier.
    pub session_id: String,
    /// Conversation history.
    pub conversation_history: Vec<Message>,
    /// Memory strategy state (if any).
    pub memory_snapshot: Option<MemorySnapshot>,
    /// Knowledge store state (if any; used by `EnhancedMemory`).
    pub knowledge_snapshot: Option<MemorySnapshot>,
    /// Plan document state (populated during Plan mode turns).
    #[serde(default)]
    pub plan_document: Option<crate::agent::modes::PlanDocument>,
    /// Durable per-message diff state for undo/redo.
    #[serde(default)]
    pub diff_state: SessionDiffState,
    /// Shared state as JSON.
    pub shared_state: Option<serde_json::Value>,
    /// Snapshot metadata.
    pub metadata: SessionMetadata,
}

/// Metadata for a persisted session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// When the session was first created (or first saved).
    pub created_at: DateTime<Utc>,
    /// When this snapshot was saved.
    pub saved_at: DateTime<Utc>,
    /// Human-readable memory strategy type (e.g. "buffer", "enhanced").
    pub memory_type: String,
    /// Total messages in conversation history.
    pub total_messages: usize,
    /// Estimated total tokens.
    pub total_tokens_estimated: usize,
}

/// Store for saving and loading session snapshots.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Save a session snapshot.
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()>;

    /// Load a session snapshot by id.
    async fn load_session(&self, session_id: &str) -> Result<Option<SessionSnapshot>>;

    /// List metadata for all stored sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionMetadata>>;

    /// Delete a session by id.
    async fn delete_session(&self, session_id: &str) -> Result<()>;
}

/// File-based session store using binary rkyv v1 snapshots (`*.cls1`) under a directory.
#[derive(Debug, Clone)]
pub struct BinaryFileSessionStore {
    dir: std::path::PathBuf,
    /// Test hook: fail a save after its new file is in place and before an
    /// old-name copy is removed.
    #[cfg(test)]
    fail_before_legacy_removal: bool,
}

impl BinaryFileSessionStore {
    /// Create a binary session store using the given directory.
    pub fn new(dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            #[cfg(test)]
            fail_before_legacy_removal: false,
        }
    }

    fn path_for(&self, session_id: &str) -> std::path::PathBuf {
        self.dir.join(format!(
            "{}{SESSION_FILE_SUFFIX}",
            encode_session_id(session_id)
        ))
    }

    /// Where the old scheme stored `session_id`: `/`, `\\` and `.` became
    /// `_`, so two IDs could share a file, and on Windows a `:` named an
    /// alternate data stream. Its stem has no `.`, so it never equals a
    /// [`Self::path_for`] name.
    fn legacy_path_for(&self, session_id: &str) -> std::path::PathBuf {
        let sanitized = session_id.replace(['/', '\\', '.'], "_");
        self.dir.join(format!("{sanitized}.cls1"))
    }

    /// The old-scheme file for `session_id`, if it holds that session.
    async fn read_legacy(&self, session_id: &str) -> Option<(std::path::PathBuf, SessionSnapshot)> {
        let path = self.legacy_path_for(session_id);
        if !path.is_file() {
            return None;
        }
        let data = tokio::fs::read(&path).await.ok()?;
        let snapshot = decode_session_snapshot_v1(&data).ok()?;
        (snapshot.session_id == session_id).then_some((path, snapshot))
    }
}

/// Write `bytes` to `path` in `dir` so a crash leaves either the old file or
/// the whole new one: write a temp file in `dir`, sync it, rename it over
/// `path`, then sync `dir` (on Unix) so the rename is durable.
fn write_durably(dir: &std::path::Path, path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    let temp = dir.join(format!(".session-{}-{nanos}.tmp", std::process::id()));
    let written = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .with_context(|| format!("create {}", temp.display()))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        std::fs::rename(&temp, path)
            .with_context(|| format!("rename {} to {}", temp.display(), path.display()))?;
        #[cfg(unix)]
        std::fs::File::open(dir)?.sync_all()?;
        Ok(())
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    written
}

/// End of every session file name. The `.id` part keeps these names apart
/// from old-scheme names, whose stem has no `.`.
const SESSION_FILE_SUFFIX: &str = ".id.cls1";

/// Windows device names, reserved in a file name with any extension.
const WINDOWS_DEVICE_NAMES: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];

/// A file-name stem for `session_id` that no other ID shares and every
/// platform accepts: ASCII letters, digits, `-` and `_` stay, every other
/// byte (`%` and `.` included) becomes `%XX`, and the first character of a
/// Windows device name (`CON`, `COM1`) is encoded too.
fn encode_session_id(session_id: &str) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(session_id.len());
    for byte in session_id.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
            out.push(char::from(byte));
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    if is_windows_device_name(&out) {
        let first = out.remove(0);
        out.insert_str(0, &format!("%{:02X}", u32::from(first)));
    }
    out
}

fn is_windows_device_name(stem: &str) -> bool {
    let upper = stem.to_ascii_uppercase();
    let numbered = (upper.starts_with("COM") || upper.starts_with("LPT"))
        && upper.len() == 4
        && upper.as_bytes().get(3).is_some_and(u8::is_ascii_digit);
    WINDOWS_DEVICE_NAMES.contains(&upper.as_str()) || numbered
}

#[async_trait]
impl SessionStore for BinaryFileSessionStore {
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()> {
        tokio::fs::create_dir_all(&self.dir).await?;
        let path = self.path_for(&snapshot.session_id);
        let bytes = encode_session_snapshot_v1(snapshot)?;
        let dir = self.dir.clone();
        tokio::task::spawn_blocking(move || write_durably(&dir, &path, &bytes))
            .await
            .context("session write task failed")??;
        #[cfg(test)]
        if self.fail_before_legacy_removal {
            anyhow::bail!("injected failure before the old-name copy is removed");
        }
        // The session's own file is now durable; only then drop its
        // old-scheme copy, so it is listed once. A failure before this point
        // leaves the old copy in place.
        if let Some((legacy, _)) = self.read_legacy(&snapshot.session_id).await {
            tokio::fs::remove_file(legacy).await?;
        }
        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<SessionSnapshot>> {
        let path = self.path_for(session_id);
        if path.is_file() {
            let data = tokio::fs::read(&path).await?;
            return decode_session_snapshot_v1(&data).map(Some);
        }
        Ok(self
            .read_legacy(session_id)
            .await
            .map(|(_, snapshot)| snapshot))
    }

    async fn list_sessions(&self) -> Result<Vec<SessionMetadata>> {
        if !self.dir.is_dir() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&self.dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "cls1")
                && let Ok(data) = tokio::fs::read(&path).await
                && let Ok(snap) = decode_session_snapshot_v1(&data)
            {
                out.push(snap.metadata);
            }
        }
        Ok(out)
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let path = self.path_for(session_id);
        if path.is_file() {
            tokio::fs::remove_file(path).await?;
        }
        if let Some((legacy, _)) = self.read_legacy(session_id).await {
            tokio::fs::remove_file(legacy).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::Message;
    use chrono::{DateTime, TimeZone, Utc};

    fn fixed_time(year: i32, month: u32, day: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("invalid fixed test time"))
    }

    fn minimal_snapshot(id: &str) -> Result<SessionSnapshot> {
        Ok(SessionSnapshot {
            session_id: id.to_string(),
            conversation_history: vec![Message::user("ping")],
            memory_snapshot: None,
            knowledge_snapshot: None,
            plan_document: None,
            diff_state: SessionDiffState::default(),
            shared_state: None,
            metadata: SessionMetadata {
                created_at: fixed_time(2024, 1, 1)?,
                saved_at: fixed_time(2024, 1, 2)?,
                memory_type: "none".to_string(),
                total_messages: 1,
                total_tokens_estimated: 1,
            },
        })
    }

    /// A snapshot whose message text is its ID, so a load shows whose file
    /// it read.
    fn tagged_snapshot(id: &str) -> Result<SessionSnapshot> {
        let mut snapshot = minimal_snapshot(id)?;
        snapshot.conversation_history = vec![Message::user(id)];
        Ok(snapshot)
    }

    /// Save every ID, then check each loads its own text and all are listed.
    async fn assert_each_has_its_own_file(ids: &[&str]) -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        for id in ids {
            store.save_session(&tagged_snapshot(id)?).await?;
        }
        for id in ids {
            let loaded = store
                .load_session(id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("{id} was not stored"))?;
            assert_eq!(loaded.session_id, *id);
            assert_eq!(loaded.conversation_history[0].as_text(), Some(*id));
        }
        assert_eq!(store.list_sessions().await?.len(), ids.len());
        Ok(())
    }

    #[test]
    fn session_file_names_encode_every_reserved_character() {
        let store = BinaryFileSessionStore::new("/sessions");
        let name = |id: &str| {
            store
                .path_for(id)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        };
        let cases = [
            (
                "0190b8c2-7d1e-7a00-8f00-5e55105e55a1",
                "0190b8c2-7d1e-7a00-8f00-5e55105e55a1.id.cls1",
            ),
            ("a_b", "a_b.id.cls1"),
            ("a/b", "a%2Fb.id.cls1"),
            ("a\\b", "a%5Cb.id.cls1"),
            ("a.b", "a%2Eb.id.cls1"),
            ("a:b", "a%3Ab.id.cls1"),
            ("a%2Fb", "a%252Fb.id.cls1"),
            ("*?\"<>|", "%2A%3F%22%3C%3E%7C.id.cls1"),
            ("CON", "%43ON.id.cls1"),
            ("nul", "%6Eul.id.cls1"),
            ("COM1", "%43OM1.id.cls1"),
            ("lpt9", "%6Cpt9.id.cls1"),
            ("COM10", "COM10.id.cls1"),
            ("trail.", "trail%2E.id.cls1"),
            ("trail ", "trail%20.id.cls1"),
            ("caf\u{e9}", "caf%C3%A9.id.cls1"),
        ];
        for (id, expected) in cases {
            assert_eq!(name(id).as_deref(), Some(expected), "{id:?}");
        }
    }

    /// IDs the old scheme mapped to one file each save, load and list on
    /// their own.
    #[tokio::test]
    async fn ids_that_differ_only_in_a_reserved_character_keep_separate_files() -> Result<()> {
        assert_each_has_its_own_file(&["a/b", "a_b", "a.b", "a\\b", "a:b", "a%2Fb", "a*b"]).await
    }

    /// Windows device names and trailing dots or spaces save, load and list.
    #[tokio::test]
    async fn device_names_and_trailing_dots_or_spaces_are_stored() -> Result<()> {
        assert_each_has_its_own_file(&["CON", "nul", "COM1", "LPT9", "aux", "trail.", "trail "])
            .await
    }

    /// A session stored under its old name loads under its own ID only, is
    /// moved to its new name on save, and deletes.
    #[tokio::test]
    async fn a_session_under_its_old_name_is_read_moved_and_deleted() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let legacy = dir.path().join("a_b.cls1");
        std::fs::write(
            &legacy,
            encode_session_snapshot_v1(&tagged_snapshot("a/b")?)?,
        )?;

        let loaded = store
            .load_session("a/b")
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected the old-name session"))?;
        assert_eq!(loaded.conversation_history[0].as_text(), Some("a/b"));
        assert!(store.load_session("a_b").await?.is_none());

        store.save_session(&tagged_snapshot("a/b")?).await?;
        assert!(!legacy.exists());
        assert_eq!(store.list_sessions().await?.len(), 1);

        store.delete_session("a/b").await?;
        assert!(store.load_session("a/b").await?.is_none());
        assert_eq!(store.list_sessions().await?.len(), 0);
        Ok(())
    }

    /// A save that fails after the new file is written and before the old
    /// copy is removed leaves the old copy, and the session still loads.
    #[tokio::test]
    async fn a_save_that_fails_before_removing_the_old_copy_keeps_it() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut store = BinaryFileSessionStore::new(dir.path());
        let legacy = dir.path().join("a_b.cls1");
        std::fs::write(
            &legacy,
            encode_session_snapshot_v1(&tagged_snapshot("a/b")?)?,
        )?;
        store.fail_before_legacy_removal = true;

        let err = store
            .save_session(&tagged_snapshot("a/b")?)
            .await
            .expect_err("injected failure");

        assert_eq!(
            err.to_string(),
            "injected failure before the old-name copy is removed"
        );
        assert!(legacy.is_file(), "the old copy was removed");
        let loaded = store
            .load_session("a/b")
            .await?
            .ok_or_else(|| anyhow::anyhow!("the session was lost"))?;
        assert_eq!(loaded.conversation_history[0].as_text(), Some("a/b"));
        let names: Vec<String> = std::fs::read_dir(dir.path())?
            .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<std::io::Result<_>>()?;
        assert!(
            names.iter().all(|name| !name.ends_with(".tmp")),
            "{names:?}"
        );
        Ok(())
    }

    /// A session the old scheme put in a Windows alternate data stream
    /// (`id:legacy.cls1`) still loads and deletes.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_session_in_a_legacy_stream_still_loads_and_deletes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let legacy = dir.path().join("id:legacy.cls1");
        std::fs::write(
            &legacy,
            encode_session_snapshot_v1(&tagged_snapshot("id:legacy")?)?,
        )?;

        let loaded = store
            .load_session("id:legacy")
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected the legacy session"))?;
        assert_eq!(loaded.conversation_history[0].as_text(), Some("id:legacy"));

        store.delete_session("id:legacy").await?;
        assert!(store.load_session("id:legacy").await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn binary_file_session_store_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let snap = minimal_snapshot("id/with:unsafe")?;
        store.save_session(&snap).await?;
        let loaded = store
            .load_session("id/with:unsafe")
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected stored session"))?;
        assert_eq!(loaded.session_id, snap.session_id);
        assert_eq!(loaded.conversation_history.len(), 1);
        assert_eq!(loaded.conversation_history[0].as_text(), Some("ping"));
        assert!(loaded.diff_state.active.is_empty());
        let listed = store.list_sessions().await?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].total_messages, 1);
        store.delete_session("id/with:unsafe").await?;
        assert!(store.load_session("id/with:unsafe").await?.is_none());
        Ok(())
    }
}
