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
            encode_session_id(session_id, Case::Escaped)
        ))
    }

    /// Earlier names `session_id` may be stored under, newest first, each
    /// different from [`Self::path_for`]. A file under one of them is used
    /// only if it holds that exact ID.
    fn legacy_paths_for(&self, session_id: &str) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::with_capacity(2);
        // Before uppercase letters were escaped: "Main" and "main" shared
        // one file on a case-insensitive file system.
        let case_kept = self.dir.join(format!(
            "{}{SESSION_FILE_SUFFIX}",
            encode_session_id(session_id, Case::Kept)
        ));
        if case_kept != self.path_for(session_id) {
            paths.push(case_kept);
        }
        paths.push(self.legacy_path_for(session_id));
        paths
    }

    /// Where the old scheme stored `session_id`: `/`, `\\` and `.` became
    /// `_`, so two IDs could share a file, and on Windows a `:` named an
    /// alternate data stream. Its stem has no `.`, so it never equals a
    /// [`Self::path_for`] name.
    fn legacy_path_for(&self, session_id: &str) -> std::path::PathBuf {
        let sanitized = session_id.replace(['/', '\\', '.'], "_");
        self.dir.join(format!("{sanitized}.cls1"))
    }

    /// The session in the file at [`Self::path_for`] `session_id`, and
    /// whether it is that ID's. On a case-insensitive file system that name
    /// is also the name another ID had before uppercase letters were escaped
    /// (`main.id.cls1` is the same file as `Main.id.cls1`, which held
    /// "Main"), so the stored ID decides.
    async fn read_at_own_name(&self, session_id: &str) -> Result<Option<(SessionSnapshot, bool)>> {
        let path = self.path_for(session_id);
        if !path.is_file() {
            return Ok(None);
        }
        let snapshot = decode_session_snapshot_v1(&tokio::fs::read(&path).await?)?;
        let own = snapshot.session_id == session_id;
        Ok(Some((snapshot, own)))
    }

    /// Before `session_id` is saved or deleted, move a file at its name that
    /// holds another ID to that ID's own name, so it is not overwritten or
    /// removed. If that ID already has a file there, refuse and leave both.
    /// A file there that cannot be read or decoded may be another ID's, so
    /// that refuses too.
    async fn move_other_session_away(&self, session_id: &str) -> Result<()> {
        let path = self.path_for(session_id);
        let other = match self.read_at_own_name(session_id).await.with_context(|| {
            format!(
                "{} cannot be read as a session, so it may hold another ID; leaving it",
                path.display()
            )
        })? {
            Some((snapshot, false)) => snapshot.session_id,
            // Absent or this ID's own.
            None | Some((_, true)) => return Ok(()),
        };
        let target = self.path_for(&other);
        let dir = self.dir.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            // A hard link fails if the target exists, so a file another save
            // puts there meanwhile is never replaced.
            match std::fs::hard_link(&path, &target) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => anyhow::bail!(
                    "{} holds session {other:?}, which also has {}; leaving both",
                    path.display(),
                    target.display()
                ),
                Err(err) => {
                    return Err(err).with_context(|| {
                        format!("link {} to {}", path.display(), target.display())
                    });
                }
            }
            std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
            sync_dir(&dir)?;
            Ok(())
        })
        .await
        .context("session move task failed")?
    }

    /// Every earlier-name file that holds `session_id`, newest scheme first.
    async fn read_legacy(&self, session_id: &str) -> Vec<(std::path::PathBuf, SessionSnapshot)> {
        let mut found = Vec::new();
        for path in self.legacy_paths_for(session_id) {
            if !path.is_file() {
                continue;
            }
            let Ok(data) = tokio::fs::read(&path).await else {
                continue;
            };
            if let Ok(snapshot) = decode_session_snapshot_v1(&data)
                && snapshot.session_id == session_id
            {
                found.push((path, snapshot));
            }
        }
        found
    }
}

/// Write `bytes` to `path` in `dir` so a crash leaves either the old file or
/// the whole new one: write a temp file in `dir`, sync it, rename it over
/// `path`, then sync `dir` (on Unix only; Windows has no directory sync) so
/// the rename is durable.
fn write_durably(dir: &std::path::Path, path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    write_durably_with(dir, &temp_path(dir, nanos), path, bytes, || Ok(()))
}

/// A temp file name in `dir` that no other save uses. The pid and the time
/// alone can repeat (two saves in one process within the clock's
/// resolution), so a per-process counter is added.
fn temp_path(dir: &std::path::Path, nanos: u128) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    dir.join(format!(
        ".session-{}-{nanos}-{}.tmp",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Make a rename in `dir` durable by syncing the directory. Only Unix
/// syncs a directory this way; elsewhere this does nothing.
fn sync_dir(dir: &std::path::Path) -> std::io::Result<()> {
    #[cfg(unix)]
    std::fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// [`write_durably`] through the temp file `temp`, calling `before_rename`
/// once it is synced. Tests inject a crash there.
fn write_durably_with(
    dir: &std::path::Path,
    temp: &std::path::Path,
    path: &std::path::Path,
    bytes: &[u8],
    before_rename: impl FnOnce() -> Result<()>,
) -> Result<()> {
    use std::io::Write as _;
    // Only a temp file this call created is removed on failure: one that
    // already exists is another save's, and removing it would fail that
    // save's rename.
    let mut created = false;
    let written = (|| -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temp)
            .with_context(|| format!("create {}", temp.display()))?;
        created = true;
        file.write_all(bytes)?;
        file.sync_all()?;
        before_rename()?;
        std::fs::rename(temp, path)
            .with_context(|| format!("rename {} to {}", temp.display(), path.display()))?;
        sync_dir(dir)?;
        Ok(())
    })();
    if written.is_err() && created {
        let _ = std::fs::remove_file(temp);
    }
    written
}

/// End of every session file name. The `.id` part keeps these names apart
/// from old-scheme names, whose stem has no `.`.
const SESSION_FILE_SUFFIX: &str = ".id.cls1";

/// Windows device names, reserved in a file name with any extension.
const WINDOWS_DEVICE_NAMES: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];

/// Whether [`encode_session_id`] keeps uppercase ASCII letters.
#[derive(Clone, Copy)]
enum Case {
    /// The current scheme: uppercase letters become `%XX`.
    Escaped,
    /// The scheme before it, read only to move old files.
    Kept,
}

/// A file-name stem for `session_id` that no other ID shares and every
/// platform accepts, even with a case-insensitive file system: lowercase
/// ASCII letters, digits, `-` and `_` stay, every other byte (uppercase
/// letters, `%` and `.` included) becomes `%XX`, and the first character of
/// a Windows device name (`con`, `com1`) is encoded too.
fn encode_session_id(session_id: &str, case: Case) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(session_id.len());
    for byte in session_id.bytes() {
        let kept_letter = match case {
            Case::Escaped => byte.is_ascii_lowercase(),
            Case::Kept => byte.is_ascii_alphabetic(),
        };
        if kept_letter || byte.is_ascii_digit() || byte == b'-' || byte == b'_' {
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
        self.move_other_session_away(&snapshot.session_id).await?;
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
        for (legacy, _) in self.read_legacy(&snapshot.session_id).await {
            tokio::fs::remove_file(legacy).await?;
        }
        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<SessionSnapshot>> {
        if let Some((snapshot, true)) = self.read_at_own_name(session_id).await? {
            return Ok(Some(snapshot));
        }
        Ok(self
            .read_legacy(session_id)
            .await
            .into_iter()
            .next()
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
        self.move_other_session_away(session_id).await?;
        let path = self.path_for(session_id);
        if path.is_file() {
            tokio::fs::remove_file(path).await?;
        }
        for (legacy, _) in self.read_legacy(session_id).await {
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
            ("a%2Fb", "a%252%46b.id.cls1"),
            ("*?\"<>|", "%2A%3F%22%3C%3E%7C.id.cls1"),
            ("Main", "%4Dain.id.cls1"),
            ("CON", "%43%4F%4E.id.cls1"),
            ("nul", "%6Eul.id.cls1"),
            ("COM1", "%43%4F%4D1.id.cls1"),
            ("lpt9", "%6Cpt9.id.cls1"),
            ("com10", "com10.id.cls1"),
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

    /// IDs that differ only in letter case keep their own files on a
    /// case-insensitive file system (APFS, NTFS), and deleting one keeps the
    /// others.
    #[tokio::test]
    async fn ids_that_differ_only_in_case_keep_separate_files() -> Result<()> {
        assert_each_has_its_own_file(&["Ab", "ab", "Main", "main", "MAIN"]).await?;

        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        store.save_session(&tagged_snapshot("Main")?).await?;
        store.save_session(&tagged_snapshot("main")?).await?;
        store.delete_session("Main").await?;

        assert!(store.load_session("Main").await?.is_none());
        let kept = store
            .load_session("main")
            .await?
            .ok_or_else(|| anyhow::anyhow!("deleting Main removed main"))?;
        assert_eq!(kept.conversation_history[0].as_text(), Some("main"));
        Ok(())
    }

    /// A session saved before uppercase letters were escaped loads, moves to
    /// its new name on save, and is listed once.
    #[tokio::test]
    async fn a_session_under_its_case_kept_name_is_read_and_moved() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let case_kept = dir.path().join("Main.id.cls1");
        std::fs::write(
            &case_kept,
            encode_session_snapshot_v1(&tagged_snapshot("Main")?)?,
        )?;

        let loaded = store
            .load_session("Main")
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected the case-kept session"))?;
        assert_eq!(loaded.conversation_history[0].as_text(), Some("Main"));

        store.save_session(&tagged_snapshot("Main")?).await?;
        let mut names: Vec<String> = std::fs::read_dir(dir.path())?
            .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<std::io::Result<_>>()?;
        names.sort();
        assert_eq!(names, vec!["%4Dain.id.cls1".to_string()]);
        Ok(())
    }

    /// Write `id`'s session, tagged with `text`, under the file name `name`,
    /// as an earlier version stored it.
    fn plant(dir: &std::path::Path, name: &str, id: &str, text: &str) -> Result<()> {
        let mut snapshot = tagged_snapshot(id)?;
        snapshot.conversation_history = vec![Message::user(text)];
        std::fs::write(dir.join(name), encode_session_snapshot_v1(&snapshot)?)?;
        Ok(())
    }

    /// The text of `id`'s loaded session, or `None` if it has none.
    async fn loaded_text(store: &BinaryFileSessionStore, id: &str) -> Result<Option<String>> {
        Ok(store.load_session(id).await?.map(|snapshot| {
            snapshot.conversation_history[0]
                .as_text()
                .unwrap_or("")
                .to_string()
        }))
    }

    /// Whether `dir` treats names that differ only in case as one file.
    fn case_insensitive(dir: &std::path::Path) -> Result<bool> {
        std::fs::write(dir.join("case-probe"), b"")?;
        let insensitive = dir.join("CASE-PROBE").exists();
        std::fs::remove_file(dir.join("case-probe"))?;
        Ok(insensitive)
    }

    /// "Main" saved before uppercase letters were escaped is at
    /// `Main.id.cls1`, which on APFS or NTFS is also `main.id.cls1`, the new
    /// name of "main". Loading "main" does not return "Main".
    #[tokio::test]
    async fn a_name_holding_another_id_is_not_loaded_as_this_id() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        plant(dir.path(), "Main.id.cls1", "Main", "Main")?;

        assert_eq!(loaded_text(&store, "main").await?, None);
        assert_eq!(loaded_text(&store, "Main").await?, Some("Main".to_string()));
        Ok(())
    }

    /// A file at "main"'s name that cannot be read may hold another ID
    /// ("Main", stored before uppercase letters were escaped). Saving or
    /// deleting "main" refuses and leaves it as it was.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_unreadable_file_at_this_name_is_not_overwritten_or_deleted() -> Result<()> {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        plant(dir.path(), "main.id.cls1", "Main", "Main")?;
        let path = dir.path().join("main.id.cls1");
        let before = std::fs::read(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o200))?;

        let saved = store.save_session(&tagged_snapshot("main")?).await;
        let deleted = store.delete_session("main").await;

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        assert!(saved.is_err(), "save over an unreadable file must refuse");
        assert!(deleted.is_err(), "delete of an unreadable file must refuse");
        assert_eq!(std::fs::read(&path)?, before);
        Ok(())
    }

    /// The same holds for a file at that name that does not decode.
    #[tokio::test]
    async fn an_undecodable_file_at_this_name_is_not_overwritten_or_deleted() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let path = dir.path().join("main.id.cls1");
        std::fs::write(&path, b"not a session")?;

        let saved = store.save_session(&tagged_snapshot("main")?).await;
        let deleted = store.delete_session("main").await;

        assert!(saved.is_err(), "save over an undecodable file must refuse");
        assert!(
            deleted.is_err(),
            "delete of an undecodable file must refuse"
        );
        assert_eq!(std::fs::read(&path)?, b"not a session".to_vec());
        Ok(())
    }

    /// Saving "main" over a file that holds an earlier "Main" moves "Main"
    /// to its own name first, so both are kept.
    #[tokio::test]
    async fn saving_an_id_keeps_another_id_stored_at_its_name() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        plant(dir.path(), "Main.id.cls1", "Main", "Main")?;

        store.save_session(&tagged_snapshot("main")?).await?;

        assert_eq!(loaded_text(&store, "Main").await?, Some("Main".to_string()));
        assert_eq!(loaded_text(&store, "main").await?, Some("main".to_string()));
        let mut ids: Vec<String> = Vec::new();
        for name in std::fs::read_dir(dir.path())? {
            let data = std::fs::read(name?.path())?;
            ids.push(decode_session_snapshot_v1(&data)?.session_id);
        }
        ids.sort();
        assert_eq!(ids, vec!["Main".to_string(), "main".to_string()]);
        Ok(())
    }

    /// Deleting "main", never saved, keeps an earlier "Main" whose file is
    /// at the same name.
    #[tokio::test]
    async fn deleting_an_id_keeps_another_id_stored_at_its_name() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        plant(dir.path(), "Main.id.cls1", "Main", "Main")?;

        store.delete_session("main").await?;

        assert_eq!(loaded_text(&store, "Main").await?, Some("Main".to_string()));
        assert_eq!(loaded_text(&store, "main").await?, None);
        Ok(())
    }

    /// The same holds when the IDs differ in the case of an escape's hex
    /// digits: "%4D" was stored at `%254D.id.cls1`, and "%4d" is now
    /// `%254d.id.cls1`.
    #[tokio::test]
    async fn ids_that_differ_in_hex_digit_case_keep_their_sessions() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        plant(dir.path(), "%254D.id.cls1", "%4D", "upper")?;

        assert_eq!(loaded_text(&store, "%4d").await?, None);
        store.save_session(&tagged_snapshot("%4d")?).await?;

        assert_eq!(loaded_text(&store, "%4D").await?, Some("upper".to_string()));
        assert_eq!(loaded_text(&store, "%4d").await?, Some("%4d".to_string()));
        Ok(())
    }

    /// If the other ID also has a file at its own name, saving refuses and
    /// both files stay as they were.
    #[tokio::test]
    async fn saving_refuses_when_the_other_id_has_both_names() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        plant(dir.path(), "Main.id.cls1", "Main", "old Main")?;
        plant(dir.path(), "%4Dain.id.cls1", "Main", "new Main")?;

        let saved = store.save_session(&tagged_snapshot("main")?).await;

        if case_insensitive(dir.path())? {
            let err = saved.expect_err("the save must refuse");
            assert_eq!(
                err.to_string(),
                format!(
                    "{} holds session \"Main\", which also has {}; leaving both",
                    store.path_for("main").display(),
                    store.path_for("Main").display()
                )
            );
            assert_eq!(
                decode_session_snapshot_v1(&std::fs::read(store.path_for("main"))?)?
                    .conversation_history[0]
                    .as_text(),
                Some("old Main")
            );
        } else {
            // Without case folding the names are different files.
            saved?;
            assert_eq!(loaded_text(&store, "main").await?, Some("main".to_string()));
        }
        assert_eq!(
            loaded_text(&store, "Main").await?,
            Some("new Main".to_string())
        );
        Ok(())
    }

    /// Two temp names taken at the same time differ, so two saves in one
    /// clock tick do not share a temp file.
    #[test]
    fn temp_names_taken_at_the_same_time_differ() {
        let dir = std::path::Path::new("/sessions");
        let first = temp_path(dir, 7);
        let second = temp_path(dir, 7);

        assert_ne!(first, second);
        let prefix = format!(".session-{}-7-", std::process::id());
        for temp in [&first, &second] {
            let name = temp.file_name().map(|n| n.to_string_lossy().into_owned());
            assert!(name.is_some_and(|n| n.starts_with(&prefix)), "{temp:?}");
        }
    }

    /// A save whose temp name is already taken fails and leaves that file,
    /// which belongs to another save, as it was.
    #[test]
    fn a_save_whose_temp_name_is_taken_leaves_that_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("s.id.cls1");
        let temp = dir.path().join(".session-taken.tmp");
        std::fs::write(&temp, b"another save's bytes")?;

        let err = write_durably_with(dir.path(), &temp, &path, b"new snapshot", || Ok(()))
            .expect_err("the temp name is taken");

        assert_eq!(err.to_string(), format!("create {}", temp.display()));
        assert_eq!(std::fs::read(&temp)?, b"another save's bytes");
        assert!(!path.exists());
        Ok(())
    }

    /// Deleting a session that exists only under its old name removes that
    /// file, so it does not come back on the next load.
    #[tokio::test]
    async fn deleting_a_session_removes_its_old_name_copy() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let legacy = dir.path().join("a_b.cls1");
        std::fs::write(
            &legacy,
            encode_session_snapshot_v1(&tagged_snapshot("a/b")?)?,
        )?;

        store.delete_session("a/b").await?;

        assert!(!legacy.exists(), "the old-name copy is still there");
        assert!(store.load_session("a/b").await?.is_none());
        Ok(())
    }

    /// A crash after the temp file is written and before the rename leaves
    /// the old file whole and no temp file behind.
    #[test]
    fn a_crash_before_the_rename_keeps_the_old_file() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("s.id.cls1");
        std::fs::write(&path, b"old snapshot")?;

        let temp = temp_path(dir.path(), 0);
        let err = write_durably_with(dir.path(), &temp, &path, b"new snapshot", || {
            anyhow::bail!("injected crash before rename")
        })
        .expect_err("injected crash");

        assert_eq!(err.to_string(), "injected crash before rename");
        assert_eq!(std::fs::read(&path)?, b"old snapshot");
        let names: Vec<String> = std::fs::read_dir(dir.path())?
            .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<std::io::Result<_>>()?;
        assert_eq!(names, vec!["s.id.cls1".to_string()]);
        Ok(())
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
