//! Directory-backed in-process backend for [`crate::Dripstone::open`].
//!
//! Vectors live in [`crate::backend::memory::MemoryBackend`]. Text is indexed
//! with the in-memory BM25 index. Both are snapshotted to `store.json` so a
//! later `open` on the same path reloads the corpus.

use std::fs::{File, OpenOptions};
use std::io::Write;
#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard};

use crate::backend::full_text_indexed::{FullTextIndexedBackend, FullTextIndexedBackendConfig};
use crate::backend::memory::{InMemoryConfig, MemoryBackend};
use crate::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::fts::InMemoryFullTextIndex;
use crate::local::{LOCAL_EMBED_DIMENSION, LOCAL_EMBEDDER_NAME};
use crate::search::FusionConfig;
use crate::traits::Backend;

const SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_FILE: &str = "store.json";
const SNAPSHOT_TEMPORARY: &str = "store.json.tmp";
const SNAPSHOT_BACKUP: &str = "store.json.bak";
const LOCK_FILE: &str = "store.lock";
const WRITE_LOCAL_STORE_SNAPSHOT: &str = "write local store snapshot";
const REPLACE_LOCAL_STORE_SNAPSHOT: &str = "replace local store snapshot";
const QUARANTINE_PREFIX: &str = "store.json.corrupt-";
const STEP_WRITE_TMP: &str = "write tmp";
const STEP_SYNC_TMP: &str = "sync tmp";
const STEP_COPY_BAK: &str = "copy bak";
const STEP_SYNC_BAK: &str = "sync bak";
const STEP_RENAME: &str = "rename";
const STEP_SYNC_DIR: &str = "sync dir";

/// Read before the chunks, so a newer format whose chunks this version cannot
/// parse is reported as unsupported instead of quarantined as unreadable.
#[derive(Debug, Deserialize)]
struct SnapshotHeader {
    version: u32,
    embedder: String,
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct SnapshotChunks {
    chunks: Vec<Chunk>,
}

enum SnapshotLoad {
    Missing,
    /// Empty or not JSON. The bytes are kept and the backup may be tried.
    Unreadable,
    /// JSON with a version, embedder, or dimension this process cannot load.
    Unsupported(&'static str),
    Chunks(Vec<Chunk>),
}

/// In-process backend that persists chunks under a directory.
pub struct OpenBackend {
    inner: FullTextIndexedBackend<MemoryBackend>,
    dir: PathBuf,
    /// The directory lock file, behind the write mutex. Each snapshot write
    /// moves an owned guard onto the blocking pool, so neither a cancelled
    /// caller nor dropping this handle releases the directory lock while that
    /// write still runs. A second `open` fails until both have finished.
    write: Arc<Mutex<File>>,
}

impl std::fmt::Debug for OpenBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenBackend")
            .field("dir", &self.dir)
            .finish_non_exhaustive()
    }
}

impl OpenBackend {
    /// Open or create a store directory and load any existing snapshot.
    ///
    /// The directory lock is exclusive. A second `open`, in this process or
    /// another, fails until this handle is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be created or is not writable,
    /// when the directory is already open, when the snapshot's version,
    /// embedder, or dimension is not supported, or when an unreadable snapshot
    /// has no usable backup. An unsupported snapshot is not replaced from the
    /// backup. An unreadable snapshot is quarantined and is not overwritten.
    /// A later open refuses while that quarantine file remains and no snapshot
    /// was restored, instead of creating an empty store.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DripstoneError> {
        let dir = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)
            .map_err(|error| io_store("create local store directory", error))?;
        let lock = lock_store(&dir)?;
        let chunks = read_snapshot(&dir)?;
        let inner = load_backend(&chunks)?;
        if !snapshot_path(&dir).exists() {
            write_snapshot(&dir, &chunks)?;
        }
        Ok(Self {
            inner,
            dir,
            write: Arc::new(Mutex::new(lock)),
        })
    }

    #[cfg(test)]
    fn memory_default_ttl_secs(&self) -> Option<u64> {
        self.inner.inner().default_ttl_secs()
    }

    async fn lock_write(&self) -> OwnedMutexGuard<File> {
        Arc::clone(&self.write).lock_owned().await
    }

    async fn persist(&self, guard: OwnedMutexGuard<File>) -> Result<(), DripstoneError> {
        let mut chunks = self.inner.list(None, None).await?;
        chunks.sort_by(|left, right| left.id.cmp(&right.id));
        let dir = self.dir.clone();
        tokio::task::spawn_blocking(move || {
            let pause = arm_write_pause(&dir);
            let result = write_snapshot_with_pause(&dir, &chunks, &pause);
            // `pause` drops after this closure returns. Drop the file lock
            // first so that signal means the directory can be opened again.
            drop(guard);
            result
        })
        .await
        .map_err(|error| {
            DripstoneError::storage_with_source("join local store snapshot task", error)
        })?
    }
}

fn load_backend(chunks: &[Chunk]) -> Result<FullTextIndexedBackend<MemoryBackend>, DripstoneError> {
    let memory = MemoryBackend::with_config(InMemoryConfig {
        default_ttl_secs: None,
        max_entries: None,
        auto_cleanup: false,
    });
    memory.insert_chunks(chunks);
    let index = std::sync::Arc::new(InMemoryFullTextIndex::new());
    index.replace_documents(chunks)?;
    Ok(FullTextIndexedBackend::new(
        memory,
        index,
        FullTextIndexedBackendConfig::default(),
    ))
}

fn lock_store(dir: &Path) -> Result<File, DripstoneError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(LOCK_FILE))
        .map_err(|error| io_store("open local store lock", error))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(std::fs::TryLockError::WouldBlock) => Err(DripstoneError::storage(
            "local store directory is already open",
        )),
        Err(std::fs::TryLockError::Error(error)) => {
            Err(io_store("lock local store directory", error))
        }
    }
}

fn io_store(action: &str, error: std::io::Error) -> DripstoneError {
    if error.kind() == std::io::ErrorKind::PermissionDenied {
        DripstoneError::storage_with_source(
            format!("{action}: local store directory is not writable"),
            error,
        )
    } else {
        DripstoneError::storage_with_source(action, error)
    }
}

fn snapshot_path(dir: &Path) -> PathBuf {
    dir.join(SNAPSHOT_FILE)
}

fn backup_path(dir: &Path) -> PathBuf {
    dir.join(SNAPSHOT_BACKUP)
}

fn read_snapshot(dir: &Path) -> Result<Vec<Chunk>, DripstoneError> {
    match load_snapshot(&snapshot_path(dir))? {
        SnapshotLoad::Chunks(chunks) => Ok(chunks),
        SnapshotLoad::Unsupported(message) => Err(DripstoneError::configuration(message)),
        SnapshotLoad::Missing => read_missing_primary(dir),
        SnapshotLoad::Unreadable => {
            quarantine_snapshot(dir)?;
            read_backup(dir, false)
        }
    }
}

fn read_missing_primary(dir: &Path) -> Result<Vec<Chunk>, DripstoneError> {
    if !backup_path(dir).exists() && quarantine_present(dir)? {
        return Err(DripstoneError::configuration(
            "local store quarantine file exists and the snapshot was not restored",
        ));
    }
    read_backup(dir, true)
}

fn quarantine_present(dir: &Path) -> Result<bool, DripstoneError> {
    let entries =
        std::fs::read_dir(dir).map_err(|error| io_store("read local store directory", error))?;
    for entry in entries {
        let entry = entry.map_err(|error| io_store("read local store directory", error))?;
        let name = entry.file_name();
        if name
            .to_str()
            .is_some_and(|name| name.starts_with(QUARANTINE_PREFIX))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn read_backup(dir: &Path, missing_backup_is_empty: bool) -> Result<Vec<Chunk>, DripstoneError> {
    match load_snapshot(&backup_path(dir))? {
        SnapshotLoad::Chunks(chunks) => Ok(chunks),
        SnapshotLoad::Missing if missing_backup_is_empty => Ok(Vec::new()),
        SnapshotLoad::Missing | SnapshotLoad::Unreadable | SnapshotLoad::Unsupported(_) => {
            Err(DripstoneError::configuration(
                "local store snapshot is unreadable and the backup cannot restore it",
            ))
        }
    }
}

fn quarantine_snapshot(dir: &Path) -> Result<(), DripstoneError> {
    let dest = quarantine_path(dir)?;
    std::fs::rename(snapshot_path(dir), &dest)
        .map_err(|error| io_store("quarantine local store snapshot", error))?;
    tracing::warn!(
        quarantined = %dest.display(),
        "unreadable local store snapshot quarantined as store.json.corrupt-<timestamp>; the backup is one write behind the last acknowledged add; this file will not be overwritten"
    );
    Ok(())
}

fn quarantine_path(dir: &Path) -> Result<PathBuf, DripstoneError> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| {
            DripstoneError::storage_with_source("read clock for quarantine name", error)
        })?
        .as_nanos();
    let mut path = dir.join(format!("{QUARANTINE_PREFIX}{nanos}"));
    let mut suffix = 0u32;
    while path.exists() {
        suffix = suffix.saturating_add(1);
        path = dir.join(format!("{QUARANTINE_PREFIX}{nanos}-{suffix}"));
        if suffix == 1000 {
            return Err(DripstoneError::storage(
                "could not allocate a quarantine name for the local store snapshot",
            ));
        }
    }
    Ok(path)
}

fn primary_is_valid(dir: &Path) -> Result<bool, DripstoneError> {
    Ok(matches!(
        load_snapshot(&snapshot_path(dir))?,
        SnapshotLoad::Chunks(_)
    ))
}

fn load_snapshot(path: &Path) -> Result<SnapshotLoad, DripstoneError> {
    if !path.exists() {
        return Ok(SnapshotLoad::Missing);
    }
    let bytes =
        std::fs::read(path).map_err(|error| io_store("read local store snapshot", error))?;
    Ok(load_snapshot_bytes(&bytes))
}

fn load_snapshot_bytes(bytes: &[u8]) -> SnapshotLoad {
    if bytes.is_empty() {
        return SnapshotLoad::Unreadable;
    }
    let Ok(header) = serde_json::from_slice::<SnapshotHeader>(bytes) else {
        return SnapshotLoad::Unreadable;
    };
    if let Some(message) = unsupported_snapshot(&header) {
        return SnapshotLoad::Unsupported(message);
    }
    let Ok(snapshot) = serde_json::from_slice::<SnapshotChunks>(bytes) else {
        return SnapshotLoad::Unreadable;
    };
    if snapshot
        .chunks
        .iter()
        .any(|chunk| chunk_is_storable(chunk).is_err())
    {
        return SnapshotLoad::Unreadable;
    }
    SnapshotLoad::Chunks(snapshot.chunks)
}

fn unsupported_snapshot(snapshot: &SnapshotHeader) -> Option<&'static str> {
    if snapshot.version != SNAPSHOT_VERSION {
        return Some("local store snapshot version is not supported");
    }
    if snapshot.embedder != LOCAL_EMBEDDER_NAME {
        return Some("local store snapshot embedder is not supported");
    }
    if snapshot.dimension != LOCAL_EMBED_DIMENSION {
        return Some("local store snapshot dimension is not supported");
    }
    None
}

fn vector_is_storable(vector: &[f32]) -> Result<(), DripstoneError> {
    if vector.len() != LOCAL_EMBED_DIMENSION {
        return Err(DripstoneError::InvalidVectorLength {
            expected: LOCAL_EMBED_DIMENSION,
            actual: vector.len(),
        });
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(DripstoneError::configuration(
            "local store vector must be finite",
        ));
    }
    if !squared_norm_is_finite(vector) {
        return Err(DripstoneError::configuration(
            "local store vector squared norm must be finite",
        ));
    }
    Ok(())
}

fn squared_norm_is_finite(vector: &[f32]) -> bool {
    let mut sum = 0.0f32;
    for value in vector {
        sum += value * value;
    }
    sum.is_finite()
}

fn chunk_is_storable(chunk: &Chunk) -> Result<(), DripstoneError> {
    vector_is_storable(&chunk.vector)?;
    if let Some(rows) = chunk.multi_vectors.as_ref() {
        multi_vectors_are_storable(rows)?;
    }
    Ok(())
}

fn multi_vectors_are_storable(rows: &crate::search::MultiVectors) -> Result<(), DripstoneError> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    let width = first.len();
    for row in rows.iter() {
        if row.len() != width {
            return Err(DripstoneError::InvalidVectorLength {
                expected: width,
                actual: row.len(),
            });
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(DripstoneError::configuration(
                "local store multi-vector must be finite",
            ));
        }
        if !squared_norm_is_finite(row) {
            return Err(DripstoneError::configuration(
                "local store multi-vector squared norm must be finite",
            ));
        }
    }
    Ok(())
}

fn reject_bad_chunks(chunks: &[Chunk]) -> Result<(), DripstoneError> {
    for chunk in chunks {
        chunk_is_storable(chunk)?;
    }
    Ok(())
}

fn snapshot_bytes_would_load(bytes: &[u8]) -> Result<(), DripstoneError> {
    match load_snapshot_bytes(bytes) {
        SnapshotLoad::Chunks(_) => Ok(()),
        SnapshotLoad::Unsupported(message) => Err(DripstoneError::configuration(message)),
        SnapshotLoad::Missing | SnapshotLoad::Unreadable => Err(DripstoneError::configuration(
            "local store snapshot would not load",
        )),
    }
}

#[derive(Serialize)]
struct OpenSnapshotRef<'a> {
    version: u32,
    embedder: &'a str,
    dimension: usize,
    chunks: &'a [Chunk],
}

fn write_snapshot(dir: &Path, chunks: &[Chunk]) -> Result<(), DripstoneError> {
    let pause = arm_write_pause(dir);
    write_snapshot_with_pause(dir, chunks, &pause)
}

fn write_snapshot_with_pause(
    dir: &Path,
    chunks: &[Chunk],
    pause: &WritePause,
) -> Result<(), DripstoneError> {
    let snapshot = OpenSnapshotRef {
        version: SNAPSHOT_VERSION,
        embedder: LOCAL_EMBEDDER_NAME,
        dimension: LOCAL_EMBED_DIMENSION,
        chunks,
    };
    for chunk in chunks {
        chunk_is_storable(chunk)?;
    }
    let bytes = serde_json::to_vec_pretty(&snapshot)?;
    snapshot_bytes_would_load(&bytes)?;
    let temporary = dir.join(SNAPSHOT_TEMPORARY);
    write_synced(&temporary, &bytes, || wait_at_pause(pause))?;
    if let Err(error) = backup_valid_primary(dir) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    replace_snapshot(&temporary, &snapshot_path(dir))?;
    sync_dir(dir)
}

fn write_synced(
    path: &Path,
    bytes: &[u8],
    before_write: impl FnOnce(),
) -> Result<(), DripstoneError> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|error| io_store(WRITE_LOCAL_STORE_SNAPSHOT, error))?;
    before_write();
    file.write_all(bytes)
        .map_err(|error| io_store(WRITE_LOCAL_STORE_SNAPSHOT, error))?;
    record_write_step(STEP_WRITE_TMP);
    sync_step(&file, STEP_SYNC_TMP, "sync local store snapshot")
}

fn backup_valid_primary(dir: &Path) -> Result<(), DripstoneError> {
    if !primary_is_valid(dir)? {
        return Ok(());
    }
    let backup = backup_path(dir);
    retry_sharing(|| std::fs::copy(snapshot_path(dir), &backup).map(|_| ()))
        .map_err(|error| sharing_or_io("back up local store snapshot", error))?;
    record_write_step(STEP_COPY_BAK);
    sync_file(&backup)
}

fn replace_snapshot(temporary: &Path, primary: &Path) -> Result<(), DripstoneError> {
    let replaced = retry_sharing(|| std::fs::rename(temporary, primary));
    if let Err(error) = replaced {
        let _ = std::fs::remove_file(temporary);
        return Err(sharing_or_io(REPLACE_LOCAL_STORE_SNAPSHOT, error));
    }
    record_write_step(STEP_RENAME);
    Ok(())
}

/// Windows `FlushFileBuffers` rejects a read-only handle, so the backup is
/// reopened for write before the flush.
fn sync_file(path: &Path) -> Result<(), DripstoneError> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| io_store("open local store file for sync", error))?;
    sync_step(&file, STEP_SYNC_BAK, "sync local store file")
}

fn sync_dir(dir: &Path) -> Result<(), DripstoneError> {
    let file = open_dir_for_sync(dir)?;
    sync_step(&file, STEP_SYNC_DIR, "sync local store directory")
}

fn sync_step(file: &File, step: &'static str, action: &str) -> Result<(), DripstoneError> {
    file.sync_all().map_err(|error| io_store(action, error))?;
    record_write_step(step);
    Ok(())
}

#[cfg(test)]
thread_local! {
    static WRITE_TRACE: std::cell::RefCell<Option<Vec<&'static str>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn record_write_step(step: &'static str) {
    WRITE_TRACE.with(|trace| {
        if let Some(steps) = trace.borrow_mut().as_mut() {
            steps.push(step);
        }
    });
}

#[cfg(not(test))]
fn record_write_step(_step: &'static str) {}

/// Test hook that stops one snapshot write in `dir` after it opens the
/// temporary file, until the test resumes it. `persist` drops the file lock
/// and then drops this hook, so the done signal means the directory can be
/// opened again.
#[cfg(test)]
struct WriteHook {
    dir: PathBuf,
    paused: std::sync::mpsc::SyncSender<()>,
    resume: std::sync::mpsc::Receiver<()>,
    done: std::sync::mpsc::SyncSender<()>,
}

#[cfg(test)]
impl Drop for WriteHook {
    fn drop(&mut self) {
        let _ = self.done.send(());
    }
}

#[cfg(test)]
static WRITE_HOOKS: std::sync::Mutex<Vec<WriteHook>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
type WritePause = Option<WriteHook>;

#[cfg(test)]
fn arm_write_pause(dir: &Path) -> WritePause {
    let mut hooks = WRITE_HOOKS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let index = hooks.iter().position(|hook| hook.dir == dir)?;
    Some(hooks.swap_remove(index))
}

#[cfg(test)]
fn wait_at_pause(pause: &WritePause) {
    if let Some(hook) = pause {
        let _ = hook.paused.send(());
        let _ = hook.resume.recv();
    }
}

#[cfg(not(test))]
struct WritePause;

#[cfg(not(test))]
fn arm_write_pause(_dir: &Path) -> WritePause {
    WritePause
}

#[cfg(not(test))]
fn wait_at_pause(_pause: &WritePause) {}

/// `FILE_FLAG_BACKUP_SEMANTICS`. A directory handle on Windows needs this flag
/// and write access; a read-only open still returns access denied.
#[cfg(windows)]
const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;

#[cfg(windows)]
fn open_dir_for_sync(dir: &Path) -> Result<File, DripstoneError> {
    OpenOptions::new()
        .write(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(dir)
        .map_err(|error| io_store("open local store directory for sync", error))
}

#[cfg(not(windows))]
fn open_dir_for_sync(dir: &Path) -> Result<File, DripstoneError> {
    File::open(dir).map_err(|error| io_store("open local store directory for sync", error))
}

/// Windows `ERROR_ACCESS_DENIED`. On Linux and macOS, raw 5 is `EIO`.
#[cfg(windows)]
const ERROR_ACCESS_DENIED: i32 = 5;
/// Windows `ERROR_SHARING_VIOLATION`. On Linux and macOS, raw 32 is `EPIPE`.
#[cfg(windows)]
const ERROR_SHARING_VIOLATION: i32 = 32;

/// Another process can hold `store.json` without `FILE_SHARE_DELETE`, or hold
/// the backup across a copy. Those codes are Windows-only.
const SHARING_ATTEMPTS: u32 = 8;

fn sharing_violation(error: &std::io::Error) -> bool {
    #[cfg(windows)]
    {
        cfg!(windows)
            && matches!(
                error.raw_os_error(),
                Some(ERROR_ACCESS_DENIED | ERROR_SHARING_VIOLATION)
            )
    }
    #[cfg(not(windows))]
    {
        let _ = error;
        false
    }
}

fn sharing_or_io(action: &str, error: std::io::Error) -> DripstoneError {
    if sharing_violation(&error) {
        return DripstoneError::storage_with_source(
            format!("{action}: the file is open in another process"),
            error,
        );
    }
    io_store(action, error)
}

fn retry_sharing(mut op: impl FnMut() -> std::io::Result<()>) -> std::io::Result<()> {
    let mut attempt = 0_u32;
    loop {
        match op() {
            Ok(()) => return Ok(()),
            Err(error) if sharing_violation(&error) && attempt + 1 < SHARING_ATTEMPTS => {
                attempt = attempt.saturating_add(1);
                pause_for_sharing_retry(attempt);
            }
            Err(error) => return Err(error),
        }
    }
}

#[expect(
    clippy::disallowed_methods,
    reason = "snapshot replace runs on the blocking pool; a Windows sharing violation needs a short pause before retry"
)]
fn pause_for_sharing_retry(attempt: u32) {
    std::thread::sleep(std::time::Duration::from_millis(5 * u64::from(attempt)));
}

#[async_trait]
impl Backend for OpenBackend {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        reject_bad_chunks(chunks)?;
        let guard = self.lock_write().await;
        self.inner.upsert(chunks).await?;
        self.persist(guard).await
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.inner.get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let guard = self.lock_write().await;
        self.inner.delete(id, namespace).await?;
        self.persist(guard).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let guard = self.lock_write().await;
        let deleted = self.inner.delete_document(doc_id, namespace).await?;
        self.persist(guard).await?;
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        self.inner.search(query).await
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.inner.search_full_text(query).await
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.inner
            .search_hybrid_full_text(dense, lexical, fusion)
            .await
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let guard = self.lock_write().await;
        self.inner.delete_namespace(namespace).await?;
        self.persist(guard).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.inner.count(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        self.inner.list(namespace, limit).await
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        let guard = self.lock_write().await;
        self.inner.flush().await?;
        self.persist(guard).await
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        self.flush().await?;
        self.inner.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_keeps_chunks_until_they_are_deleted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = OpenBackend::open(dir.path()).expect("open");
        assert_eq!(backend.memory_default_ttl_secs(), None);
    }

    /// `FlushFileBuffers` rejects a read-only handle only on Windows; a
    /// read-only `sync_all` succeeds on Linux and macOS.
    #[cfg(windows)]
    #[test]
    fn directory_and_backup_sync_use_writable_handles() {
        let dir = tempfile::tempdir().expect("tempdir");
        sync_dir(dir.path()).expect("sync dir");
        let backup = dir.path().join(SNAPSHOT_BACKUP);
        std::fs::write(&backup, b"{}").expect("backup");
        sync_file(&backup).expect("sync backup");
    }

    fn local_chunk(doc_id: &str) -> Chunk {
        Chunk::new(
            doc_id,
            0,
            doc_id,
            vec![0.1; LOCAL_EMBED_DIMENSION],
            0,
            4,
            1,
            serde_json::json!({}),
            None,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_write_does_not_clobber_the_next_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = Arc::new(OpenBackend::open(dir.path()).expect("open"));
        let (paused_tx, paused_rx) = std::sync::mpsc::sync_channel(1);
        let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        WRITE_HOOKS.lock().expect("hook").push(WriteHook {
            dir: dir.path().to_path_buf(),
            paused: paused_tx,
            resume: resume_rx,
            done: done_tx,
        });

        let first = {
            let backend = Arc::clone(&backend);
            tokio::spawn(async move { backend.upsert(&[local_chunk("alpha")]).await })
        };
        tokio::task::spawn_blocking(move || paused_rx.recv())
            .await
            .expect("join")
            .expect("first write paused");
        first.abort();
        assert!(first.await.expect_err("aborted").is_cancelled());

        let second = {
            let backend = Arc::clone(&backend);
            tokio::spawn(async move { backend.upsert(&[local_chunk("beta")]).await })
        };
        // Give the second write time to run while the first is still paused.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        resume_tx.send(()).expect("resume");
        tokio::task::spawn_blocking(move || done_rx.recv())
            .await
            .expect("join")
            .expect("first write finished");
        second.await.expect("join").expect("second add");
        drop(backend);

        let reopened = OpenBackend::open(dir.path()).expect("reopen");
        let mut docs: Vec<String> = reopened
            .list(None, None)
            .await
            .expect("list")
            .into_iter()
            .map(|chunk| chunk.doc_id)
            .collect();
        docs.sort();
        assert_eq!(docs, ["alpha", "beta"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_handle_keeps_the_directory_locked_until_its_write_ends() {
        let dir = tempfile::tempdir().expect("tempdir");
        let backend = Arc::new(OpenBackend::open(dir.path()).expect("open"));
        let (paused_tx, paused_rx) = std::sync::mpsc::sync_channel(1);
        let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(1);
        let (done_tx, done_rx) = std::sync::mpsc::sync_channel(1);
        WRITE_HOOKS.lock().expect("hook").push(WriteHook {
            dir: dir.path().to_path_buf(),
            paused: paused_tx,
            resume: resume_rx,
            done: done_tx,
        });

        let first = {
            let backend = Arc::clone(&backend);
            tokio::spawn(async move { backend.upsert(&[local_chunk("alpha")]).await })
        };
        tokio::task::spawn_blocking(move || paused_rx.recv())
            .await
            .expect("join")
            .expect("first write paused");
        first.abort();
        assert!(first.await.expect_err("aborted").is_cancelled());
        drop(backend);

        let refused = OpenBackend::open(dir.path()).expect_err("still locked");
        assert_eq!(
            refused.to_string(),
            DripstoneError::storage("local store directory is already open").to_string()
        );

        resume_tx.send(()).expect("resume");
        tokio::task::spawn_blocking(move || done_rx.recv())
            .await
            .expect("join")
            .expect("first write finished");
        // The done signal is sent after persist drops the directory lock.
        let reopened = OpenBackend::open(dir.path()).expect("reopen");
        reopened
            .upsert(&[local_chunk("beta")])
            .await
            .expect("second add");
        drop(reopened);

        let docs: Vec<String> = OpenBackend::open(dir.path())
            .expect("reopen again")
            .list(None, None)
            .await
            .expect("list")
            .into_iter()
            .map(|chunk| chunk.doc_id)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(docs, ["alpha", "beta"]);
    }

    #[test]
    fn failed_replace_removes_the_temporary_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let temporary = dir.path().join(SNAPSHOT_TEMPORARY);
        std::fs::write(&temporary, b"new").expect("tmp");
        let primary = dir.path().join(SNAPSHOT_FILE);
        std::fs::create_dir(&primary).expect("primary dir");
        std::fs::write(primary.join("held"), b"x").expect("held");
        let error = replace_snapshot(&temporary, &primary).expect_err("rename");
        assert!(
            error.to_string().contains(REPLACE_LOCAL_STORE_SNAPSHOT),
            "{error}"
        );
        assert!(!temporary.exists());
        assert!(primary.join("held").exists());
    }

    #[cfg(windows)]
    #[test]
    fn sharing_retry_bounds_access_denied_and_sharing_violation() {
        let attempts = std::cell::Cell::new(0_u32);
        retry_sharing(|| {
            let seen = attempts.get();
            attempts.set(seen + 1);
            if seen < 2 {
                Err(std::io::Error::from_raw_os_error(5))
            } else {
                Ok(())
            }
        })
        .expect("recovers");
        assert_eq!(attempts.get(), 3);

        let attempts = std::cell::Cell::new(0_u32);
        let error = retry_sharing(|| {
            attempts.set(attempts.get() + 1);
            Err(std::io::Error::from_raw_os_error(32))
        })
        .expect_err("exhausted");
        assert_eq!(attempts.get(), SHARING_ATTEMPTS);
        assert_eq!(error.raw_os_error(), Some(32));

        let attempts = std::cell::Cell::new(0_u32);
        let error = retry_sharing(|| {
            attempts.set(attempts.get() + 1);
            Err(std::io::Error::from_raw_os_error(13))
        })
        .expect_err("not a sharing code");
        assert_eq!(attempts.get(), 1);
        assert_eq!(error.raw_os_error(), Some(13));
    }

    #[cfg(not(windows))]
    #[test]
    fn eio_and_raw_32_are_not_retried() {
        for code in [5, 32] {
            let attempts = std::cell::Cell::new(0_u32);
            let error = retry_sharing(|| {
                attempts.set(attempts.get() + 1);
                Err(std::io::Error::from_raw_os_error(code))
            })
            .expect_err("io");
            assert_eq!(attempts.get(), 1, "raw {code} was retried");
            assert_eq!(error.raw_os_error(), Some(code));
            let reported = sharing_or_io("back up local store snapshot", error);
            let message = reported.to_string();
            assert!(
                message.contains("back up local store snapshot"),
                "{message}"
            );
            assert!(
                !message.contains("another process"),
                "raw {code} is not a sharing violation: {message}"
            );
        }
    }

    #[test]
    fn snapshot_write_syncs_temp_backup_and_directory_in_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_snapshot(dir.path(), &[]).expect("first snapshot");
        WRITE_TRACE.with(|trace| *trace.borrow_mut() = Some(Vec::new()));
        let write = write_snapshot(dir.path(), &[]);
        let steps = WRITE_TRACE.with(|trace| trace.borrow_mut().take().expect("trace"));
        write.expect("second snapshot");
        assert_eq!(
            steps,
            [
                STEP_WRITE_TMP,
                STEP_SYNC_TMP,
                STEP_COPY_BAK,
                STEP_SYNC_BAK,
                STEP_RENAME,
                STEP_SYNC_DIR,
            ]
        );
    }

    #[test]
    fn nan_multi_vector_is_not_written_over_a_valid_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let good = Chunk::new(
            "alpha",
            0,
            "alpha",
            vec![0.1; LOCAL_EMBED_DIMENSION],
            0,
            4,
            1,
            serde_json::json!({}),
            None,
        );
        write_snapshot(dir.path(), &[good]).expect("alpha");
        let before = std::fs::read(snapshot_path(dir.path())).expect("primary");
        let bad = Chunk::new(
            "nan",
            0,
            "nan",
            vec![0.1; LOCAL_EMBED_DIMENSION],
            0,
            4,
            1,
            serde_json::json!({}),
            None,
        )
        .with_multi_vectors(crate::search::multi_vectors_from_owned(vec![vec![
            f32::NAN;
            4
        ]]));
        let error = write_snapshot(dir.path(), &[bad]).expect_err("nan multi-vector");
        assert!(
            error.to_string().contains("finite"),
            "unexpected error: {error}"
        );
        assert_eq!(
            std::fs::read(snapshot_path(dir.path())).expect("stays"),
            before
        );
        assert!(!dir.path().join(SNAPSHOT_TEMPORARY).exists());
    }
}
