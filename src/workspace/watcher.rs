//! File system watcher for automatic re-indexing of changed files.
//!
//! This module provides continuous file watching for workspace embeddings:
//! - Detects file changes (create, modify, remove) during runtime
//! - Debounces rapid changes (1500ms after last event)
//! - Caps each reindex burst so a large save storm cannot saturate the host
//! - Filters by extensions and .gitignore patterns
//! - Re-indexes changed files automatically in background
//! - Updates the manifest with new mtimes
//!
//! Only used in interactive modes (REPL, TUI) where the agent runs long-term.

use crate::backpressure::CoalescedSignal;
use crate::rag::RagService;
use crate::workspace::config::{
    DEFAULT_MAX_FILE_SIZE, PersistenceConfig, WorkspaceIndexerConfig, index_root_for_path,
    owning_store_dir,
};
use crate::workspace::gitignore;
use crate::workspace::manifest::{self, IndexManifest};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

const WATCHER_DEBOUNCE: Duration = Duration::from_millis(1_500);
const MAX_REINDEX_BURST: usize = 16;

/// Workspace file watcher for automatic re-indexing.
pub struct WorkspaceWatcher {
    _rag_service: Arc<RagService>,
    _config: WorkspaceIndexerConfig,
    _manifest: Arc<Mutex<IndexManifest>>,
    _watcher: RecommendedWatcher,
    /// Single shutdown signal. One value is enough, and the sender lives only
    /// as long as the watcher, so this queue is not a stream of file events.
    shutdown_tx: mpsc::UnboundedSender<()>,
}

impl WorkspaceWatcher {
    /// Create and start a new workspace watcher.
    ///
    /// This spawns a background task that watches for file changes and
    /// automatically re-indexes them.
    pub fn new(
        rag_service: Arc<RagService>,
        config: WorkspaceIndexerConfig,
        manifest: IndexManifest,
    ) -> anyhow::Result<Self> {
        let manifest = Arc::new(Mutex::new(manifest));

        // File events coalesce by path. The wake channel holds one signal;
        // a full wake means the processor has not drained, and the map keeps
        // the newest kind for every path. Nothing is dropped.
        let (watch_signal, signal_rx) = CoalescedSignal::channel();
        let watch_signal = Arc::new(watch_signal);

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = mpsc::unbounded_channel();

        // Create the notify watcher
        let callback_signal = Arc::clone(&watch_signal);
        let mut watcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| match res {
                Ok(event) => record_notify_event(&callback_signal, event),
                Err(e) => {
                    warn!("File watcher: error: {}", e);
                }
            })
            .map_err(|e| anyhow::anyhow!("Failed to create file watcher: {}", e))?;

        // Watch the root directory recursively (skip if path does not exist)
        if !config.root_path.exists() {
            return Err(anyhow::anyhow!(
                "Workspace path does not exist: {}",
                config.root_path.display()
            ));
        }
        if !config.root_path.is_dir() {
            return Err(anyhow::anyhow!(
                "Workspace path is not a directory: {}",
                config.root_path.display()
            ));
        }
        // Use canonical path for watching; on macOS (kqueue/FSEvents) this can avoid ENOENT
        // when the path is symlinked or has a different resolution than the watcher expects
        let watch_path: PathBuf = std::fs::canonicalize(&config.root_path).map_err(|e| {
            anyhow::anyhow!(
                "Could not resolve workspace path '{}': {}",
                config.root_path.display(),
                e
            )
        })?;
        watcher
            .watch(&watch_path, RecursiveMode::Recursive)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to watch directory '{}' (resolved to '{}'): {}",
                    config.root_path.display(),
                    watch_path.display(),
                    e
                )
            })?;
        let memory_watch_path = crate::workspace::workspace_memory_markdown_dir(&config.root_path);
        if memory_watch_path.exists() {
            watcher
                .watch(&memory_watch_path, RecursiveMode::Recursive)
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to watch memory directory '{}': {}",
                        memory_watch_path.display(),
                        e
                    )
                })?;
        }

        info!(
            "File watcher: monitoring {} for changes",
            watch_path.display()
        );

        // Event paths from the watcher may be canonical; use config with canonical root for filtering
        let mut config_for_events = config.clone();
        config_for_events.root_path = watch_path;

        // Spawn the event processor task
        let rag_clone = Arc::clone(&rag_service);
        let manifest_clone = Arc::clone(&manifest);
        tokio::spawn(async move {
            Self::process_events(
                watch_signal,
                signal_rx,
                shutdown_rx,
                rag_clone,
                config_for_events,
                manifest_clone,
            )
            .await;
        });

        Ok(Self {
            _rag_service: rag_service,
            _config: config,
            _manifest: manifest,
            _watcher: watcher,
            shutdown_tx,
        })
    }

    /// Process file system events with debouncing and filtering.
    async fn process_events(
        watch_signal: Arc<CoalescedSignal<EventKind>>,
        mut signal_rx: mpsc::Receiver<()>,
        mut shutdown_rx: mpsc::UnboundedReceiver<()>,
        rag_service: Arc<RagService>,
        config: WorkspaceIndexerConfig,
        manifest: Arc<Mutex<IndexManifest>>,
    ) {
        let mut pending_changes: HashMap<PathBuf, Instant> = HashMap::new();
        let mut pending_removals: HashMap<PathBuf, Instant> = HashMap::new();

        loop {
            tokio::select! {
                signal = signal_rx.recv() => {
                    if signal.is_none() {
                        break;
                    }
                    apply_coalesced_paths(
                        &watch_signal.drain(),
                        &config,
                        &mut pending_changes,
                        &mut pending_removals,
                    );
                }
                _ = tokio::time::sleep(WATCHER_DEBOUNCE) => {
                    let now = Instant::now();
                    let mut ready_files =
                        take_debounced_paths(&mut pending_changes, now, WATCHER_DEBOUNCE, MAX_REINDEX_BURST);
                    let ready_removals =
                        take_debounced_paths(&mut pending_removals, now, WATCHER_DEBOUNCE, MAX_REINDEX_BURST);

                    if !ready_removals.is_empty()
                        && let Err(e) = Self::remove_files_from_index(
                            &ready_removals,
                            &rag_service,
                            &config,
                            &manifest,
                        ).await
                    {
                        warn!("File watcher: failed to remove deleted files from index: {}", e);
                    }

                    if !ready_files.is_empty() {
                        // Filter by extensions and gitignore
                        ready_files.retain(|path| Self::should_index(path, &config));

                        if !ready_files.is_empty() {
                            debug!("File watcher: detected {} changed file(s)", ready_files.len());

                            // Re-index changed files
                            if let Err(e) = Self::reindex_files(
                                &ready_files,
                                &rag_service,
                                &config,
                                &manifest,
                            ).await {
                                warn!("File watcher: failed to re-index files: {}", e);
                            }
                        }
                    }
                }
                Some(_) = shutdown_rx.recv() => {
                    info!("File watcher: shutting down");
                    break;
                }
            }
        }
    }

    /// Check if a file should be indexed based on config rules.
    fn should_index(path: &Path, config: &WorkspaceIndexerConfig) -> bool {
        if !Self::could_have_been_indexed(path, config) {
            return false;
        }

        let max_size = if config.max_file_size > 0 {
            config.max_file_size as u64
        } else {
            DEFAULT_MAX_FILE_SIZE as u64
        };

        // Quick size check
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() > max_size {
                return false;
            }
        } else {
            return false; // Can't read metadata, skip
        }

        // Use gitignore-aware single-path matching so we check *this* file,
        // not whichever file a `max_files = 1` walk happens to visit first.
        gitignore::path_allowed(
            &index_root_for_path(config, path),
            path,
            Some(max_size),
            &config.resolved_extensions(),
            &config.exclude_patterns,
        )
    }

    fn could_have_been_indexed(path: &Path, config: &WorkspaceIndexerConfig) -> bool {
        gitignore::matches_index_filters(
            &config.root_path,
            path,
            &config.resolved_extensions(),
            &config.exclude_patterns,
        )
    }

    /// Re-index a batch of changed files in small windows so a save storm
    /// cannot keep the whole burst resident in RAM.
    async fn reindex_files(
        files: &[PathBuf],
        rag_service: &Arc<RagService>,
        config: &WorkspaceIndexerConfig,
        manifest: &Arc<Mutex<IndexManifest>>,
    ) -> anyhow::Result<()> {
        let window_size = RagService::index_ingest_window_size(RagService::index_concurrency());
        for file_window in files.chunks(window_size.max(1)) {
            Self::reindex_file_window(file_window, rag_service, config, manifest).await?;
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    async fn reindex_file_window(
        files: &[PathBuf],
        rag_service: &Arc<RagService>,
        config: &WorkspaceIndexerConfig,
        manifest: &Arc<Mutex<IndexManifest>>,
    ) -> anyhow::Result<()> {
        let mut documents = Vec::new();
        let mut indexed_records = Vec::new();

        for path in files {
            let path_str = match path.to_str() {
                Some(s) => s,
                None => {
                    warn!("File watcher: invalid UTF-8 path: {}", path.display());
                    continue;
                }
            };

            let content = match Self::read_file(path_str, config).await {
                Ok(c) => c,
                Err(e) => {
                    warn!("File watcher: failed to read {}: {}", path_str, e);
                    continue;
                }
            };

            if let Ok(record) = manifest::file_record_with_content(path, &content).await {
                indexed_records.push((path_str.to_string(), record));
            }

            documents.push((path_str.to_string(), content, Some(path.clone())));
        }

        if documents.is_empty() {
            return Ok(());
        }

        let (total_chunks, total_docs, failures, _) = rag_service
            .index_batch_with_paths(documents)
            .await
            .map_err(|e| anyhow::anyhow!("RAG index_batch failed: {}", e))?;

        if !failures.is_empty() {
            warn!("File watcher: {} failures during indexing", failures.len());
        }

        info!(
            "File watcher: re-indexed {} file(s), {} chunk(s)",
            total_docs, total_chunks
        );

        let mut manifest_guard = manifest.lock().await;
        Self::persist_records(config, &mut manifest_guard, indexed_records, &[]).await?;

        Ok(())
    }

    async fn remove_files_from_index(
        files: &[PathBuf],
        rag_service: &Arc<RagService>,
        config: &WorkspaceIndexerConfig,
        manifest: &Arc<Mutex<IndexManifest>>,
    ) -> anyhow::Result<()> {
        let mut removed = 0usize;
        let mut manifest_guard = manifest.lock().await;
        for path in files {
            let Some(path_str) = path.to_str() else {
                warn!(
                    "File watcher: invalid UTF-8 removed path: {}",
                    path.display()
                );
                continue;
            };
            let deleted_chunks = rag_service.delete_document(path_str).await?;
            if deleted_chunks > 0 {
                removed += 1;
            }
        }

        let removed_paths: Vec<String> = files
            .iter()
            .filter_map(|path| path.to_str().map(str::to_string))
            .collect();
        Self::persist_records(config, &mut manifest_guard, Vec::new(), &removed_paths).await?;

        if removed > 0 {
            info!(
                "File watcher: removed {} deleted file(s) from index",
                removed
            );
        }

        Ok(())
    }

    async fn persist_records(
        config: &WorkspaceIndexerConfig,
        parent_manifest: &mut IndexManifest,
        upserts: Vec<(String, manifest::IndexedFileRecord)>,
        removed_paths: &[String],
    ) -> anyhow::Result<()> {
        let PersistenceConfig::Disk { path: parent_store } = &config.persistence else {
            parent_manifest.merge_from_indexed(upserts);
            for path in removed_paths {
                parent_manifest.remove(path);
            }
            return Ok(());
        };

        let mut upserts_by_store: HashMap<PathBuf, Vec<(String, manifest::IndexedFileRecord)>> =
            HashMap::new();
        for record in upserts {
            let store = owning_store_dir(config, Path::new(&record.0))
                .unwrap_or_else(|| parent_store.clone());
            upserts_by_store.entry(store).or_default().push(record);
        }

        let mut removals_by_store: HashMap<PathBuf, Vec<String>> = HashMap::new();
        for path in removed_paths {
            let store =
                owning_store_dir(config, Path::new(path)).unwrap_or_else(|| parent_store.clone());
            removals_by_store
                .entry(store)
                .or_default()
                .push(path.clone());
        }

        let mut stores: Vec<PathBuf> = upserts_by_store.keys().cloned().collect();
        for store in removals_by_store.keys() {
            if !stores.iter().any(|existing| existing == store) {
                stores.push(store.clone());
            }
        }

        for store_dir in stores {
            let store_upserts = upserts_by_store.remove(&store_dir).unwrap_or_default();
            let store_removals = removals_by_store.remove(&store_dir).unwrap_or_default();
            let upsert_paths: Vec<String> = store_upserts.iter().map(|(p, _)| p.clone()).collect();
            if store_dir == *parent_store {
                parent_manifest.merge_from_indexed(store_upserts);
                for path in &store_removals {
                    parent_manifest.remove(path);
                }
                if let Err(error) = parent_manifest
                    .save_delta(&store_dir, &upsert_paths, &store_removals)
                    .await
                {
                    warn!("File watcher: failed to save manifest: {error}");
                }
            } else {
                let mut nested = IndexManifest::load(&store_dir).await.unwrap_or_default();
                nested.merge_from_indexed(store_upserts);
                for path in &store_removals {
                    nested.remove(path);
                }
                if let Err(error) = nested
                    .save_delta(&store_dir, &upsert_paths, &store_removals)
                    .await
                {
                    warn!("File watcher: failed to save nested manifest: {error}");
                }
            }
        }
        Ok(())
    }

    /// Read a file for indexing with size validation.
    async fn read_file(path_str: &str, config: &WorkspaceIndexerConfig) -> anyhow::Result<String> {
        crate::workspace::indexer::read_index_file(config, Path::new(path_str)).await
    }

    /// Shutdown the watcher gracefully.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }
}

impl Drop for WorkspaceWatcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn record_notify_event(signal: &CoalescedSignal<EventKind>, event: Event) {
    let kind = event.kind;
    for path in event.paths {
        signal.push_keeping(path, kind, merge_notify_kind);
    }
}

/// Keep an index event when a later event would not change the index.
///
/// inotify reports a write as modify and then close-write. Close-write is
/// [`EventKind::Access`]. Replacing the stored kind with that access would drop
/// the modify. FSEvents can report remove and then modify for one delete;
/// [`apply_coalesced_paths`] decides from the file on disk, so a trailing
/// modify does not have to stay stored as a remove.
fn merge_notify_kind(previous: Option<EventKind>, next: EventKind) -> EventKind {
    match next {
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => next,
        _ => previous.unwrap_or(next),
    }
}

fn index_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
    )
}

/// Queue a reindex or a removal from the file on disk.
///
/// Create, modify, and remove are the events that can change the index. The
/// stored kind is not the decision: macOS FSEvents emits create, remove, then
/// modify for a delete, and a later modify used to hide the removal. A path
/// that is a file is reindexed. Any other index event for a path that could
/// have been indexed is a removal.
fn apply_coalesced_paths(
    paths: &[(PathBuf, EventKind)],
    config: &WorkspaceIndexerConfig,
    pending_changes: &mut HashMap<PathBuf, Instant>,
    pending_removals: &mut HashMap<PathBuf, Instant>,
) {
    for (path, kind) in paths {
        let indexed = index_event(kind);
        let file = path.is_file();
        let removable = WorkspaceWatcher::could_have_been_indexed(path, config);
        match (indexed, file, removable) {
            (true, true, _) => {
                pending_removals.remove(path);
                pending_changes.insert(path.clone(), Instant::now());
            }
            (true, false, true) => {
                pending_changes.remove(path);
                pending_removals.insert(path.clone(), Instant::now());
            }
            (false, _, _) | (true, false, false) => {}
        }
    }
}

fn take_debounced_paths(
    pending: &mut HashMap<PathBuf, Instant>,
    now: Instant,
    debounce: Duration,
    max: usize,
) -> Vec<PathBuf> {
    let mut ready: Vec<PathBuf> = pending
        .iter()
        .filter(|(_, last_change)| now.duration_since(**last_change) >= debounce)
        .map(|(path, _)| path.clone())
        .collect();
    ready.sort();
    ready.truncate(max);
    for path in &ready {
        pending.remove(path);
    }
    ready
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_REINDEX_BURST, apply_coalesced_paths, merge_notify_kind, record_notify_event,
        take_debounced_paths,
    };
    use crate::backpressure::CoalescedSignal;
    use crate::workspace::config::WorkspaceIndexerConfig;
    use notify::event::{AccessKind, AccessMode, CreateKind, DataChange, ModifyKind, RemoveKind};
    use notify::{Event, EventKind};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    #[test]
    fn take_debounced_paths_caps_burst_and_leaves_overflow() {
        let mut pending = HashMap::new();
        let now = Instant::now();
        let debounce = Duration::from_millis(10);
        for i in 0..40 {
            pending.insert(PathBuf::from(format!("f{i:02}.rs")), now - debounce);
        }

        let first = take_debounced_paths(&mut pending, now, debounce, MAX_REINDEX_BURST);
        assert_eq!(first.len(), MAX_REINDEX_BURST);
        assert_eq!(pending.len(), 40 - MAX_REINDEX_BURST);

        let second = take_debounced_paths(&mut pending, now, debounce, MAX_REINDEX_BURST);
        assert_eq!(second.len(), MAX_REINDEX_BURST);
        assert_ne!(first, second);
    }

    #[test]
    fn close_write_does_not_erase_a_modify_or_create() {
        let (signal, _wake) = CoalescedSignal::channel();
        record_notify_event(
            &signal,
            Event::new(EventKind::Modify(ModifyKind::Any)).add_path(PathBuf::from("a.rs")),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Access(AccessKind::Close(AccessMode::Write)))
                .add_path(PathBuf::from("a.rs")),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Create(CreateKind::File)).add_path(PathBuf::from("b.rs")),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Access(AccessKind::Close(AccessMode::Write)))
                .add_path(PathBuf::from("b.rs")),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Modify(ModifyKind::Any)).add_path(PathBuf::from("c.rs")),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Remove(RemoveKind::File)).add_path(PathBuf::from("c.rs")),
        );

        let drained: HashMap<_, _> = signal.drain().into_iter().collect();
        assert!(matches!(
            drained.get(Path::new("a.rs")).copied(),
            Some(EventKind::Modify(_))
        ));
        assert!(matches!(
            drained.get(Path::new("b.rs")).copied(),
            Some(EventKind::Create(_))
        ));
        assert!(matches!(
            drained.get(Path::new("c.rs")).copied(),
            Some(EventKind::Remove(_))
        ));
        assert_eq!(
            merge_notify_kind(Some(EventKind::Modify(ModifyKind::Any)), EventKind::Any),
            EventKind::Modify(ModifyKind::Any)
        );
    }

    fn apply_recorded(
        signal: &CoalescedSignal<EventKind>,
        config: &WorkspaceIndexerConfig,
    ) -> (HashMap<PathBuf, Instant>, HashMap<PathBuf, Instant>) {
        let mut changes = HashMap::new();
        let mut removals = HashMap::new();
        apply_coalesced_paths(&signal.drain(), config, &mut changes, &mut removals);
        (changes, removals)
    }

    #[test]
    fn deleted_path_stays_a_removal_when_modify_follows_remove() {
        let dir = tempfile::tempdir().expect("temp root");
        let root = dir.path().to_path_buf();
        let config = WorkspaceIndexerConfig::new(root.clone());
        let missing = root.join("a.rs");
        let sequence = root.join("b.rs");
        assert!(!missing.exists());
        assert!(!sequence.exists());

        let (signal, _wake) = CoalescedSignal::channel();
        record_notify_event(
            &signal,
            Event::new(EventKind::Remove(RemoveKind::File)).add_path(missing.clone()),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Modify(ModifyKind::Data(DataChange::Content)))
                .add_path(missing.clone()),
        );
        let (changes, removals) = apply_recorded(&signal, &config);
        assert!(removals.contains_key(&missing), "{removals:?}");
        assert!(!changes.contains_key(&missing), "{changes:?}");

        let (signal, _wake) = CoalescedSignal::channel();
        record_notify_event(
            &signal,
            Event::new(EventKind::Create(CreateKind::File)).add_path(sequence.clone()),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Remove(RemoveKind::File)).add_path(sequence.clone()),
        );
        record_notify_event(
            &signal,
            Event::new(EventKind::Modify(ModifyKind::Data(DataChange::Content)))
                .add_path(sequence.clone()),
        );
        let (changes, removals) = apply_recorded(&signal, &config);
        assert!(removals.contains_key(&sequence), "{removals:?}");
        assert!(!changes.contains_key(&sequence), "{changes:?}");

        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(all(test, unix))]
mod link_tests {
    use super::WorkspaceWatcher;
    use crate::workspace::config::{PersistenceConfig, WorkspaceIndexerConfig};

    #[tokio::test]
    async fn a_changed_file_that_is_a_link_is_not_read() {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.rs"), "fn secret() {}").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        std::os::unix::fs::symlink(outside.path().join("secret.rs"), ws.path().join("leak.rs"))
            .expect("file link");
        let config = WorkspaceIndexerConfig::new(ws.path().to_path_buf())
            .persistence(PersistenceConfig::InMemory);
        let leak = ws.path().join("leak.rs");

        let read = WorkspaceWatcher::read_file(leak.to_str().expect("utf-8 path"), &config).await;

        let err = read.expect_err("a link must not be read");
        assert!(!err.to_string().contains("secret"), "{err}");
    }
}
