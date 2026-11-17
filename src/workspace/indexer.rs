//! Workspace file indexer.
//!
//! Walks the current directory (or configured root), respects .gitignore and
//! .caliche_ignore, filters to text-based files, and indexes them via RAG.

use crate::agent::tools::common::path;
use crate::rag::RagService;
use crate::workspace::config::{
    DEFAULT_MAX_FILE_SIZE, PersistenceConfig, WorkspaceIndexerConfig, store_dir_for_repo,
};
use crate::workspace::federated::write_workspace_root_marker;
use crate::workspace::git_repos;
use crate::workspace::gitignore;
use crate::workspace::manifest::{self, IndexManifest};
use std::collections::HashSet;
use std::sync::Arc;
use tracing::{debug, info};

/// Result of workspace indexing.
#[derive(Debug)]
pub struct WorkspaceIndexResult {
    pub files_indexed: usize,
    pub total_chunks: usize,
    pub failures: Vec<String>,
}

enum IndexPrepare {
    Skip,
    Failed,
    Ready {
        path_str: String,
        content: String,
        record: crate::workspace::manifest::IndexedFileRecord,
    },
}

/// The roots and on-disk stores of the repos strictly below `repo`.
fn nested_stores_of(
    repo: &std::path::Path,
    stores: &[(std::path::PathBuf, Option<std::path::PathBuf>)],
) -> Vec<(std::path::PathBuf, std::path::PathBuf)> {
    stores
        .iter()
        .filter(|(other, _)| other != repo && other.starts_with(repo))
        .filter_map(|(other, store)| Some((other.clone(), store.clone()?)))
        .collect()
}

/// Indexes a workspace directory into a RAG service.
pub struct WorkspaceIndexer {
    rag_service: Arc<RagService>,
    config: WorkspaceIndexerConfig,
    /// The roots and store directories of the repos this run's tree holds
    /// below its own root, as this run's `index()` discovered them.
    nested_stores: Vec<(std::path::PathBuf, std::path::PathBuf)>,
}

impl WorkspaceIndexer {
    /// Create a new workspace indexer.
    pub fn new(rag_service: Arc<RagService>, config: WorkspaceIndexerConfig) -> Self {
        Self {
            rag_service,
            config,
            nested_stores: Vec::new(),
        }
    }

    /// Index the configured root directory.
    ///
    /// Nested git worktrees get their own store directory and manifest. The RAG
    /// backend routes writes by file path when it is federated.
    pub async fn index(&self) -> anyhow::Result<WorkspaceIndexResult> {
        let mut repos = git_repos::discover_git_repos(&self.config.root_path);
        if repos.len() <= 1 {
            self.record_root_marker(&git_repos::canonical_or_owned(&self.config.root_path));
            return self.index_tree().await;
        }

        let mut files_indexed = 0usize;
        let mut total_chunks = 0usize;
        let mut failures = Vec::new();
        // A repo's purge releases the keys of the repos below it, so it must
        // run before them whatever order `.gitmodules` lists them in.
        repos.sort_by_key(|repo| repo.components().count());
        let stores: Vec<(std::path::PathBuf, Option<std::path::PathBuf>)> = repos
            .iter()
            .map(|repo| match repo_config(&self.config, repo).persistence {
                PersistenceConfig::Disk { path } => (repo.clone(), Some(path)),
                _ => (repo.clone(), None),
            })
            .collect();
        for repo in &repos {
            let repo_config = repo_config(&self.config, repo);
            if let PersistenceConfig::Disk { path } = &repo_config.persistence {
                self.record_root_marker_at(path, repo);
            }
            let mut repo_indexer = Self::new(Arc::clone(&self.rag_service), repo_config);
            repo_indexer.nested_stores = nested_stores_of(repo, &stores);
            let result = repo_indexer.index_tree().await?;
            files_indexed = files_indexed.saturating_add(result.files_indexed);
            total_chunks = total_chunks.saturating_add(result.total_chunks);
            failures.extend(result.failures);
        }
        Ok(WorkspaceIndexResult {
            files_indexed,
            total_chunks,
            failures,
        })
    }

    fn record_root_marker(&self, repo_root: &std::path::Path) {
        let PersistenceConfig::Disk { path } = &self.config.persistence else {
            return;
        };
        self.record_root_marker_at(path, repo_root);
    }

    fn record_root_marker_at(&self, store_dir: &std::path::Path, repo_root: &std::path::Path) {
        if let Err(error) = write_workspace_root_marker(store_dir, repo_root) {
            tracing::warn!("Workspace index: failed to write workspace root marker: {error}");
        }
    }

    /// Index one worktree using this indexer's root and persistence path.
    async fn index_tree(&self) -> anyhow::Result<WorkspaceIndexResult> {
        let root = &self.config.root_path;
        let walk_root = self.walk_root();
        let gitignore::CollectedFiles {
            files: mut paths,
            mut unlisted,
        } = self.collect_index_paths(&walk_root).await?;
        if self.config.read_root.is_none() {
            match crate::workspace::list_workspace_memory_markdown_files(root).await {
                Ok(mut memory_files) => paths.append(&mut memory_files),
                Err(_) => unlisted.push(crate::workspace::config::workspace_memory_markdown_dir(
                    root,
                )),
            }
        }

        let mut manifest = match &self.config.persistence {
            PersistenceConfig::Disk { path } => {
                IndexManifest::load_with_fingerprint(path, self.config.index_fingerprint.as_deref())
                    .await?
            }
            PersistenceConfig::InMemory => IndexManifest::new(),
        };

        let mut failures: Vec<String> = unlisted
            .iter()
            .map(|dir| format!("{}: {UNLISTED_KEPT}", dir.display()))
            .collect();
        self.purge_stale_entries(&walk_root, &paths, &unlisted, &mut manifest, &mut failures)
            .await;

        if paths.is_empty() {
            info!("Workspace index: no files found under {}", root.display());
            return Ok(WorkspaceIndexResult {
                files_indexed: 0,
                total_chunks: 0,
                failures: if failures.is_empty() {
                    vec!["No files found matching criteria".to_string()]
                } else {
                    failures
                },
            });
        }

        debug!("Workspace index: {} files to index", paths.len());
        let (files_indexed, total_chunks, skipped_count) = self
            .ingest_changed_files(&paths, &mut manifest, &mut failures)
            .await?;

        if files_indexed == 0 && skipped_count > 0 {
            info!(
                "Workspace index: skipped {} unchanged files, no new files to index",
                skipped_count
            );
        } else if skipped_count > 0 {
            info!(
                "Workspace index: skipping {} unchanged, indexed {} changed/new files",
                skipped_count, files_indexed
            );
        }

        info!(
            "Workspace index: {} files indexed, {} chunks, {} failures",
            files_indexed,
            total_chunks,
            failures.len()
        );

        Ok(WorkspaceIndexResult {
            files_indexed,
            total_chunks,
            failures,
        })
    }

    /// The directory this run walks, and the form every key it writes
    /// starts with. The top root is resolved once; a nested root is already
    /// the canonical path discovery returned and is walked as is, so one
    /// swapped for a link is refused and reported as unlisted. Keying by one
    /// form means a workspace named through a link, by its real path, or in
    /// single- or multi-repo mode writes the same keys.
    fn walk_root(&self) -> std::path::PathBuf {
        if self.config.read_root.is_some() {
            self.config.root_path.clone()
        } else {
            git_repos::canonical_or_owned(&self.config.root_path)
        }
    }

    /// Collect this worktree's files below `walk_root`.
    async fn collect_index_paths(
        &self,
        walk_root: &std::path::Path,
    ) -> anyhow::Result<gitignore::CollectedFiles> {
        let walk_root = walk_root.to_path_buf();
        let max_size = self.config.max_file_size;
        let exts_clone = self.config.resolved_extensions();
        let excludes = self.config.exclude_patterns.clone();
        tokio::task::spawn_blocking(move || {
            gitignore::collect_files(
                &walk_root,
                Some(max_size as u64),
                &exts_clone,
                None,
                &excludes,
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("Workspace walk task failed: {}", e))
    }

    /// Remove the manifest entries this run did not find.
    ///
    /// Only entries this run could have found are candidates: those below
    /// `walk_root` (the form this run keys by) or the root as configured (the
    /// form older runs keyed by) and, for the top repo, its memory
    /// directory. A nested repo's
    /// store is also that repo's own workspace store, so the entries its own
    /// runs wrote (its memory notes) are not this run's to purge. An entry
    /// below a directory in `unlisted`, which the walk could not list, is
    /// kept too: not seen is not gone.
    async fn purge_stale_entries(
        &self,
        walk_root: &std::path::Path,
        paths: &[std::path::PathBuf],
        unlisted: &[std::path::PathBuf],
        manifest: &mut IndexManifest,
        failures: &mut Vec<String>,
    ) {
        let current_paths: HashSet<String> = paths
            .iter()
            .filter_map(|path| path.to_str().map(str::to_string))
            .collect();
        let root = &self.config.root_path;
        let memory = self
            .config
            .read_root
            .is_none()
            .then(|| crate::workspace::config::workspace_memory_markdown_dir(root));
        let stale_paths: Vec<String> = manifest
            .indexed
            .keys()
            .filter(|path| !current_paths.contains(*path))
            .filter(|path| {
                let path = std::path::Path::new(path);
                path.starts_with(walk_root)
                    || path.starts_with(root)
                    || memory.as_ref().is_some_and(|dir| path.starts_with(dir))
            })
            .filter(|path| {
                !unlisted
                    .iter()
                    .any(|dir| std::path::Path::new(path).starts_with(dir))
            })
            .cloned()
            .collect();
        let stale_paths = self.release_nested_entries(stale_paths, failures).await;
        let mut removed_stale = Vec::new();
        for (i, stale_path) in stale_paths.iter().enumerate() {
            if let Err(e) = self.rag_service.delete_document(stale_path).await {
                failures.push(format!("{stale_path}: failed to delete stale index: {e}"));
                continue;
            }
            manifest.remove(stale_path);
            removed_stale.push(stale_path.clone());
            if i % 8 == 7 {
                tokio::task::yield_now().await;
            }
        }
        self.persist_manifest(manifest, &[], &removed_stale).await;
    }

    /// Drop each stale key under a nested repo from that repo's manifest
    /// before its document is deleted.
    ///
    /// The federated backend deletes a document from every member, so a key
    /// this run left behind (from a run before the nested repo was found)
    /// would also delete the nested repo's copy, which its manifest still
    /// lists as indexed. With the entry gone, the nested repo's run indexes
    /// the file again if it still exists. A key whose entry cannot be
    /// dropped is kept, and its document too.
    async fn release_nested_entries(
        &self,
        stale_paths: Vec<String>,
        failures: &mut Vec<String>,
    ) -> Vec<String> {
        let mut by_store: Vec<(&std::path::Path, Vec<String>)> = Vec::new();
        let mut kept = Vec::new();
        for path in stale_paths {
            let owner = self
                .nested_stores
                .iter()
                .filter(|(root, _)| std::path::Path::new(&path).starts_with(root))
                .max_by_key(|(root, _)| root.as_os_str().len());
            match owner {
                Some((_, store)) => match by_store.iter_mut().find(|(dir, _)| *dir == store) {
                    Some((_, paths)) => paths.push(path),
                    None => by_store.push((store, vec![path])),
                },
                None => kept.push(path),
            }
        }
        for (store, paths) in by_store {
            match IndexManifest::default()
                .save_delta(store, &[], &paths)
                .await
            {
                Ok(()) => kept.extend(paths),
                Err(error) => failures.extend(paths.iter().map(|path| {
                    format!("{path}: failed to release from its nested repo manifest: {error}")
                })),
            }
        }
        kept
    }

    async fn ingest_changed_files(
        &self,
        paths: &[std::path::PathBuf],
        manifest: &mut IndexManifest,
        failures: &mut Vec<String>,
    ) -> anyhow::Result<(usize, usize, usize)> {
        let window_size = RagService::index_ingest_window_size(RagService::index_concurrency());
        let mut documents = Vec::with_capacity(window_size);
        let mut indexed_records = Vec::with_capacity(window_size);
        let mut skipped_count = 0;
        let mut files_indexed = 0;
        let mut total_chunks = 0;

        for path in paths {
            match self
                .prepare_index_document(path, manifest, failures)
                .await?
            {
                IndexPrepare::Skip => skipped_count += 1,
                IndexPrepare::Failed => {}
                IndexPrepare::Ready {
                    path_str,
                    content,
                    record,
                } => {
                    indexed_records.push((path_str.clone(), record));
                    documents.push((
                        path_str.clone(),
                        content,
                        Some(std::path::PathBuf::from(path_str)),
                    ));
                    if documents.len() >= window_size {
                        self.flush_index_window(
                            manifest,
                            &mut documents,
                            &mut indexed_records,
                            &mut files_indexed,
                            &mut total_chunks,
                            failures,
                        )
                        .await?;
                        tokio::task::yield_now().await;
                    }
                }
            }
        }

        if !documents.is_empty() {
            self.flush_index_window(
                manifest,
                &mut documents,
                &mut indexed_records,
                &mut files_indexed,
                &mut total_chunks,
                failures,
            )
            .await?;
        }
        Ok((files_indexed, total_chunks, skipped_count))
    }

    async fn prepare_index_document(
        &self,
        path: &std::path::Path,
        manifest: &IndexManifest,
        failures: &mut Vec<String>,
    ) -> anyhow::Result<IndexPrepare> {
        let Some(path_str) = path.to_str() else {
            failures.push(format!("Invalid UTF-8 path: {}", path.display()));
            return Ok(IndexPrepare::Failed);
        };
        if let Ok((mtime, size)) = manifest::file_stat(path).await
            && manifest.matches_stat(path_str, mtime, size)
        {
            debug!("Skipping unchanged (mtime+size): {}", path_str);
            return Ok(IndexPrepare::Skip);
        }
        let content = match self.read_file_for_indexing(path).await {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{}: {}", path_str, e));
                return Ok(IndexPrepare::Failed);
            }
        };
        let record = match manifest::file_record_with_content(path, &content).await {
            Ok(record) => record,
            Err(e) => {
                failures.push(format!("{}: failed to stat/hash file: {}", path_str, e));
                return Ok(IndexPrepare::Failed);
            }
        };
        if manifest.is_unchanged(path_str, &record) {
            debug!("Skipping unchanged (checksum): {}", path_str);
            return Ok(IndexPrepare::Skip);
        }
        Ok(IndexPrepare::Ready {
            path_str: path_str.to_string(),
            content,
            record,
        })
    }

    async fn flush_index_window(
        &self,
        manifest: &mut IndexManifest,
        documents: &mut Vec<(String, String, Option<std::path::PathBuf>)>,
        indexed_records: &mut Vec<(String, manifest::IndexedFileRecord)>,
        files_indexed: &mut usize,
        total_chunks: &mut usize,
        failures: &mut Vec<String>,
    ) -> anyhow::Result<()> {
        let batch = std::mem::take(documents);
        let records = std::mem::take(indexed_records);
        let (chunks, docs, batch_failures, _token_usage) = self
            .rag_service
            .index_batch_with_paths(batch)
            .await
            .map_err(|e| anyhow::anyhow!("RAG index_batch failed: {}", e))?;

        failures.extend(batch_failures);
        *files_indexed = files_indexed.saturating_add(docs);
        *total_chunks = total_chunks.saturating_add(chunks);

        let upsert_paths: Vec<String> = records.iter().map(|(p, _)| p.clone()).collect();
        manifest.merge_from_indexed(records);
        self.persist_manifest(manifest, &upsert_paths, &[]).await;
        self.flush_deferred_ann_index_after_window().await;
        Ok(())
    }

    async fn flush_deferred_ann_index_after_window(&self) {
        if !crate::data::storage::vector::deferred_ann_index_enabled_from_env() {
            return;
        }
        let backend = self.rag_service.dripstone().backend();
        if let Err(error) =
            crate::data::storage::vector::flush_deferred_ann_index(backend.as_ref()).await
        {
            tracing::warn!("Workspace index: deferred ANN flush failed: {error}");
        }
    }

    async fn persist_manifest(
        &self,
        manifest: &IndexManifest,
        upsert_paths: &[String],
        removed_paths: &[String],
    ) {
        let PersistenceConfig::Disk { path } = &self.config.persistence else {
            return;
        };
        let result = if upsert_paths.is_empty() && removed_paths.is_empty() {
            Ok(())
        } else {
            manifest.save_delta(path, upsert_paths, removed_paths).await
        };
        if let Err(e) = result {
            tracing::warn!("Workspace index: failed to save manifest: {}", e);
        }
    }

    async fn read_file_for_indexing(&self, path: &std::path::Path) -> anyhow::Result<String> {
        read_index_file(&self.config, path).await
    }
}

/// The config for `repo`, a git worktree that discovery found under
/// `top`'s root: its own root and store. A repo nested below the top root
/// also gets the top root as its `read_root`.
fn repo_config(top: &WorkspaceIndexerConfig, repo: &std::path::Path) -> WorkspaceIndexerConfig {
    let workspace = git_repos::canonical_or_owned(&top.root_path);
    let mut config = top.clone();
    config.root_path = repo.to_path_buf();
    if repo != workspace {
        config.read_root = Some(workspace);
    }
    if matches!(top.persistence, PersistenceConfig::Disk { .. }) {
        config.persistence = PersistenceConfig::Disk {
            path: store_dir_for_repo(top, repo),
        };
    }
    config
}

/// Failure for a directory the index walk could not list (a link swapped
/// in, a permission or I/O error). The files indexed below it are kept.
pub(crate) const UNLISTED_KEPT: &str = "could not be listed; its indexed files are kept";

/// Prefix of every refused or failed read in [`read_index_file`].
pub(crate) const READ_FAILED: &str = "Failed to read file: ";

/// Read `path`, a file the index walk or the file watcher found, without
/// following a link.
///
/// A file in the workspace is opened below the canonical top workspace root
/// (`read_root` for a nested repository, else `root_path`), and for the top
/// repository only a file in its shared memory directory below that
/// directory, one component at a time, so a link in the project cannot pull
/// a file from outside into the index. `path` may name the root in its configured or
/// its canonical form.
pub(crate) async fn read_index_file(
    config: &WorkspaceIndexerConfig,
    path: &std::path::Path,
) -> anyhow::Result<String> {
    use std::io::Read;
    let max_size = if config.max_file_size > 0 {
        config.max_file_size
    } else {
        DEFAULT_MAX_FILE_SIZE
    };
    let mut bases = vec![
        config
            .read_root
            .clone()
            .unwrap_or_else(|| config.root_path.clone()),
    ];
    if config.read_root.is_none() {
        bases.push(crate::workspace::config::workspace_memory_markdown_dir(
            &config.root_path,
        ));
    }
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        for base in bases {
            let Ok(root) = base.canonicalize() else {
                continue;
            };
            let Ok(below) = path.strip_prefix(&base).or_else(|_| path.strip_prefix(&root)) else {
                continue;
            };
            let mut file = path::open_walked_file(&root, &root.join(below))
                .map_err(|e| anyhow::anyhow!("{READ_FAILED}{e}"))?;
            let len = file.metadata()?.len();
            if len > max_size as u64 {
                anyhow::bail!(
                    "File size validation failed: file is {len} bytes, over the {max_size} byte limit"
                );
            }
            let mut text = String::new();
            file.read_to_string(&mut text)
                .map_err(|e| anyhow::anyhow!("{READ_FAILED}{e}"))?;
            return Ok(text);
        }
        anyhow::bail!("{READ_FAILED}'{}' is outside the index roots", path.display())
    })
    .await
    .map_err(|e| anyhow::anyhow!("Index read task failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    #[derive(Debug)]
    struct CountingEmbedding {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl EmbeddingProvider for CountingEmbedding {
        async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(EmbeddingResponse {
                embeddings: request.texts.iter().map(|_| vec![0.1_f32; 8]).collect(),
                usage: None,
                model: "stub".to_string(),
            })
        }

        fn name(&self) -> &str {
            "stub"
        }
    }

    fn sorted_keys(manifest: &IndexManifest) -> Vec<String> {
        let mut keys: Vec<String> = manifest.indexed.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn indexer(
        root: &std::path::Path,
        persist: PersistenceConfig,
    ) -> (WorkspaceIndexer, Arc<CountingEmbedding>) {
        let embedding = Arc::new(CountingEmbedding {
            calls: AtomicUsize::new(0),
        });
        let rag = Arc::new(RagService::new(embedding.clone()));
        let config = WorkspaceIndexerConfig::new(root.to_path_buf())
            .file_extensions(vec!["rs".into(), "md".into()])
            .persistence(persist);
        (WorkspaceIndexer::new(rag, config), embedding)
    }

    #[tokio::test]
    async fn incremental_index_skips_unchanged_files_without_reembedding() {
        let dir = TempDir::new().expect("temp dir");
        let persist_dir = dir.path().join("index");
        tokio::fs::create_dir_all(&persist_dir)
            .await
            .expect("persist dir");
        tokio::fs::write(dir.path().join("a.rs"), "fn a() {}")
            .await
            .expect("write a");
        tokio::fs::write(dir.path().join("b.rs"), "fn b() {}")
            .await
            .expect("write b");

        let persist = PersistenceConfig::Disk {
            path: persist_dir.clone(),
        };
        let (first, embedding) = indexer(dir.path(), persist.clone());
        let first_result = first.index().await.expect("first index");
        assert_eq!(first_result.files_indexed, 2);
        let first_calls = embedding.calls.load(Ordering::SeqCst);
        assert!(first_calls > 0, "first pass should embed");

        let (second, embedding) = indexer(dir.path(), persist);
        let second_result = second.index().await.expect("second index");
        assert_eq!(second_result.files_indexed, 0);
        assert_eq!(
            embedding.calls.load(Ordering::SeqCst),
            0,
            "unchanged files must not be re-read into the embedder"
        );
    }

    #[tokio::test]
    async fn content_change_reindexes_only_the_modified_file() {
        let dir = TempDir::new().expect("temp dir");
        let persist_dir = dir.path().join("index");
        tokio::fs::create_dir_all(&persist_dir)
            .await
            .expect("persist dir");
        tokio::fs::write(dir.path().join("a.rs"), "fn a() {}")
            .await
            .expect("write a");
        tokio::fs::write(dir.path().join("b.rs"), "fn b() {}")
            .await
            .expect("write b");

        let persist = PersistenceConfig::Disk {
            path: persist_dir.clone(),
        };
        let (first, _) = indexer(dir.path(), persist.clone());
        first.index().await.expect("first index");

        tokio::fs::write(dir.path().join("b.rs"), "fn b_changed() {}")
            .await
            .expect("rewrite b");

        let (second, _) = indexer(dir.path(), persist);
        let second_result = second.index().await.expect("second index");
        assert_eq!(second_result.files_indexed, 1);
    }

    #[tokio::test]
    async fn windowed_index_covers_every_new_file() {
        let dir = TempDir::new().expect("temp dir");
        let persist_dir = dir.path().join("index");
        tokio::fs::create_dir_all(&persist_dir)
            .await
            .expect("persist dir");
        for i in 0..20 {
            tokio::fs::write(
                dir.path().join(format!("f{i}.rs")),
                format!("fn f{i}() {{}}"),
            )
            .await
            .expect("write file");
        }

        let persist = PersistenceConfig::Disk { path: persist_dir };
        let (indexer, embedding) = indexer(dir.path(), persist);
        let result = indexer.index().await.expect("windowed index");
        assert_eq!(result.files_indexed, 20);
        assert!(
            embedding.calls.load(Ordering::SeqCst) > 0,
            "windowed ingest should still embed"
        );
    }

    #[tokio::test]
    async fn index_nested_git_repos_writes_separate_manifests() {
        let dir = TempDir::new().expect("temp dir");
        let persist_dir = dir.path().join("index");
        tokio::fs::create_dir_all(&persist_dir)
            .await
            .expect("persist dir");
        tokio::fs::create_dir_all(dir.path().join(".git"))
            .await
            .expect("parent git");
        let child = dir.path().join("child");
        tokio::fs::create_dir_all(child.join(".git"))
            .await
            .expect("child git");
        tokio::fs::write(dir.path().join("root.rs"), "fn root() {}")
            .await
            .expect("write root");
        tokio::fs::write(child.join("child.rs"), "fn child() {}")
            .await
            .expect("write child");

        let persist = PersistenceConfig::Disk {
            path: persist_dir.clone(),
        };
        let (first, _) = indexer(dir.path(), persist.clone());
        let result = first.index().await.expect("nested index");
        assert_eq!(result.files_indexed, 2);

        let parent_manifest = crate::workspace::manifest::IndexManifest::load(&persist_dir)
            .await
            .expect("parent manifest");
        assert!(
            parent_manifest
                .indexed
                .keys()
                .any(|path| path.ends_with("root.rs"))
        );
        assert!(
            !parent_manifest
                .indexed
                .keys()
                .any(|path| path.ends_with("child.rs"))
        );

        let config = WorkspaceIndexerConfig::new(dir.path().to_path_buf()).persistence(persist);
        let child_store = store_dir_for_repo(&config, &child);
        let child_manifest = crate::workspace::manifest::IndexManifest::load(&child_store)
            .await
            .expect("child manifest");
        assert!(
            child_manifest
                .indexed
                .keys()
                .any(|path| path.ends_with("child.rs"))
        );
    }

    /// Memory belongs to the top repository: the top's memory note is
    /// indexed and a nested repository's own note is not.
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn index_reads_memory_for_the_top_repo_only() {
        let catalog = TempDir::new().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical");
        let child = root.join("child");
        for repo in [&root, &child] {
            std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        }
        std::fs::write(child.join("c.rs"), "fn c() {}").expect("child file");
        for (repo, note) in [(&root, "top.md"), (&child, "child.md")] {
            let memory = crate::workspace::config::workspace_memory_markdown_dir(repo);
            std::fs::create_dir_all(&memory).expect("memory dir");
            std::fs::write(memory.join(note), "a note").expect("note");
        }

        let (indexer, _) = indexer(&root, PersistenceConfig::InMemory);
        let result = indexer.index().await.expect("index");

        assert_eq!((result.files_indexed, result.failures), (2, Vec::new()));
    }

    /// A nested repository swapped for a link after an earlier index: the
    /// walk refuses its root, nothing is read from the link's target, and
    /// the files indexed in the earlier run stay in its manifest.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_nested_repo_swapped_for_a_link_keeps_its_index() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical").join("ws");
        let sub = root.join("sub");
        std::fs::create_dir_all(root.join(".git")).expect("git dir");
        std::fs::create_dir_all(sub.join(".git")).expect("sub git dir");
        std::fs::write(sub.join("a.rs"), "fn a() {}").expect("a");
        let outside = TempDir::new().expect("outside");
        std::fs::write(outside.path().join("secret.rs"), "fn secret() {}").expect("secret");
        let persist = PersistenceConfig::Disk {
            path: dir.path().join("catalog").join("top"),
        };
        let (top, _) = indexer(&root, persist);
        let config = repo_config(&top.config, &sub);
        let PersistenceConfig::Disk { path: store } = config.persistence.clone() else {
            panic!("disk persistence");
        };
        let nested = WorkspaceIndexer::new(Arc::clone(&top.rag_service), config);
        nested.index_tree().await.expect("first index");
        std::fs::rename(&sub, root.join("moved")).expect("move away");
        std::os::unix::fs::symlink(outside.path(), &sub).expect("swap");

        let result = nested.index_tree().await.expect("second index");
        let kept = crate::workspace::manifest::IndexManifest::load(&store)
            .await
            .expect("manifest");

        assert_eq!(
            (result.files_indexed, result.failures),
            (0, vec![format!("{}: {UNLISTED_KEPT}", sub.display())])
        );
        assert_eq!(
            sorted_keys(&kept),
            vec![sub.join("a.rs").display().to_string()]
        );
    }

    /// Entries below a directory the walk could not list survive a purge;
    /// entries the walk could have seen and did not are removed.
    #[tokio::test]
    async fn a_purge_keeps_entries_below_an_unlisted_directory() {
        let dir = TempDir::new().expect("temp dir");
        let (indexer, _) = indexer(dir.path(), PersistenceConfig::InMemory);
        let mut manifest = IndexManifest::new();
        let record = || crate::workspace::manifest::IndexedFileRecord {
            mtime: 1,
            size: 1,
            checksum: "c".to_string(),
        };
        let key = |path: &str| dir.path().join(path).display().to_string();
        manifest.merge_from_indexed([
            (key("sub/kept.rs"), record()),
            (key("gone.rs"), record()),
            ("/elsewhere/other.rs".to_string(), record()),
        ]);
        let mut failures = Vec::new();

        indexer
            .purge_stale_entries(
                dir.path(),
                &[],
                &[dir.path().join("sub")],
                &mut manifest,
                &mut failures,
            )
            .await;

        let mut expected = vec!["/elsewhere/other.rs".to_string(), key("sub/kept.rs")];
        expected.sort();
        assert_eq!(sorted_keys(&manifest), expected);
    }

    /// A single-repo workspace named through a link records its canonical
    /// root, the form the catalog compares against.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_single_repo_marker_records_the_canonical_root() {
        let dir = TempDir::new().expect("temp dir");
        let real = dir.path().canonicalize().expect("canonical").join("real");
        std::fs::create_dir_all(real.join(".git")).expect("git dir");
        std::fs::write(real.join("a.rs"), "fn a() {}").expect("a");
        let named = dir.path().join("named");
        std::os::unix::fs::symlink(&real, &named).expect("root link");
        let store = dir.path().join("store");
        let (indexer, _) = indexer(
            &named,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );

        indexer.index().await.expect("index");

        assert_eq!(
            std::fs::read_to_string(store.join("workspace_root")).expect("marker"),
            real.display().to_string()
        );
    }

    /// A workspace named through a link and by its real path shares one
    /// store and one key form: a file deleted between the two runs is
    /// purged.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_deleted_file_is_purged_whichever_root_form_names_the_workspace() {
        let dir = TempDir::new().expect("temp dir");
        let real = dir.path().canonicalize().expect("canonical").join("real");
        std::fs::create_dir_all(real.join(".git")).expect("git dir");
        std::fs::write(real.join("a.rs"), "fn a() {}").expect("a");
        std::fs::write(real.join("b.rs"), "fn b() {}").expect("b");
        let named = dir.path().join("named");
        std::os::unix::fs::symlink(&real, &named).expect("root link");
        let store = dir.path().join("store");
        let (by_name, _) = indexer(
            &named,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        by_name.index().await.expect("index by name");
        std::fs::remove_file(real.join("a.rs")).expect("delete a");

        let (by_path, _) = indexer(
            &real,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        by_path.index().await.expect("index by path");
        let kept = IndexManifest::load(&store).await.expect("manifest");

        assert_eq!(
            sorted_keys(&kept),
            vec![real.join("b.rs").display().to_string()]
        );
    }

    /// A file deleted between two runs that both name the workspace
    /// through a link leaves the index: the keys are canonical.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_deleted_file_is_purged_when_a_link_names_the_workspace() {
        let dir = TempDir::new().expect("temp dir");
        let real = dir.path().canonicalize().expect("canonical").join("real");
        std::fs::create_dir_all(real.join(".git")).expect("git dir");
        std::fs::write(real.join("a.rs"), "fn a() {}").expect("a");
        std::fs::write(real.join("b.rs"), "fn b() {}").expect("b");
        let named = dir.path().join("named");
        std::os::unix::fs::symlink(&real, &named).expect("root link");
        let store = dir.path().join("store");
        let run = || {
            indexer(
                &named,
                PersistenceConfig::Disk {
                    path: store.clone(),
                },
            )
            .0
        };
        run().index().await.expect("first index");
        std::fs::remove_file(real.join("a.rs")).expect("delete a");

        run().index().await.expect("second index");
        let kept = IndexManifest::load(&store).await.expect("manifest");

        assert_eq!(
            sorted_keys(&kept),
            vec![real.join("b.rs").display().to_string()]
        );
    }

    /// A nested repo that is hidden for one run and found again keeps its
    /// document: the top run's leftover key for it does not delete the copy
    /// the nested repo's manifest still lists.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_nested_repo_found_again_keeps_its_documents() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical").join("ws");
        let sub = root.join("sub");
        std::fs::create_dir_all(root.join(".git")).expect("git dir");
        std::fs::create_dir_all(sub.join(".git")).expect("sub git dir");
        std::fs::write(root.join("b.rs"), "fn b() {}").expect("b");
        std::fs::write(sub.join("a.rs"), "fn a() {}").expect("a");
        let store = dir.path().join("store");
        let (top, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        top.index().await.expect("multi-repo run");
        std::fs::rename(sub.join(".git"), dir.path().join("sub-git")).expect("hide sub git");
        top.index().await.expect("single-repo run");
        std::fs::rename(dir.path().join("sub-git"), sub.join(".git")).expect("restore sub git");

        top.index().await.expect("multi-repo run again");
        let PersistenceConfig::Disk { path: sub_store } =
            repo_config(&top.config, &sub).persistence
        else {
            panic!("sub store is on disk");
        };
        let sub_keys = sorted_keys(&IndexManifest::load(&sub_store).await.expect("sub manifest"));

        assert_eq!(top.rag_service.stats().await.expect("stats").1, 2);
        assert_eq!(sub_keys, vec![sub.join("a.rs").display().to_string()]);
    }

    /// Only the repos strictly below a repo, with a store on disk, are its
    /// nested stores.
    #[test]
    fn the_nested_stores_are_the_repos_strictly_below() {
        let path = std::path::PathBuf::from;
        let stores = vec![
            (path("/ws"), Some(path("/s/ws"))),
            (path("/ws/a"), Some(path("/s/a"))),
            (path("/ws/a/b"), Some(path("/s/b"))),
            (path("/ws/ab"), Some(path("/s/ab"))),
            (path("/ws/a/m"), None),
        ];

        assert_eq!(
            nested_stores_of(std::path::Path::new("/ws/a"), &stores),
            vec![(path("/ws/a/b"), path("/s/b"))]
        );
    }

    /// The docs count and the keys of `repo`'s own manifest.
    async fn docs_and_keys(top: &WorkspaceIndexer, repo: &std::path::Path) -> (usize, Vec<String>) {
        let PersistenceConfig::Disk { path } = repo_config(&top.config, repo).persistence else {
            panic!("store is on disk");
        };
        let keys = sorted_keys(&IndexManifest::load(&path).await.expect("manifest"));
        (top.rag_service.stats().await.expect("stats").1, keys)
    }

    /// A child repo that `.gitmodules` lists before its parent keeps its
    /// document: the parent runs and purges first.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_listed_before_its_parent_keeps_its_documents() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical").join("ws");
        let a = root.join("a");
        let b = a.join("b");
        std::fs::create_dir_all(root.join(".git")).expect("git dir");
        std::fs::create_dir_all(a.join(".git")).expect("a git dir");
        std::fs::create_dir_all(&b).expect("b");
        std::fs::write(root.join("r.rs"), "fn r() {}").expect("r");
        std::fs::write(a.join("x.rs"), "fn x() {}").expect("x");
        std::fs::write(b.join("y.rs"), "fn y() {}").expect("y");
        std::fs::write(root.join(".gitmodules"), "[submodule \"a\"]\n\tpath = a\n")
            .expect("gitmodules");
        let (top, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: dir.path().join("store"),
            },
        );
        top.index().await.expect("first run");
        std::fs::create_dir_all(b.join(".git")).expect("b git dir");
        std::fs::write(
            root.join(".gitmodules"),
            "[submodule \"b\"]\n\tpath = a/b\n[submodule \"a\"]\n\tpath = a\n",
        )
        .expect("gitmodules");

        let result = top.index().await.expect("second run");

        assert_eq!(result.failures, Vec::<String>::new());
        assert_eq!(
            docs_and_keys(&top, &b).await,
            (3, vec![b.join("y.rs").display().to_string()])
        );
    }

    /// Three nested repos hidden for one run and found again keep their
    /// documents: each leftover key goes to the deepest repo that holds it.
    #[cfg(unix)]
    #[tokio::test]
    async fn three_nested_repos_found_again_keep_their_documents() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical").join("ws");
        let a = root.join("a");
        let b = a.join("b");
        for repo in [&root, &a, &b] {
            std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        }
        std::fs::write(root.join("r.rs"), "fn r() {}").expect("r");
        std::fs::write(a.join("x.rs"), "fn x() {}").expect("x");
        std::fs::write(b.join("y.rs"), "fn y() {}").expect("y");
        let (top, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: dir.path().join("store"),
            },
        );
        top.index().await.expect("multi-repo run");
        std::fs::rename(a.join(".git"), dir.path().join("a-git")).expect("hide a git");
        std::fs::rename(b.join(".git"), dir.path().join("b-git")).expect("hide b git");
        top.index().await.expect("single-repo run");
        std::fs::rename(dir.path().join("a-git"), a.join(".git")).expect("restore a git");
        std::fs::rename(dir.path().join("b-git"), b.join(".git")).expect("restore b git");

        top.index().await.expect("multi-repo run again");

        assert_eq!(
            docs_and_keys(&top, &b).await,
            (3, vec![b.join("y.rs").display().to_string()])
        );
    }

    /// When the nested repo's manifest cannot drop a leftover key, the top
    /// run keeps the key and its document.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_leftover_key_the_nested_manifest_cannot_release_is_kept() {
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical").join("ws");
        let sub = root.join("sub");
        std::fs::create_dir_all(root.join(".git")).expect("git dir");
        std::fs::create_dir_all(sub.join(".git")).expect("sub git dir");
        std::fs::write(root.join("b.rs"), "fn b() {}").expect("b");
        std::fs::write(sub.join("a.rs"), "fn a() {}").expect("a");
        let store = dir.path().join("store");
        let (top, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        std::fs::rename(sub.join(".git"), dir.path().join("sub-git")).expect("hide sub git");
        top.index().await.expect("single-repo run");
        std::fs::rename(dir.path().join("sub-git"), sub.join(".git")).expect("restore sub git");
        let PersistenceConfig::Disk { path: sub_store } =
            repo_config(&top.config, &sub).persistence
        else {
            panic!("sub store is on disk");
        };
        if sub_store.exists() {
            std::fs::remove_dir_all(&sub_store).expect("clear sub store");
        }
        std::fs::create_dir_all(sub_store.parent().expect("parent")).expect("store parent");
        std::fs::write(&sub_store, "not a directory").expect("block sub store");

        let result = top.index().await;
        let top_keys = sorted_keys(&IndexManifest::load(&store).await.expect("top manifest"));

        assert!(result.is_err(), "the nested run cannot open its store");
        assert_eq!(top.rag_service.stats().await.expect("stats").1, 2);
        assert_eq!(
            top_keys,
            vec![
                root.join("b.rs").display().to_string(),
                sub.join("a.rs").display().to_string(),
            ]
        );
    }

    /// A workspace that gains a nested repo between two runs switches from
    /// single- to multi-repo mode; the top store keeps one key form and
    /// drops the files that now belong to the nested repo.
    #[cfg(unix)]
    #[tokio::test]
    async fn switching_to_multi_repo_mode_keeps_one_key_form() {
        let dir = TempDir::new().expect("temp dir");
        let real = dir.path().canonicalize().expect("canonical").join("real");
        std::fs::create_dir_all(real.join(".git")).expect("git dir");
        std::fs::create_dir_all(real.join("sub")).expect("sub");
        std::fs::write(real.join("b.rs"), "fn b() {}").expect("b");
        std::fs::write(real.join("sub/a.rs"), "fn a() {}").expect("a");
        let named = dir.path().join("named");
        std::os::unix::fs::symlink(&real, &named).expect("root link");
        let store = dir.path().join("store");
        let (top, _) = indexer(
            &named,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        top.index().await.expect("single-repo index");
        std::fs::create_dir_all(real.join("sub/.git")).expect("sub git dir");

        top.index().await.expect("multi-repo index");
        let kept = IndexManifest::load(&store).await.expect("manifest");

        assert_eq!(
            sorted_keys(&kept),
            vec![real.join("b.rs").display().to_string()]
        );
    }

    /// A memory note deleted since the last run of the top repo is purged.
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn a_deleted_memory_note_is_purged() {
        let catalog = TempDir::new().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical");
        std::fs::write(root.join("a.rs"), "fn a() {}").expect("a");
        let memory = crate::workspace::config::workspace_memory_markdown_dir(&root);
        std::fs::create_dir_all(&memory).expect("memory dir");
        std::fs::write(memory.join("note.md"), "a note").expect("note");
        let store = dir.path().join("store");
        let (indexer, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        indexer.index().await.expect("first index");
        std::fs::remove_file(memory.join("note.md")).expect("delete note");

        indexer.index().await.expect("second index");
        let kept = IndexManifest::load(&store).await.expect("manifest");

        assert_eq!(
            sorted_keys(&kept),
            vec![root.join("a.rs").display().to_string()]
        );
    }

    /// Keys an older run wrote under the root as configured are purged when
    /// the run walks the canonical root.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_key_in_the_configured_root_form_is_purged() {
        let dir = TempDir::new().expect("temp dir");
        let real = dir.path().canonicalize().expect("canonical").join("real");
        std::fs::create_dir_all(&real).expect("real");
        let named = dir.path().join("named");
        std::os::unix::fs::symlink(&real, &named).expect("root link");
        let (indexer, _) = indexer(&named, PersistenceConfig::InMemory);
        let mut manifest = IndexManifest::new();
        let record = crate::workspace::manifest::IndexedFileRecord {
            mtime: 1,
            size: 1,
            checksum: "c".to_string(),
        };
        manifest.merge_from_indexed([(named.join("gone.rs").display().to_string(), record)]);
        let mut failures = Vec::new();

        indexer
            .purge_stale_entries(&real, &[], &[], &mut manifest, &mut failures)
            .await;

        assert_eq!(sorted_keys(&manifest), Vec::<String>::new());
    }

    /// A nested repo's store is also that repo's own workspace store. Indexing
    /// the parent workspace leaves the notes the nested repo's own runs
    /// indexed from its memory directory.
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn indexing_the_parent_keeps_a_nested_repos_own_memory_notes() {
        let catalog = TempDir::new().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical");
        let sub = root.join("sub");
        for repo in [&root, &sub] {
            std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        }
        std::fs::write(sub.join("a.rs"), "fn a() {}").expect("a");
        let memory = crate::workspace::config::workspace_memory_markdown_dir(&sub);
        std::fs::create_dir_all(&memory).expect("memory dir");
        std::fs::write(memory.join("note.md"), "a note").expect("note");
        let store = crate::workspace::config::workspace_shared_storage_dir(&sub);
        let (own, _) = indexer(
            &sub,
            PersistenceConfig::Disk {
                path: store.clone(),
            },
        );
        own.index().await.expect("own index");
        let (parent, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: crate::workspace::config::workspace_shared_storage_dir(&root),
            },
        );
        assert_eq!(store_dir_for_repo(&parent.config, &sub), store);

        parent.index().await.expect("parent index");
        let kept = IndexManifest::load(&store).await.expect("manifest");

        let mut expected = vec![
            sub.join("a.rs").display().to_string(),
            memory.join("note.md").display().to_string(),
        ];
        expected.sort();
        assert_eq!(sorted_keys(&kept), expected);
    }

    /// The top memory directory cannot be listed (here it is a file): the
    /// memory notes indexed in an earlier run are kept.
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn an_unreadable_memory_dir_keeps_its_indexed_notes() {
        let catalog = TempDir::new().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let dir = TempDir::new().expect("temp dir");
        let root = dir.path().canonicalize().expect("canonical");
        std::fs::write(root.join("a.rs"), "fn a() {}").expect("a");
        let memory = crate::workspace::config::workspace_memory_markdown_dir(&root);
        std::fs::create_dir_all(&memory).expect("memory dir");
        std::fs::write(memory.join("note.md"), "a note").expect("note");
        let persist_dir = root.join(".index");
        let (indexer, _) = indexer(
            &root,
            PersistenceConfig::Disk {
                path: persist_dir.clone(),
            },
        );
        indexer.index().await.expect("first index");
        std::fs::remove_dir_all(&memory).expect("remove memory dir");
        std::fs::write(&memory, "not a dir").expect("memory file");

        indexer.index().await.expect("second index");
        let kept = crate::workspace::manifest::IndexManifest::load(&persist_dir)
            .await
            .expect("manifest");

        assert_eq!(sorted_keys(&kept), {
            let mut expected = vec![
                memory.join("note.md").display().to_string(),
                root.join("a.rs").display().to_string(),
            ];
            expected.sort();
            expected
        });
    }

    #[cfg(feature = "vector-store-sqlite")]
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn window_flush_drains_deferred_ann_queue() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        env.set("VECTOR_BACKEND_DIMENSION", "8");

        let dir = TempDir::new().expect("tempdir");
        let persist_dir = dir.path().join("index");
        tokio::fs::create_dir_all(&persist_dir)
            .await
            .expect("persist dir");
        for i in 0..12 {
            tokio::fs::write(
                dir.path().join(format!("f{i}.rs")),
                format!("fn f{i}() {{}}"),
            )
            .await
            .expect("write file");
        }

        let config = WorkspaceIndexerConfig::new(dir.path().to_path_buf())
            .file_extensions(vec!["rs".into()])
            .persistence(PersistenceConfig::Disk { path: persist_dir });
        let backend = crate::workspace::build_workspace_backend(&config, Some(8))
            .await
            .expect("backend");
        let embedding = Arc::new(CountingEmbedding {
            calls: AtomicUsize::new(0),
        });
        let rag = Arc::new(RagService::new_with_backend(
            embedding, backend, false, false, false,
        ));
        let indexer = WorkspaceIndexer::new(rag.clone(), config);
        let result = indexer.index().await.expect("index");
        assert_eq!(result.files_indexed, 12);
        assert_eq!(
            rag.dripstone().backend().pending_ann_index_count(),
            0,
            "window flush should drain deferred ANN queue after ingest"
        );
    }
}

#[cfg(test)]
#[cfg(unix)]
mod link_tests {
    use super::{READ_FAILED, read_index_file, repo_config};
    use crate::workspace::config::{PersistenceConfig, WorkspaceIndexerConfig};

    /// A workspace with `ok.rs` and `leak.rs`, a link to a file outside it,
    /// and an indexer config for it in its non-canonical form.
    fn workspace() -> (tempfile::TempDir, tempfile::TempDir, WorkspaceIndexerConfig) {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.rs"), "fn secret() {}").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        std::fs::write(ws.path().join("ok.rs"), "fn ok() {}").expect("ok");
        std::os::unix::fs::symlink(outside.path().join("secret.rs"), ws.path().join("leak.rs"))
            .expect("file link");
        let config = WorkspaceIndexerConfig::new(ws.path().to_path_buf())
            .persistence(PersistenceConfig::InMemory);
        (outside, ws, config)
    }

    #[tokio::test]
    async fn an_indexed_read_takes_a_workspace_file_by_either_root_form() {
        let (_outside, ws, config) = workspace();
        let canonical = ws.path().canonicalize().expect("canonical workspace");

        let named = read_index_file(&config, &ws.path().join("ok.rs"))
            .await
            .expect("named");
        let resolved = read_index_file(&config, &canonical.join("ok.rs"))
            .await
            .expect("canonical");

        assert_eq!(named, "fn ok() {}");
        assert_eq!(resolved, "fn ok() {}");
    }

    /// A nested repository swapped for a link after discovery listed it: a
    /// read of one of its files is refused, not taken from the link's target.
    #[tokio::test]
    async fn an_indexed_read_in_a_swapped_nested_repo_is_refused() {
        let (outside, ws, top) = workspace();
        let sub = ws.path().canonicalize().expect("canonical").join("sub");
        let config = repo_config(&top, &sub);
        std::os::unix::fs::symlink(outside.path(), &sub).expect("swap");

        let read = read_index_file(&config, &sub.join("secret.rs")).await;

        let err = read.expect_err("a swapped nested repo must not be read");
        assert!(
            err.to_string()
                .starts_with(&format!("{READ_FAILED}refusing to follow symlink at 'sub'")),
            "{err}"
        );
    }

    /// A nested repository indexes no memory of its own: a file in its
    /// memory directory is outside its read roots.
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn a_nested_repo_reads_no_memory_of_its_own() {
        let catalog = tempfile::tempdir().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let (_outside, ws, top) = workspace();
        let sub = ws.path().canonicalize().expect("canonical").join("sub");
        std::fs::create_dir(&sub).expect("sub");
        let memory = crate::workspace::config::workspace_memory_markdown_dir(&sub);
        std::fs::create_dir_all(&memory).expect("memory dir");
        let note = memory.join("note.md");
        std::fs::write(&note, "nested note").expect("note");

        let read = read_index_file(&repo_config(&top, &sub), &note)
            .await
            .map_err(|err| err.to_string());

        assert_eq!(
            read,
            Err(format!(
                "{READ_FAILED}'{}' is outside the index roots",
                note.display()
            ))
        );
    }

    #[tokio::test]
    async fn an_indexed_read_refuses_a_file_that_is_a_link() {
        let (_outside, ws, config) = workspace();

        let read = read_index_file(&config, &ws.path().join("leak.rs")).await;

        let err = read.expect_err("a link must not be read");
        assert!(err.to_string().starts_with(READ_FAILED), "{err}");
        assert!(!err.to_string().contains("secret"), "{err}");
    }
}
