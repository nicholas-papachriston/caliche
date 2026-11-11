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
use tokio::fs;
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

/// Indexes a workspace directory into a RAG service.
pub struct WorkspaceIndexer {
    rag_service: Arc<RagService>,
    config: WorkspaceIndexerConfig,
}

impl WorkspaceIndexer {
    /// Create a new workspace indexer.
    pub fn new(rag_service: Arc<RagService>, config: WorkspaceIndexerConfig) -> Self {
        Self {
            rag_service,
            config,
        }
    }

    /// Index the configured root directory.
    ///
    /// Nested git worktrees get their own store directory and manifest. The RAG
    /// backend routes writes by file path when it is federated.
    pub async fn index(&self) -> anyhow::Result<WorkspaceIndexResult> {
        let repos = git_repos::discover_git_repos(&self.config.root_path);
        if repos.len() <= 1 {
            self.record_root_marker(&self.config.root_path);
            return self.index_tree().await;
        }

        let mut files_indexed = 0usize;
        let mut total_chunks = 0usize;
        let mut failures = Vec::new();
        for repo in repos {
            let mut repo_config = self.config.clone();
            repo_config.root_path = repo.clone();
            if matches!(self.config.persistence, PersistenceConfig::Disk { .. }) {
                let store_dir = store_dir_for_repo(&self.config, &repo);
                self.record_root_marker_at(&store_dir, &repo);
                repo_config.persistence = PersistenceConfig::Disk { path: store_dir };
            }
            let repo_indexer = Self::new(Arc::clone(&self.rag_service), repo_config);
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
        let mut paths = self.collect_index_paths().await?;
        if let Ok(mut memory_files) =
            crate::workspace::list_workspace_memory_markdown_files(root).await
        {
            paths.append(&mut memory_files);
        }

        let mut manifest = match &self.config.persistence {
            PersistenceConfig::Disk { path } => {
                IndexManifest::load_with_fingerprint(path, self.config.index_fingerprint.as_deref())
                    .await?
            }
            PersistenceConfig::InMemory => IndexManifest::new(),
        };

        let mut failures = Vec::new();
        self.purge_stale_entries(&paths, &mut manifest, &mut failures)
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

    async fn collect_index_paths(&self) -> anyhow::Result<Vec<std::path::PathBuf>> {
        let root_clone = self.config.root_path.clone();
        let max_size = self.config.max_file_size;
        let exts_clone = self.config.resolved_extensions();
        let excludes = self.config.exclude_patterns.clone();
        tokio::task::spawn_blocking(move || {
            gitignore::collect_files(
                &root_clone,
                Some(max_size as u64),
                &exts_clone,
                None,
                &excludes,
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("Workspace walk task failed: {}", e))
    }

    async fn purge_stale_entries(
        &self,
        paths: &[std::path::PathBuf],
        manifest: &mut IndexManifest,
        failures: &mut Vec<String>,
    ) {
        let current_paths: HashSet<String> = paths
            .iter()
            .filter_map(|path| path.to_str().map(str::to_string))
            .collect();
        let stale_paths: Vec<String> = manifest
            .indexed
            .keys()
            .filter(|path| !current_paths.contains(*path))
            .cloned()
            .collect();
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
        let max_size = if self.config.max_file_size > 0 {
            self.config.max_file_size
        } else {
            DEFAULT_MAX_FILE_SIZE
        };

        path::validate_file_size(path, max_size as u64)
            .map_err(|e| anyhow::anyhow!("File size validation failed: {}", e))?;

        fs::read_to_string(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))
    }
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
