//! Fan-out vector backend over per-repo SQLite stores.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use dripstone::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use dripstone::errors::DripstoneError;
use dripstone::search::FusionConfig;
use dripstone::traits::Backend;
use futures_util::future::join_all;

use super::git_repos::{belongs_to_repo, canonical_or_owned};

/// One git worktree and its durable vector store.
#[derive(Clone)]
pub struct WorkspaceStoreMember {
    pub root: PathBuf,
    pub backend: Arc<dyn Backend>,
}

impl std::fmt::Debug for WorkspaceStoreMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceStoreMember")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

/// Search every member store. Route writes to the worktree that owns the document path.
#[derive(Debug, Clone)]
pub struct FederatedWorkspaceBackend {
    members: Vec<WorkspaceStoreMember>,
}

impl FederatedWorkspaceBackend {
    /// Build a federated backend. Empty `members` is rejected.
    ///
    /// # Errors
    ///
    /// Returns when `members` is empty.
    pub fn new(members: Vec<WorkspaceStoreMember>) -> Result<Self, DripstoneError> {
        if members.is_empty() {
            return Err(DripstoneError::pipeline(
                "FederatedWorkspaceBackend requires at least one workspace store",
            ));
        }
        Ok(Self { members })
    }

    fn member_index_for_doc(&self, doc_id: &str) -> usize {
        let path = Path::new(doc_id);
        self.members
            .iter()
            .enumerate()
            .filter(|(_, member)| {
                belongs_to_repo(path, &member.root) || path.starts_with(&member.root)
            })
            .max_by_key(|(_, member)| member.root.as_os_str().len())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn backend_at(&self, index: usize) -> &Arc<dyn Backend> {
        &self.members[index.min(self.members.len().saturating_sub(1))].backend
    }
}

fn merge_hits(mut hits: Vec<SearchResult>, limit: usize) -> Vec<SearchResult> {
    hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    hits.truncate(limit.max(1));
    hits
}

#[async_trait]
impl Backend for FederatedWorkspaceBackend {
    fn capabilities(&self) -> Capabilities {
        self.members[0].backend.capabilities()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut buckets: Vec<Vec<Chunk>> = vec![Vec::new(); self.members.len()];
        for chunk in chunks {
            buckets[self.member_index_for_doc(&chunk.doc_id)].push(chunk.clone());
        }
        for (index, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            self.backend_at(index).upsert(&bucket).await?;
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let preferred = self.member_index_for_doc(id);
        if let Some(chunk) = self.backend_at(preferred).get(id, namespace).await? {
            return Ok(Some(chunk));
        }
        for (index, member) in self.members.iter().enumerate() {
            if index == preferred {
                continue;
            }
            if let Some(chunk) = member.backend.get(id, namespace).await? {
                return Ok(Some(chunk));
            }
        }
        Ok(None)
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let preferred = self.member_index_for_doc(id);
        self.backend_at(preferred).delete(id, namespace).await?;
        for (index, member) in self.members.iter().enumerate() {
            if index == preferred {
                continue;
            }
            member.backend.delete(id, namespace).await?;
        }
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let preferred = self.member_index_for_doc(doc_id);
        let mut deleted = self
            .backend_at(preferred)
            .delete_document(doc_id, namespace)
            .await?;
        for (index, member) in self.members.iter().enumerate() {
            if index == preferred {
                continue;
            }
            deleted += member.backend.delete_document(doc_id, namespace).await?;
        }
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let limit = query.limit.max(1);
        let futs = self.members.iter().map(|member| {
            let backend = Arc::clone(&member.backend);
            let query = query.clone();
            async move { backend.search(&query).await }
        });
        let mut hits = Vec::new();
        for result in join_all(futs).await {
            match result {
                Ok(member_hits) => hits.extend(member_hits),
                Err(error) => tracing::warn!(error = %error, "workspace store search failed"),
            }
        }
        Ok(merge_hits(hits, limit))
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let limit = query.limit.max(1);
        let futs = self.members.iter().map(|member| {
            let backend = Arc::clone(&member.backend);
            let query = query.clone();
            async move { backend.search_full_text(&query).await }
        });
        let mut hits = Vec::new();
        for result in join_all(futs).await {
            match result {
                Ok(member_hits) => hits.extend(member_hits),
                Err(_) => {}
            }
        }
        Ok(merge_hits(hits, limit))
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let limit = dense.limit.max(1);
        let futs = self.members.iter().map(|member| {
            let backend = Arc::clone(&member.backend);
            let dense = dense.clone();
            let lexical = lexical.clone();
            let fusion = fusion.clone();
            async move {
                backend
                    .search_hybrid_full_text(&dense, &lexical, &fusion)
                    .await
            }
        });
        let mut hits = Vec::new();
        for result in join_all(futs).await {
            match result {
                Ok(member_hits) => hits.extend(member_hits),
                Err(_) => {}
            }
        }
        Ok(merge_hits(hits, limit))
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let mut names = BTreeSet::new();
        for member in &self.members {
            names.extend(member.backend.namespaces().await?);
        }
        Ok(names.into_iter().collect())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        for member in &self.members {
            member.backend.delete_namespace(namespace).await?;
        }
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let mut total = 0usize;
        for member in &self.members {
            total = total.saturating_add(member.backend.count(namespace).await?);
        }
        Ok(total)
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let mut chunks = Vec::new();
        for member in &self.members {
            if limit.is_some_and(|max| chunks.len() >= max) {
                break;
            }
            let remaining = limit.map(|max| max.saturating_sub(chunks.len()));
            chunks.extend(member.backend.list(namespace, remaining).await?);
        }
        if let Some(max) = limit {
            chunks.truncate(max);
        }
        Ok(chunks)
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        for member in &self.members {
            member.backend.flush().await?;
        }
        Ok(())
    }

    async fn flush_pending_ann_index(&self) -> Result<(), DripstoneError> {
        for member in &self.members {
            member.backend.flush_pending_ann_index().await?;
        }
        Ok(())
    }

    fn pending_ann_index_count(&self) -> usize {
        self.members
            .iter()
            .map(|member| member.backend.pending_ann_index_count())
            .sum()
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        for member in &self.members {
            member.backend.close().await?;
        }
        Ok(())
    }
}

/// Recorded workspace roots that sit under `workspace_root` in `catalog_dir`.
pub fn recorded_workspace_roots(catalog_dir: &Path, workspace_root: &Path) -> Vec<PathBuf> {
    let canonical_workspace = canonical_or_owned(workspace_root);
    let entries = match std::fs::read_dir(catalog_dir) {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    let mut roots = Vec::new();
    for entry in entries.flatten() {
        let marker = entry.path().join("workspace_root");
        let Ok(contents) = std::fs::read_to_string(marker) else {
            continue;
        };
        let recorded = PathBuf::from(contents.trim());
        if (recorded == canonical_workspace || recorded.starts_with(&canonical_workspace))
            && !roots.iter().any(|existing| existing == &recorded)
        {
            roots.push(recorded);
        }
    }
    roots
}

/// Write the canonical worktree path into a store directory.
pub fn write_workspace_root_marker(store_dir: &Path, repo_root: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(store_dir)?;
    let canonical = canonical_or_owned(repo_root);
    std::fs::write(
        store_dir.join("workspace_root"),
        canonical.to_string_lossy().as_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::{FederatedWorkspaceBackend, WorkspaceStoreMember};
    use dripstone::backend::memory::MemoryBackend;
    use dripstone::core::{Chunk, Metric, SearchQuery};
    use dripstone::traits::Backend;
    use std::sync::Arc;

    fn chunk(doc_id: &str, score_dim: f32) -> Chunk {
        Chunk {
            id: format!("{doc_id}#0"),
            doc_id: doc_id.to_string(),
            content: doc_id.to_string(),
            vector: Arc::from(vec![score_dim; 4]),
            start_offset: 0,
            end_offset: 1,
            chunk_index: 0,
            total_chunks: 1,
            metadata: serde_json::json!({}),
            namespace: None,
            multi_vectors: None,
        }
    }

    #[tokio::test]
    async fn federated_search_merges_member_hits() {
        let left = Arc::new(MemoryBackend::new()) as Arc<dyn Backend>;
        let right = Arc::new(MemoryBackend::new()) as Arc<dyn Backend>;
        left.upsert(&[chunk("/tmp/a/file.rs", 0.2)])
            .await
            .expect("left upsert");
        right
            .upsert(&[chunk("/tmp/b/file.rs", 0.9)])
            .await
            .expect("right upsert");
        let federated = FederatedWorkspaceBackend::new(vec![
            WorkspaceStoreMember {
                root: std::path::PathBuf::from("/tmp/a"),
                backend: left,
            },
            WorkspaceStoreMember {
                root: std::path::PathBuf::from("/tmp/b"),
                backend: right,
            },
        ])
        .expect("federated");
        let hits = federated
            .search(&SearchQuery {
                vector: Arc::from(vec![0.9; 4]),
                limit: 2,
                metric: Metric::Cosine,
                namespace: None,
                filter: None,
                hybrid_weight: None,
                sparse_vector: None,
                query_multi_vectors: None,
                maxsim: None,
                include_content: true,
                include_vectors: false,
            })
            .await
            .expect("search");
        assert_eq!(hits.len(), 2);
        assert!(hits[0].score >= hits[1].score);
    }

    #[tokio::test]
    async fn federated_upsert_routes_by_doc_path() {
        let dir = tempfile::TempDir::new().expect("temp");
        let left_root = dir.path().join("a");
        let right_root = dir.path().join("b");
        let left = Arc::new(MemoryBackend::new()) as Arc<dyn Backend>;
        let right = Arc::new(MemoryBackend::new()) as Arc<dyn Backend>;
        let federated = FederatedWorkspaceBackend::new(vec![
            WorkspaceStoreMember {
                root: left_root.clone(),
                backend: Arc::clone(&left),
            },
            WorkspaceStoreMember {
                root: right_root.clone(),
                backend: Arc::clone(&right),
            },
        ])
        .expect("federated");
        let doc = right_root.join("file.rs");
        federated
            .upsert(&[chunk(&doc.to_string_lossy(), 0.5)])
            .await
            .expect("upsert");
        assert!(left.list(None, None).await.expect("left list").is_empty());
        assert_eq!(right.list(None, None).await.expect("right list").len(), 1);
    }
}
