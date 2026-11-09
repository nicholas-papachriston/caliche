//! Open per-repo SQLite vector stores and federate them for parent workspaces.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dripstone::backend::memory::MemoryBackend;
use dripstone::backend::sqlite::{MemoryStrategy, SqliteBackend, SqliteBackendConfig};
use dripstone::traits::Backend;

use super::config::{
    DEFAULT_SQLITE_MAX_CONNECTIONS, PersistenceConfig, WorkspaceIndexerConfig,
    sqlite_cache_size_from_env, store_dir_for_repo,
};
use super::federated::{
    FederatedWorkspaceBackend, WorkspaceStoreMember, recorded_workspace_roots,
    write_workspace_root_marker,
};
use super::git_repos;
use crate::data::storage::vector::indexed::wrap_workspace_backend_with_ann_index;

/// Catalog directory that contains per-repo store folders.
#[must_use]
pub fn catalog_dir_from_store(store_dir: &Path) -> PathBuf {
    store_dir
        .parent()
        .map_or_else(|| store_dir.to_path_buf(), Path::to_path_buf)
}

/// Git worktrees that should participate in this workspace search.
#[must_use]
pub fn workspace_store_roots(config: &WorkspaceIndexerConfig) -> Vec<PathBuf> {
    let PersistenceConfig::Disk { path } = &config.persistence else {
        return vec![git_repos::canonical_or_owned(&config.root_path)];
    };
    let mut roots = git_repos::discover_git_repos(&config.root_path);
    // A declared submodule list is the set of stores. Stale catalog markers
    // from an older tree walk must not reopen extra SQLite pools.
    if git_repos::uses_declared_submodules(&config.root_path) {
        return roots;
    }
    let catalog = catalog_dir_from_store(path);
    for recorded in recorded_workspace_roots(&catalog, &config.root_path) {
        if !roots.iter().any(|existing| existing == &recorded) {
            roots.push(recorded);
        }
    }
    roots
}

/// Open one SQLite workspace store at `store_dir`.
///
/// # Errors
///
/// Returns when the directory cannot be created or SQLite fails to open.
pub async fn open_sqlite_workspace_backend(store_dir: &Path) -> anyhow::Result<Arc<dyn Backend>> {
    tokio::fs::create_dir_all(store_dir).await?;
    let cache_size = sqlite_cache_size_from_env();
    let sqlite_config = SqliteBackendConfig {
        db_path: store_dir.join("vectors.db"),
        memory_strategy: MemoryStrategy::WriteThrough {
            max_cache_size: cache_size,
            ttl_secs: None,
        },
        max_connections: DEFAULT_SQLITE_MAX_CONNECTIONS,
        table_name: "workspace_vectors".to_string(),
        ensure_schema: true,
    };
    let backend = SqliteBackend::new(sqlite_config).await?;
    Ok(Arc::new(backend))
}

fn wrap_store_backend(
    backend: Arc<dyn Backend>,
    embed_dimension: Option<usize>,
) -> anyhow::Result<Arc<dyn Backend>> {
    let backend = backend as Arc<dyn Backend + Send + Sync>;
    let wrapped = wrap_workspace_backend_with_ann_index(backend, embed_dimension)?;
    Ok(wrapped as Arc<dyn Backend>)
}

/// Build the vector backend for a workspace agent.
///
/// Disk mode opens one SQLite file per discovered or previously recorded git
/// worktree and federates search when more than one store exists.
///
/// When `CALICHE_DEFERRED_ANN_INDEX=true` and `embed_dimension` is set (or
/// `VECTOR_BACKEND_DIMENSION` is configured), each store is wrapped with an
/// in-memory HNSW index using deferred updates for bulk ingest.
///
/// # Errors
///
/// Returns when a store cannot be created or opened.
pub async fn build_workspace_backend(
    config: &WorkspaceIndexerConfig,
    embed_dimension: Option<usize>,
) -> anyhow::Result<Arc<dyn Backend>> {
    match &config.persistence {
        PersistenceConfig::InMemory => Ok(Arc::new(MemoryBackend::new())),
        PersistenceConfig::Disk { .. } => {
            let roots = workspace_store_roots(config);
            let mut members = Vec::with_capacity(roots.len());
            for repo in roots {
                let store_dir = store_dir_for_repo(config, &repo);
                write_workspace_root_marker(&store_dir, &repo)?;
                let backend = open_sqlite_workspace_backend(&store_dir).await?;
                let backend = wrap_store_backend(backend, embed_dimension)?;
                members.push(WorkspaceStoreMember {
                    root: repo,
                    backend,
                });
            }
            match members.len() {
                0 => anyhow::bail!("workspace store list is empty"),
                1 => Ok(members.remove(0).backend),
                _ => Ok(Arc::new(FederatedWorkspaceBackend::new(members)?)),
            }
        }
    }
}

#[cfg(test)]
mod declared_root_tests {
    use super::{workspace_store_roots, write_workspace_root_marker};
    use crate::workspace::config::{PersistenceConfig, WorkspaceIndexerConfig};
    use std::fs;

    fn init_git(dir: &std::path::Path) {
        fs::create_dir_all(dir.join(".git")).expect("git dir");
    }

    #[test]
    fn declared_gitmodules_ignores_stale_catalog_markers() {
        let dir = tempfile::tempdir().expect("temp");
        init_git(dir.path());
        let listed = dir.path().join("listed");
        init_git(&listed);
        let extra = dir.path().join("extra");
        init_git(&extra);
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"listed\"]\n\tpath = listed\n",
        )
        .expect("gitmodules");

        let catalog = tempfile::tempdir().expect("catalog");
        let stale = catalog.path().join("stale");
        write_workspace_root_marker(&stale, &extra).expect("marker");
        let config = WorkspaceIndexerConfig::new(dir.path().to_path_buf()).persistence(
            PersistenceConfig::Disk {
                path: catalog.path().join("primary"),
            },
        );

        let roots = workspace_store_roots(&config);
        assert!(roots.iter().any(|path| path.ends_with("listed")));
        assert!(roots.iter().all(|path| !path.ends_with("extra")));
    }
}

#[cfg(all(test, feature = "tools-rag"))]
mod deferred_ann_tests {
    use super::*;
    use dripstone::Chunk;
    use dripstone::core::Metric;
    use dripstone::core::SearchQuery;

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn disk_workspace_wraps_sqlite_with_deferred_ann_when_enabled() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_DEFERRED_ANN_INDEX", "true");
        env.set("VECTOR_BACKEND_DIMENSION", "4");

        let dir = tempfile::tempdir().expect("tempdir");
        let config = WorkspaceIndexerConfig::new(dir.path().to_path_buf()).persistence(
            PersistenceConfig::Disk {
                path: dir.path().join("store"),
            },
        );
        let backend = build_workspace_backend(&config, None)
            .await
            .expect("workspace backend");
        let chunk = Chunk::new(
            "doc",
            0,
            "workspace chunk",
            vec![1.0_f32, 0.0, 0.0, 0.0],
            0,
            10,
            1,
            serde_json::json!({}),
            None,
        );
        backend.upsert(&[chunk]).await.expect("upsert");
        let query = SearchQuery::new(vec![1.0_f32, 0.0, 0.0, 0.0])
            .metric(Metric::Cosine)
            .limit(1);
        let hits = backend.search(&query).await.expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id, "doc#0");
    }
}
