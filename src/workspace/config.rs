//! Workspace indexer configuration.
//!
//! Environment variables (when using `WorkspaceEmbeddingMode::Auto`):
//! - `CALICHE_EMBED_PERSIST`: set to `"false"` to use in-memory only (default: disk with SQLite tiered storage)
//! - `CALICHE_EMBED_PATH`: catalog directory for per-repo stores (default: `~/.caliche/workspaces`)
//! - `CALICHE_LINE_LEVEL_BM25`: set to `"true"` to enable line-level BM25 for hybrid search
//! - `CALICHE_RAG_LATE_INTERACTION`: set to `"true"` to index token multi-vectors and search with MaxSim / PLAID
//! - `CALICHE_INDEX_CONCURRENCY`: embed jobs in flight (adaptive default: leave one core free, cap 4; max 32)
//! - `CALICHE_DEFERRED_ANN_INDEX`: set to `"true"` to wrap persisted workspace
//!   SQLite stores with deferred in-memory HNSW indexing during bulk ingest.
//!   Tail scan stays enabled for read-your-write RAG recall.
//! - `CALICHE_DEFERRED_FLUSH_THRESHOLD`: auto-flush pending ANN queue after N
//!   upserts (default 256; `0` = manual flush only).
//! - `CALICHE_DEFERRED_SEARCH_FLUSH_THRESHOLD`: flush pending queue at search
//!   when pending count reaches N (`0` = disabled).
//! - `CALICHE_DEFERRED_MAX_TAIL_SCAN`: cap tail scan to the last N pending
//!   records (`0` = full queue; lowers search cost when visibility lag is bounded).
//! - `VECTOR_BACKEND_DIMENSION`: fallback embedding dimension for deferred ANN (default 1536)
//! - `CALICHE_SQLITE_CACHE_SIZE`: in-process SQLite vector cache entries (default 2048)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Default maximum file size for indexing (100KB).
pub const DEFAULT_MAX_FILE_SIZE: usize = 102_400;

/// Default in-process SQLite vector cache size for persisted workspace embeddings.
/// 2048 entries is enough for interactive retrieval without pinning tens of thousands of vectors.
pub const DEFAULT_SQLITE_CACHE_SIZE: usize = 2_048;

/// SQLite pool size for one workspace store.
///
/// A repository with many worktrees opens one pool per worktree. One
/// connection per file keeps startup from spawning a worker thread per extra
/// pool slot. SQLite writes are serial, so a second idle connection does not
/// make the first command faster.
pub const DEFAULT_SQLITE_MAX_CONNECTIONS: u32 = 1;

/// Default text-based file extensions for workspace indexing.
pub fn default_file_extensions() -> Vec<String> {
    vec![
        "rs".into(),
        "py".into(),
        "ts".into(),
        "tsx".into(),
        "js".into(),
        "jsx".into(),
        "go".into(),
        "java".into(),
        "kt".into(),
        "c".into(),
        "cpp".into(),
        "h".into(),
        "hpp".into(),
        "md".into(),
        "json".into(),
        "yaml".into(),
        "yml".into(),
        "toml".into(),
        "sql".into(),
        "sh".into(),
        "bash".into(),
        "html".into(),
        "css".into(),
        "scss".into(),
        "vue".into(),
        "svelte".into(),
        "rb".into(),
        "php".into(),
        "swift".into(),
        "m".into(),
        "mm".into(),
        "cs".into(),
        "fs".into(),
        "fsx".into(),
        "nim".into(),
        "zig".into(),
        "lua".into(),
        "r".into(),
        "R".into(),
    ]
}

/// Compute a stable workspace hash for global shared storage paths.
pub fn workspace_hash(root_path: &Path) -> String {
    let normalized = std::fs::canonicalize(root_path).unwrap_or_else(|_| root_path.to_path_buf());
    let path_str = normalized.to_string_lossy().to_string();
    let mut hasher = DefaultHasher::new();
    path_str.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Catalog directory that holds one store folder per git worktree.
#[must_use]
pub fn workspace_catalog_root() -> PathBuf {
    if let Ok(path) = std::env::var("CALICHE_EMBED_PATH") {
        let trimmed = path.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".caliche").join("workspaces")
}

/// Resolve the global per-workspace storage directory under the embedding catalog.
pub fn workspace_shared_storage_dir(root_path: &Path) -> PathBuf {
    workspace_catalog_root().join(workspace_hash(root_path))
}

/// Store directory for `repo_root` given the agent workspace persistence config.
///
/// The current workspace uses `config.persistence`. Nested git worktrees use sibling
/// folders named by [`workspace_hash`] so a parent agent can search every repo DB.
#[must_use]
pub fn store_dir_for_repo(config: &WorkspaceIndexerConfig, repo_root: &Path) -> PathBuf {
    match &config.persistence {
        PersistenceConfig::InMemory => workspace_shared_storage_dir(repo_root),
        PersistenceConfig::Disk { path } => {
            let canonical_repo = super::git_repos::canonical_or_owned(repo_root);
            let canonical_workspace = super::git_repos::canonical_or_owned(&config.root_path);
            if canonical_repo == canonical_workspace {
                path.clone()
            } else {
                path.parent()
                    .map_or_else(|| path.clone(), Path::to_path_buf)
                    .join(workspace_hash(&canonical_repo))
            }
        }
    }
}

/// Store directory that owns `file` when persistence is on disk.
#[must_use]
pub fn owning_store_dir(config: &WorkspaceIndexerConfig, file: &Path) -> Option<PathBuf> {
    let PersistenceConfig::Disk { .. } = &config.persistence else {
        return None;
    };
    let workspace = super::git_repos::canonical_or_owned(&config.root_path);
    let repo = super::git_repos::owning_git_root(file)
        .filter(|repo| *repo == workspace || repo.starts_with(&workspace))
        .unwrap_or_else(|| config.root_path.clone());
    Some(store_dir_for_repo(config, &repo))
}

/// Git root used for ignore rules for `file` under this workspace.
#[must_use]
pub fn index_root_for_path(config: &WorkspaceIndexerConfig, file: &Path) -> PathBuf {
    let workspace = super::git_repos::canonical_or_owned(&config.root_path);
    super::git_repos::owning_git_root(file)
        .filter(|repo| *repo == workspace || repo.starts_with(&workspace))
        .unwrap_or_else(|| config.root_path.clone())
}

/// Resolve per-workspace markdown memory directory under shared storage.
pub fn workspace_memory_markdown_dir(root_path: &Path) -> PathBuf {
    workspace_shared_storage_dir(root_path).join("memory")
}

/// Persistence mode for workspace embeddings.
///
/// - **Disk** (default from `WorkspaceIndexerConfig::new`): Embeddings are stored in SQLite
///   at `path/vectors.db` with an in-memory write-through cache. Incremental indexing metadata
///   is persisted to `path/indexing.sqlite`.
/// - **InMemory**: No DB; vectors live only in process memory. Re-index on each startup.
///   The file watcher can still run for live re-indexing; the manifest is kept in memory only.
///   Set `CALICHE_EMBED_PERSIST=false` to use InMemory when using auto workspace embedding.
#[derive(Debug, Clone, Default)]
pub enum PersistenceConfig {
    /// In-memory only; re-index on each startup. File watcher updates index in memory only.
    #[default]
    InMemory,
    /// Persist to disk (e.g. `~/.caliche/workspaces/<workspace_hash>/`). SQLite + in-memory cache.
    /// Index metadata is persisted in SQLite for incremental indexing and watcher updates.
    Disk { path: PathBuf },
}

/// Configuration for the workspace indexer.
///
/// In Auto mode, [`with_env_overrides`](WorkspaceIndexerConfig::with_env_overrides) applies
/// `CALICHE_LINE_LEVEL_BM25`, `CALICHE_EMBED_PERSIST`, and `CALICHE_EMBED_PATH`.
/// Workspace RAG construction also honors `CALICHE_RAG_LATE_INTERACTION=true` for MaxSim / PLAID.
/// Index embedding concurrency is `CALICHE_INDEX_CONCURRENCY` (adaptive default, max 32).
/// Persisted SQLite cache size is `CALICHE_SQLITE_CACHE_SIZE` (default 2048).
#[derive(Debug, Clone)]
pub struct WorkspaceIndexerConfig {
    /// Root directory to index (typically current working directory).
    pub root_path: PathBuf,
    /// Maximum file size in bytes. Files larger than this are skipped.
    pub max_file_size: usize,
    /// Allowed file extensions (e.g. `["rs", "py", "md"]`). Empty means use defaults.
    pub file_extensions: Vec<String>,
    /// Extra exclude patterns (in addition to .gitignore / .caliche_ignore).
    pub exclude_patterns: Vec<String>,
    /// Persistence mode for embeddings.
    pub persistence: PersistenceConfig,
    /// Use line-level BM25: index each line, aggregate scores by chunk. Higher precision for line-specific matches. More memory.
    pub line_level_bm25: bool,
    /// When true and a graph backend is provided, also build a code graph (AST nodes) during indexing.
    /// Enables structural code exploration via GraphExplorerTool. Requires tools-ast and tools-graph.
    pub build_code_graph: bool,
    /// Optional override for workspace hashline tagging. If None, behavior is auto-derived (edit tools).
    pub workspace_hashline_enabled: Option<bool>,
    /// Optional override for workspace hashline result formatting. If None, behavior is auto-derived (edit tools).
    pub workspace_hashline_format: Option<bool>,
    /// Default RAG search options (MMR, temporal decay) applied when the RAG tool is used for workspace queries.
    #[cfg(feature = "tools-rag")]
    pub rag_search_options: Option<crate::rag::RagSearchOptions>,
    /// Provider/model/chunking fingerprint for persisted workspace embeddings.
    /// A mismatch clears the persisted vector index before re-indexing.
    pub index_fingerprint: Option<String>,
}

impl WorkspaceIndexerConfig {
    /// Create config with defaults for the given root path.
    ///
    /// Default persistence is Disk at `~/.caliche/workspaces/<workspace_hash>/`.
    /// Each git worktree gets its own SQLite store under that catalog.
    pub fn new(root_path: PathBuf) -> Self {
        let persistence = PersistenceConfig::Disk {
            path: workspace_shared_storage_dir(&root_path),
        };
        Self {
            root_path,
            max_file_size: DEFAULT_MAX_FILE_SIZE,
            file_extensions: default_file_extensions(),
            exclude_patterns: Vec::new(),
            persistence,
            line_level_bm25: false,
            build_code_graph: false,
            workspace_hashline_enabled: None,
            workspace_hashline_format: None,
            #[cfg(feature = "tools-rag")]
            rag_search_options: None,
            index_fingerprint: None,
        }
    }

    /// Set max file size.
    pub fn max_file_size(mut self, n: usize) -> Self {
        self.max_file_size = n;
        self
    }

    /// Set file extensions. Replaces defaults.
    pub fn file_extensions(mut self, ext: Vec<String>) -> Self {
        self.file_extensions = ext;
        self
    }

    /// Set exclude patterns.
    pub fn exclude_patterns(mut self, patterns: Vec<String>) -> Self {
        self.exclude_patterns = patterns;
        self
    }

    /// Enable building a code graph during indexing (when graph backend is available).
    pub fn build_code_graph(mut self, enabled: bool) -> Self {
        self.build_code_graph = enabled;
        self
    }

    /// Override workspace hashline tagging behavior.
    pub fn workspace_hashline_enabled(mut self, enabled: Option<bool>) -> Self {
        self.workspace_hashline_enabled = enabled;
        self
    }

    /// Override workspace hashline result formatting behavior.
    pub fn workspace_hashline_format(mut self, enabled: Option<bool>) -> Self {
        self.workspace_hashline_format = enabled;
        self
    }

    /// Set default RAG search options (MMR, temporal decay) for workspace RAG queries.
    #[cfg(feature = "tools-rag")]
    pub fn rag_search_options(mut self, options: Option<crate::rag::RagSearchOptions>) -> Self {
        self.rag_search_options = options;
        self
    }

    /// Set persistence config.
    pub fn persistence(mut self, p: PersistenceConfig) -> Self {
        self.persistence = p;
        self
    }

    /// Use line-level BM25 for hybrid search. Opt-in; uses more memory.
    pub fn line_level_bm25(mut self, enabled: bool) -> Self {
        self.line_level_bm25 = enabled;
        self
    }

    /// Set the persisted index compatibility fingerprint.
    pub fn index_fingerprint(mut self, fingerprint: Option<String>) -> Self {
        self.index_fingerprint = fingerprint;
        self
    }

    /// Apply environment variable overrides. Call when building config for Auto mode.
    pub fn with_env_overrides(mut self) -> Self {
        if std::env::var("CALICHE_LINE_LEVEL_BM25").as_deref() == Ok("true") {
            self.line_level_bm25 = true;
        }
        if std::env::var("CALICHE_EMBED_PERSIST").as_deref() == Ok("false") {
            self.persistence = PersistenceConfig::InMemory;
        } else if !matches!(self.persistence, PersistenceConfig::InMemory) {
            self.persistence = PersistenceConfig::Disk {
                path: workspace_shared_storage_dir(&self.root_path),
            };
        }
        self
    }

    /// Resolve file extensions to use (defaults if empty).
    pub fn resolved_extensions(&self) -> Vec<String> {
        if self.file_extensions.is_empty() {
            default_file_extensions()
        } else {
            self.file_extensions.clone()
        }
    }
}

/// Parse `CALICHE_SQLITE_CACHE_SIZE`, falling back to [`DEFAULT_SQLITE_CACHE_SIZE`].
#[must_use]
pub fn parse_sqlite_cache_size(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.parse().ok())
        .filter(|&n: &usize| n > 0)
        .unwrap_or(DEFAULT_SQLITE_CACHE_SIZE)
}

/// In-process SQLite vector cache entries for persisted workspace embeddings.
#[must_use]
pub fn sqlite_cache_size_from_env() -> usize {
    parse_sqlite_cache_size(std::env::var("CALICHE_SQLITE_CACHE_SIZE").ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_SQLITE_CACHE_SIZE, PersistenceConfig, WorkspaceIndexerConfig,
        parse_sqlite_cache_size, store_dir_for_repo, workspace_hash,
    };
    use std::path::PathBuf;

    #[test]
    fn sqlite_cache_size_defaults_and_rejects_zero() {
        assert_eq!(parse_sqlite_cache_size(None), DEFAULT_SQLITE_CACHE_SIZE);
        assert_eq!(
            parse_sqlite_cache_size(Some("0")),
            DEFAULT_SQLITE_CACHE_SIZE
        );
        assert_eq!(
            parse_sqlite_cache_size(Some("nope")),
            DEFAULT_SQLITE_CACHE_SIZE
        );
        assert_eq!(parse_sqlite_cache_size(Some("512")), 512);
    }

    #[test]
    fn store_dir_for_nested_repo_uses_sibling_hash_folder() {
        let workspace = PathBuf::from("/tmp/workspace-root");
        let nested = PathBuf::from("/tmp/workspace-root/child");
        let config =
            WorkspaceIndexerConfig::new(workspace.clone()).persistence(PersistenceConfig::Disk {
                path: PathBuf::from("/tmp/catalog/abc123"),
            });
        let nested_store = store_dir_for_repo(&config, &nested);
        assert_eq!(
            nested_store,
            PathBuf::from("/tmp/catalog").join(workspace_hash(&nested))
        );
        assert_eq!(
            store_dir_for_repo(&config, &workspace),
            PathBuf::from("/tmp/catalog/abc123")
        );
    }
}
