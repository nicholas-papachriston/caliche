//! Workspace indexing for auto-embedding the current directory.
//!
//! Used with workspace embedding: index files under the workspace root,
//! respect .gitignore and .caliche_ignore, then query embeddings
//! to guide agent context.

mod config;
mod federated;
mod git_repos;
mod gitignore;
mod indexer;
mod memory_markdown;
mod store;

#[cfg(feature = "tools-rag")]
pub mod manifest;
#[cfg(not(feature = "tools-rag"))]
mod manifest;

#[cfg(feature = "tools-rag")]
pub mod watcher;

pub use crate::workspace_declaration::{
    ApplyOptions, DEFAULT_WORKSPACE_TOML, WarmStartMcpServer, WarmStartMcpTransport, WarmStartPlan,
    WarmStartSkill, WorkspaceDeclaration, WorkspaceMcpSpec, WorkspaceRepoSpec, WorkspaceSkillSpec,
};
pub use config::{
    DEFAULT_MAX_FILE_SIZE, DEFAULT_SQLITE_CACHE_SIZE, DEFAULT_SQLITE_MAX_CONNECTIONS,
    PersistenceConfig, WorkspaceIndexerConfig, sqlite_cache_size_from_env, store_dir_for_repo,
    workspace_catalog_root, workspace_hash, workspace_memory_markdown_dir,
    workspace_shared_storage_dir,
};
pub use indexer::{WorkspaceIndexResult, WorkspaceIndexer};
pub use memory_markdown::{
    ensure_workspace_memory_dir, list_workspace_memory_markdown_files,
    write_workspace_memory_markdown,
};
pub use store::build_workspace_backend;

#[cfg(feature = "tools-rag")]
pub use watcher::WorkspaceWatcher;
