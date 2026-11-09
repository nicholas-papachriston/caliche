//! Context-provider adapters.

pub mod workspace;

#[cfg(feature = "tools-rag")]
pub use workspace::fetch_workspace_context_from_rag;
pub use workspace::{DefaultWorkspaceContextProvider, WorkspaceContextProvider};
