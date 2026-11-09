//! Lifecycle hook that records file before/after for session diff (undo/redo).

use crate::agent::hooks::{HookContext, HookResult, LifecycleHook};
use crate::agent::tools::names::{FS_OP_DELETE, FS_OP_EDIT, FS_OP_WRITE, TOOL_FILESYSTEM, fs_op};
use crate::harness::DiffTracker;
use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Hook that records file state before and after fs write/edit operations for the session diff tracker.
#[derive(Debug)]
pub struct SessionDiffHook {
    /// Shared diff tracker (harness owns this)
    pub diff_tracker: Arc<RwLock<DiffTracker>>,
    /// Optional workspace root for resolving relative paths
    pub workspace_root: Option<PathBuf>,
}

impl SessionDiffHook {
    /// Create a new session diff hook.
    pub fn new(diff_tracker: Arc<RwLock<DiffTracker>>) -> Self {
        Self {
            diff_tracker,
            workspace_root: None,
        }
    }

    /// Set workspace root for resolving relative paths.
    pub fn with_workspace_root(mut self, root: PathBuf) -> Self {
        self.workspace_root = Some(root);
        self
    }

    fn resolve_path(&self, path_str: &str) -> PathBuf {
        let p = PathBuf::from(path_str);
        if p.is_absolute() {
            return p;
        }
        if let Some(ref root) = self.workspace_root {
            return root.join(&p);
        }
        p
    }
}

#[async_trait]
impl LifecycleHook for SessionDiffHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        let name = ctx.tool_name.as_ref();
        let op = fs_op(ctx.tool_input).unwrap_or("");
        if name != TOOL_FILESYSTEM || !matches!(op, FS_OP_WRITE | FS_OP_EDIT | FS_OP_DELETE) {
            return HookResult::allow();
        }
        let path_str = match ctx.tool_input.get("path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return HookResult::allow(),
        };
        let path = self.resolve_path(path_str);
        let content = tokio::fs::read(&path).await.ok();
        let mut tracker = self.diff_tracker.write().await;
        tracker.record_before(&path, content.as_deref());
        HookResult::allow()
    }

    async fn on_post_tool_use(
        &self,
        ctx: &HookContext<'_>,
        _result: &crate::agent::tools::tool::ToolResult,
    ) -> HookResult {
        let name = ctx.tool_name.as_ref();
        let op = fs_op(ctx.tool_input).unwrap_or("");
        if name != TOOL_FILESYSTEM || !matches!(op, FS_OP_WRITE | FS_OP_EDIT | FS_OP_DELETE) {
            return HookResult::allow();
        }
        let path_str = match ctx.tool_input.get("path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return HookResult::allow(),
        };
        let path = self.resolve_path(path_str);
        let content = tokio::fs::read(&path).await.ok();
        let mut tracker = self.diff_tracker.write().await;
        tracker.record_after(&path, content.as_deref());
        HookResult::allow()
    }
}
