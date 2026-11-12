//! Path restriction hook.
//!
//! The policy gate decides whether a path outside the workspace may run. This
//! hook does not grant a path because it appeared in user text. It denies a
//! rejected path form: an alternate data stream, a device path, or an 8.3
//! short name.

use crate::agent::hooks::lifecycle::{HookContext, HookResult, LifecycleHook};
use crate::agent::tools::names::TOOL_FILESYSTEM;
use crate::policy::{FsRoots, raw_paths_from_args, resolve_fs_path};
use async_trait::async_trait;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Tools whose path arguments use the filesystem resolver.
const RESTRICTED_TOOLS: &[&str] = &[TOOL_FILESYSTEM, "lsp", "edit_file", "apply_patch"];

/// Paths named in the current turn's message.
///
/// The REPL engine still fills this set before each turn. The hook does not
/// grant access from it: the policy gate decides paths outside the workspace.
pub type AllowedPaths = Arc<RwLock<HashSet<PathBuf>>>;

/// Configuration for the path restriction hook.
#[derive(Debug, Clone)]
pub struct PathRestrictionConfig {
    /// Whether the restriction is active.  When `false`, all paths are allowed.
    pub enabled: bool,
}

impl Default for PathRestrictionConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Hook that enforces workspace-root path restrictions on all filesystem tools.
#[derive(Debug)]
pub struct PathRestrictionHook {
    workspace_root: PathBuf,
    /// The CLI still updates this set. The hook does not grant from it.
    #[expect(
        dead_code,
        reason = "the CLI keeps this handle; user text does not grant a path"
    )]
    extra_allowed: AllowedPaths,
    config: PathRestrictionConfig,
}

impl PathRestrictionHook {
    /// Create a new hook and return both the hook and the shared `AllowedPaths`
    /// handle.  The caller should hold on to the handle and update it before
    /// each user turn.
    #[must_use]
    pub fn new(workspace_root: PathBuf, config: PathRestrictionConfig) -> (Self, AllowedPaths) {
        let extra_allowed: AllowedPaths = Arc::new(RwLock::new(HashSet::new()));
        let hook = Self {
            workspace_root,
            extra_allowed: Arc::clone(&extra_allowed),
            config,
        };
        (hook, extra_allowed)
    }
}

/// Extract absolute paths from free-form text (user messages).
///
/// Looks for tokens that start with `/` or `~/` after stripping common
/// punctuation.  Returns a deduplicated list of resolved `PathBuf`s.
#[must_use]
pub fn extract_paths_from_text(text: &str) -> Vec<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    let mut seen = HashSet::new();
    let mut paths = Vec::new();

    for word in text.split(|c: char| {
        c.is_whitespace() || matches!(c, ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>')
    }) {
        let word = word.trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '.' | ':'));
        if word.is_empty() {
            continue;
        }

        let from_home = word.starts_with("~/") && !home.is_empty();
        let expanded = if from_home {
            format!("{}/{}", home, &word[2..])
        } else {
            word.to_string()
        };

        // A Windows home (`C:\Users\...`) does not start with `/`.
        if (from_home || expanded.starts_with('/')) && expanded.len() > 1 {
            let pb = PathBuf::from(&expanded);
            if seen.insert(pb.clone()) {
                paths.push(pb);
            }
        }
    }

    paths
}

#[async_trait]
impl LifecycleHook for PathRestrictionHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        if !self.config.enabled {
            return HookResult::allow();
        }
        let tool = ctx.tool_name.as_ref();
        if !RESTRICTED_TOOLS.contains(&tool) {
            return HookResult::allow();
        }
        let workspace = self.workspace_root.to_string_lossy();
        if workspace.trim().is_empty() {
            return HookResult::deny("no workspace root; filesystem path is denied");
        }
        let roots = FsRoots::new(workspace.as_ref(), workspace.as_ref());
        for raw in raw_paths_from_args(ctx.tool_input) {
            if resolve_fs_path(&raw, roots).is_err() {
                return HookResult::deny("filesystem path is rejected");
            }
        }
        HookResult::allow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_paths_from_user_text() {
        let paths =
            extract_paths_from_text("Please read /etc/hosts and also check ~/projects/foo.rs");
        assert!(paths.contains(&PathBuf::from("/etc/hosts")));
        // ~/projects/foo.rs should be expanded
        assert!(
            paths
                .iter()
                .any(|p| p.to_string_lossy().ends_with("projects/foo.rs"))
        );
    }

    #[test]
    fn returns_empty_when_text_has_no_paths() {
        let paths = extract_paths_from_text("just a normal sentence with no paths");
        assert!(paths.is_empty());
    }

    #[test]
    fn rejected_path_form_fails_and_a_relative_escape_resolves() {
        let (hook, _) =
            PathRestrictionHook::new(PathBuf::from("/work"), PathRestrictionConfig::default());
        let workspace = hook.workspace_root.to_string_lossy();
        let roots = FsRoots::new(workspace.as_ref(), workspace.as_ref());
        assert!(resolve_fs_path("notes.txt:Zone.Identifier", roots).is_err());
        let escaped = resolve_fs_path("../secret", roots).expect("escape resolves");
        assert_eq!(escaped.place, crate::policy::PathPlace::Outside);
    }

    #[tokio::test]
    async fn empty_workspace_root_denies() {
        let (hook, _) = PathRestrictionHook::new(PathBuf::new(), PathRestrictionConfig::default());
        let args = serde_json::json!({"op": "read", "path": "Cargo.toml"});
        let ctx = HookContext::new(TOOL_FILESYSTEM, &args, "call-no-root");
        assert!(hook.on_pre_tool_use(&ctx).await.is_denied());
    }
}
