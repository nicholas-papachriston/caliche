//! File ignore patterns for context management.
//!
//! Delegates to the `ignore` crate (same engine ripgrep uses) so that
//! `.caliche_ignore` files behave with full gitignore semantics rather
//! than the hand-rolled subset that shipped previously.

use anyhow::Result;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::path::Path;

/// Build a `Gitignore` matcher for the project root's `.caliche_ignore`.
///
/// Returns `None` if the file does not exist, or if the matcher cannot be built
/// (all errors are surfaced through the inner `Gitignore::matched` later on).
fn build_matcher(project_root: &Path) -> Option<Gitignore> {
    let ignore_file = project_root.join(".caliche_ignore");
    if !ignore_file.exists() {
        return None;
    }

    let mut builder = GitignoreBuilder::new(project_root);
    if let Some(err) = builder.add(&ignore_file) {
        tracing::warn!(
            "Failed to load .caliche_ignore at {}: {}",
            ignore_file.display(),
            err
        );
        return None;
    }
    match builder.build() {
        Ok(matcher) => Some(matcher),
        Err(e) => {
            tracing::warn!(
                "Failed to compile .caliche_ignore matcher for {}: {}",
                project_root.display(),
                e
            );
            None
        }
    }
}

/// Check if a file should be ignored based on `.caliche_ignore` patterns.
///
/// Uses gitignore semantics (wildcards, leading `/`, directory-only `trailing/`,
/// negation with `!`, `**` recursive globs, comments with `#`).
pub fn should_ignore_file(path: &Path, project_root: &Path) -> bool {
    let Some(matcher) = build_matcher(project_root) else {
        return false;
    };

    let is_dir = path.is_dir();
    matcher.matched(path, is_dir).is_ignore()
}

/// Load the raw pattern lines from `.caliche_ignore` for display/inspection.
///
/// This is intentionally lossy; the canonical matcher used at runtime is the
/// gitignore-semantics matcher built by [`build_matcher`].
pub fn load_ignore_patterns(project_root: &Path) -> Result<Vec<String>> {
    let ignore_file = project_root.join(".caliche_ignore");

    if !ignore_file.exists() {
        return Ok(Vec::new());
    }

    let content = std::fs::read_to_string(&ignore_file)?;
    let patterns: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    Ok(patterns)
}
