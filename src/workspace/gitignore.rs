//! Gitignore-aware file walking.
//!
//! Walks with [`walk_tree_pruned`], which lists every directory one
//! component at a time without following a link, honoring `.gitignore`,
//! `.ignore`, `.caliche_ignore` and `.git/info/exclude`, plus optional
//! size, extension and extra-exclude filters. A directory swapped for a link
//! is refused, not followed, and is reported as unlisted.

use std::ops::ControlFlow;
use std::path::Path;
use std::path::PathBuf;

use caliche_sandbox::EntryKind;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::agent::tools::tree_walk::{WalkOptions, WalkedEntry, walk_tree_pruned};

/// The files a walk collected, and the directories it could not list.
#[derive(Debug, Default)]
pub struct CollectedFiles {
    /// Files that pass the filters, named below the key root.
    pub files: Vec<PathBuf>,
    /// Directories the walk tried to list and could not (a link swapped in,
    /// a permission or I/O error), named below the key root. Nothing below
    /// them was seen.
    pub unlisted: Vec<PathBuf>,
}

/// Collect file paths under `walk_root` respecting ignore files.
///
/// - `root`: the form of the walked directory the returned paths start with
/// - `walk_root`: the canonical directory to walk; it is listed as given,
///   never resolved again, so a root that is now a link is refused
/// - `max_file_size`: skip files larger than this (bytes); `None` = no limit
/// - `extensions`: allowed extensions (e.g. `["rs", "py"]`); empty = allow all
/// - `max_files`: stop after collecting this many files; `None` = no limit
/// - `exclude_patterns`: extra gitignore-style patterns (in addition to ignore files)
///
/// Nested git worktrees are not entered. This runs blocking I/O. Call from
/// `spawn_blocking` if used in async context.
pub fn collect_files(
    root: &Path,
    walk_root: &Path,
    max_file_size: Option<u64>,
    extensions: &[String],
    max_files: Option<usize>,
    exclude_patterns: &[String],
) -> CollectedFiles {
    let extra = extra_excludes(walk_root, exclude_patterns);
    let ext_set: std::collections::HashSet<String> =
        extensions.iter().map(|s| s.to_lowercase()).collect();
    let prune = |entry: &WalkedEntry| {
        super::git_repos::is_git_root(&entry.path) || is_excluded(extra.as_ref(), entry, true)
    };
    let mut files = Vec::new();
    let report = walk_tree_pruned(walk_root, WalkOptions::default(), &prune, &mut |entry| {
        if max_files.is_some_and(|limit| files.len() >= limit) {
            return ControlFlow::Break(());
        }
        if entry.kind == EntryKind::File
            && max_file_size.is_none_or(|max| entry.len <= max)
            && (ext_set.is_empty() || extension_allowed(&entry.path, &ext_set))
            && !is_excluded(extra.as_ref(), entry, false)
        {
            files.push(root.join(&entry.relative));
        }
        ControlFlow::Continue(())
    });
    let unlisted = report
        .unlisted
        .iter()
        .map(|dir| match dir.strip_prefix(walk_root) {
            Ok(below) if below.as_os_str().is_empty() => root.to_path_buf(),
            Ok(below) => root.join(below),
            Err(_) => dir.clone(),
        })
        .collect();
    CollectedFiles { files, unlisted }
}

fn is_excluded(extra: Option<&Gitignore>, entry: &WalkedEntry, is_dir: bool) -> bool {
    is_dir == (entry.kind == EntryKind::Dir)
        && extra.is_some_and(|gi| gi.matched(&entry.path, is_dir).is_ignore())
}

/// True when `path` is under `root` or the shared workspace memory directory.
pub fn under_workspace_root(root: &Path, path: &Path) -> bool {
    let memory_dir = super::config::workspace_memory_markdown_dir(root);
    path.starts_with(root) || path.starts_with(&memory_dir)
}

/// Extension / root / extra-exclude checks that do not require the file to exist.
pub fn matches_index_filters(
    root: &Path,
    path: &Path,
    extensions: &[String],
    exclude_patterns: &[String],
) -> bool {
    if !under_workspace_root(root, path) {
        return false;
    }
    let ext_set: std::collections::HashSet<String> =
        extensions.iter().map(|s| s.to_lowercase()).collect();
    if !ext_set.is_empty() && !extension_allowed(path, &ext_set) {
        return false;
    }
    let is_dir = crate::agent::tools::common::path::entry_kind(path) == Some(EntryKind::Dir);
    extra_excludes(root, exclude_patterns).is_none_or(|gi| !gi.matched(path, is_dir).is_ignore())
}

/// True when a live file should be indexed: filters, size, and gitignore.
///
/// `root` is a canonical directory. A workspace file is found by a no-link
/// walk of the directories that lead to it, so a link anywhere on the way is
/// refused. A file in the shared memory directory is stat'ed without
/// following a final link.
pub fn path_allowed(
    root: &Path,
    path: &Path,
    max_file_size: Option<u64>,
    extensions: &[String],
    exclude_patterns: &[String],
) -> bool {
    if !matches_index_filters(root, path, extensions, exclude_patterns) {
        return false;
    }
    if under_memory_dir(root, path) {
        return std::fs::symlink_metadata(path)
            .is_ok_and(|meta| meta.is_file() && max_file_size.is_none_or(|max| meta.len() <= max));
    }
    let extra = extra_excludes(root, exclude_patterns);
    let prune = |entry: &WalkedEntry| {
        !path.starts_with(&entry.path) || is_excluded(extra.as_ref(), entry, true)
    };
    let mut allowed = false;
    walk_tree_pruned(root, WalkOptions::default(), &prune, &mut |entry| {
        if entry.path == path {
            allowed = entry.kind == EntryKind::File
                && max_file_size.is_none_or(|max| entry.len <= max)
                && !is_excluded(extra.as_ref(), entry, false);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    allowed
}

/// True when `path` is in the shared memory directory of `root`, which is
/// outside the workspace, under the embedding catalog.
pub fn under_memory_dir(root: &Path, path: &Path) -> bool {
    let memory_dir = super::config::workspace_memory_markdown_dir(root);
    path.starts_with(memory_dir)
}

fn extension_allowed(path: &Path, ext_set: &std::collections::HashSet<String>) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_lowercase)
        .unwrap_or_default();
    ext_set.contains(&ext)
}

fn extra_excludes(root: &Path, patterns: &[String]) -> Option<Gitignore> {
    if patterns.is_empty() {
        return None;
    }
    let mut builder = GitignoreBuilder::new(root);
    let mut added = false;
    for pattern in patterns {
        if builder.add_line(None, pattern).is_ok() {
            added = true;
        }
    }
    if !added {
        return None;
    }
    builder.build().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_file(dir: &Path, rel: &str, contents: &str) {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, contents).expect("write file");
    }

    #[test]
    fn collect_files_honors_extra_exclude_patterns() {
        let dir = TempDir::new().expect("temp dir");
        write_file(dir.path(), "keep.rs", "fn keep() {}");
        write_file(dir.path(), "skip.rs", "fn skip() {}");
        write_file(dir.path(), "nested/skip.rs", "fn nested() {}");

        let root = dir.path().canonicalize().expect("canonical");
        let paths = collect_files(
            &root,
            &root,
            None,
            &["rs".to_string()],
            None,
            &["skip.rs".to_string()],
        );
        let names: Vec<String> = paths
            .files
            .iter()
            .filter_map(|p| p.file_name()?.to_str().map(str::to_string))
            .collect();
        assert_eq!(names, vec!["keep.rs".to_string()]);
    }

    #[test]
    fn path_allowed_matches_the_named_file_not_the_first_walk_hit() {
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        let dir = root.as_path();
        write_file(dir, "aaa.rs", "fn a() {}");
        write_file(dir, "zzz.rs", "fn z() {}");
        let target = dir.join("zzz.rs");
        let exts = vec!["rs".to_string()];

        assert!(path_allowed(dir, &target, None, &exts, &[]));
        assert!(!path_allowed(
            dir,
            &dir.join("missing.rs"),
            None,
            &exts,
            &[],
        ));
    }

    #[test]
    fn path_allowed_rejects_gitignored_files() {
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        let dir = root.as_path();
        write_file(dir, ".gitignore", "secret.rs\n");
        write_file(dir, "secret.rs", "fn secret() {}");
        write_file(dir, "ok.rs", "fn ok() {}");
        let exts = vec!["rs".to_string()];

        assert!(!path_allowed(dir, &dir.join("secret.rs"), None, &exts, &[],));
        assert!(path_allowed(dir, &dir.join("ok.rs"), None, &exts, &[],));
    }

    #[test]
    fn collect_files_skips_nested_git_worktrees() {
        let dir = TempDir::new().expect("temp dir");
        write_file(dir.path(), ".git/HEAD", "ref: refs/heads/main\n");
        write_file(dir.path(), "keep.rs", "fn keep() {}");
        write_file(dir.path(), "child/.git/HEAD", "ref: refs/heads/main\n");
        write_file(dir.path(), "child/nested.rs", "fn nested() {}");

        let root = dir.path().canonicalize().expect("canonical");
        let paths = collect_files(&root, &root, None, &["rs".to_string()], None, &[]);
        let names: Vec<String> = paths
            .files
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_string))
            .collect();
        assert_eq!(names, vec!["keep.rs".to_string()]);
    }
}
