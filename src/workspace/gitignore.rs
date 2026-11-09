//! Gitignore-aware file walking.
//!
//! Uses the `ignore` crate to walk directories while respecting
//! .gitignore, .caliche_ignore, and optional size/extension filters.

use std::path::Path;
use std::path::PathBuf;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

/// Collect file paths under `root` respecting .gitignore and .caliche_ignore.
///
/// - `root`: directory to walk
/// - `max_file_size`: skip files larger than this (bytes); `None` = no limit
/// - `extensions`: allowed extensions (e.g. `["rs", "py"]`); empty = allow all
/// - `max_files`: stop after collecting this many files; `None` = no limit
/// - `exclude_patterns`: extra gitignore-style patterns (in addition to ignore files)
///
/// This runs blocking I/O. Call from `spawn_blocking` if used in async context.
pub fn collect_files(
    root: &Path,
    max_file_size: Option<u64>,
    extensions: &[String],
    max_files: Option<usize>,
    exclude_patterns: &[String],
) -> Vec<PathBuf> {
    let mut walker = walk_builder(root, max_file_size, exclude_patterns);
    let extra = extra_excludes(root, exclude_patterns);
    apply_exclude_filter(&mut walker, extra);

    let ext_set: std::collections::HashSet<String> =
        extensions.iter().map(|s| s.to_lowercase()).collect();
    let has_ext_filter = !ext_set.is_empty();

    let mut out = Vec::new();
    for result in walker.build() {
        if let Some(limit) = max_files
            && out.len() >= limit
        {
            break;
        }

        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }

        if has_ext_filter && !extension_allowed(path, &ext_set) {
            continue;
        }

        out.push(path.to_path_buf());
    }

    out
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
    let is_dir = path.is_dir();
    extra_excludes(root, exclude_patterns).is_none_or(|gi| !gi.matched(path, is_dir).is_ignore())
}

/// True when a live file should be indexed: filters, size, and gitignore.
///
/// Unlike walking the tree with `max_files = 1`, this checks `path` itself
/// rather than whatever file the walker happens to visit first.
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
    if let Some(max) = max_file_size {
        match std::fs::metadata(path) {
            Ok(meta) if meta.len() > max => return false,
            Ok(_) => {}
            Err(_) => return false,
        }
    }
    if is_memory_path(root, path) {
        return true;
    }
    gitignore_allows(root, path, max_file_size, exclude_patterns)
}

fn is_memory_path(root: &Path, path: &Path) -> bool {
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

fn is_nested_git_worktree(root: &Path, entry: &ignore::DirEntry) -> bool {
    if entry.depth() == 0 {
        return false;
    }
    if !entry
        .file_type()
        .is_some_and(|file_type| file_type.is_dir())
    {
        return false;
    }
    let path = entry.path();
    path != root && super::git_repos::is_git_root(path)
}

fn walk_builder(
    root: &Path,
    max_file_size: Option<u64>,
    _exclude_patterns: &[String],
) -> ignore::WalkBuilder {
    let mut walker = ignore::WalkBuilder::new(root);
    walker
        .standard_filters(true)
        .require_git(false)
        .add_custom_ignore_filename(".caliche_ignore");
    if let Some(n) = max_file_size {
        walker.max_filesize(Some(n));
    }
    let root_owned = root.to_path_buf();
    walker.filter_entry(move |entry| !is_nested_git_worktree(&root_owned, entry));
    walker
}

fn apply_exclude_filter(walker: &mut ignore::WalkBuilder, extra: Option<Gitignore>) {
    let Some(extra) = extra else {
        return;
    };
    walker.filter_entry(move |entry| {
        let is_dir = entry
            .file_type()
            .is_some_and(|file_type| file_type.is_dir());
        !extra.matched(entry.path(), is_dir).is_ignore()
    });
}

fn gitignore_allows(
    root: &Path,
    candidate: &Path,
    max_file_size: Option<u64>,
    exclude_patterns: &[String],
) -> bool {
    let extra = extra_excludes(root, exclude_patterns);
    let mut walker = walk_builder(root, max_file_size, exclude_patterns);
    let candidate_owned = candidate.to_path_buf();
    walker.filter_entry(move |entry| {
        let path = entry.path();
        let is_dir = entry
            .file_type()
            .is_some_and(|file_type| file_type.is_dir());
        if extra
            .as_ref()
            .is_some_and(|gi| gi.matched(path, is_dir).is_ignore())
        {
            return false;
        }
        candidate_owned == path || candidate_owned.starts_with(path)
    });
    walker.build().flatten().any(|entry| {
        entry.path() == candidate
            && entry
                .file_type()
                .is_some_and(|file_type| file_type.is_file())
    })
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

        let paths = collect_files(
            dir.path(),
            None,
            &["rs".to_string()],
            None,
            &["skip.rs".to_string()],
        );
        let names: Vec<String> = paths
            .iter()
            .filter_map(|p| p.file_name()?.to_str().map(str::to_string))
            .collect();
        assert_eq!(names, vec!["keep.rs".to_string()]);
    }

    #[test]
    fn path_allowed_matches_the_named_file_not_the_first_walk_hit() {
        let dir = TempDir::new().expect("temp dir");
        write_file(dir.path(), "aaa.rs", "fn a() {}");
        write_file(dir.path(), "zzz.rs", "fn z() {}");
        let target = dir.path().join("zzz.rs");
        let exts = vec!["rs".to_string()];

        assert!(path_allowed(dir.path(), &target, None, &exts, &[]));
        assert!(!path_allowed(
            dir.path(),
            &dir.path().join("missing.rs"),
            None,
            &exts,
            &[],
        ));
    }

    #[test]
    fn path_allowed_rejects_gitignored_files() {
        let dir = TempDir::new().expect("temp dir");
        write_file(dir.path(), ".gitignore", "secret.rs\n");
        write_file(dir.path(), "secret.rs", "fn secret() {}");
        write_file(dir.path(), "ok.rs", "fn ok() {}");
        let exts = vec!["rs".to_string()];

        assert!(!path_allowed(
            dir.path(),
            &dir.path().join("secret.rs"),
            None,
            &exts,
            &[],
        ));
        assert!(path_allowed(
            dir.path(),
            &dir.path().join("ok.rs"),
            None,
            &exts,
            &[],
        ));
    }

    #[test]
    fn collect_files_skips_nested_git_worktrees() {
        let dir = TempDir::new().expect("temp dir");
        write_file(dir.path(), ".git/HEAD", "ref: refs/heads/main\n");
        write_file(dir.path(), "keep.rs", "fn keep() {}");
        write_file(dir.path(), "child/.git/HEAD", "ref: refs/heads/main\n");
        write_file(dir.path(), "child/nested.rs", "fn nested() {}");

        let paths = collect_files(dir.path(), None, &["rs".to_string()], None, &[]);
        let names: Vec<String> = paths
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_string))
            .collect();
        assert_eq!(names, vec!["keep.rs".to_string()]);
    }
}
