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
    /// Files that pass the filters, named below the walked root.
    pub files: Vec<PathBuf>,
    /// Directories the walk tried to list and could not (a link swapped in,
    /// a permission or I/O error). Nothing below them was seen.
    pub unlisted: Vec<PathBuf>,
}

/// Collect file paths under `root` respecting ignore files.
///
/// - `root`: the canonical directory to walk; it is listed as given, never
///   resolved again, so a root that is now a link is refused
/// - `max_file_size`: skip files larger than this (bytes); `None` = no limit
/// - `extensions`: allowed extensions (e.g. `["rs", "py"]`); empty = allow all
/// - `max_files`: stop after collecting this many files; `None` = no limit
/// - `exclude_patterns`: extra gitignore-style patterns (in addition to ignore files)
///
/// Nested git worktrees are not entered. This runs blocking I/O. Call from
/// `spawn_blocking` if used in async context.
pub fn collect_files(
    root: &Path,
    max_file_size: Option<u64>,
    extensions: &[String],
    max_files: Option<usize>,
    exclude_patterns: &[String],
) -> CollectedFiles {
    let extra = extra_excludes(root, exclude_patterns);
    let ext_set: std::collections::HashSet<String> =
        extensions.iter().map(|s| s.to_lowercase()).collect();
    let prune = |entry: &WalkedEntry| {
        super::git_repos::is_git_root(&entry.path) || is_excluded(extra.as_ref(), entry)
    };
    let mut files = Vec::new();
    let report = walk_tree_pruned(root, WalkOptions::default(), &prune, &mut |entry| {
        if max_files.is_some_and(|limit| files.len() >= limit) {
            return ControlFlow::Break(());
        }
        if entry.kind == EntryKind::File
            && max_file_size.is_none_or(|max| entry.len <= max)
            && (ext_set.is_empty() || extension_allowed(&entry.path, &ext_set))
            && !is_excluded(extra.as_ref(), entry)
        {
            files.push(entry.path.clone());
        }
        ControlFlow::Continue(())
    });
    CollectedFiles {
        files,
        unlisted: report.unlisted,
    }
}

/// True when an extra exclude pattern matches `entry`. A walk checks each
/// directory before it enters it, so only the entry itself is matched.
fn is_excluded(extra: Option<&Gitignore>, entry: &WalkedEntry) -> bool {
    extra.is_some_and(|gi| {
        gi.matched(&entry.path, entry.kind == EntryKind::Dir)
            .is_ignore()
    })
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
    // The path may be gone, so it is matched as a file, along with every
    // directory above it: a directory pattern such as `nested/` excludes
    // the files below it.
    extra_excludes(root, exclude_patterns).is_none_or(|gi| {
        let matched = if path.starts_with(root) {
            gi.matched_path_or_any_parents(path, false)
        } else {
            gi.matched(path, false)
        };
        !matched.is_ignore()
    })
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
        return memory_file_len(path).is_some_and(|len| max_file_size.is_none_or(|max| len <= max));
    }
    // `matches_index_filters` has already checked the extra excludes
    // against `path` and every directory above it.
    let prune = |entry: &WalkedEntry| !path.starts_with(&entry.path);
    let mut allowed = false;
    walk_tree_pruned(root, WalkOptions::default(), &prune, &mut |entry| {
        if entry.path == path {
            allowed =
                entry.kind == EntryKind::File && max_file_size.is_none_or(|max| entry.len <= max);
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    });
    allowed
}

/// True when `path` is a regular file the index could hold: a memory note
/// stat'ed without following a final link, or a workspace file listed by a
/// no-link walk of its parent.
pub fn is_index_file(root: &Path, path: &Path) -> bool {
    if under_memory_dir(root, path) {
        memory_file_len(path).is_some()
    } else {
        crate::agent::tools::common::path::is_walked_file(path)
    }
}

/// The length of the memory note at `path`, or `None` when it is not a
/// regular file. A link is not followed. The memory directory is in the
/// embedding catalog, outside the workspace.
fn memory_file_len(path: &Path) -> Option<u64> {
    std::fs::symlink_metadata(path)
        .ok()
        .filter(std::fs::Metadata::is_file)
        .map(|meta| meta.len())
}

/// True when `path` is in the shared memory directory of `root`, which is
/// outside the workspace, under the embedding catalog.
fn under_memory_dir(root: &Path, path: &Path) -> bool {
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
        let paths = collect_files(&root, None, &["rs".to_string()], None, &[]);
        let names: Vec<String> = paths
            .files
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_string))
            .collect();
        assert_eq!(names, vec!["keep.rs".to_string()]);
    }

    fn file_names(files: &[PathBuf]) -> Vec<String> {
        let mut names: Vec<String> = files
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_string))
            .collect();
        names.sort();
        names
    }

    /// A directory pattern excludes the files below it, from the walk and
    /// from the single-path checks.
    #[test]
    fn a_directory_exclude_covers_the_files_below_it() {
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        write_file(&root, "keep.rs", "fn keep() {}");
        write_file(&root, "nested/inner/skip.rs", "fn skip() {}");
        let exts = vec!["rs".to_string()];
        let excludes = vec!["nested/".to_string()];
        let skip = root.join("nested/inner/skip.rs");

        let collected = collect_files(&root, None, &exts, None, &excludes);

        assert_eq!(file_names(&collected.files), vec!["keep.rs"]);
        assert!(!path_allowed(&root, &skip, None, &exts, &excludes));
        assert!(!matches_index_filters(&root, &skip, &exts, &excludes));
        assert!(path_allowed(
            &root,
            &root.join("keep.rs"),
            None,
            &exts,
            &excludes
        ));
    }

    /// A file over the size limit is neither collected nor allowed.
    #[test]
    fn a_file_over_the_size_limit_is_left_out() {
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        write_file(&root, "small.rs", "1234");
        write_file(&root, "big.rs", "12345");
        let exts = vec!["rs".to_string()];

        let collected = collect_files(&root, Some(4), &exts, None, &[]);

        assert_eq!(file_names(&collected.files), vec!["small.rs"]);
        assert!(!path_allowed(
            &root,
            &root.join("big.rs"),
            Some(4),
            &exts,
            &[]
        ));
        assert!(path_allowed(
            &root,
            &root.join("small.rs"),
            Some(4),
            &exts,
            &[]
        ));
    }

    /// Files the user's global git excludes list are not collected.
    #[test]
    fn collect_files_honors_the_global_git_excludes() {
        let tmp = TempDir::new().expect("temp dir");
        let top = tmp.path().canonicalize().expect("canonical");
        write_file(&top, "config/git/ignore", "globalsecret.toml\n");
        let root = top.join("ws");
        write_file(&root, "globalsecret.toml", "secret");
        write_file(&root, "ok.toml", "ok");
        crate::agent::tools::tree_walk::set_test_global_excludes(Some(
            top.join("config/git/ignore"),
        ));

        let collected = collect_files(&root, None, &[], None, &[]);
        let allowed = path_allowed(&root, &root.join("globalsecret.toml"), None, &[], &[]);
        crate::agent::tools::tree_walk::set_test_global_excludes(None);

        assert_eq!(file_names(&collected.files), vec!["ok.toml"]);
        assert!(!allowed);
    }

    /// A FIFO at the `commondir` a `.git` file points to (a command in the
    /// sandbox can make one) is not read: collecting returns at once.
    #[cfg(unix)]
    #[test]
    fn a_fifo_commondir_does_not_block_collecting() {
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        write_file(&root, "sub/a.rs", "a");
        std::fs::create_dir_all(root.join("sub/gd")).expect("git dir");
        std::fs::write(
            root.join("sub/.git"),
            format!("gitdir: {}\n", root.join("sub/gd").display()),
        )
        .expect("gitfile");
        let status = crate::policy::program_command("mkfifo")
            .arg(root.join("sub/gd/commondir"))
            .status()
            .expect("run mkfifo");
        assert!(status.success());

        let (tx, rx) = std::sync::mpsc::channel();
        let walked = root.join("sub");
        std::thread::spawn(move || {
            let _ = tx.send(collect_files(&walked, None, &[], None, &[]).files);
        });
        let files = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("collecting must not wait on a FIFO");

        assert_eq!(file_names(&files), vec!["a.rs"]);
    }

    /// A memory note that is a link, or over the size limit, is not allowed.
    #[cfg(unix)]
    #[test]
    #[serial_test::serial(env)]
    fn a_memory_note_that_is_a_link_is_not_allowed() {
        let catalog = TempDir::new().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let outside = TempDir::new().expect("outside");
        write_file(outside.path(), "secret.md", "secret");
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        let memory = super::super::config::workspace_memory_markdown_dir(&root);
        write_file(&memory, "note.md", "a note");
        std::os::unix::fs::symlink(outside.path().join("secret.md"), memory.join("leak.md"))
            .expect("note link");
        let exts = vec!["md".to_string()];

        assert!(!path_allowed(
            &root,
            &memory.join("leak.md"),
            None,
            &exts,
            &[]
        ));
        assert!(path_allowed(
            &root,
            &memory.join("note.md"),
            None,
            &exts,
            &[]
        ));
        assert!(!path_allowed(
            &root,
            &memory.join("note.md"),
            Some(5),
            &exts,
            &[]
        ));
    }

    /// A memory path, outside the root, is checked against the extra
    /// excludes as a path of its own; it does not stop the check.
    #[test]
    #[serial_test::serial(env)]
    fn a_memory_path_passes_the_index_filters_with_excludes_set() {
        let catalog = TempDir::new().expect("catalog");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that writes the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_EMBED_PATH", catalog.path());
        let tmp = TempDir::new().expect("temp dir");
        let root = tmp.path().canonicalize().expect("canonical");
        let note = super::super::config::workspace_memory_markdown_dir(&root).join("note.md");

        let allowed =
            matches_index_filters(&root, &note, &["md".to_string()], &["nested/".to_string()]);

        assert!(allowed);
    }
}
