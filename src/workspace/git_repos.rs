//! Nested git worktree discovery for per-repo embedding stores.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

use crate::gitmodules::{is_safe_relative, parse_gitmodules};

/// Maximum nested git worktrees to register under one workspace root.
const MAX_NESTED_REPOS: usize = 64;

/// True when `path` is a git worktree root (`.git` file or directory).
#[must_use]
pub fn is_git_root(path: &Path) -> bool {
    path.join(".git").exists()
}

/// Canonical path, or the original path when canonicalize fails.
#[must_use]
pub fn canonical_or_owned(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Innermost git worktree that contains `path`, if any.
#[must_use]
pub fn owning_git_root(path: &Path) -> Option<PathBuf> {
    let mut current = if path.is_file() {
        path.parent().map(Path::to_path_buf)?
    } else {
        path.to_path_buf()
    };
    loop {
        if is_git_root(&current) {
            return Some(canonical_or_owned(&current));
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Declared worktrees only.
///
/// `None` when `.gitmodules` is missing or lists no safe submodule paths.
/// A file that lists paths still wins when those checkouts are missing, so
/// discovery does not walk the tree looking for other `.git` directories.
fn repos_from_gitmodules(root: &Path) -> Option<Vec<PathBuf>> {
    if !uses_declared_submodules(root) {
        return None;
    }
    let mut repos = vec![canonical_or_owned(root)];
    for path in gitmodules_worktrees(root) {
        append_declared_repo(&mut repos, path);
    }
    Some(repos)
}

/// True when `.gitmodules` lists at least one safe relative submodule path.
#[must_use]
pub(crate) fn uses_declared_submodules(root: &Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(root.join(".gitmodules")) else {
        return false;
    };
    !parse_gitmodules(&contents).is_empty()
}

fn append_declared_repo(repos: &mut Vec<PathBuf>, path: PathBuf) {
    let before = repos.len();
    let children = gitmodules_worktrees(&path);
    push_unique(repos, path);
    if repos.len() == before {
        return;
    }
    for child in children {
        append_declared_repo(repos, child);
    }
}

fn push_unique(repos: &mut Vec<PathBuf>, candidate: PathBuf) {
    if repos.len() >= MAX_NESTED_REPOS {
        return;
    }
    if !repos.iter().any(|existing| existing == &candidate) {
        repos.push(candidate);
    }
}

fn skip_discovery_dir(name: &str) -> bool {
    matches!(name, "node_modules" | "target" | "dist" | "vendor")
}

fn gitmodules_worktrees(root: &Path) -> Vec<PathBuf> {
    let Ok(contents) = std::fs::read_to_string(root.join(".gitmodules")) else {
        return Vec::new();
    };
    parse_gitmodules(&contents)
        .into_iter()
        .filter_map(|entry| contained_git_root(root, &entry.relative_path))
        .collect()
}

/// Live git root for `relative` when it stays inside `root` after canonicalize.
fn contained_git_root(root: &Path, relative: &str) -> Option<PathBuf> {
    if !is_safe_relative(relative) {
        return None;
    }
    let candidate = root.join(relative);
    if !is_git_root(&candidate) {
        return None;
    }
    let canonical_root = std::fs::canonicalize(root).ok()?;
    let canonical = std::fs::canonicalize(&candidate).ok()?;
    if canonical == canonical_root || !canonical.starts_with(&canonical_root) {
        return None;
    }
    Some(canonical)
}

/// Git worktrees to index under `root`, including `root` itself.
///
/// When `.gitmodules` lists safe submodule paths, those live checkouts (and
/// their own `.gitmodules` entries) are the list. Missing checkouts do not
/// fall back to a tree walk. A root with no declared paths still walks for
/// nested `.git` directories. A non-git folder still returns `root` so loose
/// files keep a store.
#[must_use]
pub fn discover_git_repos(root: &Path) -> Vec<PathBuf> {
    let canonical_root = canonical_or_owned(root);
    if let Some(repos) = repos_from_gitmodules(&canonical_root) {
        return repos;
    }
    let mut repos = vec![canonical_root.clone()];

    let mut walker = WalkBuilder::new(root);
    walker
        .standard_filters(false)
        .hidden(true)
        .require_git(false)
        .filter_entry(|entry| {
            if entry.depth() == 0 {
                return true;
            }
            entry
                .file_name()
                .to_str()
                .is_none_or(|name| !skip_discovery_dir(name))
        });
    for entry in walker.build().flatten() {
        if repos.len() >= MAX_NESTED_REPOS {
            break;
        }
        let path = entry.path();
        if !entry.file_type().is_some_and(|kind| kind.is_dir()) {
            continue;
        }
        if path == root || path == canonical_root {
            continue;
        }
        if !is_git_root(path) {
            continue;
        }
        push_unique(&mut repos, canonical_or_owned(path));
    }

    for path in gitmodules_worktrees(root) {
        push_unique(&mut repos, path);
    }
    repos
}

/// True when `path` belongs to `root` and not to a nested git worktree.
#[must_use]
pub fn belongs_to_repo(path: &Path, root: &Path) -> bool {
    let canonical_root = canonical_or_owned(root);
    let Some(owner) = owning_git_root(path) else {
        return path.starts_with(root) || path.starts_with(&canonical_root);
    };
    owner == canonical_root
}

#[cfg(test)]
mod tests {
    use super::{belongs_to_repo, discover_git_repos, is_git_root};
    use std::fs;
    use tempfile::TempDir;

    fn init_git(dir: &std::path::Path) {
        fs::create_dir_all(dir.join(".git")).expect("git dir");
    }

    #[test]
    fn discover_git_repos_includes_nested_worktrees() {
        let dir = TempDir::new().expect("temp");
        init_git(dir.path());
        let child = dir.path().join("child");
        init_git(&child);
        fs::write(dir.path().join("root.rs"), "fn root() {}").expect("root file");
        fs::write(child.join("child.rs"), "fn child() {}").expect("child file");

        let repos = discover_git_repos(dir.path());
        assert_eq!(repos.len(), 2);
        assert!(repos.iter().any(|path| is_git_root(path)));
        assert!(repos.iter().any(|path| path.ends_with("child")));
    }

    #[test]
    fn belongs_to_repo_excludes_nested_git_files() {
        let dir = TempDir::new().expect("temp");
        init_git(dir.path());
        let child = dir.path().join("child");
        init_git(&child);
        let nested = child.join("child.rs");
        fs::write(&nested, "fn child() {}").expect("child file");
        let parent_file = dir.path().join("root.rs");
        fs::write(&parent_file, "fn root() {}").expect("root file");

        assert!(belongs_to_repo(&parent_file, dir.path()));
        assert!(!belongs_to_repo(&nested, dir.path()));
        assert!(belongs_to_repo(&nested, &child));
    }

    #[test]
    fn discover_git_repos_skips_undeclared_repos_when_gitmodules_exists() {
        let dir = TempDir::new().expect("temp");
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

        let repos = discover_git_repos(dir.path());
        assert!(repos.iter().any(|path| path.ends_with("listed")));
        assert!(repos.iter().all(|path| !path.ends_with("extra")));
    }

    #[test]
    fn discover_git_repos_follows_nested_gitmodules() {
        let dir = TempDir::new().expect("temp");
        init_git(dir.path());
        let child = dir.path().join("child");
        init_git(&child);
        let nested = child.join("nested");
        init_git(&nested);
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"child\"]\n\tpath = child\n",
        )
        .expect("root gitmodules");
        fs::write(
            child.join(".gitmodules"),
            "[submodule \"nested\"]\n\tpath = nested\n",
        )
        .expect("child gitmodules");

        let repos = discover_git_repos(dir.path());
        assert!(repos.iter().any(|path| path.ends_with("child")));
        assert!(repos.iter().any(|path| path.ends_with("nested")));
    }

    #[test]
    fn discover_git_repos_includes_gitmodules_paths() {
        let dir = TempDir::new().expect("temp");
        init_git(dir.path());
        let child = dir.path().join("vendor_like");
        init_git(&child);
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"vendor_like\"]\n\tpath = vendor_like\n",
        )
        .expect("gitmodules");

        let repos = discover_git_repos(dir.path());
        assert!(repos.iter().any(|path| path.ends_with("vendor_like")));
    }

    #[test]
    fn discover_git_repos_accepts_quoted_paths_and_skips_escape_paths() {
        let dir = TempDir::new().expect("temp");
        init_git(dir.path());
        let child = dir.path().join("my child");
        init_git(&child);
        let outside = dir.path().join("outside-name");
        init_git(&outside);
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"my child\"]\n\tpath = \"my child\"\n[submodule \"up\"]\n\tpath = ../outside\n[submodule \"abs\"]\n\tpath = /tmp/outside\n",
        )
        .expect("gitmodules");

        let repos = discover_git_repos(dir.path());
        assert!(repos.iter().any(|path| path.ends_with("my child")));
        assert!(repos.iter().all(|path| !path.ends_with("outside-name")));
    }

    #[test]
    fn missing_submodule_checkout_does_not_walk_other_repos() {
        let dir = TempDir::new().expect("temp");
        init_git(dir.path());
        let extra = dir.path().join("extra");
        init_git(&extra);
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"missing\"]\n\tpath = missing\n",
        )
        .expect("gitmodules");

        let repos = discover_git_repos(dir.path());
        assert_eq!(repos.len(), 1);
        assert!(repos.iter().all(|path| !path.ends_with("extra")));
    }

    #[cfg(unix)]
    #[test]
    fn discover_git_repos_rejects_symlink_that_leaves_the_root() {
        let dir = TempDir::new().expect("temp");
        let outside = TempDir::new().expect("outside");
        init_git(dir.path());
        init_git(outside.path());
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).expect("symlink");
        fs::write(
            dir.path().join(".gitmodules"),
            "[submodule \"link\"]\n\tpath = link\n",
        )
        .expect("gitmodules");

        let repos = discover_git_repos(dir.path());
        let outside_root = fs::canonicalize(outside.path()).expect("outside canonical");
        assert!(repos.iter().all(|path| path != &outside_root));
        assert_eq!(repos.len(), 1);
    }
}
