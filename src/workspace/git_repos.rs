//! Nested git worktree discovery for per-repo embedding stores.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use caliche_sandbox::EntryKind;

use crate::agent::tools::tree_walk::{WalkOptions, WalkedEntry, walk_tree_pruned};
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
    let base = canonical_or_owned(root);
    let mut repos = vec![base.clone()];
    for path in gitmodules_worktrees(&base, &base) {
        append_declared_repo(&base, &mut repos, path);
    }
    Some(repos)
}

/// True when `.gitmodules` lists at least one safe relative submodule path.
#[must_use]
pub(crate) fn uses_declared_submodules(root: &Path) -> bool {
    let base = canonical_or_owned(root);
    let Some(contents) = read_gitmodules(&base, &base) else {
        return false;
    };
    !parse_gitmodules(&contents).is_empty()
}

/// The `.gitmodules` text of the repository at `root`, opened from the
/// canonical top root `base` with no link followed on the way and read up
/// to `MAX_DISCOVERED_FILE_BYTES` (1 MiB). A sandboxed command can write
/// these files and swap a nested repository for a link after discovery
/// checked it, so a link anywhere below `base`, a FIFO or an oversized file
/// is skipped with a warning. Links above `base` (such as `/var` on macOS)
/// were resolved once, when `base` was made canonical.
fn read_gitmodules(base: &Path, root: &Path) -> Option<String> {
    crate::discovery::auto::read_workspace_text(base, &root.join(".gitmodules"))
}

fn append_declared_repo(base: &Path, repos: &mut Vec<PathBuf>, path: PathBuf) {
    let before = repos.len();
    let children = gitmodules_worktrees(base, &path);
    push_unique(repos, path);
    if repos.len() == before {
        return;
    }
    for child in children {
        append_declared_repo(base, repos, child);
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

fn gitmodules_worktrees(base: &Path, root: &Path) -> Vec<PathBuf> {
    let Some(contents) = read_gitmodules(base, root) else {
        return Vec::new();
    };
    parse_gitmodules(&contents)
        .into_iter()
        .filter_map(|entry| contained_git_root(base, root, &entry.relative_path))
        .collect()
}

/// Live git root for the submodule path `relative` under `root`, kept only
/// when it resolves inside the top root `base`.
fn contained_git_root(base: &Path, root: &Path, relative: &str) -> Option<PathBuf> {
    if !is_safe_relative(relative) {
        return None;
    }
    confined_git_root(base, &root.join(relative))
}

/// The canonical form of `candidate` when it is a git worktree inside the
/// canonical top root `base` (`base` itself is deduplicated by the caller).
///
/// A sandboxed command can swap a nested directory for a link at any time,
/// so the check is made on the resolved path that is returned, never on an
/// earlier resolve of a parent.
fn confined_git_root(base: &Path, candidate: &Path) -> Option<PathBuf> {
    if !is_git_root(candidate) {
        return None;
    }
    let canonical = std::fs::canonicalize(candidate).ok()?;
    if !canonical.starts_with(base) {
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

    let skipped = |entry: &WalkedEntry| {
        entry
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(skip_discovery_dir)
    };
    let options = WalkOptions {
        max_depth: None,
        include_hidden: false,
        honor_ignores: false,
    };
    walk_tree_pruned(&canonical_root, options, &skipped, &mut |entry| {
        if entry.kind != EntryKind::Dir || skipped(entry) {
            return ControlFlow::Continue(());
        }
        append_walked_repo(&canonical_root, &mut repos, &entry.path)
    });

    for path in gitmodules_worktrees(&canonical_root, &canonical_root) {
        push_unique(&mut repos, path);
    }
    repos
}

/// Append `dir`, a directory the no-link walk listed, when it is a git
/// worktree. The walk stops once `MAX_NESTED_REPOS` are found; the cap is a
/// bound on work, not a filter. `dir` can be a link by the time it is
/// checked, so it is kept only when it resolves inside the top root `base`.
fn append_walked_repo(base: &Path, repos: &mut Vec<PathBuf>, dir: &Path) -> ControlFlow<()> {
    if repos.len() >= MAX_NESTED_REPOS {
        return ControlFlow::Break(());
    }
    if let Some(repo) = confined_git_root(base, dir) {
        push_unique(repos, repo);
    }
    ControlFlow::Continue(())
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

    /// A FIFO at `.gitmodules` (a command in the sandbox can make one) is
    /// not read: discovery returns at once and walks the tree instead.
    #[cfg(unix)]
    #[test]
    fn discover_git_repos_does_not_wait_on_a_fifo_gitmodules() {
        let dir = TempDir::new().expect("temp");
        let root = dir.path().canonicalize().expect("canonical");
        init_git(&root);
        init_git(&root.join("child"));
        let status = crate::policy::program_command("mkfifo")
            .arg(root.join(".gitmodules"))
            .status()
            .expect("run mkfifo");
        assert!(status.success());

        let (tx, rx) = std::sync::mpsc::channel();
        let walked = root.clone();
        std::thread::spawn(move || {
            let _ = tx.send(discover_git_repos(&walked));
        });
        let repos = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("discovery must not wait on a FIFO");

        assert_eq!(repos, vec![root.clone(), root.join("child")]);
    }

    /// A nested repository swapped for a link after discovery checked it:
    /// its `.gitmodules` is not read through the link, so the outside
    /// repository's submodules are not listed.
    #[cfg(unix)]
    #[test]
    fn a_nested_repository_swapped_for_a_link_is_not_read() {
        let dir = TempDir::new().expect("temp");
        let base = dir.path().join("ws");
        fs::create_dir_all(&base).expect("ws");
        let base = base.canonicalize().expect("canonical");
        init_git(&base);
        fs::create_dir_all(dir.path().join("outside")).expect("outside");
        let outside = dir
            .path()
            .join("outside")
            .canonicalize()
            .expect("canonical");
        init_git(&outside);
        init_git(&outside.join("inner"));
        fs::write(
            outside.join(".gitmodules"),
            "[submodule \"inner\"]\n\tpath = inner\n",
        )
        .expect("outside gitmodules");
        let sub = base.join("sub");
        std::os::unix::fs::symlink(&outside, &sub).expect("link");

        let mut repos = Vec::new();
        super::append_declared_repo(&base, &mut repos, sub.clone());

        assert_eq!(repos, vec![sub]);
    }

    /// A workspace `ws` (the canonical top root) with a git repo `outside`
    /// next to it that holds a nested repo `inner`.
    #[cfg(unix)]
    fn workspace_and_outside_repo(dir: &TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
        fs::create_dir_all(dir.path().join("ws")).expect("ws");
        fs::create_dir_all(dir.path().join("outside")).expect("outside");
        let base = dir.path().join("ws").canonicalize().expect("canonical");
        let outside = dir
            .path()
            .join("outside")
            .canonicalize()
            .expect("canonical");
        init_git(&base);
        init_git(&outside);
        init_git(&outside.join("inner"));
        (base, outside)
    }

    /// A nested repository swapped for a link after its `.gitmodules` was
    /// read: the submodule paths it listed resolve through the link, outside
    /// the top root, so none is kept.
    #[cfg(unix)]
    #[test]
    fn a_submodule_path_through_a_swapped_link_is_not_listed() {
        let dir = TempDir::new().expect("temp");
        let (base, outside) = workspace_and_outside_repo(&dir);
        std::os::unix::fs::symlink(&outside, base.join("sub")).expect("link");

        let listed = super::contained_git_root(&base, &base.join("sub"), "inner");

        assert_eq!(listed, None);
    }

    /// A directory the walk listed and a command then swapped for a link
    /// to a repository outside the workspace is not kept.
    #[cfg(unix)]
    #[test]
    fn a_walked_dir_swapped_for_a_link_is_not_listed() {
        let dir = TempDir::new().expect("temp");
        let (base, outside) = workspace_and_outside_repo(&dir);
        init_git(&base.join("kept"));
        std::os::unix::fs::symlink(&outside, base.join("sub")).expect("link");

        let mut repos = vec![base.clone()];
        for dir in [base.join("sub"), base.join("kept")] {
            let _ = super::append_walked_repo(&base, &mut repos, &dir);
        }

        assert_eq!(repos, vec![base.clone(), base.join("kept")]);
    }

    /// A root with `listed` and `extra` repos whose `.gitmodules` text names
    /// only `listed`, placed by `plant`. Returns the canonical root and the
    /// repos discovery finds.
    fn discover_with_gitmodules(
        plant: impl FnOnce(&std::path::Path, &str),
    ) -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
        let dir = TempDir::new().expect("temp");
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).expect("ws");
        let root = root.canonicalize().expect("canonical");
        init_git(&root);
        init_git(&root.join("listed"));
        init_git(&root.join("extra"));
        plant(&root, "[submodule \"listed\"]\n\tpath = listed\n");
        let repos = discover_git_repos(&root);
        (root, repos)
    }

    /// An oversized `.gitmodules` (a command in the sandbox can make a
    /// sparse one of any size) is not read: discovery walks the tree.
    #[test]
    fn discover_git_repos_skips_an_oversized_gitmodules() {
        let (root, mut repos) = discover_with_gitmodules(|root, text| {
            let path = root.join(".gitmodules");
            fs::write(&path, text).expect("gitmodules");
            fs::File::options()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(2 << 20))
                .expect("grow");
        });
        repos.sort();
        assert_eq!(
            repos,
            vec![root.clone(), root.join("extra"), root.join("listed")]
        );
    }

    /// A `.gitmodules` that is a link is not followed: its target's
    /// `path =` lines do not choose the repos.
    #[cfg(unix)]
    #[test]
    fn discover_git_repos_does_not_follow_a_gitmodules_link() {
        let outside = TempDir::new().expect("outside");
        let target = outside.path().join("modules");
        let (root, mut repos) = discover_with_gitmodules(|root, text| {
            fs::write(&target, text).expect("target");
            std::os::unix::fs::symlink(&target, root.join(".gitmodules")).expect("link");
        });
        repos.sort();
        assert_eq!(
            repos,
            vec![root.clone(), root.join("extra"), root.join("listed")]
        );
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
