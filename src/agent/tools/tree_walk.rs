//! Walk a directory tree without following a link, honoring ignore files.
//!
//! Every directory is listed with
//! [`list_resolved_dir`](crate::agent::tools::common::path::list_resolved_dir),
//! which reaches it one component at a time from the file system root and
//! refuses a link. A directory swapped for a link while the walk runs is
//! refused instead of listed, so the walk never leaves the tree it started
//! in. Links are never followed or reported. Ignore files in the tree and
//! in the directories above it are read with the same no-link reads. The
//! user's global excludes file and a `.git` file's pointers are git
//! configuration outside the tree: they are read with plain reads, and what
//! they hold only adds ignore patterns.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use caliche_sandbox::EntryKind;
use ignore::Match;
use ignore::gitignore::{Gitignore, GitignoreBuilder};

use crate::agent::tools::common::path::{
    list_resolved_dir, open_resolved_file, read_resolved_text, resolved_metadata,
};

/// Directory names a walk never enters: version control metadata.
pub const VCS_DIRS: [&str; 4] = [".git", ".hg", ".svn", ".jj"];

/// What a walk visits.
#[derive(Debug, Clone, Copy)]
pub struct WalkOptions {
    /// How many levels below the root to visit; 1 is the root's entries.
    pub max_depth: Option<usize>,
    /// Visit names that start with `.`.
    pub include_hidden: bool,
    /// Which ignore rules apply.
    pub ignores: Ignores,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            include_hidden: false,
            ignores: Ignores::Index,
        }
    }
}

/// The ignore rules a walk applies, each as the `ignore` crate's walker
/// applied them with its standard filters for the callers this walk
/// replaced.
///
/// The sources are the `.gitignore` and `.ignore` files and the
/// `info/exclude` of a repository rooted there, for every directory walked
/// and every directory above the root, and the user's global git excludes
/// (`core.excludesFile`, else `$XDG_CONFIG_HOME/git/ignore`). A match is
/// decided by source first (`.caliche_ignore`, then `.ignore`, then
/// `.gitignore`, then `info/exclude`, then the global excludes) and, within
/// a source, by the deepest directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ignores {
    /// No ignore rules.
    Off,
    /// The rules the search and listing tools apply: the git sources count
    /// only inside a repository and stop at the nearest repository root
    /// going up, as in git (`require_git`); `.caliche_ignore` is not read.
    Tools,
    /// The rules the repository map applies: those of [`Ignores::Tools`]
    /// and `.caliche_ignore`.
    RepoMap,
    /// The rules the indexer applies: the git sources apply everywhere,
    /// through every directory above the root, and `.caliche_ignore` is
    /// read too.
    Index,
}

impl Ignores {
    /// Whether the git sources need a repository, as the `ignore` crate's
    /// `require_git`.
    fn require_git(self) -> bool {
        matches!(self, Self::Tools | Self::RepoMap)
    }

    /// Whether `.caliche_ignore` files are read.
    fn caliche(self) -> bool {
        matches!(self, Self::RepoMap | Self::Index)
    }
}

/// One file or directory the walk found.
#[derive(Debug, Clone)]
pub struct WalkedEntry {
    /// The entry's canonical path.
    pub path: PathBuf,
    /// The path relative to the walk root.
    pub relative: PathBuf,
    /// [`EntryKind::File`] or [`EntryKind::Dir`]; links are never visited.
    pub kind: EntryKind,
    /// The length of a regular file, 0 for a directory.
    pub len: u64,
}

/// Walk `root`, a canonical directory, calling `visit` for each file and
/// directory in sorted order, depth first. `visit` returns
/// [`ControlFlow::Break`] to stop.
///
/// Returns the errors met on the way, such as a directory that turned into a
/// link; the walk skips that directory and goes on.
pub fn walk_tree(
    root: &Path,
    options: WalkOptions,
    visit: &mut dyn FnMut(&WalkedEntry) -> ControlFlow<()>,
) -> Vec<String> {
    walk_tree_pruned(root, options, &|_| false, visit).errors
}

/// What [`walk_tree_pruned`] reports besides the entries it visited.
#[derive(Debug, Default)]
pub struct WalkReport {
    /// The errors met on the way, as [`walk_tree`] returns them.
    pub errors: Vec<String>,
    /// Directories the walk tried to list and could not: one swapped for a
    /// link, or a permission or I/O error, `root` included. Nothing below
    /// them was seen, so a caller must not treat their content as gone.
    #[cfg_attr(
        not(feature = "tools-rag"),
        expect(dead_code, reason = "only the workspace indexer reads it")
    )]
    pub unlisted: Vec<PathBuf>,
}

/// [`walk_tree`], but a directory for which `prune` returns true is visited
/// and not entered.
pub fn walk_tree_pruned(
    root: &Path,
    options: WalkOptions,
    prune: &dyn Fn(&WalkedEntry) -> bool,
    visit: &mut dyn FnMut(&WalkedEntry) -> ControlFlow<()>,
) -> WalkReport {
    let mut walk = Walk {
        root,
        options,
        ignores: Vec::new(),
        global: None,
        prune,
        visit,
        errors: Vec::new(),
        unlisted: Vec::new(),
    };
    if options.ignores != Ignores::Off {
        walk.global = global_excludes(root, &mut walk.errors);
        walk.ignores = outer_ignores(root, options.ignores, &mut walk.errors);
    }
    let _ = walk.dir(root, 1);
    WalkReport {
        errors: walk.errors,
        unlisted: walk.unlisted,
    }
}

/// The state one [`walk_tree`] threads through its recursion.
struct Walk<'a> {
    root: &'a Path,
    options: WalkOptions,
    /// The ignore rules of the directories above the current one, outermost
    /// first.
    ignores: Vec<DirIgnores>,
    /// The user's global git excludes.
    global: Option<Gitignore>,
    prune: &'a dyn Fn(&WalkedEntry) -> bool,
    visit: &'a mut dyn FnMut(&WalkedEntry) -> ControlFlow<()>,
    errors: Vec<String>,
    unlisted: Vec<PathBuf>,
}

impl Walk<'_> {
    fn dir(&mut self, dir: &Path, depth: usize) -> ControlFlow<()> {
        let mut entries = match list_resolved_dir(dir) {
            Ok(entries) => entries,
            Err(err) => {
                self.errors.push(format!("{}: {err}", dir.display()));
                self.unlisted.push(dir.to_path_buf());
                return ControlFlow::Continue(());
            }
        };
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        if self.options.ignores == Ignores::Off {
            return self.entries(dir, depth, &entries);
        }
        let listed = |name: &str| {
            entries
                .iter()
                .any(|entry| entry.kind == EntryKind::File && entry.name == name)
        };
        let kind_of = |name: &str| {
            entries
                .iter()
                .find(|entry| entry.name == name)
                .map(|entry| entry.kind)
        };
        let vcs = Vcs::of(dir, kind_of(".git"), kind_of(".jj"));
        let here = dir_ignores(dir, vcs, &listed, self.options.ignores, &mut self.errors);
        self.ignores.push(here);
        let flow = self.entries(dir, depth, &entries);
        self.ignores.pop();
        flow
    }

    fn entries(
        &mut self,
        dir: &Path,
        depth: usize,
        entries: &[caliche_sandbox::DirEntryBeneath],
    ) -> ControlFlow<()> {
        for entry in entries {
            let Some(name) = entry.name.to_str() else {
                continue;
            };
            let is_dir = match entry.kind {
                EntryKind::Dir => true,
                EntryKind::File => false,
                EntryKind::Other => continue,
            };
            if (is_dir && VCS_DIRS.contains(&name))
                || (!self.options.include_hidden && name.starts_with('.'))
            {
                continue;
            }
            let path = dir.join(&entry.name);
            if self.options.ignores != Ignores::Off
                && is_ignored(
                    &self.ignores,
                    self.global.as_ref(),
                    self.options.ignores.require_git(),
                    &path,
                    is_dir,
                )
            {
                continue;
            }
            let walked = WalkedEntry {
                relative: path.strip_prefix(self.root).unwrap_or(&path).to_path_buf(),
                path,
                kind: entry.kind,
                len: entry.len,
            };
            (self.visit)(&walked)?;
            if is_dir
                && self.options.max_depth.is_none_or(|max| depth < max)
                && !(self.prune)(&walked)
            {
                self.dir(&walked.path, depth + 1)?;
            }
        }
        ControlFlow::Continue(())
    }
}

/// The ignore rules of each directory above `root`, outermost first.
///
/// `root` is canonical, so every directory above it is too. The ignore
/// files there are read with no link followed; their `.git` and `.jj`
/// entries are read as in the walk, see [`Vcs::of`].
fn outer_ignores(root: &Path, rules: Ignores, errors: &mut Vec<String>) -> Vec<DirIgnores> {
    let mut above: Vec<&Path> = root.ancestors().skip(1).collect();
    above.reverse();
    above
        .into_iter()
        .map(|dir| {
            let kind_of = |name: &str| match resolved_metadata(&dir.join(name)) {
                Ok(meta) if meta.is_dir() => Some(EntryKind::Dir),
                Ok(_) => Some(EntryKind::File),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
                Err(_) => Some(EntryKind::Other),
            };
            let vcs = Vcs::of(dir, kind_of(".git"), kind_of(".jj"));
            dir_ignores(dir, vcs, &|_| true, rules, errors)
        })
        .collect()
}

/// What a directory's `.git` and `.jj` entries make it.
struct Vcs {
    /// It is a repository root: its `.git` or `.jj` resolves to a directory
    /// or a regular file, as the `ignore` crate counts one.
    is_repo: bool,
    /// The git directory whose `info/exclude` applies here.
    git_dir: Option<PathBuf>,
}

impl Vcs {
    /// Read the `.git` and `.jj` entries of `dir`, of the given kinds.
    ///
    /// A `.git` directory is the git directory, and a `.git` file points to
    /// one. A link is followed as git follows it, and counts only when it
    /// leads to a directory or a regular file: a dangling link, or one to a
    /// FIFO or a device, makes no repository.
    fn of(dir: &Path, git: Option<EntryKind>, jj: Option<EntryKind>) -> Self {
        let entry = |name: &str, kind: Option<EntryKind>| match kind? {
            EntryKind::Other => follow_link(&dir.join(name)),
            kind => Some((dir.join(name), kind)),
        };
        let git = entry(".git", git);
        let git_dir = match &git {
            Some((path, EntryKind::Dir)) => Some(path.clone()),
            Some((path, _)) => gitfile_common_dir(dir, path),
            None => None,
        };
        Self {
            is_repo: git.is_some() || entry(".jj", jj).is_some(),
            git_dir,
        }
    }
}

/// Where the link `path` leads, if it reaches a directory or a regular
/// file.
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "a .git or .jj link, followed as git follows it; the target is then described with no link followed"
    )
)]
fn follow_link(path: &Path) -> Option<(PathBuf, EntryKind)> {
    let target = std::fs::canonicalize(path).ok()?;
    let meta = resolved_metadata(&target).ok()?;
    let kind = if meta.is_dir() {
        EntryKind::Dir
    } else {
        EntryKind::File
    };
    Some((target, kind))
}

/// The ignore rules one directory adds, by source.
struct DirIgnores {
    caliche: Option<Gitignore>,
    ignore: Option<Gitignore>,
    git: Option<Gitignore>,
    /// The `info/exclude` of a repository rooted at this directory.
    exclude: Option<Gitignore>,
    /// This directory is a repository root, counted only for
    /// [`Ignores::Tools`].
    has_git: bool,
}

/// Read the ignore rules of `dir`. `listed` says which ignore file names to
/// try; a missing file adds nothing.
fn dir_ignores(
    dir: &Path,
    vcs: Vcs,
    listed: &dyn Fn(&str) -> bool,
    rules: Ignores,
    errors: &mut Vec<String>,
) -> DirIgnores {
    let mut file = |name: &str| {
        listed(name)
            .then(|| read_ignore(dir, &dir.join(name), errors))
            .flatten()
    };
    let caliche = rules.caliche().then(|| file(".caliche_ignore")).flatten();
    let ignore = file(".ignore");
    let git = file(".gitignore");
    let exclude = vcs
        .git_dir
        .and_then(|git_dir| read_ignore(dir, &git_dir.join("info").join("exclude"), errors));
    DirIgnores {
        caliche,
        ignore,
        git,
        exclude,
        has_git: rules.require_git() && vcs.is_repo,
    }
}

/// The user's global excludes as a matcher rooted at `root`, if the file
/// exists and holds a pattern.
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "git configuration outside the workspace, often a dotfiles link; it only adds ignore patterns"
    )
)]
fn global_excludes(root: &Path, errors: &mut Vec<String>) -> Option<Gitignore> {
    let path = global_excludes_path()?;
    if !path.is_file() {
        return None;
    }
    let mut builder = GitignoreBuilder::new(root);
    if let Some(err) = builder.add(&path) {
        errors.push(format!("{}: {err}", path.display()));
    }
    builder
        .build()
        .map_err(|err| errors.push(format!("{}: {err}", path.display())))
        .ok()
        .filter(|matcher| !matcher.is_empty())
}

/// Where git looks for the user's global excludes file.
#[cfg(not(test))]
fn global_excludes_path() -> Option<PathBuf> {
    ignore::gitignore::gitconfig_excludes_path()
}

#[cfg(test)]
thread_local! {
    /// The global excludes file a test's walks use; none by default, so a
    /// test neither reads nor depends on the developer's own git config.
    static TEST_GLOBAL_EXCLUDES: std::cell::RefCell<Option<PathBuf>> =
        const { std::cell::RefCell::new(None) };
}

/// Where this test thread's walks look for the global excludes file.
#[cfg(test)]
fn global_excludes_path() -> Option<PathBuf> {
    TEST_GLOBAL_EXCLUDES.with(|path| path.borrow().clone())
}

/// Use `path` as the global excludes file for walks on this test thread.
#[cfg(test)]
pub(crate) fn set_test_global_excludes(path: Option<PathBuf>) {
    TEST_GLOBAL_EXCLUDES.with(|slot| *slot.borrow_mut() = path);
}

/// The canonical common git directory a `.git` file in `dir` points to.
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "a git directory pointer, resolved as git resolves it; it only locates info/exclude, read with no link followed"
    )
)]
fn gitfile_common_dir(dir: &Path, dot_git: &Path) -> Option<PathBuf> {
    let text = read_head(dot_git)?;
    let git_dir = dir.join(text.lines().next()?.strip_prefix("gitdir:")?.trim());
    let common = match read_head(&git_dir.join("commondir")) {
        Some(text) => git_dir.join(text.lines().next()?.trim()),
        None => git_dir,
    };
    std::fs::canonicalize(common).ok()
}

/// The first 4 KiB of a small git pointer file, as text. The pointer is
/// followed as git follows it, then opened without blocking and only as a
/// regular file, so a FIFO or a device there reads as no pointer.
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "a git directory pointer, resolved as git resolves it; it is then opened with no link followed"
    )
)]
fn read_head(path: &Path) -> Option<String> {
    use std::io::Read as _;
    let mut text = String::new();
    open_resolved_file(&std::fs::canonicalize(path).ok()?)
        .ok()?
        .take(4096)
        .read_to_string(&mut text)
        .ok()?;
    Some(text)
}

/// The ignore file `file` as a matcher rooted at `root`; `None` when it is
/// missing. Read with no link followed; a link or a read error is reported.
fn read_ignore(root: &Path, file: &Path, errors: &mut Vec<String>) -> Option<Gitignore> {
    let text = match read_resolved_text(file) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return None,
        Err(err) => {
            errors.push(format!("{}: {err}", file.display()));
            return None;
        }
    };
    let mut builder = GitignoreBuilder::new(root);
    for line in text.lines() {
        let _ = builder.add_line(None, line);
    }
    builder
        .build()
        .map_err(|err| errors.push(format!("{}: {err}", file.display())))
        .ok()
}

/// Whether `path` is ignored, decided as the `ignore` crate decides: the
/// first source with a match wins, in the order `.caliche_ignore`,
/// `.ignore`, `.gitignore`, `info/exclude`, global excludes; within a
/// source, the deepest directory wins. With `require_git`, the git sources
/// count only inside a repository and only up to its root.
fn is_ignored(
    stack: &[DirIgnores],
    global: Option<&Gitignore>,
    require_git: bool,
    path: &Path,
    is_dir: bool,
) -> bool {
    let decide = |matcher: Option<&Gitignore>| match matcher?.matched(path, is_dir) {
        Match::Ignore(_) => Some(true),
        Match::Whitelist(_) => Some(false),
        Match::None => None,
    };
    let any_git = !require_git || stack.iter().any(|dir| dir.has_git);
    let mut found: [Option<bool>; 4] = [None; 4];
    let mut saw_git = false;
    for dir in stack.iter().rev() {
        let git_counts = any_git && !saw_git;
        let sources = [
            (dir.caliche.as_ref(), true),
            (dir.ignore.as_ref(), true),
            (dir.git.as_ref(), git_counts),
            (dir.exclude.as_ref(), git_counts),
        ];
        for (slot, (matcher, counts)) in found.iter_mut().zip(sources) {
            if slot.is_none() && counts {
                *slot = decide(matcher);
            }
        }
        saw_git = saw_git || dir.has_git;
    }
    let global = if any_git { decide(global) } else { None };
    found
        .into_iter()
        .chain([global])
        .flatten()
        .next()
        .unwrap_or(false)
}

/// A repository with three files a tool walk lists and an index walk
/// leaves out two of: `above.rs` is in a `.gitignore` above the repository
/// root, and `hidden.rs` is in its `.caliche_ignore`. Each holds a `main`
/// with `needle` in it. Returns the canonical repository path.
#[cfg(test)]
pub(crate) fn tool_rules_fixture(top: &Path) -> PathBuf {
    let top = top.canonicalize().expect("top");
    let repo = top.join("repo");
    std::fs::create_dir_all(repo.join(".git")).expect("git dir");
    std::fs::write(top.join(".gitignore"), "above.rs\n").expect("outer ignore");
    std::fs::write(repo.join(".caliche_ignore"), "hidden.rs\n").expect("caliche ignore");
    for name in ["a.rs", "above.rs", "hidden.rs"] {
        std::fs::write(repo.join(name), "fn main() {} // needle\n").expect("file");
    }
    repo
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn walk_names(root: &Path, options: WalkOptions) -> Vec<String> {
        let mut names = Vec::new();
        let errors = walk_tree(root, options, &mut |entry| {
            names.push(entry.relative.to_string_lossy().replace('\\', "/"));
            ControlFlow::Continue(())
        });
        assert!(errors.is_empty(), "{errors:?}");
        names
    }

    #[test]
    fn a_walk_skips_links_ignored_hidden_and_vcs_entries() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("secret.txt"), "OUTSIDE").expect("secret");
        std::fs::create_dir_all(root.join("src/deep")).expect("dirs");
        std::fs::create_dir_all(root.join(".git")).expect("git");
        std::fs::create_dir_all(root.join("target")).expect("target");
        std::fs::write(root.join("src/deep/a.rs"), "a").expect("a");
        std::fs::write(root.join("target/out.rs"), "o").expect("o");
        std::fs::write(root.join(".hidden"), "h").expect("h");
        std::fs::write(root.join(".gitignore"), "target/\n").expect("ignore");
        std::os::unix::fs::symlink(outside.path(), root.join("out")).expect("link");

        assert_eq!(
            walk_names(&root, WalkOptions::default()),
            vec!["src", "src/deep", "src/deep/a.rs"]
        );
        assert_eq!(
            walk_names(
                &root,
                WalkOptions {
                    include_hidden: true,
                    ignores: Ignores::Off,
                    max_depth: Some(1),
                }
            ),
            vec![".gitignore", ".hidden", "src", "target"]
        );
    }

    /// A directory swapped for a link after it was listed, and before the
    /// walk enters it, is refused: the walk lists nothing outside.
    #[test]
    fn a_directory_swapped_for_a_link_mid_walk_is_not_entered() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("secret.txt"), "OUTSIDE").expect("secret");
        std::fs::create_dir(root.join("sub")).expect("sub");
        std::fs::write(root.join("sub/inside.txt"), "in").expect("inside");

        let mut names = Vec::new();
        let errors = walk_tree(&root, WalkOptions::default(), &mut |entry| {
            names.push(entry.relative.to_string_lossy().replace('\\', "/"));
            if entry.relative == Path::new("sub") {
                std::fs::rename(root.join("sub"), root.join("moved")).expect("move");
                std::os::unix::fs::symlink(outside.path(), root.join("sub")).expect("swap");
            }
            ControlFlow::Continue(())
        });

        assert_eq!(names, vec!["sub"]);
        assert_eq!(errors.len(), 1, "{errors:?}");
    }

    /// A directory swapped for a link mid-walk is reported as unlisted, so a
    /// caller can tell "not seen" from "gone".
    #[test]
    fn a_directory_swapped_mid_walk_is_reported_unlisted() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::create_dir(root.join("sub")).expect("sub");

        let report = walk_tree_pruned(&root, WalkOptions::default(), &|_| false, &mut |entry| {
            if entry.relative == Path::new("sub") {
                std::fs::rename(root.join("sub"), root.join("moved")).expect("move");
                std::os::unix::fs::symlink(outside.path(), root.join("sub")).expect("swap");
            }
            ControlFlow::Continue(())
        });

        assert_eq!(report.unlisted, vec![root.join("sub")]);
    }

    /// Write `files` (path, content) below `root`, creating parents.
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn write_all(root: &Path, files: &[(&str, &str)]) {
        for (path, content) in files {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
            std::fs::write(path, content).expect("write");
        }
    }

    /// The user's global git excludes file applies to every walk.
    #[test]
    fn a_walk_honors_the_global_git_excludes() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        write_all(
            &root,
            &[
                ("config/git/ignore", "globalsecret.toml\n"),
                ("ws/globalsecret.toml", "s"),
                ("ws/ok.toml", "o"),
            ],
        );
        set_test_global_excludes(Some(root.join("config/git/ignore")));

        let names = walk_names(&root.join("ws"), WalkOptions::default());
        set_test_global_excludes(None);

        assert_eq!(names, vec!["ok.toml"]);
    }

    /// The root's own `.git/info/exclude` applies.
    #[test]
    fn a_walk_honors_the_roots_info_exclude() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        write_all(
            &root,
            &[
                (".git/info/exclude", "secret.env\n"),
                ("secret.env", "s"),
                ("ok.rs", "o"),
            ],
        );

        assert_eq!(walk_names(&root, WalkOptions::default()), vec!["ok.rs"]);
    }

    /// A walk of a folder inside a repository applies that repository's
    /// `info/exclude`, as git does.
    #[test]
    fn a_walk_honors_the_enclosing_repos_info_exclude() {
        let tmp = tempfile::tempdir().expect("tmp");
        let repo = tmp.path().canonicalize().expect("repo");
        write_all(
            &repo,
            &[
                (".git/info/exclude", "secret.env\n"),
                ("app/secret.env", "s"),
                ("app/ok.rs", "o"),
            ],
        );

        assert_eq!(
            walk_names(&repo.join("app"), WalkOptions::default()),
            vec!["ok.rs"]
        );
    }

    /// A linked worktree's `.git` file leads to the shared `info/exclude`.
    #[test]
    fn a_walk_of_a_linked_worktree_honors_the_shared_info_exclude() {
        let tmp = tempfile::tempdir().expect("tmp");
        let top = tmp.path().canonicalize().expect("top");
        write_all(
            &top,
            &[
                ("main/.git/info/exclude", "secret.env\n"),
                ("main/.git/worktrees/wt/commondir", "../..\n"),
                ("wt/secret.env", "s"),
                ("wt/ok.rs", "o"),
            ],
        );
        let gitdir = top.join("main/.git/worktrees/wt");
        std::fs::write(
            top.join("wt/.git"),
            format!("gitdir: {}\n", gitdir.display()),
        )
        .expect("gitfile");

        assert_eq!(
            walk_names(&top.join("wt"), WalkOptions::default()),
            vec!["ok.rs"]
        );
    }

    /// For the tools, git ignore files above the repository root do not
    /// apply; for the indexer they do.
    #[test]
    fn git_ignore_files_above_the_repo_root_apply_only_to_the_index_walk() {
        let tmp = tempfile::tempdir().expect("tmp");
        let top = tmp.path().canonicalize().expect("top");
        std::fs::create_dir_all(top.join("repo/.git")).expect("git dir");
        write_all(
            &top,
            &[
                (".gitignore", "result\n"),
                ("repo/result", "r"),
                ("repo/ok.rs", "o"),
            ],
        );
        let tools = WalkOptions {
            ignores: Ignores::Tools,
            ..WalkOptions::default()
        };

        assert_eq!(
            walk_names(&top.join("repo"), tools),
            vec!["ok.rs", "result"]
        );
        assert_eq!(
            walk_names(&top.join("repo"), WalkOptions::default()),
            vec!["ok.rs"]
        );
    }

    /// An ignore file in a directory above the root, inside the repository,
    /// applies; a deeper one wins over a shallower one.
    #[test]
    fn a_walk_honors_ignore_files_above_the_root() {
        let tmp = tempfile::tempdir().expect("tmp");
        let repo = tmp.path().canonicalize().expect("repo");
        std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        write_all(
            &repo,
            &[
                (".gitignore", "*.env\n"),
                ("pkg/.gitignore", "!keep.env\n"),
                ("pkg/app/secret.env", "s"),
                ("pkg/app/keep.env", "k"),
                ("pkg/app/ok.rs", "o"),
            ],
        );

        assert_eq!(
            walk_names(&repo.join("pkg/app"), WalkOptions::default()),
            vec!["keep.env", "ok.rs"]
        );
    }

    /// An ignore file above the root that is a link is not read; the walk
    /// reports it.
    #[test]
    fn a_linked_ignore_file_above_the_root_is_not_read() {
        let tmp = tempfile::tempdir().expect("tmp");
        let repo = tmp.path().canonicalize().expect("repo");
        std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        write_all(
            &repo,
            &[("real-ignore", "secret.env\n"), ("app/secret.env", "s")],
        );
        std::os::unix::fs::symlink(repo.join("real-ignore"), repo.join(".gitignore"))
            .expect("ignore link");
        let mut names = Vec::new();

        let errors = walk_tree(&repo.join("app"), WalkOptions::default(), &mut |entry| {
            names.push(entry.relative.to_string_lossy().into_owned());
            ControlFlow::Continue(())
        });

        let link = repo.join(".gitignore");
        assert_eq!(names, vec!["secret.env"]);
        assert_eq!(
            errors,
            vec![format!(
                "{}: refusing to follow symlink at '.gitignore' in '{}'",
                link.display(),
                link.display()
            )]
        );
    }

    /// The files a walk of `root` with `ignores` finds, relative to it.
    fn walk_files(root: &Path, ignores: Ignores) -> Vec<String> {
        let mut files = Vec::new();
        let options = WalkOptions {
            ignores,
            ..WalkOptions::default()
        };
        let errors = walk_tree(root, options, &mut |entry| {
            if entry.kind == EntryKind::File {
                files.push(entry.relative.to_string_lossy().replace('\\', "/"));
            }
            ControlFlow::Continue(())
        });
        assert!(errors.is_empty(), "{errors:?}");
        files
    }

    /// A `.git` that is a link to a git directory is followed, as git
    /// follows it: the repository's ignore rules and its `info/exclude`
    /// apply, and the global excludes count.
    #[test]
    fn a_linked_git_dir_keeps_the_git_rules() {
        let tmp = tempfile::tempdir().expect("tmp");
        let top = tmp.path().canonicalize().expect("top");
        write_all(
            &top,
            &[
                ("global", "*.secret\n"),
                ("store/info/exclude", "ex.txt\n"),
                ("repo/.gitignore", "gi.txt\n"),
                ("repo/a.rs", "a"),
                ("repo/ex.txt", "e"),
                ("repo/sub/b.rs", "b"),
                ("repo/sub/ex.txt", "e"),
                ("repo/sub/gi.txt", "g"),
                ("repo/sub/k.secret", "s"),
            ],
        );
        std::os::unix::fs::symlink("../store", top.join("repo/.git")).expect("git link");
        set_test_global_excludes(Some(top.join("global")));

        let tools = walk_files(&top.join("repo/sub"), Ignores::Tools);
        let index = walk_files(&top.join("repo"), Ignores::Index);
        set_test_global_excludes(None);

        assert_eq!(tools, vec!["b.rs"]);
        assert_eq!(index, vec!["a.rs", "sub/b.rs"]);
    }

    /// A dangling `.git` or `.jj` link in a subdirectory makes no
    /// repository, so the parent's `.gitignore` still applies below it.
    #[test]
    fn a_dangling_vcs_link_makes_no_repository() {
        let tmp = tempfile::tempdir().expect("tmp");
        let repo = tmp.path().canonicalize().expect("repo");
        std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        write_all(
            &repo,
            &[
                (".gitignore", "sec.txt\n"),
                ("d/ok.rs", "o"),
                ("d/sec.txt", "s"),
                ("e/sec.txt", "s"),
            ],
        );
        std::os::unix::fs::symlink("nowhere", repo.join("d/.git")).expect("git link");
        std::os::unix::fs::symlink("nowhere", repo.join("e/.jj")).expect("jj link");

        assert_eq!(walk_files(&repo, Ignores::Tools), vec!["d/ok.rs"]);
    }

    /// A pruned directory is visited and not entered.
    #[test]
    fn a_pruned_directory_is_not_entered() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        std::fs::create_dir_all(root.join("skip/inner")).expect("dirs");
        std::fs::write(root.join("skip/inner/a.rs"), "a").expect("a");
        std::fs::write(root.join("b.rs"), "b").expect("b");

        let mut names = Vec::new();
        let report = walk_tree_pruned(
            &root,
            WalkOptions::default(),
            &|entry| entry.relative == Path::new("skip"),
            &mut |entry| {
                names.push(entry.relative.to_string_lossy().replace('\\', "/"));
                ControlFlow::Continue(())
            },
        );

        assert_eq!(names, vec!["b.rs", "skip"]);
        assert!(report.unlisted.is_empty());
    }

    #[test]
    fn entry_kind_does_not_follow_a_link() {
        use crate::agent::tools::common::path::{entry_kind, is_walked_file};
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        std::fs::write(root.join("file"), "f").expect("file");
        std::os::unix::fs::symlink(root.join("file"), root.join("link")).expect("link");

        assert!(is_walked_file(&root.join("file")));
        assert!(!is_walked_file(&root.join("link")));
        assert!(!is_walked_file(&root.join("missing")));
        assert_eq!(entry_kind(&root.join("link")), Some(EntryKind::Other));
    }
}

#[cfg(all(test, unix))]
mod parity_tests;
