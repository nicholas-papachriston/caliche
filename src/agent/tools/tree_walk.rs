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

use crate::agent::tools::common::path::{list_resolved_dir, read_resolved_text};

/// Ignore files read in every directory of the walk.
pub const IGNORE_FILES: [&str; 3] = [".gitignore", ".ignore", ".caliche_ignore"];

/// Directory names a walk never enters: version control metadata.
pub const VCS_DIRS: [&str; 4] = [".git", ".hg", ".svn", ".jj"];

/// What a walk visits.
#[derive(Debug, Clone, Copy)]
pub struct WalkOptions {
    /// How many levels below the root to visit; 1 is the root's entries.
    pub max_depth: Option<usize>,
    /// Visit names that start with `.`.
    pub include_hidden: bool,
    /// Apply the user's global git excludes (`core.excludesFile`, else
    /// `$XDG_CONFIG_HOME/git/ignore`), the [`IGNORE_FILES`] of every
    /// directory above the root, the root's `.git/info/exclude`, and the
    /// [`IGNORE_FILES`] of every directory walked. A deeper file wins.
    pub honor_ignores: bool,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            max_depth: None,
            include_hidden: false,
            honor_ignores: true,
        }
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
        prune,
        visit,
        errors: Vec::new(),
        unlisted: Vec::new(),
    };
    if options.honor_ignores {
        walk.ignores = outer_ignores(root, &mut walk.errors);
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
    /// The ignore matchers of the directories above the current one.
    ignores: Vec<Gitignore>,
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
        let pushed = self.options.honor_ignores
            && push_dir_ignores(dir, &entries, &mut self.ignores, &mut self.errors);
        let flow = self.entries(dir, depth, &entries);
        if pushed {
            self.ignores.pop();
        }
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
            if self.options.honor_ignores && is_ignored(&self.ignores, &path, is_dir) {
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

/// The matchers that apply to `root` from outside the walk, least specific
/// first, in git's order:
///
/// 1. the user's global excludes (`core.excludesFile`, else
///    `$XDG_CONFIG_HOME/git/ignore`);
/// 2. the `info/exclude` of the git repository that holds `root`, which is
///    `root` itself or the nearest directory above it with a `.git`;
/// 3. the [`IGNORE_FILES`] of each directory above `root`, from that
///    repository's root down, or from the file system root when no
///    repository holds `root`.
///
/// `root` is canonical, so every directory above it is too, and the ignore
/// files above it are read with no link followed.
fn outer_ignores(root: &Path, errors: &mut Vec<String>) -> Vec<Gitignore> {
    let mut ignores = Vec::new();
    if let Some(global) = global_excludes(root, errors) {
        ignores.push(global);
    }
    let repo = enclosing_repo(root);
    if let Some(Repo {
        root: repo_root,
        exclude: Some(exclude),
    }) = &repo
        && let Some(matcher) = read_ignore(repo_root, exclude, errors)
    {
        ignores.push(matcher);
    }
    let top = repo.as_ref().map(|repo| repo.root.as_path());
    let mut above: Vec<&Path> = root
        .ancestors()
        .skip(1)
        .take_while(|dir| top.is_none_or(|top| dir.starts_with(top)))
        .collect();
    above.reverse();
    for dir in above {
        if let Some(matcher) = read_dir_ignore_files(dir, errors) {
            ignores.push(matcher);
        }
    }
    ignores
}

/// The [`IGNORE_FILES`] in `dir`, a directory above the walk, as one matcher.
fn read_dir_ignore_files(dir: &Path, errors: &mut Vec<String>) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(dir);
    let mut any = false;
    for name in IGNORE_FILES {
        match read_resolved_text(&dir.join(name)) {
            Ok(text) => {
                for line in text.lines() {
                    let _ = builder.add_line(None, line);
                }
                any = true;
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => errors.push(format!("{}: {err}", dir.join(name).display())),
        }
    }
    if !any {
        return None;
    }
    builder
        .build()
        .map_err(|err| errors.push(format!("{}: {err}", dir.display())))
        .ok()
}

/// The user's global excludes as a matcher rooted at `root`, if the file
/// exists and holds a pattern.
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

/// The git repository that holds a walk root.
struct Repo {
    /// The repository's working tree root.
    root: PathBuf,
    /// Its `info/exclude`, when its git directory could be found.
    exclude: Option<PathBuf>,
}

/// The repository whose working tree holds `root`: the nearest of `root`
/// and the directories above it that has a `.git` entry.
///
/// A `.git` directory holds `info/exclude`. A `.git` file (a linked worktree
/// or a submodule) names its git directory with `gitdir:`, and a `commondir`
/// file there names the shared one that holds `info/exclude`.
fn enclosing_repo(root: &Path) -> Option<Repo> {
    root.ancestors().find_map(|dir| {
        let dot_git = dir.join(".git");
        let meta = std::fs::symlink_metadata(&dot_git).ok()?;
        let git_dir = if meta.is_dir() {
            Some(dot_git)
        } else if meta.is_file() {
            gitfile_common_dir(dir, &dot_git)
        } else {
            None
        };
        Some(Repo {
            root: dir.to_path_buf(),
            exclude: git_dir.map(|git_dir| git_dir.join("info").join("exclude")),
        })
    })
}

/// The canonical common git directory a `.git` file in `dir` points to.
fn gitfile_common_dir(dir: &Path, dot_git: &Path) -> Option<PathBuf> {
    let text = read_head(dot_git)?;
    let git_dir = dir.join(text.lines().next()?.strip_prefix("gitdir:")?.trim());
    let common = match read_head(&git_dir.join("commondir")) {
        Some(text) => git_dir.join(text.lines().next()?.trim()),
        None => git_dir,
    };
    std::fs::canonicalize(common).ok()
}

/// The first 4 KiB of a small git pointer file, as text.
fn read_head(path: &Path) -> Option<String> {
    use std::io::Read as _;
    let mut text = String::new();
    std::fs::File::open(path)
        .ok()?
        .take(4096)
        .read_to_string(&mut text)
        .ok()?;
    Some(text)
}

/// Read the ignore files `dir` lists into one matcher.
fn push_dir_ignores(
    dir: &Path,
    entries: &[caliche_sandbox::DirEntryBeneath],
    ignores: &mut Vec<Gitignore>,
    errors: &mut Vec<String>,
) -> bool {
    let mut builder = GitignoreBuilder::new(dir);
    let mut any = false;
    for name in IGNORE_FILES {
        if !entries
            .iter()
            .any(|entry| entry.kind == EntryKind::File && entry.name == name)
        {
            continue;
        }
        match read_resolved_text(&dir.join(name)) {
            Ok(text) => {
                for line in text.lines() {
                    let _ = builder.add_line(None, line);
                }
                any = true;
            }
            Err(err) => errors.push(format!("{}: {err}", dir.join(name).display())),
        }
    }
    if !any {
        return false;
    }
    match builder.build() {
        Ok(matcher) => {
            ignores.push(matcher);
            true
        }
        Err(err) => {
            errors.push(format!("{}: {err}", dir.display()));
            false
        }
    }
}

fn read_ignore(root: &Path, file: &Path, errors: &mut Vec<String>) -> Option<Gitignore> {
    let text = read_resolved_text(file).ok()?;
    let mut builder = GitignoreBuilder::new(root);
    for line in text.lines() {
        let _ = builder.add_line(None, line);
    }
    builder
        .build()
        .map_err(|err| errors.push(format!("{}: {err}", file.display())))
        .ok()
}

/// The deepest matcher that decides wins, as in git.
fn is_ignored(ignores: &[Gitignore], path: &Path, is_dir: bool) -> bool {
    for matcher in ignores.iter().rev() {
        match matcher.matched(path, is_dir) {
            Match::Ignore(_) => return true,
            Match::Whitelist(_) => return false,
            Match::None => {}
        }
    }
    false
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
                    honor_ignores: false,
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

    /// Ignore files above the repository root do not apply.
    #[test]
    fn ignore_files_above_the_repo_root_do_not_apply() {
        let tmp = tempfile::tempdir().expect("tmp");
        let top = tmp.path().canonicalize().expect("top");
        std::fs::create_dir_all(top.join("repo/.git")).expect("git dir");
        write_all(&top, &[(".gitignore", "result\n"), ("repo/result", "r")]);

        assert_eq!(
            walk_names(&top.join("repo"), WalkOptions::default()),
            vec!["result"]
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
