//! Walk a directory tree without following a link, honoring ignore files.
//!
//! Every directory is listed with
//! [`list_resolved_dir`](crate::agent::tools::common::path::list_resolved_dir),
//! which reaches it one component at a time from the file system root and
//! refuses a link. A directory swapped for a link while the walk runs is
//! refused instead of listed, so the walk never leaves the tree it started
//! in. Links are never followed or reported. Ignore files are read with the
//! same no-link reads.

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
    /// Apply [`IGNORE_FILES`] and the root's `.git/info/exclude`.
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
    let mut errors = Vec::new();
    let mut ignores = Vec::new();
    if options.honor_ignores
        && let Some(exclude) = read_ignore(root, &root.join(".git/info/exclude"), &mut errors)
    {
        ignores.push(exclude);
    }
    let _ = walk_dir(root, root, 1, options, &mut ignores, visit, &mut errors);
    errors
}

fn walk_dir(
    root: &Path,
    dir: &Path,
    depth: usize,
    options: WalkOptions,
    ignores: &mut Vec<Gitignore>,
    visit: &mut dyn FnMut(&WalkedEntry) -> ControlFlow<()>,
    errors: &mut Vec<String>,
) -> ControlFlow<()> {
    let mut entries = match list_resolved_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            errors.push(format!("{}: {err}", dir.display()));
            return ControlFlow::Continue(());
        }
    };
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let pushed = options.honor_ignores && push_dir_ignores(dir, &entries, ignores, errors);
    let flow = visit_entries(root, dir, depth, options, &entries, ignores, visit, errors);
    if pushed {
        ignores.pop();
    }
    flow
}

#[expect(
    clippy::too_many_arguments,
    reason = "the walk state is threaded through one recursion"
)]
fn visit_entries(
    root: &Path,
    dir: &Path,
    depth: usize,
    options: WalkOptions,
    entries: &[caliche_sandbox::DirEntryBeneath],
    ignores: &mut Vec<Gitignore>,
    visit: &mut dyn FnMut(&WalkedEntry) -> ControlFlow<()>,
    errors: &mut Vec<String>,
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
            || (!options.include_hidden && name.starts_with('.'))
        {
            continue;
        }
        let path = dir.join(&entry.name);
        if options.honor_ignores && is_ignored(ignores, &path, is_dir) {
            continue;
        }
        let walked = WalkedEntry {
            relative: path.strip_prefix(root).unwrap_or(&path).to_path_buf(),
            path,
            kind: entry.kind,
            len: entry.len,
        };
        visit(&walked)?;
        if is_dir && options.max_depth.is_none_or(|max| depth < max) {
            walk_dir(
                root,
                &walked.path,
                depth + 1,
                options,
                ignores,
                visit,
                errors,
            )?;
        }
    }
    ControlFlow::Continue(())
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
