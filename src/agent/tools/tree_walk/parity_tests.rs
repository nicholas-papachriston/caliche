//! The no-link walk against the `ignore` crate's walker it replaced.
//!
//! One fixture tree holds every ignore source: `.gitignore` with negations,
//! `.ignore`, `.caliche_ignore`, a repository's `info/exclude`, a nested
//! repository, a linked worktree, a jj repository, ignore files above any
//! repository, and a global excludes file. Each test walks it from several
//! roots and asserts the file set the old walker returned, with the
//! intended differences listed literally.
//!
//! The old walker read the global excludes from the user's git config; the
//! reference here gets the same file as an explicit ignore, so neither side
//! reads the environment. That reference differs from the old walker in
//! two ways the fixture avoids:
//! - for a file outside every repository with `require_git` set, where
//!   `the_global_excludes_need_a_repo_when_git_is_required` covers it;
//! - for a global pattern with a slash in it: the walk anchors it at the
//!   walk root, the old walker at its working directory, and git at the
//!   work tree. The fixture's global pattern has no slash.

use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::path::Path;

use super::{Ignores, WalkOptions, set_test_global_excludes, walk_tree};
use caliche_sandbox::EntryKind;

/// Files and their contents, by path below the fixture top.
const FIXTURE: &[(&str, &str)] = &[
    ("global-excludes", "*.secret\n"),
    ("outer/.gitignore", "outer_git.txt\n"),
    ("outer/.ignore", "outer_ign.txt\n"),
    ("outer/top.rs", "t"),
    ("outer/outer_git.txt", "g"),
    ("outer/outer_ign.txt", "i"),
    ("outer/repo/.git/info/exclude", "excluded.txt\n"),
    ("outer/repo/.git/worktrees/wt/commondir", "../..\n"),
    ("outer/repo/.gitignore", "*.log\n!keep.log\nbuild/\n"),
    ("outer/repo/.ignore", "ign.txt\n"),
    (
        "outer/repo/.caliche_ignore",
        "cal.txt\n!z.txt\n!forced.log\n",
    ),
    ("outer/repo/a.rs", "a"),
    ("outer/repo/x.log", "x"),
    ("outer/repo/keep.log", "k"),
    ("outer/repo/forced.log", "f"),
    ("outer/repo/cal.txt", "c"),
    ("outer/repo/ign.txt", "i"),
    ("outer/repo/excluded.txt", "e"),
    ("outer/repo/outer_git.txt", "g"),
    ("outer/repo/outer_ign.txt", "i"),
    ("outer/repo/global.secret", "s"),
    ("outer/repo/build/b.rs", "b"),
    ("outer/repo/app/.gitignore", "!x.log\nz.txt\n"),
    ("outer/repo/app/.ignore", "!ign.txt\n"),
    ("outer/repo/app/c.rs", "c"),
    ("outer/repo/app/x.log", "x"),
    ("outer/repo/app/y.log", "y"),
    ("outer/repo/app/z.txt", "z"),
    ("outer/repo/app/ign.txt", "i"),
    ("outer/repo/app/excluded.txt", "e"),
    ("outer/repo/sub/.git/info/exclude", "subex.txt\n"),
    ("outer/repo/sub/.gitignore", "subgit.txt\n"),
    ("outer/repo/sub/d.rs", "d"),
    ("outer/repo/sub/subex.txt", "s"),
    ("outer/repo/sub/subgit.txt", "s"),
    ("outer/repo/sub/excluded.txt", "e"),
    ("outer/repo/sub/x.log", "x"),
    ("outer/repo/sub/cal.txt", "c"),
    ("outer/repo/sub/outer_git.txt", "g"),
    ("outer/wt/e.rs", "e"),
    ("outer/wt/excluded.txt", "e"),
    ("outer/wt/x.log", "x"),
    ("jj/.jj/repo", "r"),
    ("jj/.gitignore", "jg.txt\n"),
    ("jj/j.rs", "j"),
    ("jj/jg.txt", "g"),
    ("norepo/.gitignore", "ng.txt\n"),
    ("norepo/n.rs", "n"),
    ("norepo/ng.txt", "g"),
];

/// The walk roots, below the fixture top; "" is the top itself.
const ROOTS: [&str; 7] = [
    "",
    "outer",
    "outer/repo",
    "outer/repo/app",
    "outer/repo/sub",
    "outer/wt",
    "norepo",
];

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().expect("tmp");
    let top = tmp.path().canonicalize().expect("top");
    for (path, text) in FIXTURE {
        let path = top.join(path);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
        std::fs::write(path, text).expect("write");
    }
    let gitdir = top.join("outer/repo/.git/worktrees/wt");
    std::fs::write(
        top.join("outer/wt/.git"),
        format!("gitdir: {}\n", gitdir.display()),
    )
    .expect("gitfile");
    (tmp, top)
}

/// The files the old walker returned below `root`, relative to it.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn reference(top: &Path, root: &Path, require_git: bool, caliche: bool) -> BTreeSet<String> {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        .standard_filters(true)
        .require_git(require_git)
        .follow_links(false)
        .git_global(false);
    if caliche {
        builder.add_custom_ignore_filename(".caliche_ignore");
    }
    assert!(builder.add_ignore(top.join("global-excludes")).is_none());
    builder
        .build()
        .map(|entry| entry.expect("entry"))
        .filter(|entry| entry.file_type().is_some_and(|kind| kind.is_file()))
        .map(|entry| relative(root, entry.path()))
        .collect()
}

/// The files the no-link walk returns below `root`, relative to it.
fn walked(top: &Path, root: &Path, ignores: Ignores) -> BTreeSet<String> {
    set_test_global_excludes(Some(top.join("global-excludes")));
    let mut files = BTreeSet::new();
    let options = WalkOptions {
        ignores,
        ..WalkOptions::default()
    };
    let errors = walk_tree(root, options, &mut |entry| {
        if entry.kind == EntryKind::File {
            files.insert(entry.relative.to_string_lossy().replace('\\', "/"));
        }
        ControlFlow::Continue(())
    });
    set_test_global_excludes(None);
    assert!(errors.is_empty(), "{errors:?}");
    files
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("below root")
        .to_string_lossy()
        .replace('\\', "/")
}

/// For each root where the walks differ, the files only the old walker
/// returned and the files only the no-link walk returns.
fn differences(ignores: Ignores) -> Vec<(String, Vec<String>, Vec<String>)> {
    let require_git = ignores != Ignores::Index;
    let caliche = ignores != Ignores::Tools;
    let (_tmp, top) = fixture();
    ROOTS
        .iter()
        .map(|root| {
            let dir = top.join(root);
            let old = reference(&top, &dir, require_git, caliche);
            let new = walked(&top, &dir, ignores);
            (
                (*root).to_owned(),
                old.difference(&new).cloned().collect::<Vec<_>>(),
                new.difference(&old).cloned().collect::<Vec<_>>(),
            )
        })
        .filter(|(_, old, new)| !old.is_empty() || !new.is_empty())
        .collect()
}

/// The files the no-link walk returns from the fixture's repository.
fn repo_files(ignores: Ignores) -> Vec<String> {
    let (_tmp, top) = fixture();
    walked(&top, &top.join("outer/repo"), ignores)
        .into_iter()
        .collect()
}

fn owned(rows: &[(&str, &[&str], &[&str])]) -> Vec<(String, Vec<String>, Vec<String>)> {
    let strings = |items: &[&str]| items.iter().map(|item| (*item).to_owned()).collect();
    rows.iter()
        .map(|(root, old, new)| ((*root).to_owned(), strings(old), strings(new)))
        .collect()
}

/// The indexer walk: standard filters, `.caliche_ignore`, and
/// `require_git` clear.
///
/// Intended difference: a linked worktree's `.git` file leads to the shared
/// `info/exclude`, as in git. The old walker resolved a `.git` file only
/// with `require_git` set, so here it read no exclude for `outer/wt`.
#[test]
fn the_index_walk_matches_the_old_walker() {
    assert_eq!(
        repo_files(Ignores::Index),
        vec![
            "a.rs",
            "app/c.rs",
            "app/ign.txt",
            "app/x.log",
            "app/z.txt",
            "forced.log",
            "keep.log",
            "sub/d.rs",
        ]
    );
    assert_eq!(
        differences(Ignores::Index),
        owned(&[
            ("", &["outer/wt/excluded.txt"], &[]),
            ("outer", &["wt/excluded.txt"], &[]),
            ("outer/wt", &["excluded.txt"], &[]),
        ])
    );
}

/// The tools walk: standard filters with `require_git` set, and no
/// `.caliche_ignore`.
#[test]
fn the_tools_walk_matches_the_old_walker() {
    assert_eq!(
        repo_files(Ignores::Tools),
        vec![
            "a.rs",
            "app/c.rs",
            "app/ign.txt",
            "app/x.log",
            "cal.txt",
            "keep.log",
            "outer_git.txt",
            "sub/cal.txt",
            "sub/d.rs",
            "sub/excluded.txt",
            "sub/outer_git.txt",
            "sub/x.log",
        ]
    );
    assert_eq!(differences(Ignores::Tools), owned(&[]));
}

/// The repository map walk: standard filters with `require_git` set, and
/// `.caliche_ignore`.
#[test]
fn the_repo_map_walk_matches_the_old_walker() {
    assert_eq!(
        repo_files(Ignores::RepoMap),
        vec![
            "a.rs",
            "app/c.rs",
            "app/ign.txt",
            "app/x.log",
            "app/z.txt",
            "forced.log",
            "keep.log",
            "outer_git.txt",
            "sub/d.rs",
            "sub/excluded.txt",
            "sub/outer_git.txt",
            "sub/x.log",
        ]
    );
    assert_eq!(differences(Ignores::RepoMap), owned(&[]));
}

/// With git required, the global excludes apply only inside a repository,
/// as the old walker applied them.
#[test]
fn the_global_excludes_need_a_repo_when_git_is_required() {
    let (_tmp, top) = fixture();
    std::fs::write(top.join("norepo/plain.secret"), "s").expect("secret");

    let tools: Vec<String> = walked(&top, &top.join("norepo"), Ignores::Tools)
        .into_iter()
        .collect();
    let index: Vec<String> = walked(&top, &top.join("norepo"), Ignores::Index)
        .into_iter()
        .collect();

    assert_eq!(tools, vec!["n.rs", "ng.txt", "plain.secret"]);
    assert_eq!(index, vec!["n.rs"]);
}
