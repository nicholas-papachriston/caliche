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
//! reads the environment. That reference stands in for the old walker
//! only while the fixture's global patterns have no slash, and except for
//! a file outside every repository with `require_git` set, which
//! `the_global_excludes_need_a_repo_when_git_is_required` covers.
//!
//! The fixture leaves out, and other tests cover:
//! - a global pattern with a slash: the walk anchors it at the walk root,
//!   where git anchors it at the work tree
//!   (`a_global_pattern_with_a_slash_is_anchored_at_the_walk_root`);
//! - a `.git` or `.jj` that is a link, a FIFO or a device
//!   (`a_linked_git_dir_keeps_the_git_rules`,
//!   `a_dangling_vcs_link_makes_no_repository`,
//!   `a_vcs_entry_of_any_kind_makes_a_repository`).

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
            files.insert(entry.relative.to_string_lossy().into_owned());
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

/// A global pattern with a slash is anchored at the walk root: walking
/// `repo/app`, `sub/x.txt` hides `app/sub/x.txt`. Git anchors it at the
/// work tree instead, where it would hide `repo/sub/x.txt`; this is
/// stricter, not a leak.
#[test]
fn a_global_pattern_with_a_slash_is_anchored_at_the_walk_root() {
    let tmp = tempfile::tempdir().expect("tmp");
    let top = tmp.path().canonicalize().expect("top");
    for (path, text) in [
        ("global-excludes", "sub/x.txt\n"),
        ("repo/app/ok.rs", "o"),
        ("repo/app/sub/x.txt", "x"),
        ("repo/app/sub/y.txt", "y"),
    ] {
        let path = top.join(path);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
        std::fs::write(path, text).expect("write");
    }
    std::fs::create_dir_all(top.join("repo/.git")).expect("git dir");

    let files: Vec<String> = walked(&top, &top.join("repo/app"), Ignores::Tools)
        .into_iter()
        .collect();

    assert_eq!(files, vec!["ok.rs", "sub/y.txt"]);
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

/// A `git` command for a test oracle: no system or global config, and no
/// way to reach a network, since ssh fails and git never prompts.
pub(super) fn oracle_git() -> std::process::Command {
    let mut command = crate::policy::git_command();
    command
        .env("GIT_SSH_COMMAND", "false")
        .env("GIT_TERMINAL_PROMPT", "0");
    command
}

/// The untracked files git lists in the repository `repo` with the global
/// excludes file `global`. Hidden names are left out, since the walk never
/// lists them, and so is a nested repository, which git lists as a
/// directory and does not enter.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn git_listed(repo: &Path, global: &Path) -> BTreeSet<String> {
    let output = oracle_git()
        .arg("-c")
        .arg(format!("core.excludesFile={}", global.display()))
        .args(["-c", "core.ignorecase=false"])
        .args(["ls-files", "--others", "--exclude-standard", "-z"])
        .current_dir(repo)
        .output()
        .expect("git ls-files");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout)
        .expect("utf-8 names")
        .split('\0')
        .filter(|name| !name.is_empty() && !name.ends_with('/'))
        .filter(|name| !name.split('/').any(|part| part.starts_with('.')))
        .map(str::to_owned)
        .collect()
}

/// The files the walk with `ignores` lists in `repo`, with the global
/// excludes file `global`. Read errors are not checked here; the unit
/// tests pin them.
fn walk_listed(repo: &Path, global: &Path, ignores: Ignores) -> BTreeSet<String> {
    set_test_global_excludes(Some(global.to_path_buf()));
    let mut files = BTreeSet::new();
    let options = WalkOptions {
        ignores,
        ..WalkOptions::default()
    };
    let _ = walk_tree(repo, options, &mut |entry| {
        if entry.kind == EntryKind::File {
            files.insert(entry.relative.to_string_lossy().into_owned());
        }
        ControlFlow::Continue(())
    });
    set_test_global_excludes(None);
    files
}

/// A fresh git repository at `top/repo`; the global excludes file is
/// `top/global`, empty until a case writes it.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn git_repo(top: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let repo = top.join("repo");
    std::fs::create_dir_all(&repo).expect("repo dir");
    let status = oracle_git()
        .args(["init", "-q"])
        .current_dir(&repo)
        .status()
        .expect("git init");
    assert!(status.success());
    // git init turns folding on where the file system folds case; these
    // cases compare syntax only, so the walks read it off as git is run.
    let status = oracle_git()
        .args(["config", "core.ignorecase", "false"])
        .current_dir(&repo)
        .status()
        .expect("git config");
    assert!(status.success());
    let global = top.join("global");
    std::fs::write(&global, "").expect("global");
    (repo, global)
}

/// One walk's difference from git in one case. Rows where the two agree
/// are left out.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct GitDifference {
    case: String,
    walk: &'static str,
    /// Files git lists and the walk hides.
    only_git_lists: Vec<String>,
    /// Files the walk lists and git hides.
    only_walk_lists: Vec<String>,
}

/// One walk's difference from a reference in one case, where the reference
/// hides a file when git or the `ignore` crate's reader hides it. Rows
/// where the two agree are left out.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct ReferenceDifference {
    pub(super) case: String,
    pub(super) walk: &'static str,
    /// Files the walk lists and a reference hides.
    pub(super) leaks: Vec<String>,
    /// Files the walk hides and no reference hides.
    pub(super) over: Vec<String>,
}

fn git_differences(case: &str, repo: &Path, global: &Path) -> Vec<GitDifference> {
    let git = git_listed(repo, global);
    [("tools", Ignores::Tools), ("index", Ignores::Index)]
        .into_iter()
        .map(|(name, ignores)| {
            let walk = walk_listed(repo, global, ignores);
            GitDifference {
                case: case.to_owned(),
                walk: name,
                only_git_lists: git.difference(&walk).cloned().collect(),
                only_walk_lists: walk.difference(&git).cloned().collect(),
            }
        })
        .filter(|row| !row.only_git_lists.is_empty() || !row.only_walk_lists.is_empty())
        .collect()
}

/// Ignore file content git reads in a way a plain text read does not, and
/// the files each case adds next to `a.secret` and `ok.rs`: a byte order
/// mark before the first pattern; a line that is not UTF-8 after or before
/// the pattern; CRLF line ends, alone and with a byte order mark, an
/// escaped trailing space or a line that is not UTF-8; a NUL, which ends
/// the pattern; a negation that is not UTF-8, which git compares as bytes;
/// trailing whitespace, of which git drops only unescaped spaces; and one
/// file that mixes a pattern that does not parse, a comment and a negation
/// that are not UTF-8, and a CRLF line; braces, which git reads literally;
/// a backslash in a class; and POSIX classes such as `[:alpha:]`.
pub(super) const ENCODINGS: [(&str, &[u8], &[&str]); 35] = [
    ("bom", b"\xEF\xBB\xBF*.secret\n", &[]),
    ("bom-brace", b"\xEF\xBB\xBFa{b}\n", &["a{b}", "ab"]),
    ("bad-after", b"*.secret\n# caf\xE9\n", &[]),
    ("bad-before", b"# caf\xE9\n*.secret\n", &[]),
    ("crlf", b"*.secret\r\n", &[]),
    ("bom-crlf", b"\xEF\xBB\xBF*.secret\r\n", &[]),
    ("space-crlf", b"sp\\ \r\n", &["sp "]),
    ("bad-crlf", b"# caf\xE9\r\n*.secret\r\n", &[]),
    ("nul", b"*.secret\0junk\n", &[]),
    ("tab", b"a.secret\t\n", &["a.secret\t"]),
    (
        "bad-negation",
        b"*.txt\n!caf\xE9.txt\n",
        &["caf\u{FFFD}.txt"],
    ),
    ("tab-only", b"\t\n", &["\t"]),
    ("esc-space-space", b"sp\\  \n", &["sp "]),
    ("nbsp", b"a.secret\xC2\xA0\n", &["a.secret\u{A0}"]),
    ("negated-tab", b"*.secret\n!a.secret\t\n", &["a.secret\t"]),
    ("esc-tab", b"a.secret\\\t\n", &["a.secret\t"]),
    ("tab-space", b"a.secret\t \n", &["a.secret\t"]),
    ("esc-space-tab", b"sp\\ \t\n", &["sp \t"]),
    ("dbl-bs-space", b"a.secret\\\\ \n", &["a.secret\\"]),
    (
        "ideo-space",
        b"a.secret\xE3\x80\x80\n",
        &["a.secret\u{3000}"],
    ),
    ("formfeed", b"a.secret\x0c\n", &["a.secret\x0c"]),
    ("cr-cr", b"a.secret\r\r\n", &["a.secret\r"]),
    ("space-only-esc", b"\\ \n", &[" "]),
    ("dbl-bs-tab", b"a.secret\\\\\t\n", &["a.secret\\\t"]),
    ("trail-spaces", b"*.secret  \n", &[]),
    ("space-esc-space", b"a \\ \n", &["a", "a  "]),
    ("braces", b"a{b}.secret\n", &["a{b}.secret", "ab.secret"]),
    ("braces-comma", b"x{y,z}\n", &["x{y,z}", "xy"]),
    ("brace-tab", b"q{\t\n", &["q{\t"]),
    ("esc-in-class", b"a[\\]]b\n", &["a]b"]),
    ("posix", b"a[[:alpha:]]b\n", &["axb", "a1b"]),
    ("posix-neg", b"a[![:digit:]]b\n", &["axb", "a1b"]),
    ("posix-space", b"a[[:space:]]b\n", &["a b", "axb"]),
    ("posix-bad", b"a[[:alpha:]\xE9]b\n", &["axb", "a1b"]),
    (
        "mixed",
        b"[z-a]\n*.txt\n# caf\xE9\n!caf\xE9.txt\n*.secret\r\n",
        &["caf\u{FFFD}.txt"],
    ),
];

/// The walks against git for every encoding in every git ignore source:
/// the global excludes, `info/exclude` and `.gitignore`.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn encoding_differences() -> Vec<GitDifference> {
    let mut rows = Vec::new();
    for source in ["global", "info/exclude", ".gitignore"] {
        for &(encoding, content, extra) in ENCODINGS.iter().chain(&SYNTAX) {
            let tmp = tempfile::tempdir().expect("tmp");
            let top = tmp.path().canonicalize().expect("top");
            let (repo, global) = git_repo(&top);
            std::fs::write(repo.join("a.secret"), "s").expect("secret");
            std::fs::write(repo.join("ok.rs"), "o").expect("ok");
            for name in extra {
                let path = repo.join(name);
                std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
                std::fs::write(path, "x").expect("extra file");
            }
            let file = match source {
                "global" => global.clone(),
                "info/exclude" => repo.join(".git/info/exclude"),
                _ => repo.join(".gitignore"),
            };
            std::fs::create_dir_all(file.parent().expect("parent")).expect("dirs");
            std::fs::write(&file, content).expect("ignore file");
            rows.extend(git_differences(
                &format!("{source} {encoding}"),
                &repo,
                &global,
            ));
        }
    }
    rows
}

/// Patterns across git's wildmatch syntax in an ignore file, and the
/// files each case adds next to `a.secret` and `ok.rs`: `*`, `?` and `**`
/// in each position; leading, inner and trailing slashes; ranges, negated
/// ranges and the members `]` and `-`; backslash escapes in and out of a
/// class; every POSIX class, also with upper-case names, empty and as a
/// range end; a non-ASCII range, which git reads by bytes; braces; blank
/// lines and lines that only look like comments; `**` leading, trailing,
/// between slashes, alone and tripled; a directory and a file of one name;
/// and negations, alone, re-including files below `d/**` or below
/// directories, and with a class, an escape, braces, trailing whitespace,
/// a CRLF or a NUL.
pub(super) const SYNTAX: [(&str, &[u8], &[&str]); 110] = [
    ("star-ext", b"*.c\n", &["x.c", "d/x.c", "x.cc"]),
    ("star-mid", b"a*b\n", &["ab", "axb", "axyb", "a/b"]),
    ("star-under-dir", b"d/*\n", &["d/x", "d/e/y", "f/d/x"]),
    ("star-basename", b"x*z\n", &["xz", "d/xyz", "x/z"]),
    ("star-dir-star", b"*/x\n", &["d/x", "d/e/x", "x"]),
    ("qmark", b"?.t\n", &["a.t", "ab.t", "d/a.t"]),
    ("qmark-slash", b"a?b\n", &["axb", "a/b"]),
    (
        "dstar-lead",
        b"**/foo\n",
        &["foo", "a/foo", "a/b/foo", "a/foox"],
    ),
    (
        "dstar-lead-dir",
        b"**/d/x\n",
        &["d/x", "a/d/x", "a/b/d/x", "d/y"],
    ),
    ("dstar-trail", b"abc/**\n", &["abc/x", "abc/d/x", "abcx"]),
    (
        "dstar-mid",
        b"a/**/b\n",
        &["a/b", "a/x/b", "a/x/y/b", "a/xb"],
    ),
    ("dstar-inword", b"a**b\n", &["ab", "axb", "a/b", "a/x/b"]),
    ("dstar-after-slash", b"a/**b\n", &["a/b", "a/xb", "a/x/b"]),
    ("dstar-ext", b"**/*.c\n", &["x.c", "d/e/x.c", "d/x.h"]),
    ("lead-slash", b"/top\n", &["top", "d/top"]),
    ("lead-slash-star", b"/*.c\n", &["x.c", "d/x.c"]),
    ("mid-slash", b"d/x\n", &["d/x", "e/d/x"]),
    ("trail-slash", b"dir/\n", &["dir/x", "e/dir/y", "sub/dir"]),
    ("trail-slash-star", b"d*/\n", &["d", "dx/f"]),
    ("range", b"[a-c]x\n", &["ax", "bx", "dx"]),
    ("range-bang", b"[!a-c]x\n", &["ax", "dx"]),
    ("range-caret", b"[^a-c]x\n", &["ax", "dx"]),
    ("range-reverse", b"[c-a]x\n", &["ax", "bx"]),
    ("class-close-first", b"[]a]x\n", &["]x", "ax", "bx"]),
    ("class-neg-close", b"[!]a]x\n", &["]x", "ax", "bx"]),
    ("class-dash-end", b"[a-]x\n", &["-x", "ax", "bx"]),
    ("class-dash-start", b"[-a]x\n", &["-x", "ax", "bx"]),
    ("class-esc-close", b"[\\]]x\n", &["]x", "\\x"]),
    ("class-esc-dash", b"[a\\-c]x\n", &["-x", "ax", "bx", "cx"]),
    ("class-esc-range", b"[\\a-\\c]x\n", &["ax", "bx", "dx"]),
    ("class-slash", b"a[/]b\n", &["a/b", "axb"]),
    ("class-unclosed", b"a[bc\n", &["a[bc", "ab"]),
    ("posix-alnum", b"x[[:alnum:]]y\n", &["xay", "x1y", "x-y"]),
    ("posix-alpha", b"x[[:alpha:]]y\n", &["xay", "x1y"]),
    ("posix-blank", b"x[[:blank:]]y\n", &["x y", "x\ty", "xay"]),
    ("posix-cntrl", b"x[[:cntrl:]]y\n", &["x\u{1}y", "xay"]),
    ("posix-digit", b"x[[:digit:]]y\n", &["x1y", "xay"]),
    ("posix-graph", b"x[[:graph:]]y\n", &["x-y", "x y", "xay"]),
    ("posix-lower", b"x[[:lower:]]y\n", &["xay", "x1y"]),
    (
        "posix-print",
        b"x[[:print:]]y\n",
        &["x y", "xay", "x\u{1}y"],
    ),
    ("posix-punct", b"x[[:punct:]]y\n", &["x-y", "xay"]),
    (
        "posix-space-xy",
        b"x[[:space:]]y\n",
        &["x y", "x\ty", "xay"],
    ),
    ("posix-upper", b"x[[:upper:]]y\n", &["xAy", "x1y"]),
    ("posix-xdigit", b"x[[:xdigit:]]y\n", &["xfy", "xgy"]),
    (
        "posix-two",
        b"x[[:digit:][:punct:]]y\n",
        &["x1y", "x-y", "xay"],
    ),
    ("posix-neg-alpha", b"x[![:alpha:]]y\n", &["xay", "x1y"]),
    ("posix-unknown", b"x[[:foo:]]y\n", &["xfy", "xoy", "xf]y"]),
    ("esc-at-end", b"x\\\n", &["x\\", "x"]),
    ("comment-line", b"#x\n", &["#x"]),
    (
        "posix-first-close",
        b"[[:a]b:]\n",
        &["ab:]", "[b:]", "bb:]"],
    ),
    (
        "posix-name-unclosed",
        b"a[[:b]x\n",
        &["a[x", "a:x", "abx", "acx"],
    ),
    ("posix-empty-name", b"x[[:]y\n", &["x[y", "x:y", "x]y"]),
    (
        "posix-space-git",
        b"a[[:space:]]b\n",
        &["a\x0bb", "a\x0cb", "a b"],
    ),
    ("qmark-no-slash", b"/a?b\n", &["a/b", "axb"]),
    ("glued-dstar", b"a**/b\n", &["ab/b", "a/x/b"]),
    (
        "dstar-esc-slash",
        b"a/**\\/b\n",
        &["a/x/y/b", "a/z/b", "a/b"],
    ),
    ("esc-star", b"a\\*b\n", &["a*b", "axb"]),
    ("esc-qmark", b"a\\?b\n", &["a?b", "axb"]),
    ("esc-bracket", b"a\\[b\n", &["a[b", "ab"]),
    ("esc-hash", b"\\#x\n", &["#x"]),
    ("esc-bang", b"\\!x\n", &["!x"]),
    ("esc-backslash", b"a\\\\\n", &["a\\"]),
    ("bang-mid", b"a!b\n", &["a!b"]),
    ("braces-empty", b"x{}\n", &["x{}", "x"]),
    ("space-mid", b"a b\n", &["a b", "ab"]),
    ("negation", b"*.c\n!keep.c\n", &["keep.c", "x.c"]),
    ("negation-in-dir", b"d/*\n!d/k\n", &["d/k", "d/x"]),
    ("negation-under-dir", b"d/\n!d/x\n", &["d/x", "d/y"]),
    ("negation-star", b"*\n!*.c\n", &["x.c", "x.h", "d/y.c"]),
    (
        "negation-esc-class",
        b"*.c\n!a[\\x]b.c\n",
        &["axb.c", "ayb.c"],
    ),
    (
        "negation-posix-space",
        b"*.c\n!a[[:space:]]b.c\n",
        &["a b.c", "ayb.c"],
    ),
    (
        "negation-posix-dash",
        b"*.c\n!a[[:digit:]-]b.c\n",
        &["a1b.c", "a-b.c", "ayb.c"],
    ),
    ("negation-brace", b"*.c\n!{a}.c\n", &["{a}.c", "a.c"]),
    ("negation-escape", b"*.c\n!\\k.c\n", &["k.c"]),
    ("negation-unclosed", b"*.c\n!a[b.c\n", &["a[b.c", "ab.c"]),
    (
        "dstar-zero",
        b"a/**/b\n",
        &["a/b", "a/x/b", "a/x/y/b", "x/a/b"],
    ),
    ("dstar-lead-path", b"**/d/x\n", &["d/x", "e/d/x", "d/y"]),
    ("dstar-only", b"**\n", &["d/x"]),
    ("dstar-triple", b"a/***/b\n", &["a/b", "a/x/b", "a/x/y/b"]),
    ("dstar-reinclude", b"d/**\n!d/x\n", &["d/x", "d/y"]),
    ("star-slash", b"*/x\n", &["d/x", "d/e/x", "x"]),
    (
        "negation-dirs",
        b"*\n!*/\n!*.c\n",
        &["d/x.c", "d/x.h", "y.c"],
    ),
    ("dir-and-file", b"d\n", &["d/x", "e/d"]),
    ("blank-lines", b"\n\n*.secret\n\n", &[]),
    ("hash-after-space", b" #x\n", &[" #x", "#x"]),
    ("esc-space-lead", b"\\ a\n", &[" a", "a"]),
    (
        "posix-alpha-upper",
        b"x[[:alpha:]]y\n",
        &["xAy", "xZy", "x1y"],
    ),
    (
        "posix-alnum-upper",
        b"x[[:alnum:]]y\n",
        &["xAy", "x9y", "x-y"],
    ),
    (
        "posix-xdigit-upper",
        b"x[[:xdigit:]]y\n",
        &["xAy", "xFy", "xGy", "xby"],
    ),
    (
        "posix-upper-all",
        b"x[[:upper:]]y\n",
        &["xAy", "xZy", "xby"],
    ),
    (
        "posix-lower-all",
        b"x[[:lower:]]y\n",
        &["xay", "xzy", "xBy"],
    ),
    ("posix-empty", b"a[[:]]b\n", &["a:b", "a]b", "ab", "a[b"]),
    (
        "range-to-posix",
        b"x[a-[:alpha:]]y\n",
        &["xa]y", "xby", "x-y", "x:y"],
    ),
    (
        "range-nonascii",
        "[\u{e0}-z]*\n".as_bytes(),
        &["\u{e9}cole", "zed", "\u{e0}a", "b"],
    ),
    (
        "posix-dash-end",
        b"a[[:alpha:]-]b\n",
        &["axb", "a-b", "a1b"],
    ),
    ("class-esc-dash-mid", b"a[\\-]b\n", &["a-b", "a\\b", "axb"]),
    (
        "negation-posix-space-2",
        b"*.env\n!a[[:space:]].env\n",
        &["a .env", "ab.env"],
    ),
    (
        "negation-esc-dash",
        b"*.env\n!a[\\-].env\n",
        &["a-.env", "ab.env"],
    ),
    (
        "negation-posix-unknown",
        b"*.env\n!a[[:foo:]].env\n",
        &["af.env", "ab.env"],
    ),
    (
        "negation-posix-dash-end",
        b"*.env\n!a[[:alpha:]-].env\n",
        &["ab.env", "a-.env", "a1.env"],
    ),
    ("negation-brace-alt", b"x*\n!x{y}\n", &["x{y}", "xy"]),
    ("negation-brace-comma", b"x*\n!x{y,z}\n", &["x{y,z}", "xy"]),
    ("negation-esc-close", b"a*\n!a[\\]]\n", &["a]", "a\\"]),
    ("negation-trail-space", b"a*\n!a \n", &["a", "a "]),
    (
        "negation-after-tab",
        b"secret.env\t\n!s[[:alpha:]]cret.env\n",
        &["secret.env", "secret.env\t"],
    ),
    ("negation-crlf", b"a*\r\n!ab\r\n", &["ab", "ab\r"]),
    (
        "negation-esc-space-crlf",
        b"a*\r\n!a\\ \r\n",
        &["a ", "a\\"],
    ),
    ("negation-nul", b"a*\n!ab\0c\n", &["ab", "abc"]),
    (
        "negation-brace-tab",
        b"a*\n!a{b}\t\n",
        &["a{b}\t", "ab", "a{b}"],
    ),
    (
        "negation-posix-alpha",
        b"a*\n!a[[:alpha:]]\n",
        &["ab", "a1"],
    ),
];

/// The walks against their references for every case in [`ENCODINGS`] in
/// `.ignore` (Tools and Index) and `.caliche_ignore` (Index). git never
/// reads these files, so a file is hidden by the references when git
/// hides it with the same content as a `.gitignore`, or when the `ignore`
/// crate's own reader of the file hides it.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn caliche_only_differences() -> Vec<ReferenceDifference> {
    let sources = [
        (".ignore", "tools", Ignores::Tools),
        (".ignore", "index", Ignores::Index),
        (".caliche_ignore", "index", Ignores::Index),
    ];
    let mut rows = Vec::new();
    for (source, walk, ignores) in sources {
        for &(encoding, content, extra) in ENCODINGS.iter().chain(&SYNTAX) {
            let mut files: Vec<String> = vec!["a.secret".into(), "ok.rs".into()];
            files.extend(extra.iter().map(|name| (*name).to_owned()));
            let tmp = tempfile::tempdir().expect("tmp");
            let top = tmp.path().canonicalize().expect("top");
            let (git_ref, _) = git_repo(&top.join("git"));
            let (repo, global) = git_repo(&top.join("walk"));
            for name in &files {
                for dir in [&git_ref, &repo] {
                    let path = dir.join(name);
                    std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
                    std::fs::write(path, "x").expect("file");
                }
            }
            std::fs::write(git_ref.join(".gitignore"), content).expect(".gitignore");
            let file = repo.join(source);
            std::fs::write(&file, content).expect("ignore file");
            let git = git_listed(&git_ref, &global);
            let mut builder = ignore::gitignore::GitignoreBuilder::new(&repo);
            let _ = builder.add(&file);
            let reader = builder.build().expect("crate reader");
            let walked = walk_listed(&repo, &global, ignores);
            let hidden_by_ref = |name: &String| {
                !git.contains(name) || reader.matched(repo.join(name), false).is_ignore()
            };
            let leaks: Vec<String> = files
                .iter()
                .filter(|name| walked.contains(*name) && hidden_by_ref(name))
                .cloned()
                .collect();
            let over: Vec<String> = files
                .iter()
                .filter(|name| !walked.contains(*name) && !hidden_by_ref(name))
                .cloned()
                .collect();
            if !leaks.is_empty() || !over.is_empty() {
                rows.push(ReferenceDifference {
                    case: format!("{source} {encoding}"),
                    walk,
                    leaks,
                    over,
                });
            }
        }
    }
    rows
}

/// The walks hide every file a reference hides in `.ignore` and
/// `.caliche_ignore`, for every case in [`ENCODINGS`] and [`SYNTAX`], and
/// hide no file that no reference hides.
///
/// Limit: the crate half of the reference is the same `ignore` crate
/// builder the walks use, so this test cannot find a file the walks hide
/// that no reference hides; git is the independent half.
#[test]
fn the_walks_read_caliche_ignore_files_as_their_references_do() {
    assert_eq!(caliche_only_differences(), reference_rows(&[]));
}

/// The cases where the walks hide files git lists in a git ignore
/// source, because the `ignore` crate's reading of the same file hides
/// them, and those files. Each holds in every git source and both walks.
const GIT_SOURCE_OVER_HIDES: [(&str, &[&str]); 27] = [
    ("bom-brace", &["ab"]),
    ("tab", &["a.secret"]),
    ("nbsp", &["a.secret"]),
    ("tab-space", &["a.secret"]),
    ("ideo-space", &["a.secret"]),
    ("formfeed", &["a.secret"]),
    ("cr-cr", &["a.secret"]),
    ("braces", &["ab.secret"]),
    ("braces-comma", &["xy"]),
    ("class-esc-dash", &["bx"]),
    ("class-slash", &["a/b"]),
    ("class-unclosed", &["a[bc"]),
    ("posix-unknown", &["xf]y"]),
    ("braces-empty", &["x"]),
    ("negation-posix-space", &["a b.c"]),
    ("negation-posix-dash", &["a-b.c", "a1b.c"]),
    ("negation-brace", &["{a}.c"]),
    ("class-esc-dash-mid", &["a\\b"]),
    ("negation-posix-space-2", &["a .env"]),
    ("negation-posix-dash-end", &["a-.env", "ab.env"]),
    ("negation-brace-alt", &["x{y}"]),
    ("negation-brace-comma", &["x{y,z}"]),
    ("negation-esc-close", &["a]"]),
    ("negation-after-tab", &["secret.env"]),
    ("negation-nul", &["ab"]),
    ("negation-brace-tab", &["a{b}\t"]),
    ("negation-posix-alpha", &["ab"]),
];

/// The walks list what git lists for every case in [`ENCODINGS`] and
/// [`SYNTAX`], in every git ignore source, except the files of
/// [`GIT_SOURCE_OVER_HIDES`], which they hide. They list no file git
/// hides.
#[test]
fn the_walks_read_ignore_file_encodings_as_git_does() {
    let mut expected = Vec::new();
    for source in ["global", "info/exclude", ".gitignore"] {
        for (case, _, _) in ENCODINGS.iter().chain(&SYNTAX) {
            let over = GIT_SOURCE_OVER_HIDES.iter().find(|(name, _)| name == case);
            for walk in ["tools", "index"] {
                if let Some((_, files)) = over {
                    expected.push((format!("{source} {case}"), walk, *files));
                }
            }
        }
    }
    let expected: Vec<(&str, &'static str, &[&str], &[&str])> = expected
        .iter()
        .map(|(case, walk, files)| (case.as_str(), *walk, *files, &[][..]))
        .collect();
    assert_eq!(encoding_differences(), literal_rows(&expected));
}

/// Ignore files at several levels and in several sources, each as (case,
/// files and their content, files the case adds next to `a.secret` and
/// `ok.rs`): patterns in `d/.gitignore`, which anchor at `d`; a deeper
/// file over a shallower one; a negation below an excluded directory,
/// whose ignore file git does not read; and `.gitignore` over
/// `info/exclude` over the global excludes.
type Layered = (
    &'static str,
    &'static [(&'static str, &'static [u8])],
    &'static [&'static str],
);

pub(super) const LAYERED: [Layered; 14] = [
    (
        "nested-anchored",
        &[("d/.gitignore", b"/a\n")],
        &["d/a", "d/e/a", "a"],
    ),
    (
        "nested-middle-slash",
        &[("d/.gitignore", b"e/a\n")],
        &["d/e/a", "d/f/e/a", "e/a"],
    ),
    (
        "nested-basename",
        &[("d/.gitignore", b"a\n")],
        &["d/a", "d/e/a", "a"],
    ),
    (
        "nested-dstar-lead",
        &[("d/.gitignore", b"**/a\n")],
        &["d/a", "d/e/a", "a"],
    ),
    (
        "nested-dstar-trail",
        &[("d/.gitignore", b"e/**\n")],
        &["d/e/a", "d/e/f/a", "e/a"],
    ),
    (
        "nested-dir-only",
        &[("d/.gitignore", b"e/\n")],
        &["d/e/a", "d/f/e", "e/a"],
    ),
    (
        "nested-negation",
        &[(".gitignore", b"*.c\n"), ("d/.gitignore", b"!keep.c\n")],
        &["d/keep.c", "keep.c", "d/x.c"],
    ),
    (
        "nested-hides",
        &[(".gitignore", b"*.c\n!k.c\n"), ("d/.gitignore", b"k.c\n")],
        &["d/k.c", "k.c"],
    ),
    (
        "nested-under-excluded",
        &[(".gitignore", b"d/\n"), ("d/.gitignore", b"!x\n")],
        &["d/x", "d/y"],
    ),
    (
        "gitignore-over-global",
        &[("global", b"*.secret\n"), (".gitignore", b"!a.secret\n")],
        &[],
    ),
    (
        "gitignore-over-exclude",
        &[("info/exclude", b"x\n"), (".gitignore", b"!x\n")],
        &["x"],
    ),
    (
        "exclude-over-global",
        &[("global", b"*.c\n"), ("info/exclude", b"!y.c\n")],
        &["y.c", "z.c"],
    ),
    (
        "global-negation-loses",
        &[("global", b"!a.secret\n"), (".gitignore", b"*.secret\n")],
        &[],
    ),
    (
        "exclude-negation-loses",
        &[("info/exclude", b"!x\n"), (".gitignore", b"x\n")],
        &["x"],
    ),
];

/// The walks against git for every case in [`LAYERED`].
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn layered_differences() -> Vec<GitDifference> {
    let mut rows = Vec::new();
    for &(case, sources, extra) in &LAYERED {
        let tmp = tempfile::tempdir().expect("tmp");
        let top = tmp.path().canonicalize().expect("top");
        let (repo, global) = git_repo(&top);
        for name in ["a.secret", "ok.rs"].iter().chain(extra) {
            let path = repo.join(name);
            std::fs::create_dir_all(path.parent().expect("parent")).expect("dirs");
            std::fs::write(path, "x").expect("file");
        }
        for &(source, content) in sources {
            let file = match source {
                "global" => global.clone(),
                "info/exclude" => repo.join(".git/info/exclude"),
                _ => repo.join(source),
            };
            std::fs::create_dir_all(file.parent().expect("parent")).expect("dirs");
            std::fs::write(&file, content).expect("ignore file");
        }
        rows.extend(git_differences(case, &repo, &global));
    }
    rows
}

/// The walks apply ignore files at every level and in every git source
/// as git does, for every case in [`LAYERED`].
#[test]
fn the_walks_apply_ignore_files_by_level_and_source_as_git_does() {
    assert_eq!(layered_differences(), literal_rows(&[]));
}

/// The literal rows `(case, walk, only git lists, only the walk lists)`.
pub(super) fn literal_rows(rows: &[(&str, &'static str, &[&str], &[&str])]) -> Vec<GitDifference> {
    rows.iter()
        .map(|(case, walk, git, walked)| GitDifference {
            case: (*case).to_owned(),
            walk,
            only_git_lists: strings(git),
            only_walk_lists: strings(walked),
        })
        .collect()
}

/// The literal rows `(case, walk, leaks, over)`.
pub(super) fn reference_rows(
    rows: &[(&str, &'static str, &[&str], &[&str])],
) -> Vec<ReferenceDifference> {
    rows.iter()
        .map(|(case, walk, leaks, over)| ReferenceDifference {
            case: (*case).to_owned(),
            walk,
            leaks: strings(leaks),
            over: strings(over),
        })
        .collect()
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

/// A small seeded generator, so a failing draw can be replayed.
struct Draw(u64);

impl Draw {
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(bound).expect("bound")).expect("index")
    }

    fn pick<'a>(&mut self, parts: &[&'a str]) -> &'a str {
        parts[self.below(parts.len())]
    }

    /// One to `most` parts joined.
    fn join(&mut self, parts: &[&str], most: usize) -> String {
        let count = 1 + self.below(most);
        (0..count).map(|_| self.pick(parts)).collect()
    }
}

/// The pieces drawn patterns are made of: the bytes a matcher reads
/// specially, POSIX names known and not, and plain letters in both cases.
const PATTERN_PARTS: [&str; 39] = [
    "a",
    "b",
    "A",
    "x",
    "[",
    "]",
    ":",
    "!",
    "^",
    "-",
    "*",
    "?",
    "/",
    "\\",
    "**",
    "**/",
    "\\/",
    "**\\/",
    "/**",
    "[:alpha:]",
    "[:space:]",
    "[:lower:]",
    "[:x:]",
    "[[:",
    ":]",
    "[:a]",
    "[a-c]",
    "[A-C]",
    "[!a]",
    "[]a]",
    "[a-]",
    "[^b]",
    "[\\]]",
    "[/]",
    "[[:upper:]]",
    "[[:blank:]]",
    "[[:print:]]",
    "\\A",
    " ",
];

/// The bytes drawn file names are made of.
const NAME_BYTES: &[u8] = b"abABx[]:!^-*?\\ \t\x0b\x0c";

/// A path near `pattern`: its bytes each kept, dropped, replaced or
/// followed by another, with a `/` now and then, so the paths land on
/// both sides of what the pattern matches. Its parts are the names
/// between slashes; `None` when none is left or one would be hidden.
fn near_path(draw: &mut Draw, pattern: &str) -> Option<Vec<String>> {
    let mut text = Vec::new();
    for &byte in pattern.as_bytes() {
        let other = NAME_BYTES[draw.below(NAME_BYTES.len())];
        match draw.below(20) {
            0..=10 => text.push(byte),
            11 | 12 => {}
            13..=15 => text.push(other),
            16 | 17 => text.extend([byte, other]),
            _ => text.extend([byte, b'/']),
        }
    }
    let text = String::from_utf8(text).ok()?;
    let parts: Vec<String> = text
        .split('/')
        .filter(|part| !part.is_empty())
        .take(3)
        .map(str::to_owned)
        .collect();
    (!parts.is_empty() && !parts.iter().any(|part| part.starts_with('.'))).then_some(parts)
}

/// Drawn cases, each a `.gitignore` line and paths below its directory,
/// none two that a file system folding case would take for one, and no
/// path both a file and a directory.
fn drawn_cases(seed: u64, count: usize) -> Vec<(String, Vec<String>)> {
    let mut draw = Draw(seed);
    (0..count)
        .map(|_| {
            let pattern = draw.join(&PATTERN_PARTS, 6);
            let mut files = BTreeSet::new();
            let mut dirs = BTreeSet::new();
            let mut paths = Vec::new();
            for _ in 0..10 {
                let Some(parts) = near_path(&mut draw, &pattern) else {
                    continue;
                };
                let lower: Vec<String> = parts.iter().map(|part| part.to_lowercase()).collect();
                let prefixes: Vec<String> =
                    (1..lower.len()).map(|len| lower[..len].join("/")).collect();
                let path = lower.join("/");
                let clashes = files.contains(&path)
                    || dirs.contains(&path)
                    || prefixes.iter().any(|prefix| files.contains(prefix));
                if !clashes {
                    files.insert(path);
                    dirs.extend(prefixes);
                    paths.push(parts.join("/"));
                }
            }
            (pattern, paths)
        })
        .collect()
}

/// For drawn patterns and paths, each case a `.gitignore` in its own
/// directory, the cases where the index walk lists other files than git:
/// the pattern, the files only git lists and the files only the walk
/// lists.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn drawn_differences(seed: u64) -> Vec<(String, Vec<String>, Vec<String>)> {
    let tmp = tempfile::tempdir().expect("tmp");
    let top = tmp.path().canonicalize().expect("top");
    let (repo, global) = git_repo(&top);
    let cases = drawn_cases(seed, 800);
    for (index, (pattern, paths)) in cases.iter().enumerate() {
        let dir = repo.join(format!("c{index}"));
        std::fs::create_dir_all(&dir).expect("case dir");
        std::fs::write(dir.join(".gitignore"), format!("{pattern}\n")).expect(".gitignore");
        for path in paths {
            let file = dir.join(path);
            std::fs::create_dir_all(file.parent().expect("parent")).expect("dirs");
            std::fs::write(file, "x").expect("file");
        }
    }
    let output = oracle_git()
        .arg("-c")
        .arg(format!("core.excludesFile={}", global.display()))
        .args(["-c", "core.ignorecase=false"])
        .args(["ls-files", "--others", "--exclude-standard", "-z"])
        .current_dir(&repo)
        .output()
        .expect("git ls-files");
    assert!(output.status.success(), "{output:?}");
    let git: BTreeSet<String> = String::from_utf8(output.stdout)
        .expect("utf-8 names")
        .split('\0')
        .filter(|name| !name.is_empty() && !name.ends_with('/'))
        .map(str::to_owned)
        .collect();
    let walk = walk_listed(&repo, &global, Ignores::Index);
    let in_case = |set: &BTreeSet<String>, index: usize| -> Vec<String> {
        let prefix = format!("c{index}/");
        set.iter()
            .filter_map(|name| name.strip_prefix(&prefix))
            .filter(|name| !name.starts_with('.'))
            .map(str::to_owned)
            .collect()
    };
    cases
        .into_iter()
        .enumerate()
        .filter_map(|(index, (pattern, _))| {
            let dir = repo.join(format!("c{index}"));
            let mut builder = ignore::gitignore::GitignoreBuilder::new(&dir);
            let _ = builder.add(dir.join(".gitignore"));
            let reader = builder.build().expect("crate reader");
            // The walk prunes a hidden directory, and never decides the
            // file's own directory, so a file is hidden when it or a
            // directory between it and that directory is.
            let by_crate = |name: &String| {
                Path::new(name)
                    .ancestors()
                    .filter(|part| !part.as_os_str().is_empty())
                    .any(|part| {
                        let is_dir = part != Path::new(name);
                        reader.matched(dir.join(part), is_dir).is_ignore()
                    })
            };
            let by_ref: Vec<String> = in_case(&git, index)
                .into_iter()
                .filter(|name| !by_crate(name))
                .collect();
            let by_walk = in_case(&walk, index);
            let only_ref: Vec<String> = by_ref
                .iter()
                .filter(|name| !by_walk.contains(name))
                .cloned()
                .collect();
            let only_walk: Vec<String> = by_walk
                .iter()
                .filter(|name| !by_ref.contains(name))
                .cloned()
                .collect();
            (!only_ref.is_empty() || !only_walk.is_empty())
                .then_some((pattern, only_ref, only_walk))
        })
        .collect()
}

/// Patterns and paths drawn from the bytes a matcher reads specially are
/// listed as git lists them, less what the `ignore` crate's own reader of
/// the same file hides. The draw is seeded, so a difference names a case
/// to replay.
#[test]
fn drawn_patterns_list_what_git_and_the_crate_list() {
    let none = Vec::<(String, Vec<String>, Vec<String>)>::new();
    assert_eq!(drawn_differences(0x5eed_0001), none);
}
