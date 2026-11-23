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
    let global = top.join("global");
    std::fs::write(&global, "").expect("global");
    (repo, global)
}

/// Each walk's difference from git in one case: the case, the walk, the
/// files only git lists and the files only the walk lists. Rows where the
/// two agree are left out.
pub(super) type GitDifference = (String, &'static str, Vec<String>, Vec<String>);

fn git_differences(case: &str, repo: &Path, global: &Path) -> Vec<GitDifference> {
    let git = git_listed(repo, global);
    [("tools", Ignores::Tools), ("index", Ignores::Index)]
        .into_iter()
        .map(|(name, ignores)| {
            let walk = walk_listed(repo, global, ignores);
            (
                case.to_owned(),
                name,
                git.difference(&walk).cloned().collect::<Vec<_>>(),
                walk.difference(&git).cloned().collect::<Vec<_>>(),
            )
        })
        .filter(|(_, _, git, walk)| !git.is_empty() || !walk.is_empty())
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
pub(super) const ENCODINGS: [(&str, &[u8], &[&str]); 34] = [
    ("bom", b"\xEF\xBB\xBF*.secret\n", &[]),
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
pub(super) const SYNTAX: [(&str, &[u8], &[&str]); 100] = [
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
    ("posix-unknown", b"x[[:foo:]]y\n", &["xfy", "xoy"]),
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

/// A walk against its references for `.ignore` or `.caliche_ignore`
/// holding one case: (case, walk, files a reference hides that the walk
/// lists, files the walk hides that no reference hides).
type CalicheDifference = (String, &'static str, Vec<String>, Vec<String>);

/// The walks against their references for every case in [`ENCODINGS`] in
/// `.ignore` (Tools and Index) and `.caliche_ignore` (Index). git never
/// reads these files, so a file is hidden by the references when git
/// hides it with the same content as a `.gitignore`, or when the `ignore`
/// crate's own reader of the file hides it.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn caliche_only_differences() -> Vec<CalicheDifference> {
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
                rows.push((format!("{source} {encoding}"), walk, leaks, over));
            }
        }
    }
    rows
}

/// The walks hide every file a reference hides in `.ignore` and
/// `.caliche_ignore`, for every case in [`ENCODINGS`] and [`SYNTAX`].
///
/// Known differences: in every row the walk lists a file a reference
/// hides, for the reasons listed at
/// [`the_walks_read_ignore_file_encodings_as_git_does`]: a byte order mark,
/// a line that is not UTF-8, a NUL, trailing whitespace git keeps, braces,
/// a backslash in a class, POSIX classes, `***`, a non-ASCII range, an
/// unclosed class, and a negation the two readings do not agree on.
///
/// Limit: the crate half of the reference is the same `ignore` crate
/// builder the walks use, so this test cannot find a file the walks hide
/// that no reference hides; git is the independent half.
#[test]
fn the_walks_read_caliche_ignore_files_as_their_references_do() {
    assert_eq!(
        caliche_only_differences(),
        literal_rows(&[
            (".ignore bom", "tools", &["a.secret"], &[]),
            (".ignore bad-after", "tools", &["a.secret"], &[]),
            (".ignore bad-before", "tools", &["a.secret"], &[]),
            (".ignore bom-crlf", "tools", &["a.secret"], &[]),
            (".ignore bad-crlf", "tools", &["a.secret"], &[]),
            (".ignore nul", "tools", &["a.secret"], &[]),
            (".ignore tab", "tools", &["a.secret\t"], &[]),
            (".ignore bad-negation", "tools", &["caf\u{FFFD}.txt"], &[]),
            (".ignore tab-only", "tools", &["\t"], &[]),
            (".ignore esc-space-space", "tools", &["sp "], &[]),
            (".ignore nbsp", "tools", &["a.secret\u{a0}"], &[]),
            (".ignore negated-tab", "tools", &["a.secret"], &[]),
            (".ignore esc-tab", "tools", &["a.secret\t"], &[]),
            (".ignore tab-space", "tools", &["a.secret\t"], &[]),
            (".ignore esc-space-tab", "tools", &["sp \t"], &[]),
            (".ignore dbl-bs-space", "tools", &["a.secret\\"], &[]),
            (".ignore ideo-space", "tools", &["a.secret\u{3000}"], &[]),
            (".ignore formfeed", "tools", &["a.secret\u{c}"], &[]),
            (".ignore cr-cr", "tools", &["a.secret\r"], &[]),
            (".ignore dbl-bs-tab", "tools", &["a.secret\\\t"], &[]),
            (".ignore braces", "tools", &["a{b}.secret"], &[]),
            (".ignore braces-comma", "tools", &["x{y,z}"], &[]),
            (".ignore brace-tab", "tools", &["q{\t"], &[]),
            (".ignore esc-in-class", "tools", &["a]b"], &[]),
            (".ignore posix", "tools", &["axb"], &[]),
            (".ignore posix-neg", "tools", &["axb"], &[]),
            (".ignore posix-space", "tools", &["a b"], &[]),
            (".ignore posix-bad", "tools", &["axb"], &[]),
            (
                ".ignore mixed",
                "tools",
                &["a.secret", "caf\u{FFFD}.txt"],
                &[]
            ),
            (".ignore class-esc-close", "tools", &["]x"], &[]),
            (".ignore class-esc-dash", "tools", &["-x"], &[]),
            (".ignore class-esc-range", "tools", &["ax", "bx"], &[]),
            (".ignore posix-alnum", "tools", &["xay", "x1y"], &[]),
            (".ignore posix-alpha", "tools", &["xay"], &[]),
            (".ignore posix-blank", "tools", &["x y", "x\ty"], &[]),
            (".ignore posix-cntrl", "tools", &["x\u{1}y"], &[]),
            (".ignore posix-digit", "tools", &["x1y"], &[]),
            (".ignore posix-graph", "tools", &["x-y", "xay"], &[]),
            (".ignore posix-lower", "tools", &["xay"], &[]),
            (".ignore posix-print", "tools", &["x y", "xay"], &[]),
            (".ignore posix-punct", "tools", &["x-y"], &[]),
            (".ignore posix-space-xy", "tools", &["x y", "x\ty"], &[]),
            (".ignore posix-upper", "tools", &["xAy"], &[]),
            (".ignore posix-xdigit", "tools", &["xfy"], &[]),
            (".ignore posix-two", "tools", &["x1y", "x-y"], &[]),
            (".ignore posix-neg-alpha", "tools", &["x1y"], &[]),
            (".ignore braces-empty", "tools", &["x{}"], &[]),
            (".ignore negation-brace", "tools", &["a.c"], &[]),
            (".ignore negation-unclosed", "tools", &["a[b.c"], &[]),
            (".ignore dstar-triple", "tools", &["a/b", "a/x/y/b"], &[]),
            (".ignore posix-alpha-upper", "tools", &["xAy", "xZy"], &[]),
            (".ignore posix-alnum-upper", "tools", &["xAy", "x9y"], &[]),
            (
                ".ignore posix-xdigit-upper",
                "tools",
                &["xAy", "xFy", "xby"],
                &[]
            ),
            (".ignore posix-upper-all", "tools", &["xAy", "xZy"], &[]),
            (".ignore posix-lower-all", "tools", &["xay", "xzy"], &[]),
            (".ignore range-to-posix", "tools", &["xa]y"], &[]),
            (".ignore range-nonascii", "tools", &["école", "àa"], &[]),
            (".ignore posix-dash-end", "tools", &["axb", "a-b"], &[]),
            (".ignore negation-brace-alt", "tools", &["xy"], &[]),
            (".ignore negation-brace-comma", "tools", &["xy"], &[]),
            (
                ".ignore negation-after-tab",
                "tools",
                &["secret.env\t"],
                &[]
            ),
            (".ignore negation-brace-tab", "tools", &["ab"], &[]),
            (".ignore bom", "index", &["a.secret"], &[]),
            (".ignore bad-after", "index", &["a.secret"], &[]),
            (".ignore bad-before", "index", &["a.secret"], &[]),
            (".ignore bom-crlf", "index", &["a.secret"], &[]),
            (".ignore bad-crlf", "index", &["a.secret"], &[]),
            (".ignore nul", "index", &["a.secret"], &[]),
            (".ignore tab", "index", &["a.secret\t"], &[]),
            (".ignore bad-negation", "index", &["caf\u{FFFD}.txt"], &[]),
            (".ignore tab-only", "index", &["\t"], &[]),
            (".ignore esc-space-space", "index", &["sp "], &[]),
            (".ignore nbsp", "index", &["a.secret\u{a0}"], &[]),
            (".ignore negated-tab", "index", &["a.secret"], &[]),
            (".ignore esc-tab", "index", &["a.secret\t"], &[]),
            (".ignore tab-space", "index", &["a.secret\t"], &[]),
            (".ignore esc-space-tab", "index", &["sp \t"], &[]),
            (".ignore dbl-bs-space", "index", &["a.secret\\"], &[]),
            (".ignore ideo-space", "index", &["a.secret\u{3000}"], &[]),
            (".ignore formfeed", "index", &["a.secret\u{c}"], &[]),
            (".ignore cr-cr", "index", &["a.secret\r"], &[]),
            (".ignore dbl-bs-tab", "index", &["a.secret\\\t"], &[]),
            (".ignore braces", "index", &["a{b}.secret"], &[]),
            (".ignore braces-comma", "index", &["x{y,z}"], &[]),
            (".ignore brace-tab", "index", &["q{\t"], &[]),
            (".ignore esc-in-class", "index", &["a]b"], &[]),
            (".ignore posix", "index", &["axb"], &[]),
            (".ignore posix-neg", "index", &["axb"], &[]),
            (".ignore posix-space", "index", &["a b"], &[]),
            (".ignore posix-bad", "index", &["axb"], &[]),
            (
                ".ignore mixed",
                "index",
                &["a.secret", "caf\u{FFFD}.txt"],
                &[]
            ),
            (".ignore class-esc-close", "index", &["]x"], &[]),
            (".ignore class-esc-dash", "index", &["-x"], &[]),
            (".ignore class-esc-range", "index", &["ax", "bx"], &[]),
            (".ignore posix-alnum", "index", &["xay", "x1y"], &[]),
            (".ignore posix-alpha", "index", &["xay"], &[]),
            (".ignore posix-blank", "index", &["x y", "x\ty"], &[]),
            (".ignore posix-cntrl", "index", &["x\u{1}y"], &[]),
            (".ignore posix-digit", "index", &["x1y"], &[]),
            (".ignore posix-graph", "index", &["x-y", "xay"], &[]),
            (".ignore posix-lower", "index", &["xay"], &[]),
            (".ignore posix-print", "index", &["x y", "xay"], &[]),
            (".ignore posix-punct", "index", &["x-y"], &[]),
            (".ignore posix-space-xy", "index", &["x y", "x\ty"], &[]),
            (".ignore posix-upper", "index", &["xAy"], &[]),
            (".ignore posix-xdigit", "index", &["xfy"], &[]),
            (".ignore posix-two", "index", &["x1y", "x-y"], &[]),
            (".ignore posix-neg-alpha", "index", &["x1y"], &[]),
            (".ignore braces-empty", "index", &["x{}"], &[]),
            (".ignore negation-brace", "index", &["a.c"], &[]),
            (".ignore negation-unclosed", "index", &["a[b.c"], &[]),
            (".ignore dstar-triple", "index", &["a/b", "a/x/y/b"], &[]),
            (".ignore posix-alpha-upper", "index", &["xAy", "xZy"], &[]),
            (".ignore posix-alnum-upper", "index", &["xAy", "x9y"], &[]),
            (
                ".ignore posix-xdigit-upper",
                "index",
                &["xAy", "xFy", "xby"],
                &[]
            ),
            (".ignore posix-upper-all", "index", &["xAy", "xZy"], &[]),
            (".ignore posix-lower-all", "index", &["xay", "xzy"], &[]),
            (".ignore range-to-posix", "index", &["xa]y"], &[]),
            (".ignore range-nonascii", "index", &["école", "àa"], &[]),
            (".ignore posix-dash-end", "index", &["axb", "a-b"], &[]),
            (".ignore negation-brace-alt", "index", &["xy"], &[]),
            (".ignore negation-brace-comma", "index", &["xy"], &[]),
            (
                ".ignore negation-after-tab",
                "index",
                &["secret.env\t"],
                &[]
            ),
            (".ignore negation-brace-tab", "index", &["ab"], &[]),
            (".caliche_ignore bom", "index", &["a.secret"], &[]),
            (".caliche_ignore bad-after", "index", &["a.secret"], &[]),
            (".caliche_ignore bad-before", "index", &["a.secret"], &[]),
            (".caliche_ignore bom-crlf", "index", &["a.secret"], &[]),
            (".caliche_ignore bad-crlf", "index", &["a.secret"], &[]),
            (".caliche_ignore nul", "index", &["a.secret"], &[]),
            (".caliche_ignore tab", "index", &["a.secret\t"], &[]),
            (
                ".caliche_ignore bad-negation",
                "index",
                &["caf\u{FFFD}.txt"],
                &[]
            ),
            (".caliche_ignore tab-only", "index", &["\t"], &[]),
            (".caliche_ignore esc-space-space", "index", &["sp "], &[]),
            (".caliche_ignore nbsp", "index", &["a.secret\u{a0}"], &[]),
            (".caliche_ignore negated-tab", "index", &["a.secret"], &[]),
            (".caliche_ignore esc-tab", "index", &["a.secret\t"], &[]),
            (".caliche_ignore tab-space", "index", &["a.secret\t"], &[]),
            (".caliche_ignore esc-space-tab", "index", &["sp \t"], &[]),
            (
                ".caliche_ignore dbl-bs-space",
                "index",
                &["a.secret\\"],
                &[]
            ),
            (
                ".caliche_ignore ideo-space",
                "index",
                &["a.secret\u{3000}"],
                &[]
            ),
            (".caliche_ignore formfeed", "index", &["a.secret\u{c}"], &[]),
            (".caliche_ignore cr-cr", "index", &["a.secret\r"], &[]),
            (
                ".caliche_ignore dbl-bs-tab",
                "index",
                &["a.secret\\\t"],
                &[]
            ),
            (".caliche_ignore braces", "index", &["a{b}.secret"], &[]),
            (".caliche_ignore braces-comma", "index", &["x{y,z}"], &[]),
            (".caliche_ignore brace-tab", "index", &["q{\t"], &[]),
            (".caliche_ignore esc-in-class", "index", &["a]b"], &[]),
            (".caliche_ignore posix", "index", &["axb"], &[]),
            (".caliche_ignore posix-neg", "index", &["axb"], &[]),
            (".caliche_ignore posix-space", "index", &["a b"], &[]),
            (".caliche_ignore posix-bad", "index", &["axb"], &[]),
            (
                ".caliche_ignore mixed",
                "index",
                &["a.secret", "caf\u{FFFD}.txt"],
                &[]
            ),
            (".caliche_ignore class-esc-close", "index", &["]x"], &[]),
            (".caliche_ignore class-esc-dash", "index", &["-x"], &[]),
            (
                ".caliche_ignore class-esc-range",
                "index",
                &["ax", "bx"],
                &[]
            ),
            (".caliche_ignore posix-alnum", "index", &["xay", "x1y"], &[]),
            (".caliche_ignore posix-alpha", "index", &["xay"], &[]),
            (
                ".caliche_ignore posix-blank",
                "index",
                &["x y", "x\ty"],
                &[]
            ),
            (".caliche_ignore posix-cntrl", "index", &["x\u{1}y"], &[]),
            (".caliche_ignore posix-digit", "index", &["x1y"], &[]),
            (".caliche_ignore posix-graph", "index", &["x-y", "xay"], &[]),
            (".caliche_ignore posix-lower", "index", &["xay"], &[]),
            (".caliche_ignore posix-print", "index", &["x y", "xay"], &[]),
            (".caliche_ignore posix-punct", "index", &["x-y"], &[]),
            (
                ".caliche_ignore posix-space-xy",
                "index",
                &["x y", "x\ty"],
                &[]
            ),
            (".caliche_ignore posix-upper", "index", &["xAy"], &[]),
            (".caliche_ignore posix-xdigit", "index", &["xfy"], &[]),
            (".caliche_ignore posix-two", "index", &["x1y", "x-y"], &[]),
            (".caliche_ignore posix-neg-alpha", "index", &["x1y"], &[]),
            (".caliche_ignore braces-empty", "index", &["x{}"], &[]),
            (".caliche_ignore negation-brace", "index", &["a.c"], &[]),
            (
                ".caliche_ignore negation-unclosed",
                "index",
                &["a[b.c"],
                &[]
            ),
            (
                ".caliche_ignore dstar-triple",
                "index",
                &["a/b", "a/x/y/b"],
                &[]
            ),
            (
                ".caliche_ignore posix-alpha-upper",
                "index",
                &["xAy", "xZy"],
                &[]
            ),
            (
                ".caliche_ignore posix-alnum-upper",
                "index",
                &["xAy", "x9y"],
                &[]
            ),
            (
                ".caliche_ignore posix-xdigit-upper",
                "index",
                &["xAy", "xFy", "xby"],
                &[]
            ),
            (
                ".caliche_ignore posix-upper-all",
                "index",
                &["xAy", "xZy"],
                &[]
            ),
            (
                ".caliche_ignore posix-lower-all",
                "index",
                &["xay", "xzy"],
                &[]
            ),
            (".caliche_ignore range-to-posix", "index", &["xa]y"], &[]),
            (
                ".caliche_ignore range-nonascii",
                "index",
                &["école", "àa"],
                &[]
            ),
            (
                ".caliche_ignore posix-dash-end",
                "index",
                &["axb", "a-b"],
                &[]
            ),
            (".caliche_ignore negation-brace-alt", "index", &["xy"], &[]),
            (
                ".caliche_ignore negation-brace-comma",
                "index",
                &["xy"],
                &[]
            ),
            (
                ".caliche_ignore negation-after-tab",
                "index",
                &["secret.env\t"],
                &[]
            ),
            (".caliche_ignore negation-brace-tab", "index", &["ab"], &[]),
        ])
    );
}

/// The walks list what git lists for every case in [`ENCODINGS`] and
/// [`SYNTAX`], in every git ignore source.
///
/// Known differences, each row as the files only git lists and the files
/// only the walk lists:
/// - a byte order mark hides the first pattern of `info/exclude` and
///   `.gitignore`, and a line that is not UTF-8 drops the whole of either
///   file, the negation case's `*.txt` with it; in the global excludes it
///   drops every later pattern;
/// - a NUL does not end the pattern, so `*.secret` never matches;
/// - trailing whitespace other than unescaped spaces is trimmed, where git
///   keeps it: a tab, a form feed, a carriage return left by CRLF, a
///   no-break space or an ideographic space. The pattern then hides
///   `a.secret` in place of the file whose name ends in that character,
///   and a negation un-hides `a.secret`. A pattern of only a tab is
///   dropped, and one whose trailing whitespace starts with an escaped
///   character is left ending in a backslash and matches nothing, and
///   a tab after an escaped backslash hides the name without the tab;
/// - the crate keeps the trailing space after an escaped backslash, which
///   git drops, so the pattern matches nothing.
/// - braces are read as alternatives, where git reads them literally: the
///   walk lists `a{b}.secret`, `x{y,z}` and `x{}` and hides `ab.secret`,
///   `xy` and `x`, and `q{` with a trailing tab does not parse;
/// - a class does not read a backslash as an escape, and POSIX classes
///   such as `[:alpha:]` are not supported, so classes with either hide
///   other files than git's;
/// - the walk hides `a/b` for `a[/]b` and `a[bc` for an unclosed `a[bc`,
///   which git lists, and a negation with an unclosed class un-hides a
///   file;
/// - `***` is read as `**`, where git reads it as `*`, so `a/***/b` hides
///   `a/b` and `a/x/y/b`;
/// - a range is compared by character, where git compares bytes, so
///   `[à-z]` does not parse and `[à-z]*` hides nothing;
/// - a POSIX class is not supported in any position, also with upper-case
///   names or as the end of a range (`x[a-[:alpha:]]y`);
/// - a negation with braces, a class git reads otherwise, or trailing
///   whitespace un-hides other files than git's.
#[test]
fn the_walks_read_ignore_file_encodings_as_git_does() {
    assert_eq!(
        encoding_differences(),
        literal_rows(&[
            ("global bad-before", "tools", &[], &["a.secret"]),
            ("global bad-before", "index", &[], &["a.secret"]),
            ("global bad-crlf", "tools", &[], &["a.secret"]),
            ("global bad-crlf", "index", &[], &["a.secret"]),
            ("global nul", "tools", &[], &["a.secret"]),
            ("global nul", "index", &[], &["a.secret"]),
            ("global tab", "tools", &["a.secret"], &["a.secret\t"]),
            ("global tab", "index", &["a.secret"], &["a.secret\t"]),
            ("global tab-only", "tools", &[], &["\t"]),
            ("global tab-only", "index", &[], &["\t"]),
            ("global esc-space-space", "tools", &[], &["sp "]),
            ("global esc-space-space", "index", &[], &["sp "]),
            ("global nbsp", "tools", &["a.secret"], &["a.secret\u{a0}"]),
            ("global nbsp", "index", &["a.secret"], &["a.secret\u{a0}"]),
            ("global negated-tab", "tools", &[], &["a.secret"]),
            ("global negated-tab", "index", &[], &["a.secret"]),
            ("global esc-tab", "tools", &[], &["a.secret\t"]),
            ("global esc-tab", "index", &[], &["a.secret\t"]),
            ("global tab-space", "tools", &["a.secret"], &["a.secret\t"]),
            ("global tab-space", "index", &["a.secret"], &["a.secret\t"]),
            ("global esc-space-tab", "tools", &[], &["sp \t"]),
            ("global esc-space-tab", "index", &[], &["sp \t"]),
            ("global dbl-bs-space", "tools", &[], &["a.secret\\"]),
            ("global dbl-bs-space", "index", &[], &["a.secret\\"]),
            (
                "global ideo-space",
                "tools",
                &["a.secret"],
                &["a.secret\u{3000}"]
            ),
            (
                "global ideo-space",
                "index",
                &["a.secret"],
                &["a.secret\u{3000}"]
            ),
            (
                "global formfeed",
                "tools",
                &["a.secret"],
                &["a.secret\u{c}"]
            ),
            (
                "global formfeed",
                "index",
                &["a.secret"],
                &["a.secret\u{c}"]
            ),
            ("global cr-cr", "tools", &["a.secret"], &["a.secret\r"]),
            ("global cr-cr", "index", &["a.secret"], &["a.secret\r"]),
            ("global dbl-bs-tab", "tools", &[], &["a.secret\\\t"]),
            ("global dbl-bs-tab", "index", &[], &["a.secret\\\t"]),
            ("global braces", "tools", &["ab.secret"], &["a{b}.secret"]),
            ("global braces", "index", &["ab.secret"], &["a{b}.secret"]),
            ("global braces-comma", "tools", &["xy"], &["x{y,z}"]),
            ("global braces-comma", "index", &["xy"], &["x{y,z}"]),
            ("global brace-tab", "tools", &[], &["q{\t"]),
            ("global brace-tab", "index", &[], &["q{\t"]),
            ("global esc-in-class", "tools", &[], &["a]b"]),
            ("global esc-in-class", "index", &[], &["a]b"]),
            ("global posix", "tools", &[], &["axb"]),
            ("global posix", "index", &[], &["axb"]),
            ("global posix-neg", "tools", &[], &["axb"]),
            ("global posix-neg", "index", &[], &["axb"]),
            ("global posix-space", "tools", &[], &["a b"]),
            ("global posix-space", "index", &[], &["a b"]),
            ("global posix-bad", "tools", &[], &["axb"]),
            ("global posix-bad", "index", &[], &["axb"]),
            ("global mixed", "tools", &[], &["a.secret"]),
            ("global mixed", "index", &[], &["a.secret"]),
            ("global class-esc-close", "tools", &[], &["]x"]),
            ("global class-esc-close", "index", &[], &["]x"]),
            ("global class-esc-dash", "tools", &["bx"], &["-x"]),
            ("global class-esc-dash", "index", &["bx"], &["-x"]),
            ("global class-esc-range", "tools", &[], &["ax", "bx"]),
            ("global class-esc-range", "index", &[], &["ax", "bx"]),
            ("global class-slash", "tools", &["a/b"], &[]),
            ("global class-slash", "index", &["a/b"], &[]),
            ("global class-unclosed", "tools", &["a[bc"], &[]),
            ("global class-unclosed", "index", &["a[bc"], &[]),
            ("global posix-alnum", "tools", &[], &["x1y", "xay"]),
            ("global posix-alnum", "index", &[], &["x1y", "xay"]),
            ("global posix-alpha", "tools", &[], &["xay"]),
            ("global posix-alpha", "index", &[], &["xay"]),
            ("global posix-blank", "tools", &[], &["x\ty", "x y"]),
            ("global posix-blank", "index", &[], &["x\ty", "x y"]),
            ("global posix-cntrl", "tools", &[], &["x\u{1}y"]),
            ("global posix-cntrl", "index", &[], &["x\u{1}y"]),
            ("global posix-digit", "tools", &[], &["x1y"]),
            ("global posix-digit", "index", &[], &["x1y"]),
            ("global posix-graph", "tools", &[], &["x-y", "xay"]),
            ("global posix-graph", "index", &[], &["x-y", "xay"]),
            ("global posix-lower", "tools", &[], &["xay"]),
            ("global posix-lower", "index", &[], &["xay"]),
            ("global posix-print", "tools", &[], &["x y", "xay"]),
            ("global posix-print", "index", &[], &["x y", "xay"]),
            ("global posix-punct", "tools", &[], &["x-y"]),
            ("global posix-punct", "index", &[], &["x-y"]),
            ("global posix-space-xy", "tools", &[], &["x\ty", "x y"]),
            ("global posix-space-xy", "index", &[], &["x\ty", "x y"]),
            ("global posix-upper", "tools", &[], &["xAy"]),
            ("global posix-upper", "index", &[], &["xAy"]),
            ("global posix-xdigit", "tools", &[], &["xfy"]),
            ("global posix-xdigit", "index", &[], &["xfy"]),
            ("global posix-two", "tools", &[], &["x-y", "x1y"]),
            ("global posix-two", "index", &[], &["x-y", "x1y"]),
            ("global posix-neg-alpha", "tools", &[], &["x1y"]),
            ("global posix-neg-alpha", "index", &[], &["x1y"]),
            ("global braces-empty", "tools", &["x"], &["x{}"]),
            ("global braces-empty", "index", &["x"], &["x{}"]),
            ("global negation-posix-space", "tools", &["a b.c"], &[]),
            ("global negation-posix-space", "index", &["a b.c"], &[]),
            (
                "global negation-posix-dash",
                "tools",
                &["a-b.c", "a1b.c"],
                &[]
            ),
            (
                "global negation-posix-dash",
                "index",
                &["a-b.c", "a1b.c"],
                &[]
            ),
            ("global negation-brace", "tools", &["{a}.c"], &["a.c"]),
            ("global negation-brace", "index", &["{a}.c"], &["a.c"]),
            ("global negation-unclosed", "tools", &[], &["a[b.c"]),
            ("global negation-unclosed", "index", &[], &["a[b.c"]),
            ("global dstar-triple", "tools", &[], &["a/b", "a/x/y/b"]),
            ("global dstar-triple", "index", &[], &["a/b", "a/x/y/b"]),
            ("global posix-alpha-upper", "tools", &[], &["xAy", "xZy"]),
            ("global posix-alpha-upper", "index", &[], &["xAy", "xZy"]),
            ("global posix-alnum-upper", "tools", &[], &["x9y", "xAy"]),
            ("global posix-alnum-upper", "index", &[], &["x9y", "xAy"]),
            (
                "global posix-xdigit-upper",
                "tools",
                &[],
                &["xAy", "xFy", "xby"]
            ),
            (
                "global posix-xdigit-upper",
                "index",
                &[],
                &["xAy", "xFy", "xby"]
            ),
            ("global posix-upper-all", "tools", &[], &["xAy", "xZy"]),
            ("global posix-upper-all", "index", &[], &["xAy", "xZy"]),
            ("global posix-lower-all", "tools", &[], &["xay", "xzy"]),
            ("global posix-lower-all", "index", &[], &["xay", "xzy"]),
            ("global range-to-posix", "tools", &[], &["xa]y"]),
            ("global range-to-posix", "index", &[], &["xa]y"]),
            ("global range-nonascii", "tools", &[], &["àa", "école"]),
            ("global range-nonascii", "index", &[], &["àa", "école"]),
            ("global posix-dash-end", "tools", &[], &["a-b", "axb"]),
            ("global posix-dash-end", "index", &[], &["a-b", "axb"]),
            ("global class-esc-dash-mid", "tools", &["a\\b"], &[]),
            ("global class-esc-dash-mid", "index", &["a\\b"], &[]),
            ("global negation-posix-space-2", "tools", &["a .env"], &[]),
            ("global negation-posix-space-2", "index", &["a .env"], &[]),
            (
                "global negation-posix-dash-end",
                "tools",
                &["a-.env", "ab.env"],
                &[]
            ),
            (
                "global negation-posix-dash-end",
                "index",
                &["a-.env", "ab.env"],
                &[]
            ),
            ("global negation-brace-alt", "tools", &["x{y}"], &["xy"]),
            ("global negation-brace-alt", "index", &["x{y}"], &["xy"]),
            ("global negation-brace-comma", "tools", &["x{y,z}"], &["xy"]),
            ("global negation-brace-comma", "index", &["x{y,z}"], &["xy"]),
            ("global negation-esc-close", "tools", &["a]"], &[]),
            ("global negation-esc-close", "index", &["a]"], &[]),
            (
                "global negation-after-tab",
                "tools",
                &["secret.env"],
                &["secret.env\t"]
            ),
            (
                "global negation-after-tab",
                "index",
                &["secret.env"],
                &["secret.env\t"]
            ),
            ("global negation-nul", "tools", &["ab"], &[]),
            ("global negation-nul", "index", &["ab"], &[]),
            ("global negation-brace-tab", "tools", &["a{b}\t"], &["ab"]),
            ("global negation-brace-tab", "index", &["a{b}\t"], &["ab"]),
            ("global negation-posix-alpha", "tools", &["ab"], &[]),
            ("global negation-posix-alpha", "index", &["ab"], &[]),
            ("info/exclude bom", "tools", &[], &["a.secret"]),
            ("info/exclude bom", "index", &[], &["a.secret"]),
            ("info/exclude bad-after", "tools", &[], &["a.secret"]),
            ("info/exclude bad-after", "index", &[], &["a.secret"]),
            ("info/exclude bad-before", "tools", &[], &["a.secret"]),
            ("info/exclude bad-before", "index", &[], &["a.secret"]),
            ("info/exclude bom-crlf", "tools", &[], &["a.secret"]),
            ("info/exclude bom-crlf", "index", &[], &["a.secret"]),
            ("info/exclude bad-crlf", "tools", &[], &["a.secret"]),
            ("info/exclude bad-crlf", "index", &[], &["a.secret"]),
            ("info/exclude nul", "tools", &[], &["a.secret"]),
            ("info/exclude nul", "index", &[], &["a.secret"]),
            ("info/exclude tab", "tools", &["a.secret"], &["a.secret\t"]),
            ("info/exclude tab", "index", &["a.secret"], &["a.secret\t"]),
            (
                "info/exclude bad-negation",
                "tools",
                &[],
                &["caf\u{FFFD}.txt"]
            ),
            (
                "info/exclude bad-negation",
                "index",
                &[],
                &["caf\u{FFFD}.txt"]
            ),
            ("info/exclude tab-only", "tools", &[], &["\t"]),
            ("info/exclude tab-only", "index", &[], &["\t"]),
            ("info/exclude esc-space-space", "tools", &[], &["sp "]),
            ("info/exclude esc-space-space", "index", &[], &["sp "]),
            (
                "info/exclude nbsp",
                "tools",
                &["a.secret"],
                &["a.secret\u{a0}"]
            ),
            (
                "info/exclude nbsp",
                "index",
                &["a.secret"],
                &["a.secret\u{a0}"]
            ),
            ("info/exclude negated-tab", "tools", &[], &["a.secret"]),
            ("info/exclude negated-tab", "index", &[], &["a.secret"]),
            ("info/exclude esc-tab", "tools", &[], &["a.secret\t"]),
            ("info/exclude esc-tab", "index", &[], &["a.secret\t"]),
            (
                "info/exclude tab-space",
                "tools",
                &["a.secret"],
                &["a.secret\t"]
            ),
            (
                "info/exclude tab-space",
                "index",
                &["a.secret"],
                &["a.secret\t"]
            ),
            ("info/exclude esc-space-tab", "tools", &[], &["sp \t"]),
            ("info/exclude esc-space-tab", "index", &[], &["sp \t"]),
            ("info/exclude dbl-bs-space", "tools", &[], &["a.secret\\"]),
            ("info/exclude dbl-bs-space", "index", &[], &["a.secret\\"]),
            (
                "info/exclude ideo-space",
                "tools",
                &["a.secret"],
                &["a.secret\u{3000}"]
            ),
            (
                "info/exclude ideo-space",
                "index",
                &["a.secret"],
                &["a.secret\u{3000}"]
            ),
            (
                "info/exclude formfeed",
                "tools",
                &["a.secret"],
                &["a.secret\u{c}"]
            ),
            (
                "info/exclude formfeed",
                "index",
                &["a.secret"],
                &["a.secret\u{c}"]
            ),
            (
                "info/exclude cr-cr",
                "tools",
                &["a.secret"],
                &["a.secret\r"]
            ),
            (
                "info/exclude cr-cr",
                "index",
                &["a.secret"],
                &["a.secret\r"]
            ),
            ("info/exclude dbl-bs-tab", "tools", &[], &["a.secret\\\t"]),
            ("info/exclude dbl-bs-tab", "index", &[], &["a.secret\\\t"]),
            (
                "info/exclude braces",
                "tools",
                &["ab.secret"],
                &["a{b}.secret"]
            ),
            (
                "info/exclude braces",
                "index",
                &["ab.secret"],
                &["a{b}.secret"]
            ),
            ("info/exclude braces-comma", "tools", &["xy"], &["x{y,z}"]),
            ("info/exclude braces-comma", "index", &["xy"], &["x{y,z}"]),
            ("info/exclude brace-tab", "tools", &[], &["q{\t"]),
            ("info/exclude brace-tab", "index", &[], &["q{\t"]),
            ("info/exclude esc-in-class", "tools", &[], &["a]b"]),
            ("info/exclude esc-in-class", "index", &[], &["a]b"]),
            ("info/exclude posix", "tools", &[], &["axb"]),
            ("info/exclude posix", "index", &[], &["axb"]),
            ("info/exclude posix-neg", "tools", &[], &["axb"]),
            ("info/exclude posix-neg", "index", &[], &["axb"]),
            ("info/exclude posix-space", "tools", &[], &["a b"]),
            ("info/exclude posix-space", "index", &[], &["a b"]),
            ("info/exclude posix-bad", "tools", &[], &["axb"]),
            ("info/exclude posix-bad", "index", &[], &["axb"]),
            (
                "info/exclude mixed",
                "tools",
                &[],
                &["a.secret", "caf\u{FFFD}.txt"]
            ),
            (
                "info/exclude mixed",
                "index",
                &[],
                &["a.secret", "caf\u{FFFD}.txt"]
            ),
            ("info/exclude class-esc-close", "tools", &[], &["]x"]),
            ("info/exclude class-esc-close", "index", &[], &["]x"]),
            ("info/exclude class-esc-dash", "tools", &["bx"], &["-x"]),
            ("info/exclude class-esc-dash", "index", &["bx"], &["-x"]),
            ("info/exclude class-esc-range", "tools", &[], &["ax", "bx"]),
            ("info/exclude class-esc-range", "index", &[], &["ax", "bx"]),
            ("info/exclude class-slash", "tools", &["a/b"], &[]),
            ("info/exclude class-slash", "index", &["a/b"], &[]),
            ("info/exclude class-unclosed", "tools", &["a[bc"], &[]),
            ("info/exclude class-unclosed", "index", &["a[bc"], &[]),
            ("info/exclude posix-alnum", "tools", &[], &["x1y", "xay"]),
            ("info/exclude posix-alnum", "index", &[], &["x1y", "xay"]),
            ("info/exclude posix-alpha", "tools", &[], &["xay"]),
            ("info/exclude posix-alpha", "index", &[], &["xay"]),
            ("info/exclude posix-blank", "tools", &[], &["x\ty", "x y"]),
            ("info/exclude posix-blank", "index", &[], &["x\ty", "x y"]),
            ("info/exclude posix-cntrl", "tools", &[], &["x\u{1}y"]),
            ("info/exclude posix-cntrl", "index", &[], &["x\u{1}y"]),
            ("info/exclude posix-digit", "tools", &[], &["x1y"]),
            ("info/exclude posix-digit", "index", &[], &["x1y"]),
            ("info/exclude posix-graph", "tools", &[], &["x-y", "xay"]),
            ("info/exclude posix-graph", "index", &[], &["x-y", "xay"]),
            ("info/exclude posix-lower", "tools", &[], &["xay"]),
            ("info/exclude posix-lower", "index", &[], &["xay"]),
            ("info/exclude posix-print", "tools", &[], &["x y", "xay"]),
            ("info/exclude posix-print", "index", &[], &["x y", "xay"]),
            ("info/exclude posix-punct", "tools", &[], &["x-y"]),
            ("info/exclude posix-punct", "index", &[], &["x-y"]),
            (
                "info/exclude posix-space-xy",
                "tools",
                &[],
                &["x\ty", "x y"]
            ),
            (
                "info/exclude posix-space-xy",
                "index",
                &[],
                &["x\ty", "x y"]
            ),
            ("info/exclude posix-upper", "tools", &[], &["xAy"]),
            ("info/exclude posix-upper", "index", &[], &["xAy"]),
            ("info/exclude posix-xdigit", "tools", &[], &["xfy"]),
            ("info/exclude posix-xdigit", "index", &[], &["xfy"]),
            ("info/exclude posix-two", "tools", &[], &["x-y", "x1y"]),
            ("info/exclude posix-two", "index", &[], &["x-y", "x1y"]),
            ("info/exclude posix-neg-alpha", "tools", &[], &["x1y"]),
            ("info/exclude posix-neg-alpha", "index", &[], &["x1y"]),
            ("info/exclude braces-empty", "tools", &["x"], &["x{}"]),
            ("info/exclude braces-empty", "index", &["x"], &["x{}"]),
            (
                "info/exclude negation-posix-space",
                "tools",
                &["a b.c"],
                &[]
            ),
            (
                "info/exclude negation-posix-space",
                "index",
                &["a b.c"],
                &[]
            ),
            (
                "info/exclude negation-posix-dash",
                "tools",
                &["a-b.c", "a1b.c"],
                &[]
            ),
            (
                "info/exclude negation-posix-dash",
                "index",
                &["a-b.c", "a1b.c"],
                &[]
            ),
            ("info/exclude negation-brace", "tools", &["{a}.c"], &["a.c"]),
            ("info/exclude negation-brace", "index", &["{a}.c"], &["a.c"]),
            ("info/exclude negation-unclosed", "tools", &[], &["a[b.c"]),
            ("info/exclude negation-unclosed", "index", &[], &["a[b.c"]),
            (
                "info/exclude dstar-triple",
                "tools",
                &[],
                &["a/b", "a/x/y/b"]
            ),
            (
                "info/exclude dstar-triple",
                "index",
                &[],
                &["a/b", "a/x/y/b"]
            ),
            (
                "info/exclude posix-alpha-upper",
                "tools",
                &[],
                &["xAy", "xZy"]
            ),
            (
                "info/exclude posix-alpha-upper",
                "index",
                &[],
                &["xAy", "xZy"]
            ),
            (
                "info/exclude posix-alnum-upper",
                "tools",
                &[],
                &["x9y", "xAy"]
            ),
            (
                "info/exclude posix-alnum-upper",
                "index",
                &[],
                &["x9y", "xAy"]
            ),
            (
                "info/exclude posix-xdigit-upper",
                "tools",
                &[],
                &["xAy", "xFy", "xby"]
            ),
            (
                "info/exclude posix-xdigit-upper",
                "index",
                &[],
                &["xAy", "xFy", "xby"]
            ),
            (
                "info/exclude posix-upper-all",
                "tools",
                &[],
                &["xAy", "xZy"]
            ),
            (
                "info/exclude posix-upper-all",
                "index",
                &[],
                &["xAy", "xZy"]
            ),
            (
                "info/exclude posix-lower-all",
                "tools",
                &[],
                &["xay", "xzy"]
            ),
            (
                "info/exclude posix-lower-all",
                "index",
                &[],
                &["xay", "xzy"]
            ),
            ("info/exclude range-to-posix", "tools", &[], &["xa]y"]),
            ("info/exclude range-to-posix", "index", &[], &["xa]y"]),
            (
                "info/exclude range-nonascii",
                "tools",
                &[],
                &["àa", "école"]
            ),
            (
                "info/exclude range-nonascii",
                "index",
                &[],
                &["àa", "école"]
            ),
            ("info/exclude posix-dash-end", "tools", &[], &["a-b", "axb"]),
            ("info/exclude posix-dash-end", "index", &[], &["a-b", "axb"]),
            ("info/exclude class-esc-dash-mid", "tools", &["a\\b"], &[]),
            ("info/exclude class-esc-dash-mid", "index", &["a\\b"], &[]),
            (
                "info/exclude negation-posix-space-2",
                "tools",
                &["a .env"],
                &[]
            ),
            (
                "info/exclude negation-posix-space-2",
                "index",
                &["a .env"],
                &[]
            ),
            (
                "info/exclude negation-posix-dash-end",
                "tools",
                &["a-.env", "ab.env"],
                &[]
            ),
            (
                "info/exclude negation-posix-dash-end",
                "index",
                &["a-.env", "ab.env"],
                &[]
            ),
            (
                "info/exclude negation-brace-alt",
                "tools",
                &["x{y}"],
                &["xy"]
            ),
            (
                "info/exclude negation-brace-alt",
                "index",
                &["x{y}"],
                &["xy"]
            ),
            (
                "info/exclude negation-brace-comma",
                "tools",
                &["x{y,z}"],
                &["xy"]
            ),
            (
                "info/exclude negation-brace-comma",
                "index",
                &["x{y,z}"],
                &["xy"]
            ),
            ("info/exclude negation-esc-close", "tools", &["a]"], &[]),
            ("info/exclude negation-esc-close", "index", &["a]"], &[]),
            (
                "info/exclude negation-after-tab",
                "tools",
                &["secret.env"],
                &["secret.env\t"]
            ),
            (
                "info/exclude negation-after-tab",
                "index",
                &["secret.env"],
                &["secret.env\t"]
            ),
            ("info/exclude negation-nul", "tools", &["ab"], &[]),
            ("info/exclude negation-nul", "index", &["ab"], &[]),
            (
                "info/exclude negation-brace-tab",
                "tools",
                &["a{b}\t"],
                &["ab"]
            ),
            (
                "info/exclude negation-brace-tab",
                "index",
                &["a{b}\t"],
                &["ab"]
            ),
            ("info/exclude negation-posix-alpha", "tools", &["ab"], &[]),
            ("info/exclude negation-posix-alpha", "index", &["ab"], &[]),
            (".gitignore bom", "tools", &[], &["a.secret"]),
            (".gitignore bom", "index", &[], &["a.secret"]),
            (".gitignore bad-after", "tools", &[], &["a.secret"]),
            (".gitignore bad-after", "index", &[], &["a.secret"]),
            (".gitignore bad-before", "tools", &[], &["a.secret"]),
            (".gitignore bad-before", "index", &[], &["a.secret"]),
            (".gitignore bom-crlf", "tools", &[], &["a.secret"]),
            (".gitignore bom-crlf", "index", &[], &["a.secret"]),
            (".gitignore bad-crlf", "tools", &[], &["a.secret"]),
            (".gitignore bad-crlf", "index", &[], &["a.secret"]),
            (".gitignore nul", "tools", &[], &["a.secret"]),
            (".gitignore nul", "index", &[], &["a.secret"]),
            (".gitignore tab", "tools", &["a.secret"], &["a.secret\t"]),
            (".gitignore tab", "index", &["a.secret"], &["a.secret\t"]),
            (
                ".gitignore bad-negation",
                "tools",
                &[],
                &["caf\u{FFFD}.txt"]
            ),
            (
                ".gitignore bad-negation",
                "index",
                &[],
                &["caf\u{FFFD}.txt"]
            ),
            (".gitignore tab-only", "tools", &[], &["\t"]),
            (".gitignore tab-only", "index", &[], &["\t"]),
            (".gitignore esc-space-space", "tools", &[], &["sp "]),
            (".gitignore esc-space-space", "index", &[], &["sp "]),
            (
                ".gitignore nbsp",
                "tools",
                &["a.secret"],
                &["a.secret\u{a0}"]
            ),
            (
                ".gitignore nbsp",
                "index",
                &["a.secret"],
                &["a.secret\u{a0}"]
            ),
            (".gitignore negated-tab", "tools", &[], &["a.secret"]),
            (".gitignore negated-tab", "index", &[], &["a.secret"]),
            (".gitignore esc-tab", "tools", &[], &["a.secret\t"]),
            (".gitignore esc-tab", "index", &[], &["a.secret\t"]),
            (
                ".gitignore tab-space",
                "tools",
                &["a.secret"],
                &["a.secret\t"]
            ),
            (
                ".gitignore tab-space",
                "index",
                &["a.secret"],
                &["a.secret\t"]
            ),
            (".gitignore esc-space-tab", "tools", &[], &["sp \t"]),
            (".gitignore esc-space-tab", "index", &[], &["sp \t"]),
            (".gitignore dbl-bs-space", "tools", &[], &["a.secret\\"]),
            (".gitignore dbl-bs-space", "index", &[], &["a.secret\\"]),
            (
                ".gitignore ideo-space",
                "tools",
                &["a.secret"],
                &["a.secret\u{3000}"]
            ),
            (
                ".gitignore ideo-space",
                "index",
                &["a.secret"],
                &["a.secret\u{3000}"]
            ),
            (
                ".gitignore formfeed",
                "tools",
                &["a.secret"],
                &["a.secret\u{c}"]
            ),
            (
                ".gitignore formfeed",
                "index",
                &["a.secret"],
                &["a.secret\u{c}"]
            ),
            (".gitignore cr-cr", "tools", &["a.secret"], &["a.secret\r"]),
            (".gitignore cr-cr", "index", &["a.secret"], &["a.secret\r"]),
            (".gitignore dbl-bs-tab", "tools", &[], &["a.secret\\\t"]),
            (".gitignore dbl-bs-tab", "index", &[], &["a.secret\\\t"]),
            (
                ".gitignore braces",
                "tools",
                &["ab.secret"],
                &["a{b}.secret"]
            ),
            (
                ".gitignore braces",
                "index",
                &["ab.secret"],
                &["a{b}.secret"]
            ),
            (".gitignore braces-comma", "tools", &["xy"], &["x{y,z}"]),
            (".gitignore braces-comma", "index", &["xy"], &["x{y,z}"]),
            (".gitignore brace-tab", "tools", &[], &["q{\t"]),
            (".gitignore brace-tab", "index", &[], &["q{\t"]),
            (".gitignore esc-in-class", "tools", &[], &["a]b"]),
            (".gitignore esc-in-class", "index", &[], &["a]b"]),
            (".gitignore posix", "tools", &[], &["axb"]),
            (".gitignore posix", "index", &[], &["axb"]),
            (".gitignore posix-neg", "tools", &[], &["axb"]),
            (".gitignore posix-neg", "index", &[], &["axb"]),
            (".gitignore posix-space", "tools", &[], &["a b"]),
            (".gitignore posix-space", "index", &[], &["a b"]),
            (".gitignore posix-bad", "tools", &[], &["axb"]),
            (".gitignore posix-bad", "index", &[], &["axb"]),
            (
                ".gitignore mixed",
                "tools",
                &[],
                &["a.secret", "caf\u{FFFD}.txt"]
            ),
            (
                ".gitignore mixed",
                "index",
                &[],
                &["a.secret", "caf\u{FFFD}.txt"]
            ),
            (".gitignore class-esc-close", "tools", &[], &["]x"]),
            (".gitignore class-esc-close", "index", &[], &["]x"]),
            (".gitignore class-esc-dash", "tools", &["bx"], &["-x"]),
            (".gitignore class-esc-dash", "index", &["bx"], &["-x"]),
            (".gitignore class-esc-range", "tools", &[], &["ax", "bx"]),
            (".gitignore class-esc-range", "index", &[], &["ax", "bx"]),
            (".gitignore class-slash", "tools", &["a/b"], &[]),
            (".gitignore class-slash", "index", &["a/b"], &[]),
            (".gitignore class-unclosed", "tools", &["a[bc"], &[]),
            (".gitignore class-unclosed", "index", &["a[bc"], &[]),
            (".gitignore posix-alnum", "tools", &[], &["x1y", "xay"]),
            (".gitignore posix-alnum", "index", &[], &["x1y", "xay"]),
            (".gitignore posix-alpha", "tools", &[], &["xay"]),
            (".gitignore posix-alpha", "index", &[], &["xay"]),
            (".gitignore posix-blank", "tools", &[], &["x\ty", "x y"]),
            (".gitignore posix-blank", "index", &[], &["x\ty", "x y"]),
            (".gitignore posix-cntrl", "tools", &[], &["x\u{1}y"]),
            (".gitignore posix-cntrl", "index", &[], &["x\u{1}y"]),
            (".gitignore posix-digit", "tools", &[], &["x1y"]),
            (".gitignore posix-digit", "index", &[], &["x1y"]),
            (".gitignore posix-graph", "tools", &[], &["x-y", "xay"]),
            (".gitignore posix-graph", "index", &[], &["x-y", "xay"]),
            (".gitignore posix-lower", "tools", &[], &["xay"]),
            (".gitignore posix-lower", "index", &[], &["xay"]),
            (".gitignore posix-print", "tools", &[], &["x y", "xay"]),
            (".gitignore posix-print", "index", &[], &["x y", "xay"]),
            (".gitignore posix-punct", "tools", &[], &["x-y"]),
            (".gitignore posix-punct", "index", &[], &["x-y"]),
            (".gitignore posix-space-xy", "tools", &[], &["x\ty", "x y"]),
            (".gitignore posix-space-xy", "index", &[], &["x\ty", "x y"]),
            (".gitignore posix-upper", "tools", &[], &["xAy"]),
            (".gitignore posix-upper", "index", &[], &["xAy"]),
            (".gitignore posix-xdigit", "tools", &[], &["xfy"]),
            (".gitignore posix-xdigit", "index", &[], &["xfy"]),
            (".gitignore posix-two", "tools", &[], &["x-y", "x1y"]),
            (".gitignore posix-two", "index", &[], &["x-y", "x1y"]),
            (".gitignore posix-neg-alpha", "tools", &[], &["x1y"]),
            (".gitignore posix-neg-alpha", "index", &[], &["x1y"]),
            (".gitignore braces-empty", "tools", &["x"], &["x{}"]),
            (".gitignore braces-empty", "index", &["x"], &["x{}"]),
            (".gitignore negation-posix-space", "tools", &["a b.c"], &[]),
            (".gitignore negation-posix-space", "index", &["a b.c"], &[]),
            (
                ".gitignore negation-posix-dash",
                "tools",
                &["a-b.c", "a1b.c"],
                &[]
            ),
            (
                ".gitignore negation-posix-dash",
                "index",
                &["a-b.c", "a1b.c"],
                &[]
            ),
            (".gitignore negation-brace", "tools", &["{a}.c"], &["a.c"]),
            (".gitignore negation-brace", "index", &["{a}.c"], &["a.c"]),
            (".gitignore negation-unclosed", "tools", &[], &["a[b.c"]),
            (".gitignore negation-unclosed", "index", &[], &["a[b.c"]),
            (".gitignore dstar-triple", "tools", &[], &["a/b", "a/x/y/b"]),
            (".gitignore dstar-triple", "index", &[], &["a/b", "a/x/y/b"]),
            (
                ".gitignore posix-alpha-upper",
                "tools",
                &[],
                &["xAy", "xZy"]
            ),
            (
                ".gitignore posix-alpha-upper",
                "index",
                &[],
                &["xAy", "xZy"]
            ),
            (
                ".gitignore posix-alnum-upper",
                "tools",
                &[],
                &["x9y", "xAy"]
            ),
            (
                ".gitignore posix-alnum-upper",
                "index",
                &[],
                &["x9y", "xAy"]
            ),
            (
                ".gitignore posix-xdigit-upper",
                "tools",
                &[],
                &["xAy", "xFy", "xby"]
            ),
            (
                ".gitignore posix-xdigit-upper",
                "index",
                &[],
                &["xAy", "xFy", "xby"]
            ),
            (".gitignore posix-upper-all", "tools", &[], &["xAy", "xZy"]),
            (".gitignore posix-upper-all", "index", &[], &["xAy", "xZy"]),
            (".gitignore posix-lower-all", "tools", &[], &["xay", "xzy"]),
            (".gitignore posix-lower-all", "index", &[], &["xay", "xzy"]),
            (".gitignore range-to-posix", "tools", &[], &["xa]y"]),
            (".gitignore range-to-posix", "index", &[], &["xa]y"]),
            (".gitignore range-nonascii", "tools", &[], &["àa", "école"]),
            (".gitignore range-nonascii", "index", &[], &["àa", "école"]),
            (".gitignore posix-dash-end", "tools", &[], &["a-b", "axb"]),
            (".gitignore posix-dash-end", "index", &[], &["a-b", "axb"]),
            (".gitignore class-esc-dash-mid", "tools", &["a\\b"], &[]),
            (".gitignore class-esc-dash-mid", "index", &["a\\b"], &[]),
            (
                ".gitignore negation-posix-space-2",
                "tools",
                &["a .env"],
                &[]
            ),
            (
                ".gitignore negation-posix-space-2",
                "index",
                &["a .env"],
                &[]
            ),
            (
                ".gitignore negation-posix-dash-end",
                "tools",
                &["a-.env", "ab.env"],
                &[]
            ),
            (
                ".gitignore negation-posix-dash-end",
                "index",
                &["a-.env", "ab.env"],
                &[]
            ),
            (".gitignore negation-brace-alt", "tools", &["x{y}"], &["xy"]),
            (".gitignore negation-brace-alt", "index", &["x{y}"], &["xy"]),
            (
                ".gitignore negation-brace-comma",
                "tools",
                &["x{y,z}"],
                &["xy"]
            ),
            (
                ".gitignore negation-brace-comma",
                "index",
                &["x{y,z}"],
                &["xy"]
            ),
            (".gitignore negation-esc-close", "tools", &["a]"], &[]),
            (".gitignore negation-esc-close", "index", &["a]"], &[]),
            (
                ".gitignore negation-after-tab",
                "tools",
                &["secret.env"],
                &["secret.env\t"]
            ),
            (
                ".gitignore negation-after-tab",
                "index",
                &["secret.env"],
                &["secret.env\t"]
            ),
            (".gitignore negation-nul", "tools", &["ab"], &[]),
            (".gitignore negation-nul", "index", &["ab"], &[]),
            (
                ".gitignore negation-brace-tab",
                "tools",
                &["a{b}\t"],
                &["ab"]
            ),
            (
                ".gitignore negation-brace-tab",
                "index",
                &["a{b}\t"],
                &["ab"]
            ),
            (".gitignore negation-posix-alpha", "tools", &["ab"], &[]),
            (".gitignore negation-posix-alpha", "index", &["ab"], &[]),
        ])
    );
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

pub(super) fn literal_rows(rows: &[(&str, &'static str, &[&str], &[&str])]) -> Vec<GitDifference> {
    let strings = |items: &[&str]| items.iter().map(|item| (*item).to_owned()).collect();
    rows.iter()
        .map(|(case, walk, git, walked)| ((*case).to_owned(), *walk, strings(git), strings(walked)))
        .collect()
}
