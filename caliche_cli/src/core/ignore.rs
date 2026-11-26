//! File ignore patterns for context management.
//!
//! Delegates to the `ignore` crate (same engine ripgrep uses) so that
//! `.caliche_ignore` files behave with full gitignore semantics rather
//! than the hand-rolled subset that shipped previously.

use anyhow::Result;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use std::path::Path;

/// Build a `Gitignore` matcher for the project root's `.caliche_ignore`.
///
/// Returns `None` if the file does not exist, or if the matcher cannot be built
/// (all errors are surfaced through the inner `Gitignore::matched` later on).
fn build_matcher(project_root: &Path) -> Option<Gitignore> {
    let ignore_file = project_root.join(".caliche_ignore");
    if !ignore_file.exists() {
        return None;
    }

    let mut builder = GitignoreBuilder::new(project_root);
    if let Some(err) = builder.add(&ignore_file) {
        tracing::warn!(
            "Failed to load .caliche_ignore at {}: {}",
            ignore_file.display(),
            err
        );
        return None;
    }
    match builder.build() {
        Ok(matcher) => Some(matcher),
        Err(e) => {
            tracing::warn!(
                "Failed to compile .caliche_ignore matcher for {}: {}",
                project_root.display(),
                e
            );
            None
        }
    }
}

/// Check if a file should be ignored based on `.caliche_ignore` patterns.
///
/// Uses gitignore semantics (wildcards, leading `/`, directory-only `trailing/`,
/// negation with `!`, `**` recursive globs, comments with `#`).
pub fn should_ignore_file(path: &Path, project_root: &Path) -> bool {
    let Some(matcher) = build_matcher(project_root) else {
        return false;
    };

    let is_dir = path.is_dir();
    matcher.matched(path, is_dir).is_ignore()
}

/// Load the raw pattern lines from `.caliche_ignore` for display/inspection.
///
/// This is intentionally lossy; the canonical matcher used at runtime is the
/// gitignore-semantics matcher built by [`build_matcher`].
pub fn load_ignore_patterns(project_root: &Path) -> Result<Vec<String>> {
    let ignore_file = project_root.join(".caliche_ignore");

    if !ignore_file.exists() {
        return Ok(Vec::new());
    }

    let content = std::fs::read_to_string(&ignore_file)?;
    let patterns: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();

    Ok(patterns)
}

#[cfg(test)]
mod tests {
    use super::should_ignore_file;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// `.caliche_ignore` content read in a way a plain text read does not,
    /// and the files each case adds next to `a.secret` and `ok.rs`.
    const CASES: [(&str, &[u8], &[&str]); 25] = [
        ("plain", b"*.secret\n", &[]),
        ("bad-pattern", b"[z-a]\n*.secret\n", &[]),
        ("bom", b"\xEF\xBB\xBF*.secret\n", &[]),
        ("bad-before", b"# caf\xE9\n*.secret\n", &[]),
        ("bad-after", b"*.secret\n# caf\xE9\n", &[]),
        ("crlf", b"*.secret\r\n", &[]),
        ("space-crlf", b"sp\\ \r\n", &["sp "]),
        ("nul", b"*.secret\0junk\n", &[]),
        ("tab", b"a.secret\t\n", &["a.secret\t"]),
        ("esc-tab", b"a.secret\\\t\n", &["a.secret\t"]),
        (
            "ideo-space",
            b"a.secret\xE3\x80\x80\n",
            &["a.secret\u{3000}"],
        ),
        ("trail-spaces", b"*.secret  \n", &[]),
        ("space-esc-space", b"a \\ \n", &["a", "a  "]),
        (
            "bad-negation",
            b"*.txt\n!caf\xE9.txt\n",
            &["caf\u{FFFD}.txt"],
        ),
        ("braces", b"a{b}.secret\n", &["a{b}.secret", "ab.secret"]),
        ("braces-comma", b"x{y,z}\n", &["x{y,z}", "xy"]),
        ("esc-in-class", b"a[\\]]b\n", &["a]b"]),
        ("neg-brace", b"x*\n!x{y}\n", &["x{y}", "xy"]),
        (
            "neg-after-tab",
            b"secret.env\t\n!s[[:alpha:]]cret.env\n",
            &["secret.env"],
        ),
        (
            "neg-posix-space",
            b"*.env\n!a[[:space:]].env\n",
            &["a .env", "ab.env"],
        ),
        ("neg-unclosed", b"*.c\n!a[b.c\n", &["a[b.c"]),
        ("posix-upper-name", b"x[[:alpha:]]y\n", &["xAy", "x1y"]),
        ("posix", b"a[[:alpha:]]b\n", &["axb", "a1b"]),
        ("posix-bad", b"a[[:alpha:]\xE9]b\n", &["axb", "a1b"]),
        (
            "mixed",
            b"[z-a]\n*.txt\n# caf\xE9\n!caf\xE9.txt\n*.secret\r\n",
            &["caf\u{FFFD}.txt"],
        ),
    ];

    /// A `git` command for the oracle: no system or global config, no
    /// inherited variable that points it at another repository, and no
    /// way to reach a network, since ssh fails and git never prompts.
    fn oracle_git() -> std::process::Command {
        let mut command = caliche::policy::git_command();
        command
            .env("GIT_SSH_COMMAND", "false")
            .env("GIT_TERMINAL_PROMPT", "0");
        command
    }

    /// The files of `dir` that git lists as untracked when `content` is
    /// its `.gitignore`.
    fn git_listed(dir: &Path, content: &[u8]) -> BTreeSet<String> {
        std::fs::write(dir.join(".gitignore"), content).expect(".gitignore");
        let init = oracle_git()
            .args(["init", "-q"])
            .current_dir(dir)
            .status()
            .expect("git init");
        assert!(init.success());
        let output = oracle_git()
            .args(["-c", "core.excludesFile=", "-c", "core.ignorecase=false"])
            .args(["ls-files", "--others"])
            .args(["--exclude-standard", "-z"])
            .current_dir(dir)
            .output()
            .expect("git ls-files");
        assert!(output.status.success(), "{output:?}");
        String::from_utf8(output.stdout)
            .expect("utf-8 names")
            .split('\0')
            .filter(|name| !name.is_empty() && !name.starts_with('.'))
            .map(str::to_owned)
            .collect()
    }

    /// For each case, with `content` as the `.caliche_ignore`: the files a
    /// reference hides that `should_ignore_file` keeps, and the files it
    /// hides that no reference hides. git never reads `.caliche_ignore`, so
    /// a file is hidden by the references when git hides it with the same
    /// content as a `.gitignore`, or when the `ignore` crate's own reader
    /// of the file hides it.
    fn differences() -> Vec<(&'static str, Vec<String>, Vec<String>)> {
        let mut rows = Vec::new();
        for (case, content, extra) in CASES {
            let mut files = vec!["a.secret", "ok.rs"];
            files.extend(extra);
            let git_dir = tempfile::tempdir().expect("git dir");
            let caliche_dir = tempfile::tempdir().expect("caliche dir");
            let git_root = git_dir.path().canonicalize().expect("git root");
            let root = caliche_dir.path().canonicalize().expect("caliche root");
            for name in &files {
                std::fs::write(git_root.join(name), "x").expect("file");
                std::fs::write(root.join(name), "x").expect("file");
            }
            let git = git_listed(&git_root, content);
            let file = root.join(".caliche_ignore");
            std::fs::write(&file, content).expect(".caliche_ignore");
            let mut builder = ignore::gitignore::GitignoreBuilder::new(&root);
            let _ = builder.add(&file);
            let reader = builder.build().expect("crate reader");
            let hidden_by_ref = |name: &str| {
                !git.contains(name) || reader.matched(root.join(name), false).is_ignore()
            };
            let hidden = |name: &str| should_ignore_file(&root.join(name), &root);
            let leaks: Vec<String> = files
                .iter()
                .filter(|name| !hidden(name) && hidden_by_ref(name))
                .map(|name| (*name).to_owned())
                .collect();
            let over: Vec<String> = files
                .iter()
                .filter(|name| hidden(name) && !hidden_by_ref(name))
                .map(|name| (*name).to_owned())
                .collect();
            if !leaks.is_empty() || !over.is_empty() {
                rows.push((case, leaks, over));
            }
        }
        rows
    }

    /// `.caliche_ignore` hides every file a reference hides.
    ///
    /// Known differences, each a file a reference hides that the reader
    /// keeps:
    /// - a pattern that does not parse, or a line that is not UTF-8, drops
    ///   the whole file;
    /// - a NUL does not end the pattern;
    /// - a trailing tab or ideographic space is trimmed, and an escaped
    ///   trailing tab leaves a pattern that matches nothing;
    /// - braces are read as alternatives, where git reads them literally,
    ///   so a negation `!x{y}` un-hides `xy`;
    /// - a class that is not closed is read literally, where git's pattern
    ///   matches nothing, so a negation with one un-hides a file;
    /// - a backslash in a class and POSIX classes such as `[:alpha:]` are
    ///   not read as git reads them.
    ///
    /// Limit: the crate half of the reference is the engine under test,
    /// so only the git half can show a file the reader hides wrongly.
    #[test]
    fn the_caliche_ignore_file_hides_what_its_references_hide() {
        let strings = |items: &[&str]| -> Vec<String> {
            items.iter().map(|item| (*item).to_owned()).collect()
        };
        let known: Vec<(&str, Vec<String>, Vec<String>)> = [
            ("bad-pattern", &["a.secret"][..], &[][..]),
            ("bad-before", &["a.secret"][..], &[][..]),
            ("bad-after", &["a.secret"][..], &[][..]),
            ("nul", &["a.secret"][..], &[][..]),
            ("tab", &["a.secret\t"][..], &[][..]),
            ("esc-tab", &["a.secret\t"][..], &[][..]),
            ("ideo-space", &["a.secret\u{3000}"][..], &[][..]),
            ("bad-negation", &["caf\u{FFFD}.txt"][..], &[][..]),
            ("braces", &["a{b}.secret"][..], &[][..]),
            ("braces-comma", &["x{y,z}"][..], &[][..]),
            ("esc-in-class", &["a]b"][..], &[][..]),
            ("neg-brace", &["xy"][..], &[][..]),
            ("neg-unclosed", &["a[b.c"][..], &[][..]),
            ("posix-upper-name", &["xAy"][..], &[][..]),
            ("posix", &["axb"][..], &[][..]),
            ("posix-bad", &["axb"][..], &[][..]),
            ("mixed", &["a.secret", "caf\u{FFFD}.txt"][..], &[][..]),
        ]
        .into_iter()
        .map(|(case, leaks, over)| (case, strings(leaks), strings(over)))
        .collect();
        assert_eq!(differences(), known);
    }
}
