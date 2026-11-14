//! The one way to build a `git` command.
//!
//! Repository config, and anything an inherited environment adds, is
//! hostile input: a workspace repository can name programs for git to run
//! (`diff.external`, a textconv or filter driver, hooks, a pager, an
//! fsmonitor). [`git_command`] starts `git` with no system or global config,
//! with the inherited variables that inject config or programs removed, and
//! with every program key git consults on its own turned off.
//! [`refused_program_keys`] names the keys a repository's own config may
//! not set at all, since a driver name cannot be turned off from outside.

use crate::{GIT_CONFIG_GLOBAL_VAR, GIT_CONFIG_NOSYSTEM_VAR, ISOLATED_GIT_CONFIG};

/// The variable that names the system config file.
pub const GIT_CONFIG_SYSTEM_VAR: &str = "GIT_CONFIG_SYSTEM";

/// Inherited variables that add config or name a program for git to run.
pub const GIT_HOSTILE_VARS: [&str; 12] = [
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_EXTERNAL_DIFF",
    "GIT_DIFF_OPTS",
    "GIT_PAGER",
    "GIT_EDITOR",
    "GIT_SEQUENCE_EDITOR",
    "GIT_ASKPASS",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_EXEC_PATH",
    "GIT_TEMPLATE_DIR",
];

/// Prefixes of the numbered `GIT_CONFIG_KEY_<n>` / `GIT_CONFIG_VALUE_<n>`
/// variables, removed with [`GIT_HOSTILE_VARS`].
pub const GIT_CONFIG_PAIR_PREFIXES: [&str; 2] = ["GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"];

/// Config every `git` command gets on its command line, which wins over the
/// repository's config: no fsmonitor, no hooks, no pager.
///
/// `diff.external` cannot be turned off this way: an empty value makes git
/// try to run an empty program. Diffs pass [`GIT_DIFF_SAFETY_FLAGS`]
/// instead, and a repository that sets it is refused by its config check.
pub const GIT_PROGRAM_OVERRIDES: [(&str, &str); 3] = [
    ("core.fsmonitor", "false"),
    ("core.hooksPath", ISOLATED_GIT_CONFIG),
    ("core.pager", "cat"),
];

/// Flags every `git diff`, `show` and `log -p` gets, so no external diff and
/// no textconv driver runs.
pub const GIT_DIFF_SAFETY_FLAGS: [&str; 2] = ["--no-ext-diff", "--no-textconv"];

/// A `git` command with no system or global config, no hostile inherited
/// variable, and [`GIT_PROGRAM_OVERRIDES`] on its command line. The caller
/// adds the subcommand and pins the repository.
#[must_use]
pub fn git_command() -> std::process::Command {
    hardened_git(std::ffi::OsStr::new("git"))
}

fn hardened_git(program: &std::ffi::OsStr) -> std::process::Command {
    #[expect(
        clippy::disallowed_methods,
        reason = "the one place a git command is built"
    )]
    let mut command = std::process::Command::new(program);
    for name in GIT_HOSTILE_VARS {
        command.env_remove(name);
    }
    for (name, _) in std::env::vars_os() {
        let Some(name) = name.to_str() else {
            continue;
        };
        if GIT_CONFIG_PAIR_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            command.env_remove(name);
        }
    }
    command
        .env(GIT_CONFIG_NOSYSTEM_VAR, "1")
        .env(GIT_CONFIG_SYSTEM_VAR, ISOLATED_GIT_CONFIG)
        .env(GIT_CONFIG_GLOBAL_VAR, ISOLATED_GIT_CONFIG);
    for (key, value) in GIT_PROGRAM_OVERRIDES {
        command.arg("-c").arg(format!("{key}={value}"));
    }
    command
}

/// A command that runs `program`. A program named `git` (any directory,
/// any case, with or without `.exe`) gets [`git_command`], so no caller can
/// build a `git` command without the hardening.
#[must_use]
pub fn program_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let program = program.as_ref();
    if is_git_program(program) {
        return hardened_git(program);
    }
    #[expect(
        clippy::disallowed_methods,
        reason = "every other program is built here, after the git check"
    )]
    std::process::Command::new(program)
}

fn is_git_program(program: &std::ffi::OsStr) -> bool {
    std::path::Path::new(program)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.eq_ignore_ascii_case("git"))
}

/// Config keys, as `git config --list --name-only` prints them (section and
/// key lowercased), that name a program git may run and that a repository
/// may not set. A driver's name is chosen by the repository, so these are
/// matched by section and last part.
#[must_use]
pub fn refused_program_keys<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    names
        .into_iter()
        .filter(|name| is_program_key(name))
        .map(str::to_string)
        .collect()
}

/// The program keys a repository's config sets, from the output of
/// `git config --list --name-only --show-scope -z`. The `command` scope,
/// which holds [`GIT_PROGRAM_OVERRIDES`], is skipped.
#[must_use]
pub fn refused_program_keys_in_listing(listing: &[u8]) -> Vec<String> {
    let fields: Vec<String> = listing
        .split(|byte| *byte == 0)
        .map(|field| String::from_utf8_lossy(field).into_owned())
        .collect();
    let names = fields
        .chunks_exact(2)
        .filter(|pair| pair[0] != "command")
        .map(|pair| pair[1].as_str());
    refused_program_keys(names)
}

fn is_program_key(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let section = name.split('.').next().unwrap_or_default();
    let last = name.rsplit('.').next().unwrap_or_default();
    match section {
        "filter" => matches!(last, "clean" | "smudge" | "process"),
        "diff" => matches!(last, "external" | "textconv" | "command"),
        "merge" | "mergetool" | "difftool" => matches!(last, "driver" | "cmd"),
        "core" => matches!(
            last,
            "fsmonitor" | "hookspath" | "pager" | "editor" | "sshcommand" | "askpass" | "gitproxy"
        ),
        "gpg" => last == "program",
        "credential" => last == "helper",
        "sequence" => last == "editor",
        "pager" | "include" | "includeif" => true,
        "uploadpack" => last == "packobjectshook",
        "remote" => matches!(last, "uploadpack" | "receivepack" | "proxy"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_keys_are_refused_by_section_and_last_part() {
        let listed = [
            "core.repositoryformatversion",
            "core.bare",
            "diff.external",
            "diff.png.textconv",
            "filter.lfs.smudge",
            "filter.lfs.required",
            "core.fsmonitor",
            "core.hookspath",
            "user.name",
            "pager.log",
            "include.path",
            "includeif.gitdir:/x.path",
            "merge.ours.driver",
            "remote.origin.url",
            "remote.origin.uploadpack",
        ];

        assert_eq!(
            refused_program_keys(listed),
            vec![
                "diff.external",
                "diff.png.textconv",
                "filter.lfs.smudge",
                "core.fsmonitor",
                "core.hookspath",
                "pager.log",
                "include.path",
                "includeif.gitdir:/x.path",
                "merge.ours.driver",
                "remote.origin.uploadpack",
            ]
        );
    }
}
