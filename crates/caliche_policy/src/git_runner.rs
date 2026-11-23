//! The one way to build a `git` command.
//!
//! Repository config, and anything an inherited environment adds, is
//! hostile input: a workspace repository can name programs for git to run
//! (`diff.external`, a textconv or filter driver, hooks, a pager, an
//! fsmonitor). [`git_command`] starts `git` with no system or global config,
//! with the inherited variables that inject config or programs removed, and
//! with every program key git consults on its own turned off.
//!
//! An inherited environment can also name another repository: a git hook or
//! `git rebase --exec` exports `GIT_DIR` and `GIT_INDEX_FILE` to what it
//! starts. [`git_command`] removes every variable that picks the
//! repository, so git finds it from the working directory, or from what
//! the caller sets on the command.
//! [`refused_program_keys`] names the keys a repository's own config may
//! not set at all, since a driver name cannot be turned off from outside.

use crate::{GIT_CONFIG_GLOBAL_VAR, GIT_CONFIG_NOSYSTEM_VAR, ISOLATED_GIT_CONFIG};

/// The variable that names the system config file.
pub const GIT_CONFIG_SYSTEM_VAR: &str = "GIT_CONFIG_SYSTEM";

/// Inherited variables that add config or name a program for git to run.
/// Git reads `GIT_CONFIG_KEY_<n>` and `GIT_CONFIG_VALUE_<n>` only when
/// `GIT_CONFIG_COUNT` is set, so removing the count turns those off too.
/// `GIT_CONFIG` makes `git config` read another file in place of the
/// repository's, so a value read from it, such as an editor, would come
/// from elsewhere.
pub const GIT_HOSTILE_VARS: [&str; 13] = [
    "GIT_CONFIG",
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

/// Inherited variables that pick the repository, its work tree, index,
/// object store, refs or history files, or widen how git searches for it.
/// With one of them left in place, a command run in one repository's
/// directory reads or writes another.
///
/// `GIT_CEILING_DIRECTORIES` is not here: it only stops discovery, so an
/// inherited value can only make git find less, and it keeps git out of a
/// repository the user fenced off above the working directory.
pub const GIT_REPOSITORY_VARS: [&str; 11] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_REPLACE_REF_BASE",
    "GIT_GRAFT_FILE",
    "GIT_SHALLOW_FILE",
];

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
/// no textconv driver runs, and git does not read a submodule's repository,
/// which the git dir check does not cover.
pub const GIT_DIFF_SAFETY_FLAGS: [&str; 3] =
    ["--no-ext-diff", "--no-textconv", "--ignore-submodules=all"];

/// Flags every `git status` gets, so git does not read a submodule's
/// repository.
pub const GIT_STATUS_SAFETY_FLAGS: [&str; 1] = ["--ignore-submodules=all"];

/// `git status` arguments for a NUL-separated porcelain v1 listing with no
/// rename detection, after [`GIT_STATUS_SAFETY_FLAGS`]; `untracked` is the
/// `--untracked-files=` option, such as `--untracked-files=no`.
#[must_use]
pub fn porcelain_status_args(untracked: &'static str) -> Vec<&'static str> {
    let mut args = vec!["status"];
    args.extend(GIT_STATUS_SAFETY_FLAGS);
    args.extend(["--porcelain=v1", "-z", "--no-renames", untracked]);
    args
}

/// A `git` command with no system or global config, no hostile inherited
/// variable, no inherited variable that picks the repository
/// ([`GIT_REPOSITORY_VARS`]), and [`GIT_PROGRAM_OVERRIDES`] on its command
/// line. The caller adds the subcommand and pins the repository.
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
    for name in GIT_HOSTILE_VARS.into_iter().chain(GIT_REPOSITORY_VARS) {
        command.env_remove(name);
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
/// key lowercased), that name a program git may run, or make git read a
/// submodule's repository, and that a repository may not set. A driver's name is chosen by the repository, so these are
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
        // `diff.submodule` and `status.submodule*` show a submodule's
        // commits or content, read from its repository.
        "diff" => matches!(last, "external" | "textconv" | "command" | "submodule"),
        "status" => last.starts_with("submodule"),
        "submodule" => last == "recurse",
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

    /// The variables a `git` command removes from what it inherits.
    fn removed(command: &std::process::Command) -> Vec<String> {
        command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn git_commands_drop_inherited_repository_variables() {
        let expected = [
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_ASKPASS",
            "GIT_COMMON_DIR",
            "GIT_CONFIG",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_PARAMETERS",
            "GIT_DIFF_OPTS",
            "GIT_DIR",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM",
            "GIT_EDITOR",
            "GIT_EXEC_PATH",
            "GIT_EXTERNAL_DIFF",
            "GIT_GRAFT_FILE",
            "GIT_INDEX_FILE",
            "GIT_NAMESPACE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_PAGER",
            "GIT_REPLACE_REF_BASE",
            "GIT_SEQUENCE_EDITOR",
            "GIT_SHALLOW_FILE",
            "GIT_SSH",
            "GIT_SSH_COMMAND",
            "GIT_TEMPLATE_DIR",
            "GIT_WORK_TREE",
        ];
        assert_eq!(removed(&git_command()), expected);
        assert_eq!(removed(&program_command("/usr/bin/git")), expected);
    }

    #[test]
    fn porcelain_status_args_start_with_the_safety_flags() {
        assert_eq!(
            porcelain_status_args("--untracked-files=no"),
            [
                "status",
                "--ignore-submodules=all",
                "--porcelain=v1",
                "-z",
                "--no-renames",
                "--untracked-files=no",
            ]
        );
    }

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
            "diff.submodule",
            "status.submodulesummary",
            "submodule.recurse",
            "submodule.sub.url",
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
                "diff.submodule",
                "status.submodulesummary",
                "submodule.recurse",
            ]
        );
    }
}
