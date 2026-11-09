//! Tracks the last workspace-mutating or shell command so coding verification
//! can reuse a test run the agent already completed after its last edit.

use crate::agent::hooks::{HookContext, HookResult, LifecycleHook};
use crate::agent::tools::names::{
    TOOL_APPLY_PATCH, TOOL_BASH, TOOL_FILESYSTEM, fs_op, is_fs_write_op,
};
use crate::agent::tools::tool::ToolResult;
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Flags that are harmless extras in the recorded command (do not restrict the
/// set of tests that ran).
const CARGO_BENIGN_FLAGS: &[&str] = &[
    "--offline",
    "-q",
    "--quiet",
    "--color",
    "--color=never",
    "--color=always",
    "--color=auto",
    "--nocapture",
];

/// Extract the command string from a JSON value that may be a string or an
/// array of strings (argv-style).
fn extract_command_str(value: &serde_json::Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        return Some(s.to_owned());
    }
    if let Some(arr) = value.as_array() {
        let parts: Option<Vec<&str>> = arr.iter().map(|v| v.as_str()).collect();
        return parts.map(|p| p.join(" "));
    }
    None
}

/// Strip leading `NAME=value` shell variable assignments from a token iterator
/// and return the remaining tokens as a `Vec<String>`.
fn strip_env_assignments(tokens: &[String]) -> &[String] {
    let mut i = 0;
    while i < tokens.len() {
        // A leading assignment looks like `FOO=bar` or `FOO=`
        if tokens[i].contains('=')
            && tokens[i].split_once('=').is_some_and(|(key, _)| {
                !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            })
        {
            i += 1;
        } else {
            break;
        }
    }
    &tokens[i..]
}

/// Strip a leading `cd <path> &&` from a token slice.
fn strip_cd_prefix<'a>(tokens: &'a [String]) -> &'a [String] {
    if tokens.len() >= 3 && tokens[0] == "cd" && tokens[2] == "&&" {
        return &tokens[3..];
    }
    tokens
}

/// Drop env assignments, `cd <path> &&`, and flags that do not change which
/// tests ran, so `cargo test --offline` and `cargo test -- --test-threads=1`
/// compare equal to `cargo test`.
fn core_command_tokens(tokens: &[String]) -> Vec<String> {
    let stripped = strip_cd_prefix(strip_env_assignments(tokens));
    let mut out = Vec::with_capacity(stripped.len());
    let mut skip_next = false;
    let mut after_dashdash = false;
    for tok in stripped {
        if skip_next {
            skip_next = false;
            continue;
        }
        if tok.as_str() == "--" {
            after_dashdash = true;
            continue;
        }
        if after_dashdash {
            if tok.as_str() == "--nocapture" {
                continue;
            }
            if tok.as_str() == "--test-threads" {
                skip_next = true;
                continue;
            }
            if tok.starts_with("--test-threads=") {
                continue;
            }
            out.push(tok.clone());
            continue;
        }
        if tok.as_str() == "--color" {
            skip_next = true;
            continue;
        }
        if tok.starts_with("--color=") || CARGO_BENIGN_FLAGS.contains(&tok.as_str()) {
            continue;
        }
        out.push(tok.clone());
    }
    out
}

fn tokens_cover(recorded: &[String], wanted: &[String]) -> bool {
    let recorded_core = core_command_tokens(recorded);
    if recorded_core.is_empty() {
        return false;
    }
    recorded_core == core_command_tokens(wanted)
}

/// Last workspace-relevant tool outcome for a coding session.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
enum LastCommandEvent {
    #[default]
    Idle,
    Mutation,
    Ran {
        /// Whitespace-normalized token list of the recorded command.
        tokens: Vec<String>,
        success: bool,
    },
}

/// Session ledger of the most recent edit or shell command.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CommandLedger {
    last: LastCommandEvent,
}

impl CommandLedger {
    /// Record a successful filesystem mutation. Later verification cannot be reused
    /// until the same command is run again.
    pub fn record_mutation(&mut self) {
        self.last = LastCommandEvent::Mutation;
    }

    /// Record a shell command executed at the workspace root.
    pub fn record_command(&mut self, command: &str, success: bool) {
        let tokens = tokenize(command);
        if tokens.is_empty() {
            return;
        }
        self.last = LastCommandEvent::Ran { tokens, success };
    }

    /// True when `command` was the last workspace event and it succeeded, with
    /// the recorded command being allowed to have extra benign flags.
    #[must_use]
    pub fn covers(&self, command: &str) -> bool {
        match &self.last {
            LastCommandEvent::Ran {
                tokens: ran,
                success: true,
            } => {
                let wanted = tokenize(command);
                if wanted.is_empty() {
                    return false;
                }
                // Exact match (after tokenization) is always fine.
                if ran == &wanted {
                    return true;
                }
                // Otherwise check if recorded covers wanted with only benign extras.
                tokens_cover(ran, &wanted)
            }
            _ => false,
        }
    }
}

/// Split a shell command string into whitespace-normalized tokens.
#[must_use]
pub fn tokenize(command: &str) -> Vec<String> {
    command.split_whitespace().map(str::to_owned).collect()
}

/// Collapse ASCII whitespace so `cargo test` matches `cargo  test`.
/// Kept for backward compatibility with existing callers.
#[must_use]
pub fn normalize_shell_command(command: &str) -> String {
    tokenize(command).join(" ")
}

fn command_ran_in_workspace(working_directory: Option<&str>, workspace: &Path) -> bool {
    let Some(cwd) = working_directory
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return true;
    };
    if cwd == "." {
        return true;
    }
    let path = Path::new(cwd);
    if path == workspace {
        return true;
    }
    match (
        std::fs::canonicalize(path),
        std::fs::canonicalize(workspace),
    ) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

/// Lifecycle hook that feeds [`CommandLedger`] from fs/bash tool results.
#[derive(Debug)]
pub(super) struct CommandLedgerHook {
    ledger: Arc<RwLock<CommandLedger>>,
    workspace_root: PathBuf,
}

impl CommandLedgerHook {
    pub(super) const fn new(ledger: Arc<RwLock<CommandLedger>>, workspace_root: PathBuf) -> Self {
        Self {
            ledger,
            workspace_root,
        }
    }
}

#[async_trait]
impl LifecycleHook for CommandLedgerHook {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_post_tool_use(&self, ctx: &HookContext<'_>, result: &ToolResult) -> HookResult {
        let name = ctx.tool_name.as_ref();
        if name == TOOL_FILESYSTEM {
            if result.success && is_fs_write_op(fs_op(ctx.tool_input).unwrap_or("")) {
                self.ledger.write().await.record_mutation();
            }
            return HookResult::allow();
        }
        if name == TOOL_APPLY_PATCH {
            if result.success {
                self.ledger.write().await.record_mutation();
            }
            return HookResult::allow();
        }
        if name == TOOL_BASH {
            let Some(command) = ctx.tool_input.get("command").and_then(extract_command_str) else {
                return HookResult::allow();
            };
            let working_directory = ctx
                .tool_input
                .get("working_directory")
                .and_then(serde_json::Value::as_str);
            if !command_ran_in_workspace(working_directory, &self.workspace_root) {
                return HookResult::allow();
            }
            self.ledger
                .write()
                .await
                .record_command(&command, result.success);
        }
        HookResult::allow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_only_the_last_successful_matching_command() {
        let mut ledger = CommandLedger::default();
        assert!(!ledger.covers("cargo test"));

        ledger.record_command("cargo  test", true);
        assert!(ledger.covers("cargo test"));

        ledger.record_mutation();
        assert!(!ledger.covers("cargo test"));

        ledger.record_command("cargo test", true);
        ledger.record_command("ls src", true);
        assert!(!ledger.covers("cargo test"));

        ledger.record_command("cargo test", false);
        assert!(!ledger.covers("cargo test"));
    }

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_shell_command("  cargo   test\n"), "cargo test");
        assert_eq!(normalize_shell_command(""), "");
    }

    // --- new tests for covers() with benign extras ---

    #[test]
    fn covers_with_env_prefix() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("CARGO_TERM_COLOR=never cargo test", true);
        assert!(ledger.covers("cargo test"), "env prefix should be stripped");
    }

    #[test]
    fn covers_with_cd_prefix() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cd /workspace && cargo test", true);
        assert!(ledger.covers("cargo test"), "cd prefix should be stripped");
    }

    #[test]
    fn covers_with_env_and_cd_prefix() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("FOO=bar cd /ws && cargo test", true);
        // FOO=bar is env, then cd /ws && is cd prefix
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn covers_with_offline_flag() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --offline", true);
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn covers_with_color_flag_space() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --color never", true);
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn covers_with_color_flag_equals() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --color=never", true);
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn covers_with_quiet_flag() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test -q", true);
        assert!(ledger.covers("cargo test"));
        let mut ledger2 = CommandLedger::default();
        ledger2.record_command("cargo test --quiet", true);
        assert!(ledger2.covers("cargo test"));
    }

    #[test]
    fn covers_with_nocapture_after_dashdash() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test -- --nocapture", true);
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn covers_when_wanted_adds_benign_test_threads() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --lib write_file", true);
        assert!(ledger.covers("cargo test --lib write_file -- --test-threads=1"));
    }

    #[test]
    fn does_not_cover_with_lib_flag() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --lib", true);
        assert!(!ledger.covers("cargo test"), "--lib restricts subset");
    }

    #[test]
    fn does_not_cover_with_package_flag() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test -p my_crate", true);
        assert!(!ledger.covers("cargo test"), "-p restricts subset");
    }

    #[test]
    fn does_not_cover_with_test_flag() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --test integration", true);
        assert!(!ledger.covers("cargo test"), "--test restricts subset");
    }

    #[test]
    fn does_not_cover_unrecognized_extra_flag() {
        let mut ledger = CommandLedger::default();
        ledger.record_command("cargo test --release", true);
        assert!(!ledger.covers("cargo test"), "--release is not benign");
    }

    #[test]
    fn covers_argv_array_string() {
        // Simulate the hook receiving the command as a JSON array.
        let value = serde_json::json!(["cargo", "test"]);
        let cmd = extract_command_str(&value).unwrap();
        let mut ledger = CommandLedger::default();
        ledger.record_command(&cmd, true);
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn covers_exact_match_with_benign_env_prefix_recorded() {
        let mut ledger = CommandLedger::default();
        // Recorded has env prefix and --offline; wanted is bare.
        ledger.record_command("CARGO_TERM_COLOR=never cargo test --offline", true);
        assert!(ledger.covers("cargo test"));
    }

    #[test]
    fn workspace_cwd_accepts_missing_dot_and_exact_root() {
        let root = Path::new("/tmp/ws");
        assert!(command_ran_in_workspace(None, root));
        assert!(command_ran_in_workspace(Some(""), root));
        assert!(command_ran_in_workspace(Some("."), root));
        assert!(command_ran_in_workspace(Some("/tmp/ws"), root));
        assert!(!command_ran_in_workspace(Some("/tmp/other"), root));
    }

    #[cfg(unix)]
    #[test]
    fn workspace_cwd_accepts_symlink_alias() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let real = tmp.path().join("real");
        std::fs::create_dir(&real).expect("mkdir");
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        let Some(link_str) = link.to_str() else {
            return;
        };
        assert!(command_ran_in_workspace(Some(link_str), &real));
    }
}
