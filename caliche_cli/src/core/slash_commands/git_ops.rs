use super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;
use caliche::AgentService;
use std::path::Path;

/// `cmd.exe` looks up a bare program name in the current directory before
/// `PATH`. With this set to `1` it does not, so a `cargo.cmd` committed to
/// the repository cannot stand in for `cargo` when `/check` runs.
#[cfg(windows)]
const NO_CWD_EXE_LOOKUP: &str = "NoDefaultCurrentDirectoryInExePath";

impl SlashCommandExecutor {
    pub(crate) async fn handle_check(args: &[String]) -> Result<SlashCommandResult> {
        let command = if args.is_empty() {
            match Self::detect_project_check_command().await {
                Some(command) => command,
                None => {
                    return Ok(SlashCommandResult::Error(
                        "No default check command detected. Use /check <command>, e.g. /check cargo check".to_string(),
                    ));
                }
            }
        } else {
            args.join(" ")
        };

        let cwd = std::env::current_dir()?;
        let started = std::time::Instant::now();
        let output = Self::run_shell_command(&command, &cwd).await;
        let elapsed_ms = started.elapsed().as_millis();
        match output {
            Ok(output) => {
                let status = output.status.code().map_or_else(
                    || "terminated by signal".to_string(),
                    |code| code.to_string(),
                );
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let content = Self::format_check_output(
                    &command,
                    &status,
                    output.status.success(),
                    elapsed_ms,
                    &stdout,
                    &stderr,
                );
                let title = if output.status.success() {
                    "Check Passed"
                } else {
                    "Check Failed"
                };
                Ok(SlashCommandResult::ShowPopup {
                    title: title.to_string(),
                    content,
                })
            }
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Failed to run check command: {e}"
            ))),
        }
    }

    async fn detect_project_check_command() -> Option<String> {
        let cwd = std::env::current_dir().ok()?;
        if tokio::fs::metadata(cwd.join("Cargo.toml")).await.is_ok() {
            return Some("cargo check".to_string());
        }
        if tokio::fs::metadata(cwd.join("package.json")).await.is_ok() {
            if tokio::fs::metadata(cwd.join("bun.lock")).await.is_ok()
                || tokio::fs::metadata(cwd.join("bun.lockb")).await.is_ok()
            {
                return Some("bun test".to_string());
            }
            return Some("npm test".to_string());
        }
        if tokio::fs::metadata(cwd.join("pyproject.toml"))
            .await
            .is_ok()
        {
            return Some("uv run pytest".to_string());
        }
        None
    }

    async fn run_shell_command(command: &str, cwd: &Path) -> std::io::Result<std::process::Output> {
        #[cfg(windows)]
        {
            tokio::process::Command::new("cmd")
                .args(["/C", command])
                .current_dir(cwd)
                .env(NO_CWD_EXE_LOOKUP, "1")
                .output()
                .await
        }
        #[cfg(not(windows))]
        {
            tokio::process::Command::new("sh")
                .args(["-lc", command])
                .current_dir(cwd)
                .output()
                .await
        }
    }

    fn format_check_output(
        command: &str,
        status: &str,
        success: bool,
        elapsed_ms: u128,
        stdout: &str,
        stderr: &str,
    ) -> String {
        const MAX_SECTION_CHARS: usize = 12_000;
        let mut out = String::new();
        out.push_str(&format!("Command: {command}\n"));
        out.push_str(&format!("Status: {status}\n"));
        out.push_str(&format!(
            "Result: {}\n",
            if success { "passed" } else { "failed" }
        ));
        out.push_str(&format!("Duration: {elapsed_ms}ms\n"));

        if !stdout.trim().is_empty() {
            out.push_str("\nstdout:\n");
            out.push_str(&Self::truncate_middle(stdout.trim(), MAX_SECTION_CHARS));
            out.push('\n');
        }
        if !stderr.trim().is_empty() {
            out.push_str("\nstderr:\n");
            out.push_str(&Self::truncate_middle(stderr.trim(), MAX_SECTION_CHARS));
            out.push('\n');
        }
        out
    }

    fn truncate_middle(value: &str, max_chars: usize) -> String {
        let char_count = value.chars().count();
        if char_count <= max_chars {
            return value.to_string();
        }

        let keep_each_side = max_chars.saturating_sub(80) / 2;
        let prefix: String = value.chars().take(keep_each_side).collect();
        let suffix: String = value
            .chars()
            .rev()
            .take(keep_each_side)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        format!("{prefix}\n\n[output truncated: {char_count} chars total]\n\n{suffix}")
    }

    pub(crate) async fn handle_pr(service: &mut AgentService) -> Result<SlashCommandResult> {
        use crate::handlers::git::run_pr_command;

        let mut buf = Vec::new();
        match run_pr_command(service, &mut buf).await {
            Ok(()) => match String::from_utf8(buf) {
                Ok(content) => Ok(SlashCommandResult::ShowPopup {
                    title: "PR Description".to_string(),
                    content,
                }),
                Err(e) => Ok(SlashCommandResult::Error(format!("Output encoding: {}", e))),
            },
            Err(e) => Ok(SlashCommandResult::Error(format!("PR failed: {}", e))),
        }
    }

    pub(crate) async fn handle_commit(service: &mut AgentService) -> Result<SlashCommandResult> {
        use crate::handlers::git::run_commit_command;

        let mut buf = Vec::new();
        match run_commit_command(service, &mut buf, false).await {
            Ok(()) => match String::from_utf8(buf) {
                Ok(content) => Ok(SlashCommandResult::ShowPopup {
                    title: "Commit Message".to_string(),
                    content,
                }),
                Err(e) => Ok(SlashCommandResult::Error(format!("Output encoding: {}", e))),
            },
            Err(e) => Ok(SlashCommandResult::Error(format!("Commit failed: {}", e))),
        }
    }

    pub(crate) async fn handle_review(service: &mut AgentService) -> Result<SlashCommandResult> {
        use crate::handlers::git::run_review_command;

        let mut buf = Vec::new();
        match run_review_command(service, &mut buf).await {
            Ok(()) => match String::from_utf8(buf) {
                Ok(content) => Ok(SlashCommandResult::ShowPopup {
                    title: "Code Review".to_string(),
                    content,
                }),
                Err(e) => Ok(SlashCommandResult::Error(format!("Output encoding: {}", e))),
            },
            Err(e) => Ok(SlashCommandResult::Error(format!("Review failed: {}", e))),
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    const CHILD_RUN: &str = "CALICHE_CWD_LOOKUP_TEST_CHILD";

    /// Re-runs itself in a child test process with the host-wide
    /// NoDefaultCurrentDirectoryInExePath removed, so only run_shell_command
    /// can block the lookup and no other test sees a changed environment.
    #[test]
    fn check_does_not_run_a_program_from_the_working_directory() {
        if std::env::var_os(CHILD_RUN).is_some() {
            check_planted_program_does_not_run();
            return;
        }
        let module = module_path!();
        let name = format!(
            "{}::check_does_not_run_a_program_from_the_working_directory",
            &module[module.find("::").expect("crate prefix") + 2..]
        );
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args(["--exact", name.as_str(), "--test-threads=1"])
            .env(CHILD_RUN, "1")
            .env_remove(NO_CWD_EXE_LOOKUP)
            .output()
            .expect("rerun the test in a child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(output.status.success(), "{stdout}");
        assert!(stdout.contains("1 passed"), "{stdout}");
    }

    fn check_planted_program_does_not_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("planted.cmd"),
            "@echo off\r\necho planted\r\n",
        )
        .expect("plant a program in the working directory");
        let output = tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(SlashCommandExecutor::run_shell_command(
                "planted",
                dir.path(),
            ))
            .expect("run cmd");
        assert_eq!(String::from_utf8_lossy(&output.stdout), "");
        assert!(!output.status.success());
    }
}

#[cfg(all(test, unix))]
mod credential_tests {
    use super::*;

    #[tokio::test]
    async fn check_command_does_not_get_stored_credentials() {
        let name = "CALICHE_CHECK_COMMAND_TEST_API_KEY";
        caliche_contracts::credential_env::set(name, "stored-check-key");

        let output = SlashCommandExecutor::run_shell_command(
            &format!("printenv {name} || echo absent"),
            Path::new("."),
        )
        .await
        .expect("run check command");

        assert_eq!(String::from_utf8_lossy(&output.stdout), "absent\n");
    }
}
