//! Auto-formatter runner for post-write/edit hooks.

use std::path::Path;
use tokio::process::Command;

/// A single configured formatter: the file extensions it applies to and the
/// shell command to run (`$FILE` is replaced with the absolute path).
#[derive(Debug, Clone)]
pub struct FormatterRule {
    pub extensions: Vec<String>,
    pub command: String,
}

/// Run the first matching formatter for `path`, if any.
///
/// `$FILE` is replaced with a POSIX-quoted absolute path so a crafted
/// filename cannot break out of `sh -c`. Runs as fire-and-forget (errors are
/// logged at debug level, never propagated) so a broken formatter command can
/// never stall or crash the agent. Returns an optional warning string to
/// surface in the tool result.
pub async fn run_formatter_if_match(rules: &[FormatterRule], path: &Path) -> Option<String> {
    let ext = path.extension()?.to_str()?;
    let abs_path = path.canonicalize().ok()?;
    let abs_str = abs_path.to_string_lossy();

    for rule in rules {
        if rule.extensions.iter().any(|e| e == ext) {
            let cmd_str = rule.command.replace("$FILE", &posix_single_quote(&abs_str));
            tracing::debug!(formatter = %cmd_str, file = %abs_str, "running formatter");

            let result = Command::new("sh")
                .arg("-c")
                .arg(&cmd_str)
                .env_clear()
                .envs(caliche_contracts::child_env::host_child_env())
                .output()
                .await;

            match result {
                Ok(out) if out.status.success() => {
                    tracing::debug!(formatter = %cmd_str, "formatter succeeded");
                    return None;
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let warning = format!(
                        "Formatter `{}` exited with status {}. stderr: {}",
                        rule.command,
                        out.status,
                        stderr.trim()
                    );
                    tracing::debug!(%warning, "formatter failed (non-fatal)");
                    return Some(warning);
                }
                Err(e) => {
                    tracing::debug!(err = %e, "formatter could not be spawned (non-fatal)");
                    return None;
                }
            }
        }
    }
    None
}

/// Quote `value` for POSIX `sh -c` so metacharacters in a path stay data.
pub(crate) fn posix_single_quote(value: &str) -> String {
    let mut out = String::from("'");
    for (i, chunk) in value.split('\'').enumerate() {
        if i > 0 {
            out.push_str(r"'\''");
        }
        out.push_str(chunk);
    }
    out.push('\'');
    out
}

#[cfg(test)]
mod tests {
    use super::{FormatterRule, posix_single_quote, run_formatter_if_match};

    #[cfg(unix)]
    #[tokio::test]
    async fn formatter_does_not_receive_host_credentials() {
        use crate::testing::planted_credential::{PRINT_PLANTED, rerun_with_planted_credential};
        if rerun_with_planted_credential(
            "harness::formatter::tests::formatter_does_not_receive_host_credentials",
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("probe.rs");
        std::fs::write(&file, "").expect("write probe");
        let rules = [FormatterRule {
            extensions: vec!["rs".to_string()],
            command: format!(
                "{{ {PRINT_PLANTED}; printf ' HOME=%s' \"${{HOME:+set}}\"; }} >&2; exit 1"
            ),
        }];
        let warning = run_formatter_if_match(&rules, &file)
            .await
            .expect("formatter warning");
        assert!(warning.ends_with("stderr: unset HOME=set"), "{warning}");
    }

    /// A scrubbed child gets `PATH` without its relative entries, so a
    /// command it looks up never resolves inside its working directory.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_scrubbed_child_path_has_no_relative_entries() {
        use crate::testing::planted_credential::rerun_in_child;
        use caliche_contracts::child_env::PATH_VAR;
        if rerun_in_child(
            "harness::formatter::tests::a_scrubbed_child_path_has_no_relative_entries",
            &[(
                PATH_VAR,
                std::ffi::OsStr::new(".:/usr/bin::./node_modules/.bin:/bin"),
            )],
            None,
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let file = dir.path().join("probe.rs");
        std::fs::write(&file, "").expect("write probe");
        let rules = [FormatterRule {
            extensions: vec!["rs".to_string()],
            command: "printf %s \"$PATH\" >&2; exit 1".to_string(),
        }];

        let warning = run_formatter_if_match(&rules, &file)
            .await
            .expect("formatter warning");

        assert!(warning.ends_with("stderr: /usr/bin:/bin"), "{warning}");
    }

    /// Runs the formatter in a child test process whose `PATH` is `path`
    /// and whose cwd holds a planted `./sh` and `./bin/sh`. Each writes a
    /// marker through an absolute `touch` and exits 0. The real `sh` must
    /// run with the default `PATH`, and no marker may be written.
    #[cfg(unix)]
    async fn only_relative_path_entries_give_the_default_path(test: &str, path: &str) {
        use crate::testing::planted_credential::rerun_in_child;
        use caliche_contracts::child_env::{DEFAULT_SEARCH_PATH, PATH_VAR};
        use std::os::unix::fs::PermissionsExt;
        let repo = tempfile::tempdir().expect("repo");
        let marker = repo.path().join("planted-sh-ran");
        std::fs::create_dir(repo.path().join("bin")).expect("bin");
        for planted in ["sh", "bin/sh"] {
            let script = repo.path().join(planted);
            std::fs::write(
                &script,
                format!("#!/bin/sh\n/usr/bin/touch '{}'\n", marker.display()),
            )
            .expect("planted sh");
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }
        if rerun_in_child(
            test,
            &[(PATH_VAR, std::ffi::OsStr::new(path))],
            Some(repo.path()),
        ) {
            assert!(!marker.exists(), "a planted sh ran with PATH={path}");
            return;
        }
        let file = std::env::current_dir().expect("cwd").join("probe.rs");
        std::fs::write(&file, "").expect("write probe");
        let rules = [FormatterRule {
            extensions: vec!["rs".to_string()],
            command: "printf %s \"$PATH\" >&2; exit 1".to_string(),
        }];

        let warning = run_formatter_if_match(&rules, &file)
            .await
            .expect("formatter warning");

        assert!(
            warning.ends_with(&format!("stderr: {DEFAULT_SEARCH_PATH}")),
            "{warning}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_with_only_dot_on_path_gets_the_default_path() {
        only_relative_path_entries_give_the_default_path(
            "harness::formatter::tests::a_child_with_only_dot_on_path_gets_the_default_path",
            ".",
        )
        .await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_child_with_only_a_relative_bin_on_path_gets_the_default_path() {
        only_relative_path_entries_give_the_default_path(
            "harness::formatter::tests::a_child_with_only_a_relative_bin_on_path_gets_the_default_path",
            "./bin",
        )
        .await;
    }

    #[test]
    fn posix_single_quote_neutralizes_metacharacters() {
        assert_eq!(posix_single_quote("/tmp/foo.rs"), "'/tmp/foo.rs'");
        assert_eq!(posix_single_quote("/tmp/foo$(id).rs"), "'/tmp/foo$(id).rs'");

        let injected = "/tmp/foo'; echo INJECTED; '.rs";
        let quoted = posix_single_quote(injected);
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("printf %s {quoted}"))
            .output()
            .expect("sh");
        assert!(
            output.status.success(),
            "stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), injected);
    }
}
