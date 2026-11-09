//! Stdio approval resolver and CLI amendment persistence.
//!
//! When ExecPolicy returns Prompt/RequireApproval, this prompts the user via stdin/stdout
//! and can persist allow rules to config when the user chooses "always".

use async_trait::async_trait;
use caliche::agent::hooks::{AmendmentPersister, ApprovalOutcome, ApprovalResolver};
use futures::future::BoxFuture;
use std::io::{self, Write};
use std::path::PathBuf;

/// Prompts the user via stdin/stdout for [y]es / [n]o / [a]lways.
#[derive(Debug)]
pub struct StdioApprovalResolver;

impl StdioApprovalResolver {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StdioApprovalResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ApprovalResolver for StdioApprovalResolver {
    fn resolve_approval(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
        reason: &str,
        proposed_amendment: Option<&[String]>,
    ) -> BoxFuture<'_, anyhow::Result<ApprovalOutcome>> {
        let card = approval_card_text(tool_name, tool_input, reason);
        let amendment = proposed_amendment.map(|p| p.to_vec());
        Box::pin(async move {
            println!("\n{card}");

            let has_always = amendment.as_ref().map(|p| !p.is_empty()).unwrap_or(false);
            if has_always {
                print!("  [y]es / [n]o / [a]lways: ");
            } else {
                print!("  [y]es / [n]o: ");
            }
            io::stdout().flush()?;

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                return Ok(ApprovalOutcome::Deny);
            }

            let response = input.trim().to_lowercase();
            match response.as_str() {
                "y" | "yes" => Ok(ApprovalOutcome::Allow),
                "n" | "no" => Ok(ApprovalOutcome::Deny),
                "a" | "always" if has_always => {
                    let Some(rule_prefix) = amendment else {
                        return Ok(ApprovalOutcome::Deny);
                    };
                    Ok(ApprovalOutcome::AllowAndPersist { rule_prefix })
                }
                _ => Ok(ApprovalOutcome::Deny),
            }
        })
    }
}

/// Full argument card for the CLI door.
///
/// The text comes from the policy broker card. It does not print `<no command>`.
pub(crate) fn approval_card_text(
    tool_name: &str,
    tool_input: &serde_json::Value,
    reason: &str,
) -> String {
    caliche::policy::ActionCard::from_call(tool_name, tool_input, reason).render()
}

/// Persists allow rules to the user config.
#[derive(Debug)]
pub struct CliAmendmentPersister {
    config_path: Option<PathBuf>,
}

impl CliAmendmentPersister {
    /// Create a persister that writes to the given config path, or to the
    /// user config when `None`. Never the project config: its permissions
    /// are ignored as untrusted, so a rule written there would not apply.
    pub fn new(config_path: Option<PathBuf>) -> Self {
        let path =
            config_path.or_else(|| Some(crate::config::FileConfigSource::user_config_path()));
        Self { config_path: path }
    }
}

#[async_trait]
impl AmendmentPersister for CliAmendmentPersister {
    fn persist_allow_rule(&self, rule_prefix: &[String]) -> BoxFuture<'_, anyhow::Result<()>> {
        if rule_prefix.is_empty() {
            return Box::pin(
                async move { Err(anyhow::anyhow!("Cannot persist empty rule prefix")) },
            );
        }
        let path = self.config_path.clone();
        let rule_prefix = rule_prefix.to_vec();
        Box::pin(async move {
            let path = path.ok_or_else(|| {
                anyhow::anyhow!("No config file path available for amendment persistence")
            })?;
            crate::config::append_bash_allow_rule(&path, &rule_prefix).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CliAmendmentPersister, approval_card_text};
    use serde_json::json;

    /// The lookup reads the current directory, which tests may not change:
    /// the test re-runs itself in a child started inside a repository that
    /// has a project config.
    #[test]
    fn allow_rules_go_to_the_user_config() {
        const MARKER: &str = "CALICHE_PERSISTER_PROBE_IN_REPO";
        if std::env::var_os(MARKER).is_none() {
            let repo = tempfile::tempdir().expect("tempdir");
            std::fs::write(repo.path().join("caliche.toml"), "").expect("project config");
            let test = "approval_resolver::tests::allow_rules_go_to_the_user_config";
            let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", test, "--nocapture"])
                .current_dir(repo.path())
                .env(MARKER, "1")
                .output()
                .expect("re-run test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        assert!(crate::config::ProjectTomlConfigSource::find_config_file().is_some());
        assert_eq!(
            CliAmendmentPersister::new(None).config_path,
            Some(crate::config::FileConfigSource::user_config_path())
        );
    }

    #[test]
    fn fs_card_shows_the_path_and_not_the_missing_command_placeholder() {
        let text = approval_card_text(
            "fs",
            &json!({"op": "write", "path": "src/lib.rs"}),
            "needs review",
        );
        assert!(text.contains("Fs write: src/lib.rs"));
        assert!(!text.contains("<no command>"));
        assert!(!text.contains("Command:"));
    }

    #[test]
    fn bash_card_shows_the_command() {
        let text = approval_card_text("bash", &json!({"command": "git status"}), "prompt");
        assert!(text.contains("Shell command: git status"));
    }
}
