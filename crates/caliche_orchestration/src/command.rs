//! Command allowlist and interpreter rejection for external harness subprocesses.
//!
//! Production (`CALICHE_ENV=production`) requires
//! [`HARNESS_COMMAND_ALLOWLIST_ENV`] unless
//! `CALICHE_ALLOW_INSECURE_PRODUCTION` is set. Bare names match only bare
//! names: listing `pi` does not authorize `/tmp/pi`.

use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;

/// Environment variable listing allowed harness executables.
pub const HARNESS_COMMAND_ALLOWLIST_ENV: &str = "CALICHE_HARNESS_COMMAND_ALLOWLIST";

/// Shells and other interpreters that must never be the harness executable.
///
/// A spec of `sh` + `["-c"]` + prompt-as-last-arg is RCE. The prompt is data;
/// it must not become a shell command string.
const FORBIDDEN_SHELLS: &[&str] = &[
    "sh",
    "bash",
    "zsh",
    "fish",
    "dash",
    "ksh",
    "csh",
    "tcsh",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
];

/// Policy applied to a harness executable before spawn.
#[derive(Debug, Clone, Default)]
pub struct CommandPolicy {
    allowlist: Option<HashSet<String>>,
    fail_closed: bool,
}

impl CommandPolicy {
    /// Read [`HARNESS_COMMAND_ALLOWLIST_ENV`] and the production profile.
    ///
    /// # Errors
    ///
    /// Returns an error when the variable is set but empty.
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            allowlist: harness_allowlist_from_env()?,
            fail_closed: production_fail_closed(),
        })
    }

    /// No allowlist and not fail-closed. Use in tests and non-production hosts
    /// that still reject shells and static-executable violations.
    #[must_use]
    pub fn open() -> Self {
        Self {
            allowlist: None,
            fail_closed: false,
        }
    }

    /// Explicit allowlist; not fail-closed beyond that list.
    #[must_use]
    pub fn with_allowlist(allowlist: HashSet<String>) -> Self {
        Self {
            allowlist: Some(allowlist),
            fail_closed: false,
        }
    }

    /// Require an allowlist even when the environment variable is unset.
    #[must_use]
    pub fn fail_closed(mut self) -> Self {
        self.fail_closed = true;
        self
    }

    /// Validate `command` against shells, static-executable rules, and the allowlist.
    ///
    /// # Errors
    ///
    /// Returns an error when the command is empty, contains NUL or `{{`, is a
    /// forbidden shell, is missing from a required allowlist, or uses a path
    /// that is not listed exactly.
    pub fn validate(&self, command: &str) -> Result<()> {
        reject_static_executable(command)?;
        if is_forbidden_shell(command) {
            anyhow::bail!(
                "harness command '{}' is a shell interpreter; pass the harness executable directly and keep the prompt as argv",
                command.trim()
            );
        }
        match &self.allowlist {
            Some(allowlist) => command_allowed_by_allowlist(command, allowlist),
            None if self.fail_closed => anyhow::bail!(
                "external harness commands require {HARNESS_COMMAND_ALLOWLIST_ENV} when CALICHE_ENV=production"
            ),
            None => Ok(()),
        }
    }
}

/// Parse [`HARNESS_COMMAND_ALLOWLIST_ENV`] when set.
///
/// # Errors
///
/// Returns an error when the variable is set but empty.
pub fn harness_allowlist_from_env() -> Result<Option<HashSet<String>>> {
    let Ok(raw) = std::env::var(HARNESS_COMMAND_ALLOWLIST_ENV) else {
        return Ok(None);
    };
    let allowlist = raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    if allowlist.is_empty() {
        anyhow::bail!("{HARNESS_COMMAND_ALLOWLIST_ENV} must list at least one command");
    }
    Ok(Some(allowlist))
}

/// Whether `command` is a known shell or command interpreter (by basename).
#[must_use]
pub fn is_forbidden_shell(command: &str) -> bool {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return false;
    }
    let name = trimmed
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(trimmed)
        .to_ascii_lowercase();
    FORBIDDEN_SHELLS.contains(&name.as_str())
}

/// Bare names match allowlist names. Path commands must be listed exactly.
///
/// # Errors
///
/// Returns an error when the command is empty, contains NUL, or is not listed.
pub fn command_allowed_by_allowlist(command: &str, allowlist: &HashSet<String>) -> Result<()> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        anyhow::bail!("harness transport requires command");
    }
    if trimmed.contains('\0') {
        anyhow::bail!("harness command contains interior NUL");
    }

    let has_path_separator = Path::new(trimmed).components().count() > 1
        || trimmed.contains('/')
        || trimmed.contains('\\');

    if has_path_separator {
        if allowlist.contains(trimmed) {
            return Ok(());
        }
        anyhow::bail!(
            "harness command {trimmed} uses a path; list the exact executable in {HARNESS_COMMAND_ALLOWLIST_ENV}"
        );
    }

    if allowlist.contains(trimmed) {
        return Ok(());
    }

    anyhow::bail!("harness command {trimmed} is not in {HARNESS_COMMAND_ALLOWLIST_ENV}");
}

fn reject_static_executable(command: &str) -> Result<()> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        anyhow::bail!("harness command is empty");
    }
    if trimmed.contains('\0') {
        anyhow::bail!("harness command contains interior NUL");
    }
    if trimmed.contains("{{") {
        anyhow::bail!(
            "harness command must be a static executable; interpolate only the prompt as argv"
        );
    }
    Ok(())
}

fn production_fail_closed() -> bool {
    is_production_profile() && !allow_insecure_production()
}

fn is_production_profile() -> bool {
    std::env::var("CALICHE_ENV")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "production" | "prod"
            )
        })
        .unwrap_or(false)
}

fn allow_insecure_production() -> bool {
    matches!(
        std::env::var("CALICHE_ALLOW_INSECURE_PRODUCTION")
            .ok()
            .as_deref()
            .map(str::trim),
        Some("1" | "true" | "yes" | "on")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basename_does_not_authorize_path() {
        let allowlist = HashSet::from(["pi".to_string()]);
        command_allowed_by_allowlist("pi", &allowlist).expect("bare name");
        command_allowed_by_allowlist("/tmp/pi", &allowlist).expect_err("path");
        command_allowed_by_allowlist("./pi", &allowlist).expect_err("relative");
    }

    #[test]
    fn forbidden_shells_include_path_basenames() {
        assert!(is_forbidden_shell("sh"));
        assert!(is_forbidden_shell("/bin/bash"));
        assert!(is_forbidden_shell(r"C:\Windows\System32\cmd.exe"));
        assert!(is_forbidden_shell("pwsh"));
        assert!(!is_forbidden_shell("pi"));
        assert!(!is_forbidden_shell("hermes"));
        assert!(!is_forbidden_shell("/usr/local/bin/pi"));
    }

    #[test]
    fn open_policy_still_rejects_shells_and_templates() {
        let policy = CommandPolicy::open();
        policy.validate("sh").expect_err("shell");
        policy.validate("bash").expect_err("shell");
        policy
            .validate("{{cmd}}")
            .expect_err("template interpolation");
        policy.validate("").expect_err("empty");
        policy.validate("pi").expect("bare harness name");
    }

    #[test]
    fn allowlist_and_fail_closed() {
        let policy = CommandPolicy::with_allowlist(HashSet::from(["hermes".to_string()]));
        policy.validate("hermes").expect("listed");
        policy.validate("pi").expect_err("not listed");

        let closed = CommandPolicy::open().fail_closed();
        closed
            .validate("pi")
            .expect_err("production requires allowlist");
    }
}
