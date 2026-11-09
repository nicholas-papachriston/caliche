//! Allowlist for MCP stdio commands.

use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;

/// Parse `CALICHE_MCP_COMMAND_ALLOWLIST` when set.
///
/// # Errors
///
/// Returns an error when the variable is set but empty.
pub fn stdio_command_allowlist_from_env() -> Result<Option<HashSet<String>>> {
    let Ok(raw) = std::env::var("CALICHE_MCP_COMMAND_ALLOWLIST") else {
        return Ok(None);
    };
    let allowlist = raw
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(ToString::to_string)
        .collect::<HashSet<_>>();
    if allowlist.is_empty() {
        anyhow::bail!("CALICHE_MCP_COMMAND_ALLOWLIST must list at least one command");
    }
    Ok(Some(allowlist))
}

/// Validate a stdio command against an explicit allowlist.
///
/// Bare names match allowlist names. Commands that contain a path separator
/// must be listed exactly; a basename of `npx` does not authorize `/tmp/npx`.
///
/// # Errors
///
/// Returns an error when the command is empty, contains NUL, or is not listed.
pub fn command_allowed_by_allowlist(command: &str, allowlist: &HashSet<String>) -> Result<()> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        anyhow::bail!("stdio transport requires command");
    }
    if trimmed.contains('\0') {
        anyhow::bail!("stdio command contains interior NUL");
    }

    let has_path_separator = Path::new(trimmed).components().count() > 1
        || trimmed.contains('/')
        || trimmed.contains('\\');

    if has_path_separator {
        if allowlist.contains(trimmed) {
            return Ok(());
        }
        anyhow::bail!(
            "MCP stdio command {trimmed} uses a path; list the exact executable in CALICHE_MCP_COMMAND_ALLOWLIST"
        );
    }

    if allowlist.contains(trimmed) {
        return Ok(());
    }

    anyhow::bail!("MCP stdio command {trimmed} is not in CALICHE_MCP_COMMAND_ALLOWLIST");
}

/// Require `CALICHE_MCP_COMMAND_ALLOWLIST` and match `command`.
///
/// # Errors
///
/// Returns an error when the allowlist is unset or the command is not listed.
pub fn validate_stdio_command_required(command: &str) -> Result<()> {
    validate_stdio_command_against(command, stdio_command_allowlist_from_env()?.as_ref())
}

/// Require an allowlist and match `command` against it.
///
/// # Errors
///
/// Returns an error when `allowlist` is `None` or the command is not listed.
pub fn validate_stdio_command_against(
    command: &str,
    allowlist: Option<&HashSet<String>>,
) -> Result<()> {
    let allowlist = allowlist.ok_or_else(|| {
        anyhow::anyhow!("stdio MCP commands require CALICHE_MCP_COMMAND_ALLOWLIST to be set")
    })?;
    command_allowed_by_allowlist(command, allowlist)
}

/// Library/CLI stdio MCP: apply the allowlist when set; require it in production.
///
/// # Errors
///
/// Returns an error when production has no allowlist or the command is not listed.
pub fn validate_stdio_command_for_library(command: &str) -> Result<()> {
    match stdio_command_allowlist_from_env()? {
        Some(allowlist) => command_allowed_by_allowlist(command, &allowlist),
        None if mcp_stdio_fail_closed() => anyhow::bail!(
            "stdio MCP commands require CALICHE_MCP_COMMAND_ALLOWLIST when CALICHE_ENV=production"
        ),
        None => Ok(()),
    }
}

fn mcp_stdio_fail_closed() -> bool {
    crate::security::is_production_profile() && !crate::security::allow_insecure_production()
}

#[cfg(test)]
mod tests {
    use super::command_allowed_by_allowlist;
    use std::collections::HashSet;

    #[test]
    fn basename_does_not_authorize_path() {
        let allowlist = HashSet::from(["npx".to_string()]);
        command_allowed_by_allowlist("npx", &allowlist).expect("bare name");
        command_allowed_by_allowlist("/tmp/npx", &allowlist).expect_err("path");
        command_allowed_by_allowlist("./npx", &allowlist).expect_err("relative");
    }
}
