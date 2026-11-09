//! Top-level `share` / `unshare` CLI commands.

use crate::client::AgentClient;
use crate::errors::{CliError, CliResult};
use crate::storage::SessionStore;
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use async_trait::async_trait;
use caliche_contracts::sessions::{ShareCreateRequestDto, ShareVisibilityDto};
use std::io::Write;

/// Arguments for `caliche share`.
#[derive(Debug, Clone, Default)]
pub struct ShareArgs {
    pub session_id: Option<String>,
    pub visibility: Option<String>,
}

impl ShareArgs {
    #[must_use]
    pub fn from_flags(session_id: Option<&str>, visibility: Option<&str>) -> Self {
        Self {
            session_id: session_id.map(str::to_string),
            visibility: visibility.map(str::to_string),
        }
    }
}

/// Arguments for `caliche unshare`.
#[derive(Debug, Clone, Default)]
pub struct UnshareArgs {
    pub session_id: Option<String>,
}

impl UnshareArgs {
    #[must_use]
    pub fn from_flags(session_id: Option<&str>) -> Self {
        Self {
            session_id: session_id.map(str::to_string),
        }
    }
}

/// Create a redacted share for a remote session and print the URL.
pub struct ShareHandler;

#[async_trait]
impl CommandHandler for ShareHandler {
    type Args = ShareArgs;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let session_id = resolve_session_id(args.session_id.as_deref()).await?;
        let visibility = parse_visibility(args.visibility.as_deref())?;
        let client = require_remote_client().await?;
        let response = client
            .share_session_with_options(&session_id, ShareCreateRequestDto { visibility })
            .await
            .map_err(|error| CliError::Command(error.to_string()))?;

        let mut out = std::io::stdout().lock();
        writeln!(out, "Share URL: {}", response.url)
            .map_err(|error| CliError::Command(error.to_string()))?;
        writeln!(out, "Slug: {}", response.slug)
            .map_err(|error| CliError::Command(error.to_string()))?;
        let visibility = match response.visibility {
            ShareVisibilityDto::Public => "public",
            ShareVisibilityDto::Authenticated => "authenticated",
        };
        writeln!(out, "Visibility: {visibility}")
            .map_err(|error| CliError::Command(error.to_string()))?;
        if response.redaction_count > 0 {
            writeln!(out, "Redactions: {}", response.redaction_count)
                .map_err(|error| CliError::Command(error.to_string()))?;
        }
        Ok(CommandResult::Success(Some(response.url)))
    }

    fn name(&self) -> &'static str {
        "share"
    }

    fn description(&self) -> &'static str {
        "Create a redacted public share URL for a remote session"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// Revoke share links for a remote session.
pub struct UnshareHandler;

#[async_trait]
impl CommandHandler for UnshareHandler {
    type Args = UnshareArgs;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let session_id = resolve_session_id(args.session_id.as_deref()).await?;
        let client = require_remote_client().await?;
        client
            .unshare_session(&session_id)
            .await
            .map_err(|error| CliError::Command(error.to_string()))?;

        let mut out = std::io::stdout().lock();
        writeln!(out, "Unshared session {session_id}")
            .map_err(|error| CliError::Command(error.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "unshare"
    }

    fn description(&self) -> &'static str {
        "Revoke share links for a remote session"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

async fn require_remote_client() -> CliResult<AgentClient> {
    let client = AgentClient::discover().map_err(|error| CliError::Command(error.to_string()))?;
    if client.base_url().is_none() {
        return Err(CliError::Command(
            "No live agent server found. Start one with `caliche --server` or open a thin-client REPL first."
                .to_string(),
        ));
    }
    if !client.health().await.unwrap_or(false) {
        return Err(CliError::Command(
            "Discovered agent server failed health check".to_string(),
        ));
    }
    Ok(client)
}

async fn resolve_session_id(explicit: Option<&str>) -> CliResult<String> {
    if let Some(session_id) = explicit.filter(|value| !value.trim().is_empty()) {
        return Ok(session_id.trim().to_string());
    }
    let store = SessionStore::new()
        .await
        .map_err(|error| CliError::Command(error.to_string()))?;
    store
        .get_last_session_id()
        .map_err(|error| CliError::Command(error.to_string()))?
        .ok_or_else(|| {
            CliError::Command(
                "No session id available. Pass --session <id> or create a remote session first."
                    .to_string(),
            )
        })
}

fn parse_visibility(raw: Option<&str>) -> CliResult<Option<ShareVisibilityDto>> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "public" => Ok(Some(ShareVisibilityDto::Public)),
        "authenticated" => Ok(Some(ShareVisibilityDto::Authenticated)),
        other => Err(CliError::Command(format!(
            "unknown --visibility '{other}' (expected public or authenticated)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_visibility_accepts_known_values() {
        assert_eq!(
            parse_visibility(Some("public")).unwrap(),
            Some(ShareVisibilityDto::Public)
        );
        assert_eq!(
            parse_visibility(Some("Authenticated")).unwrap(),
            Some(ShareVisibilityDto::Authenticated)
        );
        assert!(parse_visibility(None).unwrap().is_none());
        assert!(parse_visibility(Some("friends")).is_err());
    }
}
