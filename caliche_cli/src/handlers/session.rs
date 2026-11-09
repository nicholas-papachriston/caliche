//! Session management command handlers.
//!
//! Handles session listing, resuming, deleting, renaming, and exporting.

use crate::commands::SessionAction;
use crate::cost::{format_tokens, get_pricing};
use crate::errors::{CliError, CliResult, SessionError};
use crate::session::AggregateStats;
use crate::storage::SessionStore;
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;

#[path = "session_actions.rs"]
mod session_actions;

/// Handler for session commands.
pub struct SessionHandler;

#[async_trait]
impl CommandHandler for SessionHandler {
    type Args = SessionAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        if let SessionAction::Resume { session_id } = &args {
            let resolved = resolve_and_validate_session(session_id)
                .await
                .map_err(|e| CliError::Session(SessionError::InvalidState(e.to_string())))?;
            return Ok(CommandResult::ResumeSession(resolved));
        }
        handle_session_command_exit(args)
            .await
            .map_err(|e| CliError::Session(SessionError::InvalidState(e.to_string())))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "session"
    }

    fn description(&self) -> &'static str {
        "Manage conversation sessions"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// Resolve and validate a session ID (supports prefix matching). Returns the full session ID.
async fn resolve_and_validate_session(session_id: &str) -> Result<String> {
    let session_store = SessionStore::new().await?;
    if session_id.eq_ignore_ascii_case("last") {
        if let Some(last) = session_store.load_last_async().await? {
            return Ok(last.id);
        }
        anyhow::bail!("No previous sessions found");
    }
    if session_store.exists_async(session_id).await? {
        return Ok(session_id.to_string());
    }
    let sessions = session_store.list_async().await?;
    let matches: Vec<_> = sessions
        .iter()
        .filter(|s| s.starts_with(session_id))
        .collect();
    if matches.is_empty() {
        anyhow::bail!("Session '{}' not found", session_id);
    }
    if matches.len() == 1 {
        return Ok(matches[0].clone());
    }
    anyhow::bail!(
        "Multiple sessions match '{}': {}",
        session_id,
        matches
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Handle session subcommands that exit after completion (list, delete, rename, show, export, stats).
async fn handle_session_command_exit(action: SessionAction) -> Result<()> {
    if matches!(&action, SessionAction::Resume { .. }) {
        anyhow::bail!("Resume should be handled separately");
    }

    if let SessionAction::InspectFile { path, verbose } = &action {
        return session_actions::inspect_session_file(path, *verbose);
    }

    let session_store = SessionStore::new().await?;
    session_actions::dispatch_session_action(session_store, action).await
}

/// Print formatted aggregate statistics.
pub(super) fn print_aggregate_stats(stats: &AggregateStats) {
    println!("\n=== Session Statistics ===\n");

    if stats.total_sessions == 0 {
        println!("No sessions found matching the filters.");
        return;
    }

    println!("Period:          {}", stats.date_range());
    println!("Total Sessions:  {}", stats.total_sessions);
    println!("Total Messages:  {}", stats.total_messages);
    println!("Total Tokens:    {}", format_tokens(stats.total_tokens));
    println!();

    let providers = stats.providers_by_usage();
    if !providers.is_empty() {
        println!("Provider Usage:");
        for (provider, (sessions, tokens)) in providers.iter().take(5) {
            println!(
                "  {:<15} {:>3} sessions   {:>10} tokens",
                provider,
                sessions,
                format_tokens(*tokens)
            );
        }
        println!();
    }

    let models = stats.models_by_usage();
    if !models.is_empty() {
        println!("Top Models:");
        for (model, (sessions, tokens)) in models.iter().take(5) {
            let display_model = if model.len() > 25 {
                format!("{}...", caliche_contracts::text::truncate_utf8(model, 22))
            } else {
                model.to_string()
            };
            println!(
                "  {:<25} {:>3} sessions   {:>10} tokens",
                display_model,
                sessions,
                format_tokens(*tokens)
            );
        }
        println!();
    }

    if let Some((top_provider, _)) = providers.first() {
        if let Some((top_model, _)) = models.first() {
            if let Some(pricing) = get_pricing(top_provider, top_model) {
                let input_tokens = stats.total_tokens / 2;
                let output_tokens = stats.total_tokens / 2;
                let estimated_cost = pricing.calculate(input_tokens, output_tokens);
                println!(
                    "Estimated Cost:  ${:.2} (based on {} pricing)",
                    estimated_cost, top_model
                );
                println!("  Note: This is an estimate assuming 50/50 input/output token split.");
            }
        }
    }

    println!();
}
