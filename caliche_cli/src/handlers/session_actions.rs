//! Named helpers for session subcommands that exit after completion.

use super::print_aggregate_stats;
use crate::commands::SessionAction;
use crate::storage::SessionStore;
use anyhow::Result;
use caliche::agent::session::load_session_snapshot_from_path;
use std::io::Write;
use std::path::Path;

fn short_id(id: &str) -> &str {
    caliche_contracts::text::truncate_utf8(id, 8)
}

fn cwd_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| ".".to_string())
}

fn message_preview(content: &str, max_chars: usize) -> String {
    if content.len() > max_chars {
        format!(
            "{}...",
            caliche_contracts::text::truncate_utf8(content, max_chars)
        )
    } else {
        content.to_string()
    }
}

pub(super) fn inspect_session_file(path: &Path, verbose: bool) -> Result<()> {
    let snap = load_session_snapshot_from_path(path)?;
    let m = &snap.metadata;
    println!("\nSession snapshot file: {}", path.display());
    println!("session_id:              {}", snap.session_id);
    println!("created_at:              {}", m.created_at);
    println!("saved_at:                {}", m.saved_at);
    println!("memory_type:             {}", m.memory_type);
    println!("total_messages (meta):   {}", m.total_messages);
    println!("total_tokens_estimated: {}", m.total_tokens_estimated);
    println!(
        "conversation_messages:   {}",
        snap.conversation_history.len()
    );
    println!(
        "memory_snapshot:         {}",
        snap.memory_snapshot.is_some()
    );
    println!(
        "knowledge_snapshot:      {}",
        snap.knowledge_snapshot.is_some()
    );
    if verbose {
        print_inspect_verbose(&snap);
    }
    crate::shutdown::exit(0);
}

fn print_inspect_verbose(snap: &caliche::agent::session::SessionSnapshot) {
    if let Some(ref state) = snap.shared_state {
        if let Some(obj) = state.as_object() {
            println!(
                "shared_state keys:      {:?}",
                obj.keys().collect::<Vec<_>>()
            );
        } else {
            println!("shared_state:            {state}");
        }
    } else {
        println!("shared_state:            (none)");
    }
    let preview_n = 5.min(snap.conversation_history.len());
    println!("\nFirst {preview_n} messages (role / preview):");
    for msg in snap.conversation_history.iter().take(preview_n) {
        let role = format!("{:?}", msg.role);
        let content = msg.content.extract_text();
        let preview = message_preview(&content, 200);
        println!("  [{}] {}", role, preview.replace('\n', " "));
    }
}

pub(super) async fn list_sessions(session_store: &SessionStore, verbose: bool) -> Result<()> {
    let summaries = session_store.list_detailed_async().await?;
    if summaries.is_empty() {
        println!("No saved sessions found.");
        println!("\nSessions are stored in SQLite database: ~/.caliche/caliche.db");
        crate::shutdown::exit(0);
    }
    println!("\nSaved Sessions\n");
    if verbose {
        print_verbose_session_table(&summaries);
    } else {
        print_compact_session_list(&summaries);
    }
    println!("\n{} session(s) found", summaries.len());
    crate::shutdown::exit(0);
}

fn print_verbose_session_table(summaries: &[crate::session::SessionSummary]) {
    println!(
        "{:<10} {:<20} {:<20} {:>8} {:>12} {:<20}",
        "ID", "Name", "CWD", "Messages", "Tokens", "Updated"
    );
    println!("{}", "-".repeat(96));
    for summary in summaries {
        let name = summary.name.as_deref().unwrap_or("-");
        println!(
            "{:<10} {:<20} {:<20} {:>8} {:>12} {:<20}",
            short_id(&summary.id),
            name,
            cwd_name(&summary.working_directory),
            summary.message_count,
            summary.total_tokens,
            summary.updated_at.format("%Y-%m-%d %H:%M")
        );
    }
}

fn print_compact_session_list(summaries: &[crate::session::SessionSummary]) {
    for summary in summaries {
        let name = summary.name.as_deref().unwrap_or("-");
        println!(
            "  {} - {} ({} messages, cwd={})",
            short_id(&summary.id),
            name,
            summary.message_count,
            cwd_name(&summary.working_directory)
        );
    }
}

pub(super) async fn delete_session(
    session_store: &SessionStore,
    session_id: String,
    force: bool,
) -> Result<()> {
    if !session_store.exists_async(&session_id).await? {
        anyhow::bail!("Session '{session_id}' not found");
    }
    if !force {
        print!("Delete session '{session_id}'? [y/N] ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            crate::shutdown::exit(0);
        }
    }
    session_store.delete_async(&session_id).await?;
    println!("Session '{session_id}' deleted.");
    crate::shutdown::exit(0);
}

pub(super) async fn rename_session(
    session_store: &SessionStore,
    session_id: String,
    new_name: String,
) -> Result<()> {
    if !session_store.exists_async(&session_id).await? {
        anyhow::bail!("Session '{session_id}' not found");
    }
    session_store.rename_async(&session_id, &new_name).await?;
    println!("Session '{session_id}' renamed to '{new_name}'");
    crate::shutdown::exit(0);
}

pub(super) async fn show_session(session_store: &SessionStore, session_id: String) -> Result<()> {
    let session = session_store.load_async(&session_id).await?;
    let summary = session.summary();
    println!("\nSession Details\n");
    println!("ID:              {}", summary.id);
    if let Some(ref name) = summary.name {
        println!("Name:            {name}");
    }
    println!("Working Dir:     {}", summary.working_directory.display());
    println!("Messages:        {}", summary.message_count);
    println!("Total Tokens:    {}", summary.total_tokens);
    if let Some(ref provider) = summary.provider {
        println!("Provider:        {provider}");
    }
    if let Some(ref model) = summary.model {
        println!("Model:           {model}");
    }
    println!("Created:         {}", summary.created_at);
    println!("Updated:         {}", summary.updated_at);
    if !session.context_files.is_empty() {
        println!("\nContext Files:");
        for file in &session.context_files {
            println!("  - {}", file.display());
        }
    }
    println!(
        "\nConversation Preview (last {} messages):",
        3.min(session.conversation_history.len())
    );
    for msg in session.conversation_history.iter().rev().take(3).rev() {
        let role = format!("{:?}", msg.role);
        let content = msg.content.extract_text();
        let preview = message_preview(&content, 100);
        println!("  [{}] {}", role, preview.replace('\n', " "));
    }
    crate::shutdown::exit(0);
}

pub(super) async fn export_session(
    session_store: &SessionStore,
    session_id: String,
    output: Option<String>,
    format: String,
    include_thinking: bool,
) -> Result<()> {
    let session = session_store.load_async(&session_id).await?;
    let output_path = output.unwrap_or_else(|| format!("{session_id}.{format}"));
    let content = session.export_text(&format, include_thinking)?;
    tokio::fs::write(&output_path, content).await?;
    println!("Session exported to: {output_path}");
    crate::shutdown::exit(0);
}

pub(super) async fn print_session_stats(
    session_store: &SessionStore,
    provider: Option<String>,
    model: Option<String>,
    days: Option<u32>,
) -> Result<()> {
    let stats = session_store
        .aggregate_stats_async(provider.as_deref(), model.as_deref(), days)
        .await?;
    print_aggregate_stats(&stats);
    crate::shutdown::exit(0);
}

pub(super) async fn dispatch_session_action(
    session_store: SessionStore,
    action: SessionAction,
) -> Result<()> {
    match action {
        SessionAction::Resume { .. } => Err(anyhow::anyhow!(
            "internal routing error: SessionAction::Resume must be handled by SessionHandler before reaching handle_session_action"
        )),
        SessionAction::List { verbose } => list_sessions(&session_store, verbose).await,
        SessionAction::Delete { session_id, force } => {
            delete_session(&session_store, session_id, force).await
        }
        SessionAction::Rename {
            session_id,
            new_name,
        } => rename_session(&session_store, session_id, new_name).await,
        SessionAction::Show { session_id } => show_session(&session_store, session_id).await,
        SessionAction::Export {
            session_id,
            output,
            format,
            include_thinking,
        } => export_session(&session_store, session_id, output, format, include_thinking).await,
        SessionAction::Stats {
            provider,
            model,
            days,
        } => print_session_stats(&session_store, provider, model, days).await,
        SessionAction::InspectFile { .. } => Err(anyhow::anyhow!(
            "internal routing error: SessionAction::InspectFile must be handled before SessionStore::new in handle_session_action"
        )),
    }
}
