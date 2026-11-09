//! Ensure the CLI SQLite file has the current conversation/settings tables.

use anyhow::{Context, Result};
use caliche::{
    messages::Message,
    persistence::{
        PersistenceBackend, SqliteConversationStore, SqliteCredentialStore,
        SqlitePersistenceBackend,
    },
};
use chrono::{DateTime, Utc};
use serde_json::json;
use sqlx::Row;
use std::collections::HashSet;
use std::path::PathBuf;

/// Create current tables if missing, and copy leftover `sessions`/`settings` rows if present.
pub async fn ensure_schema(db_path: &PathBuf) -> Result<()> {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::str::FromStr;

    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create database directory")?;
    }

    let database_url = format!("sqlite:{}", db_path.display());
    let options = SqliteConnectOptions::from_str(&database_url)?.create_if_missing(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .context("Failed to connect to database")?;

    // Single query for all table existence checks (4 round-trips -> 1)
    let names: Vec<String> = sqlx::query_scalar(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'settings', 'conversations')",
    )
    .fetch_all(&pool)
    .await
    .unwrap_or_default();
    let existing_tables: HashSet<String> = names.into_iter().collect();

    let leftover_sessions = existing_tables.contains("sessions");
    let leftover_settings = existing_tables.contains("settings");
    let has_conversations = existing_tables.contains("conversations");

    if has_conversations && !leftover_sessions && !leftover_settings {
        return Ok(());
    }

    let conversation_store = SqliteConversationStore::from_path(db_path).await?;
    let _credential_store = SqliteCredentialStore::from_path(db_path).await?;
    let kv_store = SqlitePersistenceBackend::from_path(db_path).await?;

    if leftover_sessions {
        copy_sessions_table(&pool, &conversation_store).await?;
    }

    if leftover_settings {
        copy_settings_table(&pool, &kv_store).await?;
    }

    Ok(())
}

async fn copy_sessions_table(
    pool: &sqlx::SqlitePool,
    conversation_store: &SqliteConversationStore,
) -> Result<()> {
    let rows = sqlx::query(
        "SELECT id, name, working_directory, conversation, context_files, permissions,
                total_tokens, provider, model, is_autosave, created_at, updated_at
         FROM sessions WHERE is_autosave = 0",
    )
    .fetch_all(pool)
    .await?;

    for row in rows {
        let id: String = row.try_get(0)?;
        let name: Option<String> = row.try_get(1)?;
        let working_directory: String = row.try_get(2)?;
        let conversation_blob: Vec<u8> = row.try_get(3)?;
        let context_files_json: String = row.try_get(4)?;
        let permissions_json: String = row.try_get(5)?;
        let total_tokens: i64 = row.try_get(6)?;
        let provider: Option<String> = row.try_get(7)?;
        let model: Option<String> = row.try_get(8)?;
        let _is_autosave: i64 = row.try_get(9)?; // Skip is_autosave (already filtered in WHERE clause)
        let created_at_str: String = row.try_get(10)?;
        let updated_at_str: String = row.try_get(11)?;

        let conversation_history: Vec<Message> = rmp_serde::from_slice(&conversation_blob)
            .context("Failed to decode conversation history")?;

        let context_files: Vec<PathBuf> =
            serde_json::from_str(&context_files_json).unwrap_or_default();
        let permissions_granted: Vec<String> =
            serde_json::from_str(&permissions_json).unwrap_or_default();

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .context("Failed to parse created_at")?
            .with_timezone(&Utc);
        let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
            .context("Failed to parse updated_at")?
            .with_timezone(&Utc);

        let metadata = json!({
            "name": name,
            "working_directory": working_directory,
            "context_files": context_files.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
            "permissions": permissions_granted,
            "total_tokens": total_tokens,
            "provider": provider,
            "model": model,
            "is_autosave": false,
        });

        let record = caliche::persistence::ConversationRecord {
            id,
            agent_id: "cli".to_string(),
            messages: conversation_history,
            metadata,
            created_at,
            updated_at,
            expires_at: None,
        };

        conversation_store.save(&record).await?;
    }

    Ok(())
}

async fn copy_settings_table(
    pool: &sqlx::SqlitePool,
    kv_store: &SqlitePersistenceBackend,
) -> Result<()> {
    let rows = sqlx::query("SELECT key, value FROM settings")
        .fetch_all(pool)
        .await?;

    for row in rows {
        let key: String = row.try_get(0)?;
        let value: String = row.try_get(1)?;

        let kv_key = format!("setting:{}", key);
        kv_store.set(&kv_key, value.as_bytes(), None).await?;
    }

    Ok(())
}
