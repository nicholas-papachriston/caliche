//! Command history storage in SQLite

use anyhow::{Context, Result};
use chrono::Utc;
use sqlx::Row;

use crate::storage::Database;

/// Store for command history operations
#[derive(Clone, Debug)]
pub struct HistoryStore {
    db: Database,
}

impl HistoryStore {
    /// Create a new history store
    pub fn new(db: Database) -> Self {
        Self { db }
    }

    /// Add a command to history
    pub async fn add(&self, session_id: Option<&str>, command: &str) -> Result<()> {
        let timestamp = Utc::now().to_rfc3339();

        sqlx::query(
            "INSERT INTO command_history (session_id, command, timestamp) VALUES (?1, ?2, ?3)",
        )
        .bind(session_id)
        .bind(command)
        .bind(&timestamp)
        .execute(self.db.pool())
        .await
        .context("Failed to add command to history")?;

        Ok(())
    }

    /// Get recent commands (session-specific or global)
    pub async fn get_recent(&self, session_id: Option<&str>, limit: usize) -> Result<Vec<String>> {
        let rows = if let Some(sid) = session_id {
            sqlx::query(
                "SELECT command FROM command_history 
                 WHERE session_id = ?1 
                 ORDER BY timestamp DESC 
                 LIMIT ?2",
            )
            .bind(sid)
            .bind(limit as i64)
            .fetch_all(self.db.pool())
            .await?
        } else {
            sqlx::query(
                "SELECT command FROM command_history 
                 WHERE session_id IS NULL 
                 ORDER BY timestamp DESC 
                 LIMIT ?1",
            )
            .bind(limit as i64)
            .fetch_all(self.db.pool())
            .await?
        };

        let commands: Vec<String> = rows
            .into_iter()
            .map(|row| row.try_get::<String, _>(0))
            .collect::<Result<_, _>>()?;

        Ok(commands)
    }

    /// Search commands by query string
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let search_pattern = format!("%{}%", query);
        let rows = sqlx::query(
            "SELECT DISTINCT command FROM command_history 
             WHERE command LIKE ?1 
             ORDER BY timestamp DESC 
             LIMIT ?2",
        )
        .bind(&search_pattern)
        .bind(limit as i64)
        .fetch_all(self.db.pool())
        .await?;

        let commands: Vec<String> = rows
            .into_iter()
            .map(|row| row.try_get::<String, _>(0))
            .collect::<Result<_, _>>()?;

        Ok(commands)
    }

    /// Clear history (session-specific or global)
    pub async fn clear(&self, session_id: Option<&str>) -> Result<()> {
        if let Some(sid) = session_id {
            sqlx::query("DELETE FROM command_history WHERE session_id = ?1")
                .bind(sid)
                .execute(self.db.pool())
                .await?;
        } else {
            sqlx::query("DELETE FROM command_history WHERE session_id IS NULL")
                .execute(self.db.pool())
                .await?;
        }

        Ok(())
    }

    /// Get all commands for a session (for loading into REPL history)
    pub async fn get_all(&self, session_id: Option<&str>) -> Result<Vec<String>> {
        let rows = if let Some(sid) = session_id {
            sqlx::query(
                "SELECT command FROM command_history 
                 WHERE session_id = ?1 
                 ORDER BY timestamp ASC",
            )
            .bind(sid)
            .fetch_all(self.db.pool())
            .await?
        } else {
            sqlx::query(
                "SELECT command FROM command_history 
                 WHERE session_id IS NULL 
                 ORDER BY timestamp ASC",
            )
            .fetch_all(self.db.pool())
            .await?
        };

        let commands: Vec<String> = rows
            .into_iter()
            .map(|row| row.try_get::<String, _>(0))
            .collect::<Result<_, _>>()?;

        Ok(commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOB_COMMAND: &str = "error occurred while decoding column 0: mismatched types; Rust type `alloc::string::String` (as SQL type `TEXT`) is not compatible with SQL type `BLOB`: mismatched types; Rust type `alloc::string::String` (as SQL type `TEXT`) is not compatible with SQL type `BLOB`";

    #[tokio::test]
    async fn a_mistyped_command_fails_every_read_naming_the_column() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open_at(&dir.path().join("caliche.db"))
            .await
            .expect("open db");
        sqlx::query(
            "INSERT INTO command_history (session_id, command, timestamp)
             VALUES (NULL, X'00', '2026-11-11T00:00:00Z')",
        )
        .execute(db.pool())
        .await
        .expect("plant row");
        let store = HistoryStore::new(db);

        let recent = store.get_recent(None, 10).await.expect_err("get_recent");
        let found = store.search("", 10).await.expect_err("search");
        let all = store.get_all(None).await.expect_err("get_all");

        assert_eq!(
            [recent, found, all].map(|e| format!("{e:#}")),
            [BLOB_COMMAND; 3]
        );
    }
}
