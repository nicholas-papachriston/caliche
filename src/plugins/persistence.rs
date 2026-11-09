//! SQLite-backed storage for plugin definitions.

use crate::plugins::schema::PluginDefinition;
use anyhow::{Context, Result};
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous},
};
use std::str::FromStr;
use std::time::Duration;

/// SQLite-backed store for plugin definitions.
#[derive(Debug, Clone)]
pub struct PluginStore {
    pool: SqlitePool,
}

impl PluginStore {
    /// Create a new store and initialize schema.
    pub async fn new(database_url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(database_url)
            .context("invalid database URL")?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal);

        let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
            .connect_with(options)
            .await
            .context("failed to connect to SQLite")?;

        let store = Self { pool };
        store.init().await?;
        Ok(store)
    }

    /// Create from a file path (e.g. `~/.caliche/plugins.db` or `.caliche/plugins.db`).
    pub async fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("failed to create database directory")?;
        }
        let url = format!("sqlite:{}", path.display());
        Self::new(&url).await
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS plugins (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                version TEXT NOT NULL,
                plugin_type TEXT NOT NULL,
                definition BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("failed to create plugins table")?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_plugins_type ON plugins(plugin_type)")
            .execute(&self.pool)
            .await
            .context("failed to create type index")?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_plugins_name ON plugins(name)")
            .execute(&self.pool)
            .await
            .context("failed to create name index")?;

        Ok(())
    }

    /// Insert or replace a plugin definition.
    pub async fn put(&self, def: &PluginDefinition) -> Result<()> {
        let definition_json = serde_json::to_vec(def).context("serialize plugin definition")?;
        let created_at = def.created_at.to_rfc3339();
        let updated_at = def.updated_at.to_rfc3339();
        let plugin_type = format!("{:?}", def.plugin_type).to_lowercase();

        sqlx::query(
            r#"
            INSERT INTO plugins (id, name, version, plugin_type, definition, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                version = excluded.version,
                plugin_type = excluded.plugin_type,
                definition = excluded.definition,
                updated_at = excluded.updated_at
            "#,
        )
        .bind(&def.id)
        .bind(&def.name)
        .bind(&def.version)
        .bind(&plugin_type)
        .bind(&definition_json)
        .bind(&created_at)
        .bind(&updated_at)
        .execute(&self.pool)
        .await
        .context("failed to put plugin")?;

        Ok(())
    }

    /// Get a plugin by name.
    pub async fn get_by_name(&self, name: &str) -> Result<Option<PluginDefinition>> {
        let row = sqlx::query("SELECT definition FROM plugins WHERE name = ?1")
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .context("failed to get plugin")?;

        let Some(row) = row else {
            return Ok(None);
        };
        let bytes: Vec<u8> = row.try_get("definition")?;
        let def: PluginDefinition =
            serde_json::from_slice(&bytes).context("deserialize plugin definition")?;
        Ok(Some(def))
    }

    /// Get a plugin by id.
    pub async fn get_by_id(&self, id: &str) -> Result<Option<PluginDefinition>> {
        let row = sqlx::query("SELECT definition FROM plugins WHERE id = ?1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .context("failed to get plugin by id")?;

        let Some(row) = row else {
            return Ok(None);
        };
        let bytes: Vec<u8> = row.try_get("definition")?;
        let def: PluginDefinition =
            serde_json::from_slice(&bytes).context("deserialize plugin definition")?;
        Ok(Some(def))
    }

    /// List all plugin names (and optionally types).
    pub async fn list(&self) -> Result<Vec<(String, String)>> {
        let rows = sqlx::query("SELECT name, plugin_type FROM plugins ORDER BY name")
            .fetch_all(&self.pool)
            .await
            .context("failed to list plugins")?;
        let out = rows
            .into_iter()
            .map(|r| {
                let name: String = r.try_get("name").unwrap_or_default();
                let plugin_type: String = r.try_get("plugin_type").unwrap_or_default();
                (name, plugin_type)
            })
            .collect();
        Ok(out)
    }

    /// Load all plugin definitions.
    pub async fn load_all(&self) -> Result<Vec<PluginDefinition>> {
        let rows = sqlx::query("SELECT definition FROM plugins ORDER BY name")
            .fetch_all(&self.pool)
            .await
            .context("failed to load plugins")?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let bytes: Vec<u8> = row.try_get("definition")?;
            let def: PluginDefinition =
                serde_json::from_slice(&bytes).context("deserialize plugin definition")?;
            out.push(def);
        }
        Ok(out)
    }

    /// Delete a plugin by name.
    pub async fn delete_by_name(&self, name: &str) -> Result<()> {
        sqlx::query("DELETE FROM plugins WHERE name = ?1")
            .bind(name)
            .execute(&self.pool)
            .await
            .context("failed to delete plugin")?;
        Ok(())
    }
}
