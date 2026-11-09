//! SQLite database connection and schema management using sqlx

use anyhow::{Context, Result};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::PathBuf;
use std::str::FromStr;

const CURRENT_SCHEMA_VERSION: i32 = 3;

/// SQLite database wrapper for CLI storage
#[derive(Clone, Debug)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Initialize or open the database at the default location
    pub async fn open() -> Result<Self> {
        let db_path = Self::default_db_path()?;
        Self::open_at(&db_path).await
    }

    /// Open or create database at specified path
    pub async fn open_at(path: &PathBuf) -> Result<Self> {
        // Ensure directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        // Ensure conversation/settings tables exist before opening our pool
        crate::storage::schema::ensure_schema(path).await?;

        // Set restrictive permissions on Unix (before opening)
        #[cfg(unix)]
        {
            if !path.exists()
                && let Ok(file) = std::fs::File::create(path)
            {
                drop(file);
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).ok(); // Best effort
            }
        }

        let database_url = format!("sqlite:{}", path.display());
        let options = SqliteConnectOptions::from_str(&database_url)?.create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1) // SQLite works best with single connection
            .connect_with(options)
            .await
            .context("Failed to open database")?;

        let db = Self { pool };

        db.init_schema().await?;

        Ok(db)
    }

    /// Get the default database path
    pub fn default_db_path() -> Result<PathBuf> {
        let home = crate::storage::cli_home_dir()?;
        Ok(home.join(".caliche").join("caliche.db"))
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Initialize database schema
    async fn init_schema(&self) -> Result<()> {
        // Schema version tracking
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )",
        )
        .execute(&self.pool)
        .await?;

        // Check current version
        let current_version: Option<i32> =
            sqlx::query_scalar("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
                .fetch_optional(&self.pool)
                .await?;

        let version = current_version.unwrap_or(0);

        if version < CURRENT_SCHEMA_VERSION {
            if version == 0 {
                self.create_tables().await?;
            }

            if version < 2 {
                self.ensure_command_history().await?;
            }

            if version < 3 {
                self.ensure_credential_scope_tables().await?;
            }

            // Update version
            sqlx::query("INSERT OR REPLACE INTO schema_version (version) VALUES (?1)")
                .bind(CURRENT_SCHEMA_VERSION)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    /// Create all tables
    async fn create_tables(&self) -> Result<()> {
        // Provider credentials metadata (actual secrets in keyring/encrypted file)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS credentials (
                provider TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                is_default INTEGER DEFAULT 0,
                default_model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // AWS credentials metadata (separate table for complex credentials)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS aws_credentials_meta (
                provider TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                access_key_id TEXT,
                region TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Sessions with full conversation history
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                working_directory TEXT NOT NULL,
                conversation BLOB NOT NULL,
                context_files TEXT,
                permissions TEXT,
                total_tokens INTEGER DEFAULT 0,
                provider TEXT,
                model TEXT,
                is_autosave INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Index for efficient session queries
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sessions_autosave ON sessions(is_autosave, updated_at DESC)",
        )
        .execute(&self.pool)
        .await?;

        // Custom commands
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS commands (
                name TEXT PRIMARY KEY,
                description TEXT,
                prompt TEXT NOT NULL,
                model TEXT,
                provider TEXT,
                agent TEXT,
                confirm INTEGER DEFAULT 0,
                args TEXT,
                source_path TEXT,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Settings/preferences
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Credential scope registry (also for fresh installs)
        self.ensure_credential_scope_tables().await?;

        Ok(())
    }

    /// Save a setting to the database
    pub async fn save_setting(&self, key: &str, value: &str) -> Result<()> {
        sqlx::query("INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)")
            .bind(key)
            .bind(value)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Load a setting from the database
    pub async fn load_setting(&self, key: &str) -> Result<Option<String>> {
        let value: Option<String> = sqlx::query_scalar("SELECT value FROM settings WHERE key = ?1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(value)
    }

    /// Create credential scope tables if they are missing.
    async fn ensure_credential_scope_tables(&self) -> Result<()> {
        // Registry of known credential scopes (directories with stored credentials)
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS credential_scopes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_path TEXT NOT NULL,
                scope_hash TEXT NOT NULL UNIQUE,
                label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(&self.pool)
        .await?;

        // Links between scopes (allows sharing credentials across directories)
        // provider = '*' means all providers
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS credential_scope_links (
                source_hash TEXT NOT NULL,
                target_hash TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT '*',
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_hash, target_hash, provider)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_scope_links_target ON credential_scope_links(target_hash)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Create the command_history table if it is missing.
    async fn ensure_command_history(&self) -> Result<()> {
        // Command history table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS command_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                command TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )",
        )
        .execute(&self.pool)
        .await?;

        // Indexes for efficient history queries
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_history_session ON command_history(session_id, timestamp DESC)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_history_global ON command_history(timestamp DESC) WHERE session_id IS NULL",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}
