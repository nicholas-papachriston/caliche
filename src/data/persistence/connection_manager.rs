use anyhow::{Context, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
#[cfg(feature = "mysql")]
use sqlx::MySqlPool;
use sqlx::{PgPool, SqlitePool};
use std::sync::Arc;
use std::time::Duration;

/// Database type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseType {
    PostgreSQL,
    MySQL,
    SQLite,
}

impl DatabaseType {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PostgreSQL => "postgresql",
            Self::MySQL => "mysql",
            Self::SQLite => "sqlite",
        }
    }
}

/// Unified database pool type
#[derive(Debug)]
pub enum DatabasePool {
    Postgres(PgPool),
    Sqlite(SqlitePool),
    #[cfg(feature = "mysql")]
    MySql(MySqlPool),
}

impl DatabasePool {
    /// Test the connection by executing a simple query
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn test_connection(&self) -> Result<()> {
        match self {
            Self::Postgres(pool) => {
                sqlx::query("SELECT 1")
                    .execute(pool)
                    .await
                    .context("PostgreSQL connection test failed")?;
            }
            Self::Sqlite(pool) => {
                sqlx::query("SELECT 1")
                    .execute(pool)
                    .await
                    .context("SQLite connection test failed")?;
            }
            #[cfg(feature = "mysql")]
            Self::MySql(pool) => {
                sqlx::query("SELECT 1")
                    .execute(pool)
                    .await
                    .context("MySQL connection test failed")?;
            }
        }
        Ok(())
    }
}

/// Database connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub database_type: DatabaseType,
    pub connection_string: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_secs: u64,
    pub idle_timeout_secs: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            database_type: DatabaseType::PostgreSQL,
            connection_string: String::new(),
            max_connections: 10,
            min_connections: 2,
            connection_timeout_secs: 30,
            idle_timeout_secs: 600,
        }
    }
}

/// A named database connection
#[derive(Debug)]
pub struct DatabaseConnection {
    pub name: String,
    pub database_type: DatabaseType,
    pub pool: DatabasePool,
    pub config: DatabaseConfig,
}

impl DatabaseConnection {
    /// Create a new database connection from config
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn new(name: impl Into<String>, config: DatabaseConfig) -> Result<Self> {
        let pool = match config.database_type {
            DatabaseType::PostgreSQL => {
                let pool = sqlx::postgres::PgPoolOptions::new()
                    .max_connections(config.max_connections)
                    .min_connections(config.min_connections)
                    .acquire_timeout(Duration::from_secs(config.connection_timeout_secs))
                    .idle_timeout(Duration::from_secs(config.idle_timeout_secs))
                    .connect(&config.connection_string)
                    .await
                    .context("Failed to create PostgreSQL connection pool")?;
                DatabasePool::Postgres(pool)
            }
            DatabaseType::SQLite => {
                let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
                    .max_connections(config.max_connections)
                    .acquire_timeout(Duration::from_secs(config.connection_timeout_secs))
                    .connect(&config.connection_string)
                    .await
                    .context("Failed to create SQLite connection pool")?;
                DatabasePool::Sqlite(pool)
            }
            DatabaseType::MySQL => {
                #[cfg(feature = "mysql")]
                {
                    let pool = sqlx::mysql::MySqlPoolOptions::new()
                        .max_connections(config.max_connections)
                        .min_connections(config.min_connections)
                        .acquire_timeout(Duration::from_secs(config.connection_timeout_secs))
                        .idle_timeout(Duration::from_secs(config.idle_timeout_secs))
                        .connect(&config.connection_string)
                        .await
                        .context("Failed to create MySQL connection pool")?;
                    DatabasePool::MySql(pool)
                }
                #[cfg(not(feature = "mysql"))]
                {
                    return Err(anyhow::anyhow!(
                        "MySQL support requires enabling 'mysql' feature in sqlx"
                    ));
                }
            }
        };

        let connection = Self {
            name: name.into(),
            database_type: config.database_type,
            pool,
            config,
        };

        // Test the connection
        connection.pool.test_connection().await?;

        Ok(connection)
    }

    /// Get a reference to the `PostgreSQL` pool if this is a `PostgreSQL` connection
    #[must_use]
    pub const fn as_postgres(&self) -> Option<&PgPool> {
        match &self.pool {
            DatabasePool::Postgres(pool) => Some(pool),
            DatabasePool::Sqlite(_) => None,
            #[cfg(feature = "mysql")]
            DatabasePool::MySql(_) => None,
        }
    }

    /// Get a reference to the `SQLite` pool if this is a `SQLite` connection
    #[must_use]
    pub const fn as_sqlite(&self) -> Option<&SqlitePool> {
        match &self.pool {
            DatabasePool::Sqlite(pool) => Some(pool),
            DatabasePool::Postgres(_) => None,
            #[cfg(feature = "mysql")]
            DatabasePool::MySql(_) => None,
        }
    }

    /// Get a reference to the MySQL pool if this is a MySQL connection
    #[cfg(feature = "mysql")]
    pub fn as_mysql(&self) -> Option<&MySqlPool> {
        match &self.pool {
            DatabasePool::MySql(pool) => Some(pool),
            _ => None,
        }
    }
}

/// Manager for multiple named database connections
#[derive(Debug, Clone)]
pub struct DatabaseConnectionManager {
    connections: Arc<DashMap<String, Arc<DatabaseConnection>>>,
}

impl DatabaseConnectionManager {
    /// Create a new connection manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
        }
    }

    /// Register a new database connection
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn register_connection(
        &self,
        name: impl Into<String>,
        config: DatabaseConfig,
    ) -> Result<()> {
        let name = name.into();
        let connection = DatabaseConnection::new(name.clone(), config).await?;
        self.connections.insert(name, Arc::new(connection));
        Ok(())
    }

    /// Get a connection by name
    #[must_use]
    pub fn get_connection(&self, name: &str) -> Option<Arc<DatabaseConnection>> {
        self.connections.get(name).map(|entry| Arc::clone(&entry))
    }

    /// Remove a connection
    #[must_use]
    pub fn remove_connection(&self, name: &str) -> Option<Arc<DatabaseConnection>> {
        self.connections.remove(name).map(|(_, conn)| conn)
    }

    /// List all connection names
    #[must_use]
    pub fn list_connections(&self) -> Vec<String> {
        self.connections
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Test a connection
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn test_connection(&self, name: &str) -> Result<()> {
        let connection = self
            .get_connection(name)
            .with_context(|| format!("Connection '{name}' not found"))?;
        connection.pool.test_connection().await
    }
}

impl Default for DatabaseConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}
