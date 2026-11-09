use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "mysql")]
use crate::data::persistence::adapters::MySqlSchemaIntrospector;
use crate::data::persistence::adapters::SqliteSchemaIntrospector;
use crate::data::persistence::{
    connection_manager::{
        DatabaseConfig, DatabaseConnection, DatabaseConnectionManager, DatabaseType,
    },
    schema::{DatabaseIntrospector, PostgresSchemaIntrospector},
    schema_cache::SchemaCache,
};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

const NAME: &str = "database_management";
const DESCRIPTION: &str = "Manage database connections, introspect schemas, and manage schema cache. \
Supports registering connections, testing connections, introspecting database schemas, \
and refreshing schema cache.";

#[derive(Debug)]
pub struct DatabaseManagementTool {
    connection_manager: Arc<DatabaseConnectionManager>,
    schema_cache: Arc<SchemaCache>,
}

impl DatabaseManagementTool {
    #[must_use]
    pub const fn new(
        connection_manager: Arc<DatabaseConnectionManager>,
        schema_cache: Arc<SchemaCache>,
    ) -> Self {
        Self {
            connection_manager,
            schema_cache,
        }
    }

    async fn execute_register_connection(&self, arguments: &serde_json::Value) -> ToolResult {
        let connection_name = match params::extract_string(arguments, "connection_name", None) {
            Ok(name) => name,
            Err(e) => return ToolResult::error(e),
        };
        let connection_string = match params::extract_string(arguments, "connection_string", None) {
            Ok(conn) => conn,
            Err(e) => return ToolResult::error(e),
        };
        let db_type_str = params::extract_string_or(arguments, "database_type", "postgresql");
        let database_type = match parse_database_type(&db_type_str) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let config = DatabaseConfig {
            database_type,
            connection_string,
            max_connections: 10,
            min_connections: 2,
            connection_timeout_secs: 30,
            idle_timeout_secs: 600,
        };
        match self
            .connection_manager
            .register_connection(connection_name.clone(), config)
            .await
        {
            Ok(()) => ToolResult::success(
                json!({
                    "message": format!("Connection '{}' registered successfully", connection_name),
                    "connection_name": connection_name
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("Failed to register connection: {e}")),
        }
    }

    async fn execute_test_connection(&self, arguments: &serde_json::Value) -> ToolResult {
        let connection_name = match params::extract_string(arguments, "connection_name", None) {
            Ok(name) => name,
            Err(e) => return ToolResult::error(e),
        };
        match self
            .connection_manager
            .test_connection(&connection_name)
            .await
        {
            Ok(()) => ToolResult::success(
                json!({
                    "message": format!("Connection '{}' test successful", connection_name),
                    "connection_name": connection_name
                })
                .to_string(),
            ),
            Err(e) => ToolResult::error(format!("Connection test failed: {e}")),
        }
    }

    async fn execute_introspect_schema(&self, arguments: &serde_json::Value) -> ToolResult {
        let connection_name = match params::extract_string(arguments, "connection_name", None) {
            Ok(name) => name,
            Err(e) => return ToolResult::error(e),
        };
        let cache_ttl_secs = params::extract_number_or(arguments, "cache_ttl_secs", 300);
        let ttl = std::time::Duration::from_secs(cache_ttl_secs);
        if let Some(cached_schema) = self
            .schema_cache
            .get_with_ttl_check(&connection_name, Some(ttl))
            .await
        {
            return ToolResult::success(
                json!({
                    "connection_name": connection_name,
                    "schema": cached_schema,
                    "cached": true
                })
                .to_string(),
            );
        }
        let schema = match self.introspect_connection(&connection_name).await {
            Ok(s) => s,
            Err(e) => return e,
        };
        self.schema_cache
            .put_with_ttl(
                connection_name.clone(),
                schema.clone(),
                Some(cache_ttl_secs),
            )
            .await;
        ToolResult::success(
            json!({
                "connection_name": connection_name,
                "schema": schema,
                "cached": false
            })
            .to_string(),
        )
    }

    async fn execute_introspect_table(&self, arguments: &serde_json::Value) -> ToolResult {
        let connection_name = match params::extract_string(arguments, "connection_name", None) {
            Ok(name) => name,
            Err(e) => return ToolResult::error(e),
        };
        let table_name = match params::extract_string(arguments, "table_name", None) {
            Ok(name) => name,
            Err(e) => return ToolResult::error(e),
        };
        let schema_name = params::extract_string_or(arguments, "schema", "public");
        let db_schema = match self.introspect_connection(&connection_name).await {
            Ok(s) => s,
            Err(e) => return e,
        };
        let table = db_schema.tables.iter().find(|t| {
            t.name == table_name.to_lowercase() && t.schema == schema_name.to_lowercase()
        });
        match table {
            Some(t) => ToolResult::success(
                json!({
                    "connection_name": connection_name,
                    "table": t
                })
                .to_string(),
            ),
            None => ToolResult::error(format!("Table '{schema_name}.{table_name}' not found")),
        }
    }

    async fn introspect_connection(
        &self,
        connection_name: &str,
    ) -> Result<crate::data::persistence::schema::DatabaseSchema, ToolResult> {
        let connection = match self.connection_manager.get_connection(connection_name) {
            Some(conn) => conn,
            None => {
                return Err(ToolResult::error(format!(
                    "Connection '{connection_name}' not found"
                )));
            }
        };
        introspect_by_type(&connection, connection_name).await
    }

    async fn execute_refresh_schema_cache(&self, arguments: &serde_json::Value) -> ToolResult {
        let connection_name = match params::extract_string(arguments, "connection_name", None) {
            Ok(name) => name,
            Err(e) => return ToolResult::error(e),
        };
        self.schema_cache.invalidate(&connection_name).await;
        ToolResult::success(
            json!({
                "message": format!("Schema cache invalidated for '{}'", connection_name),
                "connection_name": connection_name
            })
            .to_string(),
        )
    }

    async fn execute_invalidate_schema_cache(&self) -> ToolResult {
        self.schema_cache.invalidate_all().await;
        ToolResult::success(
            json!({
                "message": "All schema caches invalidated"
            })
            .to_string(),
        )
    }
}

impl ToolMeta for DatabaseManagementTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "register_connection",
                        "test_connection",
                        "introspect_schema",
                        "introspect_table",
                        "list_connections",
                        "refresh_schema_cache",
                        "invalidate_schema_cache"
                    ],
                    "description": "The operation to perform"
                },
                "connection_name": {
                    "type": "string",
                    "description": "Name of the database connection (required for most operations)"
                },
                "connection_string": {
                    "type": "string",
                    "description": "Database connection string (required for register_connection)"
                },
                "database_type": {
                    "type": "string",
                    "enum": ["postgresql", "mysql", "sqlite"],
                    "description": "Type of database (required for register_connection)"
                },
                "table_name": {
                    "type": "string",
                    "description": "Name of table to introspect (required for introspect_table)"
                },
                "schema": {
                    "type": "string",
                    "description": "Schema name (optional, for introspect_table)"
                },
                "cache_ttl_secs": {
                    "type": "integer",
                    "description": "Cache TTL in seconds (optional, for introspect_schema)"
                }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for DatabaseManagementTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = match params::extract_string(&arguments, "operation", None) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };
        match operation.as_str() {
            "register_connection" => self.execute_register_connection(&arguments).await,
            "test_connection" => self.execute_test_connection(&arguments).await,
            "introspect_schema" => self.execute_introspect_schema(&arguments).await,
            "introspect_table" => self.execute_introspect_table(&arguments).await,
            "list_connections" => ToolResult::success(
                json!({
                    "connections": self.connection_manager.list_connections()
                })
                .to_string(),
            ),
            "refresh_schema_cache" => self.execute_refresh_schema_cache(&arguments).await,
            "invalidate_schema_cache" => self.execute_invalidate_schema_cache().await,
            _ => ToolResult::error(format!("Unknown operation: {operation}")),
        }
    }
}

fn parse_database_type(db_type_str: &str) -> Result<DatabaseType, ToolResult> {
    match db_type_str.to_lowercase().as_str() {
        "postgresql" => Ok(DatabaseType::PostgreSQL),
        "mysql" => Ok(DatabaseType::MySQL),
        "sqlite" => Ok(DatabaseType::SQLite),
        _ => Err(ToolResult::error(format!(
            "Unsupported database type: {db_type_str}"
        ))),
    }
}

async fn introspect_by_type(
    connection: &DatabaseConnection,
    connection_name: &str,
) -> Result<crate::data::persistence::schema::DatabaseSchema, ToolResult> {
    match connection.database_type {
        DatabaseType::PostgreSQL => {
            let pool = connection
                .as_postgres()
                .ok_or_else(|| ToolResult::error("Not a PostgreSQL connection"))?;
            let introspector =
                PostgresSchemaIntrospector::new(pool.clone(), connection_name.to_string());
            introspector
                .introspect()
                .await
                .map_err(|e| ToolResult::error(format!("Schema introspection failed: {e}")))
        }
        #[cfg(feature = "mysql")]
        DatabaseType::MySQL => {
            let pool = connection
                .as_mysql()
                .ok_or_else(|| ToolResult::error("Not a MySQL connection"))?;
            let introspector =
                MySqlSchemaIntrospector::new(pool.clone(), connection_name.to_string());
            introspector
                .introspect()
                .await
                .map_err(|e| ToolResult::error(format!("Schema introspection failed: {e}")))
        }
        #[cfg(not(feature = "mysql"))]
        DatabaseType::MySQL => Err(ToolResult::error("MySQL support requires 'mysql' feature")),
        DatabaseType::SQLite => {
            let pool = connection
                .as_sqlite()
                .ok_or_else(|| ToolResult::error("Not a SQLite connection"))?;
            let introspector =
                SqliteSchemaIntrospector::new(pool.clone(), connection_name.to_string());
            introspector
                .introspect()
                .await
                .map_err(|e| ToolResult::error(format!("Schema introspection failed: {e}")))
        }
    }
}
