use crate::agent::tools::common::params;
use crate::agent::tools::sql_tenant_filter::TenantFilterConfig;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::data::persistence::safety::SqlSafetyChecker;
use crate::utils::user_context::UserContext;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_util::TryStreamExt;
use serde_json::{Map, Value};
#[cfg(feature = "mysql")]
use sqlx::mysql::{MySqlColumn, MySqlPool, MySqlPoolOptions, MySqlRow, MySqlValueRef};
use sqlx::postgres::{PgColumn, PgPoolOptions, PgRow, PgValueRef};
use sqlx::sqlite::{SqliteColumn, SqlitePool, SqliteRow, SqliteValueRef};
use sqlx::{Column, PgPool, Row, TypeInfo, ValueRef};
use std::sync::Arc;
use std::time::Instant;

fn sql_safety_check_enabled() -> bool {
    std::env::var("SQL_SAFETY_CHECK")
        .ok()
        .map_or(true, |value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
}

/// Shared SQL guardrails for agent tools and plugin `db_query` steps.
#[derive(Debug)]
pub struct SqlExecutionPolicy {
    default_max_rows: usize,
    tenant_filter_config: TenantFilterConfig,
    safety_checker: Option<Arc<SqlSafetyChecker>>,
}

impl SqlExecutionPolicy {
    #[must_use]
    pub fn new(default_max_rows: usize) -> Self {
        Self {
            default_max_rows: default_max_rows.max(1),
            tenant_filter_config: TenantFilterConfig::from_env(),
            safety_checker: None,
        }
    }

    #[must_use]
    pub fn from_env(default_max_rows: usize) -> Self {
        let mut policy = Self::new(default_max_rows);
        if sql_safety_check_enabled() {
            policy.safety_checker = Some(Arc::new(SqlSafetyChecker::new()));
        }
        policy
    }

    #[must_use]
    pub fn with_tenant_filter(mut self, config: TenantFilterConfig) -> Self {
        self.tenant_filter_config = config;
        self
    }

    #[must_use]
    pub fn with_safety_checker(mut self, checker: Arc<SqlSafetyChecker>) -> Self {
        self.safety_checker = Some(checker);
        self
    }

    #[must_use]
    pub const fn default_max_rows(&self) -> usize {
        self.default_max_rows
    }

    pub async fn prepare_and_execute(
        &self,
        executor: &dyn SqlExecutor,
        sql: &str,
        max_rows: usize,
        dry_run: bool,
        context: Option<&UserContext>,
    ) -> Result<SqlExecutionResult> {
        self.tenant_filter_config.check(context)?;

        let sql = sql.trim();
        if sql.is_empty() {
            bail!("SQL statement cannot be empty");
        }
        if sql.contains(';') {
            bail!(
                "Multiple statements are not supported. Please submit one SQL statement at a time."
            );
        }

        if let Some(ref checker) = self.safety_checker {
            let safety_result = checker.check_safety(sql);
            if !safety_result.is_safe {
                let risks_summary: Vec<String> = safety_result
                    .risks
                    .iter()
                    .map(|r| format!("{:?}: {}", r.risk_type, r.description))
                    .collect();
                bail!(
                    "SQL safety check failed (severity: {:?}). Risks: {}",
                    safety_result.severity,
                    risks_summary.join("; ")
                );
            }
        }

        executor
            .execute_sql(sql, max_rows.max(1), dry_run)
            .await
            .context("SQL execution failed")
    }
}

#[derive(Debug)]
pub struct SqlExecutionTool {
    executor: Arc<dyn SqlExecutor>,
    policy: SqlExecutionPolicy,
}

impl SqlExecutionTool {
    pub fn new(executor: Arc<dyn SqlExecutor>) -> Self {
        Self {
            executor,
            policy: SqlExecutionPolicy::from_env(200),
        }
    }

    #[must_use]
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.policy.default_max_rows = max_rows.max(1);
        self
    }

    #[must_use]
    pub fn with_tenant_filter(mut self, config: TenantFilterConfig) -> Self {
        self.policy.tenant_filter_config = config;
        self
    }

    #[must_use]
    pub fn with_safety_checker(mut self, checker: Arc<SqlSafetyChecker>) -> Self {
        self.policy.safety_checker = Some(checker);
        self
    }

    #[must_use]
    pub fn with_sql_execution_policy(mut self, policy: SqlExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }
}

const NAME: &str = "sql_execute";
const DESCRIPTION: &str = "Execute SQL statements against a configured database. \
Supports read queries (SELECT/EXPLAIN) and write operations, returning results as JSON.";

impl ToolMeta for SqlExecutionTool {
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
        serde_json::json!({
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SQL statement to execute. Multiple statements separated by semicolons are not allowed."
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum number of rows to return for read queries",
                    "minimum": 1,
                    "maximum": 1000
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "When true, validates the query plan (EXPLAIN) without executing.",
                    "default": false
                }
            },
            "required": ["sql"]
        })
    }
}

#[async_trait]
impl Tool for SqlExecutionTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        self.execute_with_context(arguments, None).await
    }

    async fn execute_with_context(
        &self,
        arguments: serde_json::Value,
        context: Option<&UserContext>,
    ) -> ToolResult {
        let sql = match params::extract_string(&arguments, "sql", None) {
            Ok(sql) => sql,
            Err(e) => return ToolResult::error(e),
        };

        let max_rows =
            params::extract_number_or(&arguments, "max_rows", self.policy.default_max_rows as u64)
                as usize;
        let dry_run = params::extract_bool(&arguments, "dry_run", false);

        match self
            .policy
            .prepare_and_execute(self.executor.as_ref(), &sql, max_rows, dry_run, context)
            .await
        {
            Ok(result) => {
                let payload = serde_json::json!({
                    "sql": sql,
                    "dry_run": dry_run,
                    "rows": result.rows,
                    "rows_returned": result.rows.len(),
                    "rows_affected": result.rows_affected,
                    "execution_time_ms": result.execution_time_ms,
                    "truncated": result.truncated,
                });

                ToolResult::success(payload.to_string())
            }
            Err(err) => ToolResult::error(err.to_string()),
        }
    }
}

#[async_trait]
pub trait SqlExecutor: Send + Sync + std::fmt::Debug {
    async fn execute_sql(
        &self,
        sql: &str,
        max_rows: usize,
        dry_run: bool,
    ) -> Result<SqlExecutionResult>;
}

pub struct PostgresSqlExecutor {
    pool: PgPool,
}

impl std::fmt::Debug for PostgresSqlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresSqlExecutor")
            .field("pool", &"PgPool")
            .finish()
    }
}

impl PostgresSqlExecutor {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL")?;
        Ok(Self { pool })
    }

    #[must_use]
    pub const fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl SqlExecutor for PostgresSqlExecutor {
    async fn execute_sql(
        &self,
        sql: &str,
        max_rows: usize,
        dry_run: bool,
    ) -> Result<SqlExecutionResult> {
        let trimmed = sql.trim();
        let statement_type = trimmed
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let is_read_query = matches!(
            statement_type.as_str(),
            "select" | "with" | "show" | "describe" | "explain"
        );

        let start_time = Instant::now();

        if dry_run && !statement_type.eq("explain") {
            let explain_sql = format!("EXPLAIN {trimmed}");
            let explain_result = query_to_json(&self.pool, &explain_sql, max_rows).await?;

            return Ok(SqlExecutionResult {
                rows: explain_result.rows,
                rows_affected: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: explain_result.truncated,
            });
        }

        if is_read_query {
            let query_result = query_to_json(&self.pool, trimmed, max_rows).await?;
            Ok(SqlExecutionResult {
                rows: query_result.rows,
                rows_affected: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: query_result.truncated,
            })
        } else {
            let rows_affected = sqlx::query(trimmed)
                .execute(&self.pool)
                .await
                .context("Write query failed")?
                .rows_affected();

            Ok(SqlExecutionResult {
                rows: Vec::new(),
                rows_affected: Some(rows_affected),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: false,
            })
        }
    }
}

/// `SQLite` executor implementation
pub struct SqliteSqlExecutor {
    pool: SqlitePool,
}

impl std::fmt::Debug for SqliteSqlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteSqlExecutor")
            .field("pool", &"SqlitePool")
            .finish()
    }
}

impl SqliteSqlExecutor {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
            .connect(database_url)
            .await
            .context("Failed to connect to SQLite")?;
        Ok(Self { pool })
    }

    #[must_use]
    pub const fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl SqlExecutor for SqliteSqlExecutor {
    async fn execute_sql(
        &self,
        sql: &str,
        max_rows: usize,
        dry_run: bool,
    ) -> Result<SqlExecutionResult> {
        let trimmed = sql.trim();
        let statement_type = trimmed
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let is_read_query = matches!(statement_type.as_str(), "select" | "with" | "explain");

        let start_time = Instant::now();

        if dry_run && !statement_type.eq("explain") {
            let explain_sql = format!("EXPLAIN QUERY PLAN {trimmed}");
            let explain_result = query_sqlite_to_json(&self.pool, &explain_sql, max_rows).await?;

            return Ok(SqlExecutionResult {
                rows: explain_result.rows,
                rows_affected: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: explain_result.truncated,
            });
        }

        if is_read_query {
            let query_result = query_sqlite_to_json(&self.pool, trimmed, max_rows).await?;
            Ok(SqlExecutionResult {
                rows: query_result.rows,
                rows_affected: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: query_result.truncated,
            })
        } else {
            let rows_affected = sqlx::query(trimmed)
                .execute(&self.pool)
                .await
                .context("Write query failed")?
                .rows_affected();

            Ok(SqlExecutionResult {
                rows: Vec::new(),
                rows_affected: Some(rows_affected),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: false,
            })
        }
    }
}

#[cfg(feature = "mysql")]
/// MySQL executor implementation
pub struct MySqlSqlExecutor {
    pool: MySqlPool,
}

#[cfg(feature = "mysql")]
impl std::fmt::Debug for MySqlSqlExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MySqlSqlExecutor")
            .field("pool", &"MySqlPool")
            .finish()
    }
}

#[cfg(feature = "mysql")]
impl MySqlSqlExecutor {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = MySqlPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .context("Failed to connect to MySQL")?;
        Ok(Self { pool })
    }

    pub fn from_pool(pool: MySqlPool) -> Self {
        Self { pool }
    }
}

#[cfg(feature = "mysql")]
#[async_trait]
impl SqlExecutor for MySqlSqlExecutor {
    async fn execute_sql(
        &self,
        sql: &str,
        max_rows: usize,
        dry_run: bool,
    ) -> Result<SqlExecutionResult> {
        let trimmed = sql.trim();
        let statement_type = trimmed
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let is_read_query = matches!(
            statement_type.as_str(),
            "select" | "with" | "show" | "describe" | "explain"
        );

        let start_time = Instant::now();

        if dry_run && !statement_type.eq("explain") {
            let explain_sql = format!("EXPLAIN {}", trimmed);
            let explain_result = query_mysql_to_json(&self.pool, &explain_sql, max_rows).await?;

            return Ok(SqlExecutionResult {
                rows: explain_result.rows,
                rows_affected: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: explain_result.truncated,
            });
        }

        if is_read_query {
            let query_result = query_mysql_to_json(&self.pool, trimmed, max_rows).await?;
            Ok(SqlExecutionResult {
                rows: query_result.rows,
                rows_affected: None,
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: query_result.truncated,
            })
        } else {
            let rows_affected = sqlx::query(trimmed)
                .execute(&self.pool)
                .await
                .context("Write query failed")?
                .rows_affected();

            Ok(SqlExecutionResult {
                rows: Vec::new(),
                rows_affected: Some(rows_affected),
                execution_time_ms: start_time.elapsed().as_millis() as u64,
                truncated: false,
            })
        }
    }
}

pub struct SqlExecutionResult {
    pub rows: Vec<Value>,
    pub rows_affected: Option<u64>,
    pub execution_time_ms: u64,
    pub truncated: bool,
}

struct QueryResult {
    rows: Vec<Value>,
    truncated: bool,
}

async fn query_to_json(pool: &PgPool, sql: &str, max_rows: usize) -> Result<QueryResult> {
    let mut stream = sqlx::query(sql).fetch(pool);
    let mut rows = Vec::new();
    let mut truncated = false;

    while let Some(row) = stream.try_next().await.context("Query execution failed")? {
        let json_row = row_to_json(&row)?;
        rows.push(json_row);

        if rows.len() >= max_rows {
            truncated = stream.try_next().await?.is_some();
            break;
        }
    }

    Ok(QueryResult { rows, truncated })
}

fn row_to_json(row: &PgRow) -> Result<Value> {
    let mut object = Map::new();
    for (index, column) in row.columns().iter().enumerate() {
        let name = column.name();
        let value = decode_value(row, column, index)?;
        object.insert(name.to_string(), value);
    }
    Ok(Value::Object(object))
}

fn decode_value(row: &PgRow, column: &PgColumn, index: usize) -> Result<Value> {
    let type_name = column.type_info().name().to_ascii_lowercase();
    let value_ref: PgValueRef<'_> = row.try_get_raw(index)?;

    if value_ref.is_null() {
        return Ok(Value::Null);
    }

    if let Some(value) = decode_number(row, index, type_name.as_str())? {
        return Ok(value);
    }
    if let Some(value) = decode_time_value(row, index, type_name.as_str())? {
        return Ok(value);
    }
    decode_text_value(row, index, &type_name)
}

fn decode_number(row: &PgRow, index: usize, type_name: &str) -> Result<Option<Value>> {
    let value = match type_name {
        "bool" => Value::Bool(row.try_get(index)?),
        "int2" => Value::from(i64::from(row.try_get::<i16, _>(index)?)),
        "int4" => Value::from(i64::from(row.try_get::<i32, _>(index)?)),
        "int8" => Value::from(row.try_get::<i64, _>(index)?),
        "float4" => Value::from(f64::from(row.try_get::<f32, _>(index)?)),
        "float8" => Value::from(row.try_get::<f64, _>(index)?),
        "numeric" => match row.try_get::<String, _>(index) {
            Ok(text) => Value::String(text),
            Err(_) => Value::from(row.try_get::<f64, _>(index)?),
        },
        _ => return Ok(None),
    };
    Ok(Some(value))
}

fn decode_time_value(row: &PgRow, index: usize, type_name: &str) -> Result<Option<Value>> {
    let value = match type_name {
        "timestamp" => Value::String(row.try_get::<chrono::NaiveDateTime, _>(index)?.to_string()),
        "timestamptz" => {
            let ts: chrono::DateTime<chrono::Utc> = row.try_get(index)?;
            Value::String(ts.to_rfc3339())
        }
        "date" => {
            let date: chrono::NaiveDate = row.try_get(index)?;
            Value::String(date.to_string())
        }
        "time" => {
            let time: chrono::NaiveTime = row.try_get(index)?;
            Value::String(time.to_string())
        }
        _ => return Ok(None),
    };
    Ok(Some(value))
}

fn decode_text_value(row: &PgRow, index: usize, type_name: &str) -> Result<Value> {
    let value = match type_name {
        "text" | "varchar" | "bpchar" | "citext" => Value::String(row.try_get(index)?),
        "uuid" => Value::String(row.try_get::<uuid::Uuid, _>(index)?.to_string()),
        "json" | "jsonb" => row.try_get(index)?,
        "bytea" => {
            let bytes: Vec<u8> = row.try_get(index)?;
            Value::String(BASE64.encode(bytes))
        }
        _ => Value::String(format!("<unsupported type {type_name}>")),
    };
    Ok(value)
}

async fn query_sqlite_to_json(
    pool: &SqlitePool,
    sql: &str,
    max_rows: usize,
) -> Result<QueryResult> {
    let mut stream = sqlx::query(sql).fetch(pool);
    let mut rows = Vec::new();
    let mut truncated = false;

    while let Some(row) = stream.try_next().await.context("Query execution failed")? {
        let json_row = row_sqlite_to_json(&row)?;
        rows.push(json_row);

        if rows.len() >= max_rows {
            truncated = stream.try_next().await?.is_some();
            break;
        }
    }

    Ok(QueryResult { rows, truncated })
}

fn row_sqlite_to_json(row: &SqliteRow) -> Result<Value> {
    let mut object = Map::new();
    for (index, column) in row.columns().iter().enumerate() {
        let name = column.name();
        let value = decode_sqlite_value(row, column, index)?;
        object.insert(name.to_string(), value);
    }
    Ok(Value::Object(object))
}

fn decode_sqlite_value(row: &SqliteRow, column: &SqliteColumn, index: usize) -> Result<Value> {
    let type_name = column.type_info().name().to_ascii_lowercase();
    let value_ref: SqliteValueRef<'_> = row.try_get_raw(index)?;

    if value_ref.is_null() {
        return Ok(Value::Null);
    }

    let value = match type_name.as_str() {
        "integer" => Value::from(row.try_get::<i64, _>(index)?),
        "real" => Value::from(row.try_get::<f64, _>(index)?),
        "text" | "blob" => Value::String(row.try_get(index)?),
        "numeric" => match row.try_get::<String, _>(index) {
            Ok(text) => Value::String(text),
            Err(_) => Value::from(row.try_get::<f64, _>(index)?),
        },
        _ => Value::String(
            row.try_get::<String, _>(index)
                .unwrap_or_else(|_| "<unknown>".to_string()),
        ),
    };

    Ok(value)
}

#[cfg(feature = "mysql")]
async fn query_mysql_to_json(pool: &MySqlPool, sql: &str, max_rows: usize) -> Result<QueryResult> {
    let mut stream = sqlx::query(sql).fetch(pool);
    let mut rows = Vec::new();
    let mut truncated = false;

    while let Some(row) = stream.try_next().await.context("Query execution failed")? {
        let json_row = row_mysql_to_json(&row)?;
        rows.push(json_row);

        if rows.len() >= max_rows {
            truncated = stream.try_next().await?.is_some();
            break;
        }
    }

    Ok(QueryResult { rows, truncated })
}

#[cfg(feature = "mysql")]
fn row_mysql_to_json(row: &MySqlRow) -> Result<Value> {
    let mut object = Map::new();
    for (index, column) in row.columns().iter().enumerate() {
        let name = column.name();
        let value = decode_mysql_value(row, column, index)?;
        object.insert(name.to_string(), value);
    }
    Ok(Value::Object(object))
}

#[cfg(feature = "mysql")]
fn decode_mysql_value(row: &MySqlRow, column: &MySqlColumn, index: usize) -> Result<Value> {
    let type_name = column.type_info().name().to_ascii_lowercase();
    let value_ref: MySqlValueRef<'_> = row.try_get_raw(index)?;

    if value_ref.is_null() {
        return Ok(Value::Null);
    }

    let value = match type_name.as_str() {
        "tinyint" | "smallint" | "mediumint" | "int" => {
            Value::from(row.try_get::<i32, _>(index)? as i64)
        }
        "bigint" => Value::from(row.try_get::<i64, _>(index)?),
        "float" | "double" => Value::from(row.try_get::<f64, _>(index)?),
        "decimal" | "numeric" => match row.try_get::<String, _>(index) {
            Ok(text) => Value::String(text),
            Err(_) => Value::from(row.try_get::<f64, _>(index)?),
        },
        "varchar" | "char" | "text" | "longtext" | "mediumtext" | "tinytext" => {
            Value::String(row.try_get(index)?)
        }
        "date" | "datetime" | "timestamp" | "time" => Value::String(
            row.try_get::<String, _>(index)
                .unwrap_or_else(|_| "<unknown>".to_string()),
        ),
        "json" => {
            let json_value: serde_json::Value = row.try_get(index)?;
            json_value
        }
        _ => Value::String(
            row.try_get::<String, _>(index)
                .unwrap_or_else(|_| "<unknown>".to_string()),
        ),
    };

    Ok(value)
}
