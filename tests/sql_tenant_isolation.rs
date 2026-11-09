//! With tenant filtering enabled, the SQL tool refuses every query before it
//! reaches the database, because it has no way to isolate tenants.
#![cfg(feature = "tools-sql")]

use anyhow::Result;
use async_trait::async_trait;
use caliche::UserContext;
use caliche::agent::tools::sql_execution::{SqlExecutionPolicy, SqlExecutionResult};
use caliche::agent::tools::sql_tenant_filter::IsolationStrategy;
use caliche::agent::tools::{SqlExecutionTool, SqlExecutor, TenantFilterConfig, Tool};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

const UNSUPPORTED: &str = "Tenant isolation needs a separate database connection for each \
tenant, which the SQL tools do not provide";

/// Records the SQL it receives and returns one row.
#[derive(Debug, Default)]
struct RecordingExecutor {
    received: Mutex<Vec<String>>,
}

impl RecordingExecutor {
    fn received(&self) -> Vec<String> {
        self.received
            .lock()
            .map(|sql| sql.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl SqlExecutor for RecordingExecutor {
    async fn execute_sql(
        &self,
        sql: &str,
        _max_rows: usize,
        _dry_run: bool,
    ) -> Result<SqlExecutionResult> {
        if let Ok(mut received) = self.received.lock() {
            received.push(sql.to_string());
        }
        Ok(SqlExecutionResult {
            rows: vec![json!({"id": 1})],
            rows_affected: None,
            execution_time_ms: 0,
            truncated: false,
        })
    }
}

fn tool(config: TenantFilterConfig) -> (Arc<RecordingExecutor>, SqlExecutionTool) {
    let executor = Arc::new(RecordingExecutor::default());
    let tool = SqlExecutionTool::new(Arc::clone(&executor) as Arc<dyn SqlExecutor>)
        .with_sql_execution_policy(SqlExecutionPolicy::new(200).with_tenant_filter(config));
    (executor, tool)
}

fn enabled(strategy: IsolationStrategy) -> TenantFilterConfig {
    TenantFilterConfig {
        enabled: true,
        strategy,
        ..TenantFilterConfig::default()
    }
}

/// Loads the config the way the SQL tools do, with filtering enabled.
#[expect(
    unsafe_code,
    clippy::disallowed_methods,
    reason = "the test sets the variables that the config loader reads"
)]
fn enabled_config_from_env(strategy: &str) -> TenantFilterConfig {
    // SAFETY: every test in this file is serial, so nothing reads the environment meanwhile.
    unsafe {
        std::env::set_var("SQL_TENANT_FILTERING_ENABLED", "true");
        std::env::set_var("SQL_TENANT_ISOLATION_STRATEGY", strategy);
    }
    let config = TenantFilterConfig::from_env();
    // SAFETY: as above.
    unsafe {
        std::env::remove_var("SQL_TENANT_FILTERING_ENABLED");
        std::env::remove_var("SQL_TENANT_ISOLATION_STRATEGY");
    }
    config
}

fn acme() -> UserContext {
    let mut context = UserContext::new("user-1");
    context.tenant_id = Some("acme".to_string());
    context
}

async fn run(tool: &SqlExecutionTool, sql: &str, context: Option<&UserContext>) -> Option<String> {
    tool.execute_with_context(json!({ "sql": sql }), context)
        .await
        .error
}

fn unsupported(strategy: &str) -> Option<String> {
    Some(format!(
        "SQL query refused: tenant isolation strategy '{strategy}' is not supported. {UNSUPPORTED}"
    ))
}

#[tokio::test]
#[serial_test::serial]
async fn row_level_refuses_queries_the_text_rewrite_let_through() {
    let (executor, tool) = tool(enabled(IsolationStrategy::RowLevel));
    let context = acme();
    for sql in [
        "SELECT * FROM t WHERE a = 1 OR 1=1",
        "SELECT * FROM t WHERE tenant_id = 'acme' OR 1=1",
        "SELECT * FROM t\tWHERE a = 1",
        "SELECT * FROM t\nWHERE a = 1",
        "SELECT * FROM (SELECT * FROM t) s",
        "SELECT a FROM t UNION SELECT a FROM u",
        "SELECT * FROM t",
    ] {
        assert_eq!(
            run(&tool, sql, Some(&context)).await,
            unsupported("row_level"),
            "{sql}"
        );
    }
    assert_eq!(executor.received(), Vec::<String>::new());
}

#[tokio::test]
#[serial_test::serial]
async fn schema_strategy_refuses() {
    let (executor, tool) = tool(enabled(IsolationStrategy::Schema));
    let context = acme();
    for sql in ["SELECT * FROM t", "SELECT * FROM other.t"] {
        assert_eq!(
            run(&tool, sql, Some(&context)).await,
            unsupported("schema"),
            "{sql}"
        );
    }
    assert_eq!(executor.received(), Vec::<String>::new());
}

/// No code selects a connection per tenant, so `database` refuses too.
#[tokio::test]
#[serial_test::serial]
async fn database_strategy_refuses() {
    let (executor, tool) = tool(enabled(IsolationStrategy::Database));
    assert_eq!(
        run(&tool, "SELECT * FROM t", Some(&acme())).await,
        unsupported("database")
    );
    assert_eq!(executor.received(), Vec::<String>::new());
}

#[tokio::test]
#[serial_test::serial]
async fn none_strategy_with_filtering_enabled_refuses() {
    let (executor, tool) = tool(enabled(IsolationStrategy::None));
    assert_eq!(
        run(&tool, "SELECT * FROM t", Some(&acme())).await,
        unsupported("none")
    );
    assert_eq!(executor.received(), Vec::<String>::new());
}

#[tokio::test]
#[serial_test::serial]
async fn enabled_without_tenant_id_refuses() {
    let (executor, tool) = tool(enabled(IsolationStrategy::RowLevel));
    let no_claim = UserContext::new("user-1");
    let expected = Some(
        "SQL query refused: tenant filtering is enabled and the request has no tenant id"
            .to_string(),
    );
    assert_eq!(run(&tool, "SELECT * FROM t", None).await, expected);
    assert_eq!(
        run(&tool, "SELECT * FROM t", Some(&no_claim)).await,
        expected
    );
    assert_eq!(executor.received(), Vec::<String>::new());
}

/// Plugin `db_query` steps call the policy with no user context.
#[tokio::test]
#[serial_test::serial]
async fn policy_without_context_refuses_when_enabled() {
    let executor = RecordingExecutor::default();
    let policy =
        SqlExecutionPolicy::new(10).with_tenant_filter(enabled(IsolationStrategy::Database));
    let error = policy
        .prepare_and_execute(&executor, "SELECT * FROM t", 10, false, None)
        .await
        .err()
        .map(|err| err.to_string());
    assert_eq!(
        error,
        Some(
            "SQL query refused: tenant filtering is enabled and the request has no tenant id"
                .to_string()
        )
    );
    assert_eq!(executor.received(), Vec::<String>::new());
}

#[tokio::test]
#[serial_test::serial]
async fn unknown_strategy_from_env_refuses() {
    let (executor, tool) = tool(enabled_config_from_env("rowlevel"));
    assert_eq!(
        run(&tool, "SELECT * FROM t", Some(&acme())).await,
        Some(
            "SQL query refused: SQL_TENANT_ISOLATION_STRATEGY value 'rowlevel' is not recognized"
                .to_string()
        )
    );
    assert_eq!(executor.received(), Vec::<String>::new());
}

#[tokio::test]
#[serial_test::serial]
async fn disabled_runs_sql_unchanged() {
    let (executor, tool) = tool(TenantFilterConfig::default());
    let sql = "SELECT * FROM t WHERE a = 1 OR 1=1";
    let result = tool
        .execute_with_context(json!({ "sql": sql }), Some(&acme()))
        .await;
    assert_eq!(result.error, None);
    let payload: Value = serde_json::from_str(&result.content).unwrap_or_default();
    assert_eq!(payload["sql"], json!(sql));
    assert_eq!(payload["rows"], json!([{"id": 1}]));
    assert_eq!(payload.get("tenant_filter_applied"), None);
    assert_eq!(executor.received(), vec![sql.to_string()]);
}
