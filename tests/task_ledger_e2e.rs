//! End to end: the real bash and subagent paths write into an installed
//! task ledger.
//!
//! The ledger is process-global and its writer runs on the runtime that
//! installed it, so every scenario runs inside one test.

#![cfg(all(feature = "sqlite-store", feature = "testing"))]

use caliche::agent::tools::{BashTool, Tool};
use caliche::harness::config::PermissionModeConfig;
use caliche::harness::run_recorded_child;
use caliche::llm::ProviderType;
use caliche::persistence::task_ledger::{self, TASKS_DB_FILE};
use caliche::testing::MockProvider;
use caliche::{Agent, AgentHarness, AgentService, HarnessConfig, ToolRegistry};
use serde_json::json;
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, sqlx::FromRow)]
struct Row {
    id: String,
    kind: String,
    state: String,
    terminal_reason: Option<String>,
    parent_id: Option<String>,
}

#[expect(
    clippy::panic,
    reason = "test helper: a failure fails the calling test"
)]
async fn task_named(pool: &SqlitePool, name: &str) -> Row {
    sqlx::query_as("SELECT id, kind, state, terminal_reason, parent_id FROM tasks WHERE name = ?")
        .bind(name)
        .fetch_one(pool)
        .await
        .unwrap_or_else(|error| panic!("no task named {name:?}: {error}"))
}

async fn bash(args: serde_json::Value) {
    let _ = BashTool.execute(args).await;
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn mock_child_harness() -> AgentHarness {
    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(BashTool));
    let provider = Arc::new(MockProvider::with_tool_call(
        "bash",
        json!({ "command": "echo e2e-child" }),
    ));
    let service = AgentService::new_with_agent(Agent::new(provider, tools), ProviderType::OpenAI)
        .expect("mock agent service");
    AgentHarness::from_service(
        service,
        HarnessConfig {
            max_turns: 1,
            auto_checkpoint: false,
            permission_mode: PermissionModeConfig::AcceptAll,
            ..HarnessConfig::default()
        },
    )
    .expect("mock harness")
}

#[tokio::test]
async fn real_shell_and_subagent_paths_record_into_the_ledger() {
    let dir = tempfile::tempdir().unwrap();
    task_ledger::install(dir.path()).await.unwrap();
    let pool = SqlitePool::connect_with(
        SqliteConnectOptions::new().filename(dir.path().join(TASKS_DB_FILE)),
    )
    .await
    .unwrap();

    bash(json!({ "command": "echo e2e-ok" })).await;
    bash(json!({ "command": "exit 3" })).await;
    bash(json!({ "command": "sleep 30 # e2e-timeout", "timeout_secs": 1 })).await;
    // Dropping the tool future mid-run is how a cancelled turn ends a shell.
    let cancelled = tokio::time::timeout(
        Duration::from_millis(500),
        bash(json!({ "command": "sleep 30 # e2e-cancel" })),
    )
    .await;
    assert!(cancelled.is_err(), "the cancel scenario must be cut short");

    let mut harness = mock_child_harness();
    let session_id = harness.session_id().to_owned();
    run_recorded_child(&mut harness, false, "run the child command")
        .await
        .expect("child run");

    task_ledger::shutdown().await.unwrap();

    let ok = task_named(&pool, "echo e2e-ok").await;
    assert_eq!(
        (ok.kind.as_str(), ok.state.as_str()),
        ("shell", "succeeded")
    );

    let failed = task_named(&pool, "exit 3").await;
    assert_eq!(failed.state, "failed");
    assert_eq!(failed.terminal_reason.as_deref(), Some("exit code 3"));

    let timed_out = task_named(&pool, "sleep 30 # e2e-timeout").await;
    assert_eq!(timed_out.state, "failed");
    assert_eq!(timed_out.terminal_reason.as_deref(), Some("timeout"));

    let cancel = task_named(&pool, "sleep 30 # e2e-cancel").await;
    assert_eq!(cancel.state, "cancelled");
    assert_eq!(cancel.terminal_reason.as_deref(), Some("dropped"));

    let child = task_named(&pool, &session_id).await;
    assert_eq!(
        (child.kind.as_str(), child.state.as_str()),
        ("subagent", "succeeded")
    );
    let child_shell = task_named(&pool, "echo e2e-child").await;
    assert_eq!(child_shell.state, "succeeded");
    assert_eq!(child_shell.parent_id, Some(child.id));
    assert_eq!(ok.parent_id, None);
}
