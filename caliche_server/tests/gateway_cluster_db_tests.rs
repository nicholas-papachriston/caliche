//! The gateway cluster listings read rows written by other processes. A NULL
//! in a column the handler expects is a 500 that names the column, not a
//! panic.
//!
//! The cluster db lives under `$HOME`, and the handler opens it once per
//! process, so this binary holds a single test that plants both tables.
#![cfg(unix)]

mod harness;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serial_test::serial;
use sqlx::sqlite::SqliteConnectOptions;
use std::str::FromStr;
use tower::ServiceExt;

const NULL_BASE_URL: &str = "gateway cluster db: column \"base_url\" is NULL";
const NULL_TARGET: &str = "gateway cluster db: column \"target_agent_id\" is NULL";

#[expect(clippy::expect_used, reason = "test helper panics with context")]
async fn get(app: axum::Router, path: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(Request::get(path).body(Body::empty()).expect("request"))
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (
        status,
        String::from_utf8(bytes.to_vec()).expect("utf-8 body"),
    )
}

/// Create the cluster db as an older or foreign writer could have left it:
/// the same tables without NOT NULL, holding one row with a NULL column each.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
async fn plant_null_rows(home: &std::path::Path) {
    let dir = home.join(".caliche");
    std::fs::create_dir_all(&dir).expect("cluster dir");
    let dsn = format!("sqlite://{}", dir.join("gateway-cluster.db").display());
    let options = SqliteConnectOptions::from_str(&dsn)
        .expect("dsn")
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(options)
        .await
        .expect("open cluster db");
    for statement in [
        "CREATE TABLE gateway_cluster_nodes (agent_id TEXT PRIMARY KEY, cwd TEXT, base_url TEXT, is_main INTEGER, updated_at TEXT)",
        "CREATE TABLE gateway_cluster_routes (route_key TEXT PRIMARY KEY, target_agent_id TEXT, updated_at TEXT)",
        "INSERT INTO gateway_cluster_nodes VALUES ('worker-1', '/w', NULL, 0, '2026-11-11 00:00:00')",
        "INSERT INTO gateway_cluster_routes VALUES ('chat-1', NULL, '2026-11-11 00:00:00')",
    ] {
        sqlx::query(statement)
            .execute(&pool)
            .await
            .expect("plant row");
    }
    pool.close().await;
}

#[tokio::test]
#[serial]
async fn a_null_column_in_the_cluster_db_is_a_500_naming_it() {
    let home = tempfile::tempdir().expect("home");
    plant_null_rows(home.path()).await;
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and is the only test in this binary; the
    // router reads HOME on this thread while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("HOME", home.path());
    let app = harness::mock_router().await;

    let nodes = get(app.clone(), "/api/gateway/cluster/nodes").await;
    let routes = get(app, "/api/gateway/cluster/routes").await;

    assert_eq!(
        nodes,
        (StatusCode::INTERNAL_SERVER_ERROR, NULL_BASE_URL.to_string())
    );
    assert_eq!(
        routes,
        (StatusCode::INTERNAL_SERVER_ERROR, NULL_TARGET.to_string())
    );
}
