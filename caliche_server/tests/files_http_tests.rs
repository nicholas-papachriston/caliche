//! `/api/files/read` through the router: the file browsing switch, the
//! status and JSON mapping, and the blocking read task.

mod harness;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{Value, json};
use serial_test::serial;
use tower::ServiceExt;

const READ_NOTES: &str = "/api/files/read?path=notes.txt";

#[expect(clippy::expect_used, reason = "test helper panics with context")]
async fn get(path: &str) -> (StatusCode, Value) {
    let app = harness::mock_router().await;
    let response = app
        .oneshot(Request::get(path).body(Body::empty()).expect("request"))
        .await
        .expect("response");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, serde_json::from_slice(&bytes).expect("json body"))
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn workspace() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical workspace");
    (dir, root)
}

#[tokio::test]
#[serial]
async fn read_returns_the_file_as_json() {
    let (_dir, root) = workspace();
    std::fs::write(root.join("notes.txt"), "one\ntwo\n").expect("notes");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and runs on a current-thread runtime; the
    // blocking read task reads no environment variable.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_FILE_BROWSING", "true");
    env.set("CALICHE_WORKSPACE", &root);

    let (status, body) = get(READ_NOTES).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body,
        json!({
            "path": root.join("notes.txt").to_string_lossy(),
            "content": "one\ntwo\n",
            "size": 8,
            "truncated": false,
            "line_count": 2
        })
    );
}

#[tokio::test]
#[serial]
async fn read_is_403_while_file_browsing_is_off() {
    let (_dir, root) = workspace();
    std::fs::write(root.join("notes.txt"), "one\n").expect("notes");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and runs on a current-thread runtime; the
    // blocking read task reads no environment variable.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.remove("CALICHE_FILE_BROWSING");
    env.set("CALICHE_WORKSPACE", &root);

    let (status, body) = get(READ_NOTES).await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(
        body,
        json!({
            "error": "File browsing is not enabled. Set CALICHE_FILE_BROWSING=true to enable."
        })
    );
}

#[tokio::test]
#[serial]
async fn read_of_a_directory_is_400_with_the_error_as_json() {
    let (_dir, root) = workspace();
    std::fs::create_dir(root.join("notes.txt")).expect("directory");
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and runs on a current-thread runtime; the
    // blocking read task reads no environment variable.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_FILE_BROWSING", "true");
    env.set("CALICHE_WORKSPACE", &root);

    let (status, body) = get(READ_NOTES).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(
        body,
        json!({ "error": caliche_server::handlers::files::NOT_A_FILE })
    );
}
