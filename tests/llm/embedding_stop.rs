//! Workspace search stops for the session after the first embedding error
//! a retry cannot fix, instead of sending one request per file.
//!
//! A stub on 127.0.0.1 plays LM Studio's embeddings endpoint. It answers
//! the first `ok_replies` requests with a vector and every later one with
//! the scripted error, and counts the requests.

use caliche::AgentService;
use caliche::llm::{EmbeddingProvider, EmbeddingStopped, LMStudioConfig};
use caliche::workspace::{PersistenceConfig, WorkspaceIndexer, WorkspaceIndexerConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const MODEL_NOT_FOUND: &str = r#"{"error":{"message":"Model not found: text-embedding-qwen3-embedding-4b","type":"invalid_request_error"}}"#;
const VECTOR: &str = r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3,0.4]}],"model":"text-embedding-qwen3-embedding-4b","usage":{"prompt_tokens":1,"total_tokens":1}}"#;

/// Files in the test workspace: before the fix each one cost a request.
const FILES: usize = 24;

struct Stub {
    base_url: String,
    embedding_requests: Arc<AtomicUsize>,
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_embeddings(ok_replies: usize, status: &'static str) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let embedding_requests = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&embedding_requests);
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            let count = Arc::clone(&count);
            tokio::spawn(async move {
                let request = read_request(&mut socket).await;
                let (status, body) = if request.starts_with("POST /v1/embeddings") {
                    if count.fetch_add(1, Ordering::SeqCst) < ok_replies {
                        ("200 OK", VECTOR)
                    } else {
                        (status, MODEL_NOT_FOUND)
                    }
                } else {
                    ("404 Not Found", r#"{"error":"unexpected route"}"#)
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(reply.as_bytes()).await;
            });
        }
    });
    Stub {
        base_url: format!("http://{addr}/v1"),
        embedding_requests,
    }
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let n = socket.read(&mut chunk).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&buf);
        if let Some(end) = text.find("\r\n\r\n") {
            let length = text[..end]
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            if buf.len() >= end + 4 + length {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn lm_studio(base_url: &str) -> Arc<dyn EmbeddingProvider> {
    Arc::new(
        LMStudioConfig::builder()
            .base_url(base_url)
            .model("m")
            .build()
            .expect("provider"),
    )
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn workspace(
    persistence: impl FnOnce(&std::path::Path) -> PersistenceConfig,
) -> (tempfile::TempDir, WorkspaceIndexerConfig) {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("ws");
    std::fs::create_dir_all(&root).expect("ws");
    for i in 0..FILES {
        std::fs::write(root.join(format!("f{i}.rs")), format!("fn f{i}() {{}}\n")).expect("file");
    }
    let config = WorkspaceIndexerConfig::new(root).persistence(persistence(dir.path()));
    (dir, config)
}

/// What a host does at startup: build the workspace tool and, when that
/// works, index the workspace. Returns the build error, if any.
async fn start_workspace_search(
    config: &WorkspaceIndexerConfig,
    provider: Arc<dyn EmbeddingProvider>,
) -> Option<String> {
    match AgentService::build_workspace_rag_tool(config, provider).await {
        Ok(tool) => {
            let _ = WorkspaceIndexer::new(tool.service(), config.clone())
                .index()
                .await;
            None
        }
        Err(error) => {
            assert!(error.is::<EmbeddingStopped>(), "{error:#}");
            Some(error.to_string())
        }
    }
}

/// LM Studio answers 400 for a model that is not on disk: one request,
/// then workspace search stops with the server's reason.
#[tokio::test]
async fn a_400_model_not_found_sends_one_request_and_stops() {
    let stub = stub_embeddings(0, "400 Bad Request").await;
    let (_dir, config) = workspace(|dir| PersistenceConfig::Disk {
        path: dir.join("index"),
    });

    let error = start_workspace_search(&config, lm_studio(&stub.base_url)).await;

    assert_eq!(stub.embedding_requests.load(Ordering::SeqCst), 1);
    assert_eq!(
        error.as_deref(),
        Some(
            "lmstudio API error (400 Bad Request): Model not found: text-embedding-qwen3-embedding-4b"
        )
    );
}

/// The same with 404 and an in-memory index.
#[tokio::test]
async fn a_404_model_not_found_sends_one_request_and_stops() {
    let stub = stub_embeddings(0, "404 Not Found").await;
    let (_dir, config) = workspace(|_| PersistenceConfig::InMemory);

    let error = start_workspace_search(&config, lm_studio(&stub.base_url)).await;

    assert_eq!(stub.embedding_requests.load(Ordering::SeqCst), 1);
    assert_eq!(
        error.as_deref(),
        Some(
            "lmstudio API error (404 Not Found): Model not found: text-embedding-qwen3-embedding-4b"
        )
    );
}

/// A server error may clear on a retry: workspace search still starts.
#[tokio::test]
async fn a_503_at_startup_does_not_stop_workspace_search() {
    let stub = stub_embeddings(0, "503 Service Unavailable").await;
    let (_dir, config) = workspace(|_| PersistenceConfig::InMemory);

    let error = start_workspace_search(&config, lm_studio(&stub.base_url)).await;

    assert_eq!(error, None);
}

/// The model goes away after the probe: indexing stops with the reason,
/// and no request goes out per file. At most the requests already in
/// flight (the index concurrency, at most 4 by default) are sent.
#[tokio::test]
async fn a_model_that_goes_away_mid_index_stops_the_index() {
    let stub = stub_embeddings(1, "404 Not Found").await;
    let (_dir, config) = workspace(|_| PersistenceConfig::InMemory);

    let tool = AgentService::build_workspace_rag_tool(&config, lm_studio(&stub.base_url))
        .await
        .expect("probe succeeds");
    let result = WorkspaceIndexer::new(tool.service(), config.clone())
        .index()
        .await;
    let index_error = result.err().map(|error| format!("{error:#}"));

    let requests = stub.embedding_requests.load(Ordering::SeqCst);
    assert!((2..=5).contains(&requests), "{requests} embedding requests");
    assert_eq!(
        index_error.as_deref(),
        Some(
            "RAG index_batch failed: lmstudio API error (404 Not Found): Model not found: text-embedding-qwen3-embedding-4b"
        )
    );
}
