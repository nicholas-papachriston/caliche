//! LM Studio without the model in memory: the provider asks LM Studio to
//! load it, waits, then sends the request once.
//!
//! A stub on 127.0.0.1 plays LM Studio with just-in-time loading off. A chat
//! or embedding request for a model that is not loaded gets LM Studio's
//! `No models loaded` reply. `POST /api/v1/models/load` starts a load, and
//! `GET /api/v0/models/<id>` then reports `loading` for a few reads and
//! `loaded` after them.

use caliche::llm::provider::Provider;
use caliche::llm::provider_utils::first_token::LocalStreamLimits;
use caliche::llm::{EmbeddingProvider, EmbeddingRequest, LMStudioConfig};
use caliche::messages::{CompletionRequest, Message, StreamEvent};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// LM Studio's reply when the model is not loaded and just-in-time loading
/// is off.
const NO_MODELS_LOADED: &str = r#"{"error":{"message":"No models loaded. Please load a model in the developer page or use the 'lms load' command.","type":"invalid_request_error","param":"model","code":null}}"#;

/// How the stub answers a load request.
#[derive(Clone, Copy)]
enum Load {
    /// Accept it; the model reads `loading` this many times, then `loaded`.
    Finishes(usize),
    /// Accept it; the model never finishes loading.
    Never,
    /// Refuse it with this status and body.
    Fails(&'static str, &'static str),
}

#[derive(Default)]
struct Seen {
    /// The `model` of each load request.
    loads: Vec<String>,
    /// The `model` of each chat request.
    chats: Vec<String>,
    /// The `model` of each embedding request.
    embeddings: Vec<String>,
    /// Reads of the model state left before the load finishes; `None`
    /// until a load request arrives.
    loading_reads: Option<usize>,
}

struct Stub {
    base_url: String,
    seen: Arc<Mutex<Seen>>,
}

impl Stub {
    fn seen<T>(&self, read: impl FnOnce(&Seen) -> T) -> T {
        read(&self.seen.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_lm_studio(load: Load) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let seen = Arc::new(Mutex::new(Seen::default()));
    let shared = Arc::clone(&seen);
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            tokio::spawn(answer(socket, Arc::clone(&shared), load));
        }
    });
    Stub {
        base_url: format!("http://{addr}/v1"),
        seen,
    }
}

async fn answer(mut socket: TcpStream, seen: Arc<Mutex<Seen>>, load: Load) {
    let (line, body) = read_request(&mut socket).await;
    let model = body["model"].as_str().unwrap_or_default().to_string();
    let (status, content_type, reply) = {
        let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        route(&mut seen, load, &line, &body, model)
    };
    let response = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
        reply.len()
    );
    let _ = socket.write_all(response.as_bytes()).await;
}

/// The state the stub reports for the model; a read while loading counts
/// down.
fn state(seen: &mut Seen, load: Load) -> &'static str {
    match (seen.loading_reads, load) {
        (None, _) | (Some(_), Load::Fails(..)) => "not-loaded",
        (Some(_), Load::Never) => "loading",
        (Some(0), Load::Finishes(_)) => "loaded",
        (Some(left), Load::Finishes(_)) => {
            seen.loading_reads = Some(left - 1);
            "loading"
        }
    }
}

fn loaded(seen: &Seen, load: Load) -> bool {
    matches!((seen.loading_reads, load), (Some(0), Load::Finishes(_)))
}

fn route(
    seen: &mut Seen,
    load: Load,
    line: &str,
    body: &serde_json::Value,
    model: String,
) -> (&'static str, &'static str, String) {
    const JSON: &str = "application/json";
    if line.starts_with("GET /api/v0/models/") {
        let id = line
            .trim_start_matches("GET /api/v0/models/")
            .split(' ')
            .next()
            .unwrap_or_default()
            .to_string();
        let state = state(seen, load);
        return (
            "200 OK",
            JSON,
            format!(r#"{{"id":"{id}","state":"{state}"}}"#),
        );
    }
    if line.starts_with("POST /api/v1/models/load ") {
        seen.loads.push(model.clone());
        return match load {
            Load::Fails(status, reply) => (status, JSON, reply.to_string()),
            Load::Finishes(reads) => {
                seen.loading_reads = Some(reads);
                ("200 OK", JSON, load_reply(&model))
            }
            Load::Never => {
                seen.loading_reads = Some(usize::MAX);
                ("200 OK", JSON, load_reply(&model))
            }
        };
    }
    if line.starts_with("POST /v1/chat/completions ") {
        seen.chats.push(model);
        if !loaded(seen, load) {
            return ("400 Bad Request", JSON, NO_MODELS_LOADED.to_string());
        }
        if body["stream"].as_bool() == Some(true) {
            let chunk = r#"{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}]}"#;
            return (
                "200 OK",
                "text/event-stream",
                format!("data: {chunk}\n\ndata: [DONE]\n\n"),
            );
        }
        let reply = r#"{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        return ("200 OK", JSON, reply.to_string());
    }
    if line.starts_with("POST /v1/embeddings ") {
        seen.embeddings.push(model);
        if !loaded(seen, load) {
            return ("400 Bad Request", JSON, NO_MODELS_LOADED.to_string());
        }
        let reply = r#"{"object":"list","data":[{"object":"embedding","embedding":[0.5,0.5],"index":0}],"model":"e","usage":{"prompt_tokens":1,"total_tokens":1}}"#;
        return ("200 OK", JSON, reply.to_string());
    }
    (
        "404 Not Found",
        JSON,
        r#"{"error":"unexpected route"}"#.to_string(),
    )
}

/// LM Studio's answer to an accepted load.
fn load_reply(model: &str) -> String {
    format!(r#"{{"type":"llm","instance_id":"{model}","load_time_seconds":1.5,"status":"loaded"}}"#)
}

/// The request line and JSON body of one request.
async fn read_request(socket: &mut TcpStream) -> (String, serde_json::Value) {
    let mut buf = Vec::new();
    let mut chunk = [0_u8; 4096];
    let mut head_end = None;
    loop {
        let n = socket.read(&mut chunk).await.unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let end = head_end.or_else(|| buf.windows(4).position(|w| w == b"\r\n\r\n"));
        head_end = end;
        if let Some(end) = end {
            let head = String::from_utf8_lossy(&buf[..end]);
            let length = head
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
    let end = head_end.unwrap_or(buf.len());
    let line = String::from_utf8_lossy(&buf[..end])
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    let body = buf
        .get(end + 4..)
        .and_then(|body| serde_json::from_slice(body).ok())
        .unwrap_or(serde_json::Value::Null);
    (line, body)
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn lm_studio(base_url: &str, limit: Duration) -> impl Provider + EmbeddingProvider + 'static {
    LMStudioConfig::builder()
        .base_url(base_url)
        .model("m")
        .stream_limits(LocalStreamLimits {
            first_token: Some(limit),
            stall: Some(Duration::from_secs(5)),
        })
        .build()
        .expect("provider")
}

fn hey() -> CompletionRequest {
    CompletionRequest::new(Arc::new(vec![Message::user("hey")]))
}

/// What a turn sees: status lines, text, and the error, in order.
async fn transcript(provider: &impl Provider) -> Vec<String> {
    let mut seen = Vec::new();
    let mut stream = provider.stream(hey());
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Status { message }) => seen.push(format!("status: {message}")),
            Ok(StreamEvent::TextChunk { content }) if !content.is_empty() => {
                seen.push(format!("text: {content}"));
            }
            Ok(_) => {}
            Err(error) => {
                seen.push(format!("error: {error:#}"));
                break;
            }
        }
    }
    seen
}

/// No model loaded: one load request for the selected model, one status
/// line, the model reads `loading` three times, then the turn is sent once
/// and the reply arrives.
#[tokio::test]
async fn a_turn_loads_the_model_waits_and_is_sent_once() {
    let stub = stub_lm_studio(Load::Finishes(3)).await;
    let provider = lm_studio(&stub.base_url, Duration::from_secs(10));

    assert_eq!(
        transcript(&provider).await,
        vec!["status: Loading m in LM Studio...", "text: Hello"]
    );
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["m"]);
    assert_eq!(stub.seen(|seen| seen.chats.clone()), vec!["m"]);
}

/// The same wait on the non-streamed path (`complete`).
#[tokio::test]
async fn a_completion_loads_the_model_first() {
    let stub = stub_lm_studio(Load::Finishes(2)).await;
    let provider = lm_studio(&stub.base_url, Duration::from_secs(10));

    let response = provider.complete(hey()).await.expect("completion");

    assert_eq!(response.content.as_text(), "Hello");
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["m"]);
    assert_eq!(stub.seen(|seen| seen.chats.clone()), vec!["m"]);
}

/// A load that never finishes fails the turn after the limit, with one line
/// that names the model. No second load request, and no chat request.
#[tokio::test]
async fn a_load_that_never_finishes_fails_after_the_limit() {
    let stub = stub_lm_studio(Load::Never).await;
    let provider = lm_studio(&stub.base_url, Duration::from_secs(2));

    assert_eq!(
        transcript(&provider).await,
        vec![
            "status: Loading m in LM Studio...",
            "error: LM Studio did not finish loading m within 2 seconds.",
        ]
    );
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["m"]);
    assert!(stub.seen(|seen| seen.chats.is_empty()));
}

/// LM Studio refuses the load: the turn fails with its reason, quoted once.
#[tokio::test]
async fn a_load_error_fails_the_turn_with_the_reason_quoted_once() {
    let stub = stub_lm_studio(Load::Fails(
        "500 Internal Server Error",
        r#"{"error":{"message":"Failed to load model: not enough memory."}}"#,
    ))
    .await;
    let provider = lm_studio(&stub.base_url, Duration::from_secs(10));

    let transcript = transcript(&provider).await;

    assert_eq!(
        transcript,
        vec![
            "status: Loading m in LM Studio...",
            "error: LM Studio could not load m: lmstudio API error (500 Internal Server Error): Failed to load model: not enough memory.",
        ]
    );
    assert_eq!(transcript.concat().matches("not enough memory").count(), 1);
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["m"]);
    assert!(stub.seen(|seen| seen.chats.is_empty()));
}

/// The embedding model is loaded the same way before the first embedding.
#[tokio::test]
async fn an_embedding_loads_the_embedding_model_first() {
    let stub = stub_lm_studio(Load::Finishes(2)).await;
    let provider = lm_studio(&stub.base_url, Duration::from_secs(10));

    let response = provider
        .embed(EmbeddingRequest {
            texts: vec!["hello".to_string()],
            model: Some("e".to_string()),
        })
        .await
        .expect("embedding");

    assert_eq!(response.embeddings, vec![vec![0.5, 0.5]]);
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["e"]);
    assert_eq!(stub.seen(|seen| seen.embeddings.clone()), vec!["e"]);
}

/// A refused embedding load is permanent: embeddings stop for the session
/// with that reason, and nothing more is sent.
#[tokio::test]
async fn a_refused_embedding_load_stops_embeddings_with_the_reason() {
    let stub = stub_lm_studio(Load::Fails(
        "500 Internal Server Error",
        r#"{"error":{"message":"Failed to load model: not enough memory."}}"#,
    ))
    .await;
    let provider = caliche::llm::StopOnPermanentError::new(Arc::new(lm_studio(
        &stub.base_url,
        Duration::from_secs(10),
    )));
    let request = || EmbeddingRequest {
        texts: vec!["hello".to_string()],
        model: Some("e".to_string()),
    };

    let first = provider.embed(request()).await.expect_err("load refused");
    let second = provider.embed(request()).await.expect_err("stopped");

    let reason = "LM Studio could not load e: lmstudio API error (500 Internal Server Error): Failed to load model: not enough memory.";
    assert_eq!(first.to_string(), reason);
    assert_eq!(provider.stopped_reason().as_deref(), Some(reason));
    assert_eq!(second.to_string(), reason);
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["e"]);
    assert!(stub.seen(|seen| seen.embeddings.is_empty()));
}

/// Two turns at once while the model loads: one load request, and both
/// turns are sent after it.
#[tokio::test]
async fn a_load_under_way_is_not_requested_again() {
    let stub = stub_lm_studio(Load::Finishes(6)).await;
    let provider = lm_studio(&stub.base_url, Duration::from_secs(10));

    let (first, second) = tokio::join!(transcript(&provider), transcript(&provider));

    assert_eq!(
        first,
        vec!["status: Loading m in LM Studio...", "text: Hello"]
    );
    assert_eq!(
        second,
        vec!["status: Loading m in LM Studio...", "text: Hello"]
    );
    assert_eq!(stub.seen(|seen| seen.loads.clone()), vec!["m"]);
    assert_eq!(stub.seen(|seen| seen.chats.clone()), vec!["m", "m"]);
}
