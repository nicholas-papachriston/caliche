//! A local provider waits for its first token while the server loads the
//! model, with a status line; a stall after tokens flow still fails; cloud
//! providers keep their limits.
//!
//! A stub on 127.0.0.1 plays LM Studio: headers at once, then silence while
//! it "loads", then the reply. The limits are short so the tests run fast.

use caliche::AgentBuilder;
use caliche::llm::provider::Provider;
use caliche::llm::provider_utils::first_token::{LocalStreamLimits, ModelLoadProbe};
use caliche::llm::{
    LMStudioConfig, OllamaConfig, OpenAICompatConfig, OpenAIConfig, OpenRouterConfig,
    PerplexityConfig,
};
use caliche::messages::{CompletionRequest, Message, StreamEvent};
use futures_util::StreamExt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One piece of the scripted chat reply: wait, then send these SSE lines.
struct Step {
    after: Duration,
    lines: &'static [&'static str],
}

/// A stub LM Studio. `GET /api/v0/models/<id>` reports `state`; `POST
/// /v1/chat/completions` sends headers, then `steps`, then closes. The flag
/// is set when the client closes the chat connection before the end.
struct Stub {
    base_url: String,
    closed_early: Arc<AtomicBool>,
    chat_requests: Arc<AtomicUsize>,
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_lm_studio(state: &'static str, steps: Vec<Step>) -> Stub {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let closed_early = Arc::new(AtomicBool::new(false));
    let steps = Arc::new(steps);
    let chat_requests = Arc::new(AtomicUsize::new(0));
    let flag = Arc::clone(&closed_early);
    let requests = Arc::clone(&chat_requests);
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let steps = Arc::clone(&steps);
            let flag = Arc::clone(&flag);
            let requests = Arc::clone(&requests);
            tokio::spawn(async move {
                let request = read_request(&mut socket).await;
                if request.starts_with("GET ") {
                    let body = format!(r#"{{"id":"m","state":"{state}"}}"#);
                    let reply = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(reply.as_bytes()).await;
                    return;
                }
                requests.fetch_add(1, Ordering::SeqCst);
                let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n";
                if socket.write_all(head.as_bytes()).await.is_err() {
                    return;
                }
                for step in steps.iter() {
                    if wait_or_closed(&mut socket, step.after).await {
                        flag.store(true, Ordering::SeqCst);
                        return;
                    }
                    for line in step.lines {
                        let data = format!("{line}\n\n");
                        let chunk = format!("{:x}\r\n{data}\r\n", data.len());
                        if socket.write_all(chunk.as_bytes()).await.is_err() {
                            flag.store(true, Ordering::SeqCst);
                            return;
                        }
                    }
                }
                let _ = socket.write_all(b"0\r\n\r\n").await;
            });
        }
    });
    Stub {
        base_url: format!("http://{addr}/v1"),
        closed_early,
        chat_requests,
    }
}

/// A stub that answers every chat request with `status` and a JSON `body`.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_reply(status: &'static str, body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let request = read_request(&mut socket).await;
                let body = if request.starts_with("GET ") {
                    r#"{"id":"m","state":"loaded"}"#
                } else {
                    body
                };
                let status = if request.starts_with("GET ") {
                    "200 OK"
                } else {
                    status
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json; charset=utf-8\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(reply.as_bytes()).await;
            });
        }
    });
    format!("http://{addr}/v1")
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

/// Sleep `wait`; true when the client closed the connection meanwhile.
async fn wait_or_closed(socket: &mut tokio::net::TcpStream, wait: Duration) -> bool {
    let mut byte = [0_u8; 1];
    tokio::select! {
        () = tokio::time::sleep(wait) => false,
        read = socket.read(&mut byte) => matches!(read, Ok(0) | Err(_)),
    }
}

const ROLE: &str = r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#;
const HELLO: &str = r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;
const THERE: &str = r#"data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}]}"#;
const DONE: &str = "data: [DONE]";

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn lm_studio(base_url: &str, first_token: Duration, stall: Duration) -> impl Provider + 'static {
    LMStudioConfig::builder()
        .base_url(base_url)
        .model("m")
        .stream_limits(LocalStreamLimits {
            first_token: Some(first_token),
            stall: Some(stall),
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

/// The first byte comes after the stall limit (1s) but inside the
/// first-token limit: the turn succeeds and a status line shows while the
/// server reads the prompt. (A model that is not loaded is loaded first;
/// see `local_model_load`.)
#[tokio::test]
async fn a_cold_first_token_after_the_stall_limit_succeeds_with_a_status_line() {
    let stub = stub_lm_studio(
        "loaded",
        vec![Step {
            after: Duration::from_secs(3),
            lines: &[ROLE, HELLO, THERE, DONE],
        }],
    )
    .await;
    let provider = lm_studio(
        &stub.base_url,
        Duration::from_secs(10),
        Duration::from_secs(1),
    );

    assert_eq!(
        transcript(&provider).await,
        vec![
            "status: Waiting for the first token...",
            "text: Hello",
            "text:  there",
        ]
    );
}

/// No first token inside the first-token limit: the error says so.
#[tokio::test]
async fn no_first_token_inside_the_limit_says_the_model_may_still_be_loading() {
    let stub = stub_lm_studio(
        "loaded",
        vec![Step {
            after: Duration::from_secs(30),
            lines: &[ROLE, HELLO, DONE],
        }],
    )
    .await;
    let provider = lm_studio(
        &stub.base_url,
        Duration::from_secs(3),
        Duration::from_secs(1),
    );

    assert_eq!(
        transcript(&provider).await,
        vec![
            "status: Waiting for the first token...",
            "error: no first token after 3s (model still loading?)",
        ]
    );
}

/// Once tokens flow, a gap longer than the stall limit fails as a stall.
#[tokio::test]
async fn a_stall_after_tokens_flow_fails_with_the_stall_message() {
    let stub = stub_lm_studio(
        "loaded",
        vec![
            Step {
                after: Duration::ZERO,
                lines: &[ROLE, HELLO],
            },
            Step {
                after: Duration::from_secs(30),
                lines: &[THERE, DONE],
            },
        ],
    )
    .await;
    let provider = lm_studio(
        &stub.base_url,
        Duration::from_secs(10),
        Duration::from_secs(1),
    );

    assert_eq!(
        transcript(&provider).await,
        vec![
            "text: Hello",
            "error: SSE stream stalled: no data received for 1 seconds",
        ]
    );
}

/// Dropping the stream while the first token is awaited (Ctrl+C) closes
/// the connection, so the local server can stop.
#[tokio::test]
async fn dropping_the_stream_during_the_wait_closes_the_connection() {
    let stub = stub_lm_studio(
        "loaded",
        vec![Step {
            after: Duration::from_secs(30),
            lines: &[ROLE, HELLO, DONE],
        }],
    )
    .await;
    let provider = lm_studio(
        &stub.base_url,
        Duration::from_secs(60),
        Duration::from_secs(1),
    );
    let mut stream = provider.stream(hey());
    let first = stream
        .next()
        .await
        .map(|item| item.map_err(|e| e.to_string()));
    drop(stream);
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert!(matches!(
        first,
        Some(Ok(StreamEvent::Status { ref message })) if message == "Waiting for the first token..."
    ));
    assert!(stub.closed_early.load(Ordering::SeqCst));
}

/// The error a turn ends with, as the user sees it.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn turn_error(stub: &Stub, first_token: Duration) -> String {
    let provider = lm_studio(&stub.base_url, first_token, Duration::from_secs(1));
    let agent = AgentBuilder::new()
        .provider_instance(Arc::new(provider))
        .build()
        .await
        .expect("agent");
    let mut stream = agent.stream_chat("hey");
    while let Some(item) = stream.next().await {
        if let Err(error) = item {
            return format!("{error:#}");
        }
    }
    "no error".to_string()
}

/// A stall fails the turn with the cause in one line, and the prompt is
/// sent once.
#[tokio::test]
async fn a_turn_that_stalls_names_the_stall_and_does_not_resend() {
    let stub = stub_lm_studio(
        "loaded",
        vec![
            Step {
                after: Duration::ZERO,
                lines: &[ROLE, HELLO],
            },
            Step {
                after: Duration::from_secs(30),
                lines: &[THERE, DONE],
            },
        ],
    )
    .await;

    assert_eq!(
        turn_error(&stub, Duration::from_secs(10)).await,
        "Stream error: SSE stream stalled: no data received for 1 seconds"
    );
    assert_eq!(stub.chat_requests.load(Ordering::SeqCst), 1);
}

/// No first token inside the limit fails the turn with that cause, and the
/// prompt is sent once.
#[tokio::test]
async fn a_turn_without_a_first_token_names_the_limit_and_does_not_resend() {
    let stub = stub_lm_studio(
        "loaded",
        vec![Step {
            after: Duration::from_secs(30),
            lines: &[ROLE, HELLO, DONE],
        }],
    )
    .await;

    assert_eq!(
        turn_error(&stub, Duration::from_secs(3)).await,
        "Stream error: no first token after 3s (model still loading?)"
    );
    assert_eq!(stub.chat_requests.load(Ordering::SeqCst), 1);
}

/// Ctrl+C during the first-token wait ends the turn at once and closes the
/// connection to the local server.
#[tokio::test]
async fn ctrl_c_during_the_wait_cancels_the_turn() {
    let stub = stub_lm_studio(
        "loaded",
        vec![Step {
            after: Duration::from_secs(30),
            lines: &[ROLE, HELLO, DONE],
        }],
    )
    .await;
    let provider = lm_studio(
        &stub.base_url,
        Duration::from_secs(60),
        Duration::from_secs(1),
    );
    let agent = AgentBuilder::new()
        .provider_instance(Arc::new(provider))
        .build()
        .await
        .expect("agent");
    let cancel = Arc::new(AtomicBool::new(false));
    let mut stream = agent.stream_chat_with_cancel_flag("hey", Some(Arc::clone(&cancel)));
    let setter = Arc::clone(&cancel);
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(3)).await;
        setter.store(true, Ordering::SeqCst);
    });
    let started = tokio::time::Instant::now();
    let mut seen = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::Status { message }) => seen.push(format!("status: {message}")),
            Ok(_) => {}
            Err(error) => {
                seen.push(format!("error: {error:#}"));
                break;
            }
        }
    }
    drop(stream);
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert_eq!(
        seen,
        vec![
            "status: Waiting for the first token...",
            "error: Stream cancelled by user",
        ]
    );
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(stub.closed_early.load(Ordering::SeqCst));
    assert_eq!(stub.chat_requests.load(Ordering::SeqCst), 1);
}

/// LM Studio's reply when no model can serve the request (captured from
/// LM Studio on 127.0.0.1:1234). The user sees its message.
const LM_STUDIO_NO_MODEL: &str = r#"{
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'lms load' command.",
        "type": "invalid_request_error",
        "param": "model",
        "code": null
    }
}"#;

#[tokio::test]
async fn a_lm_studio_error_reply_shows_its_message() {
    let base_url = stub_reply("400 Bad Request", LM_STUDIO_NO_MODEL).await;
    let provider = lm_studio(&base_url, Duration::from_secs(10), Duration::from_secs(1));

    assert_eq!(
        transcript(&provider).await,
        vec![
            "error: lmstudio API error (400 Bad Request): No models loaded. Please load a model in the developer page or use the 'lms load' command."
        ]
    );
}

/// Local configs probe their server and wait long for the first token;
/// cloud configs keep the stall limit for the first byte and probe nothing.
#[test]
fn only_local_providers_wait_for_a_first_token() {
    let cloud = [
        limits(&OpenAIConfig::new("k", "gpt")),
        limits(&OpenRouterConfig::new("k", "m")),
        limits(&PerplexityConfig::new("k", "m")),
    ];
    let lm_studio = LMStudioConfig::new("http://127.0.0.1:1234/v1", "m");
    let ollama = OllamaConfig::new("http://127.0.0.1:11434/v1", "llama3");

    assert_eq!(
        cloud,
        [
            (None, Duration::from_secs(60), None),
            (None, Duration::from_secs(60), None),
            (None, Duration::from_secs(60), None),
        ]
    );
    assert_eq!(
        limits(&lm_studio),
        (
            Some(Duration::from_secs(600)),
            Duration::from_secs(60),
            Some(ModelLoadProbe::LmStudio {
                url: "http://127.0.0.1:1234/api/v0/models/m".to_string(),
                model: "m".to_string(),
            })
        )
    );
    assert_eq!(
        limits(&ollama).2,
        Some(ModelLoadProbe::Ollama {
            url: "http://127.0.0.1:11434/api/ps".to_string(),
            model: "llama3".to_string(),
        })
    );
}

fn limits(
    config: &impl OpenAICompatConfig,
) -> (Option<Duration>, Duration, Option<ModelLoadProbe>) {
    (
        config.first_token_timeout(),
        config.stall_timeout(),
        config.model_load_probe(),
    )
}

/// A stub that answers every GET with `body`.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_get(body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        while let Ok((mut socket, _)) = listener.accept().await {
            tokio::spawn(async move {
                let _ = read_request(&mut socket).await;
                let reply = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(reply.as_bytes()).await;
            });
        }
    });
    format!("http://{addr}")
}

async fn reported_window(probe: ModelLoadProbe) -> Option<usize> {
    probe.context_window(&reqwest::Client::new()).await
}

/// LM Studio reports the loaded context length, else the model's maximum.
#[tokio::test]
async fn lm_studio_reports_the_loaded_context_window() {
    let loaded = stub_get(
        r#"{"id":"m","state":"loaded","max_context_length":262144,"loaded_context_length":32768}"#,
    )
    .await;
    let not_loaded =
        stub_get(r#"{"id":"m","state":"not-loaded","max_context_length":262144}"#).await;
    let silent = stub_get(r#"{"id":"m","state":"loaded"}"#).await;
    let probe = |root: &str| {
        lm_studio(
            &format!("{root}/v1"),
            Duration::from_secs(1),
            Duration::from_secs(1),
        )
        .model_load_probe()
    };

    assert_eq!(
        reported_window(probe(&loaded).expect("probe")).await,
        Some(32_768)
    );
    assert_eq!(
        reported_window(probe(&not_loaded).expect("probe")).await,
        Some(262_144)
    );
    assert_eq!(reported_window(probe(&silent).expect("probe")).await, None);
}

/// Ollama reports the context length of the loaded model only.
#[tokio::test]
async fn ollama_reports_the_context_window_of_the_loaded_model() {
    let root = stub_get(
        r#"{"models":[{"name":"other:latest","context_length":8192},{"name":"llama3:latest","context_length":16384}]}"#,
    )
    .await;
    let probe = |model: &str| ModelLoadProbe::Ollama {
        url: format!("{root}/api/ps"),
        model: model.to_string(),
    };

    assert_eq!(reported_window(probe("llama3")).await, Some(16_384));
    assert_eq!(reported_window(probe("qwen3")).await, None);
}
