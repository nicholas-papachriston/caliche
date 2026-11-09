//! A local provider's service across model switches: the base URL it was
//! built with, and the context window its server reports.
//!
//! Stubs on 127.0.0.1 play LM Studio: chat completions, and the model-state
//! endpoint, where `small-model` has a 16384-token loaded window and
//! `big-model` 200000. For the base URL, the service is built with
//! `with_base_url` pointing at stub A, while the environment URL for the same
//! provider points at stub B. After a switch the next turn must reach stub A,
//! and stub B must get nothing.
//!
//! The environment URLs are set in the in-process credential store, which
//! the whole process shares. The tests that set them run one at a time
//! (`serial(environment_urls)`), and each restores the store when it ends.

use caliche::llm::ProviderType;
use caliche::messages::Message;
use caliche::{AgentService, AgentServiceConfig};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// The request line and JSON body of each request a stub received.
type Requests = Arc<Mutex<Vec<(String, serde_json::Value)>>>;

/// Start a stub; returns its `/v1` base URL and what it receives.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn stub_server() -> (String, Requests) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let requests: Requests = Arc::default();
    let seen = Arc::clone(&requests);
    tokio::spawn(async move {
        while let Ok((socket, _)) = listener.accept().await {
            tokio::spawn(answer(socket, Arc::clone(&seen)));
        }
    });
    (format!("http://{addr}/v1"), requests)
}

/// Read one request, record it, and answer a chat completion.
async fn answer(mut socket: TcpStream, seen: Requests) {
    let mut raw = Vec::new();
    let mut buf = [0_u8; 8192];
    let head_end = loop {
        let Ok(n) = socket.read(&mut buf).await else {
            return;
        };
        if n == 0 {
            return;
        }
        raw.extend_from_slice(&buf[..n]);
        if let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let head = String::from_utf8_lossy(&raw[..head_end]).to_string();
    let length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    while raw.len() < head_end + length {
        let Ok(n) = socket.read(&mut buf).await else {
            return;
        };
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
    }
    let body: serde_json::Value =
        serde_json::from_slice(&raw[head_end..]).unwrap_or(serde_json::Value::Null);
    let request_line = head.lines().next().unwrap_or_default().to_string();
    let stream = body["stream"].as_bool() == Some(true);
    let is_chat = request_line.starts_with("POST /v1/chat/completions ");
    let model_state = model_state(&request_line);
    seen.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push((request_line, body));
    let reply = match (is_chat, stream, model_state) {
        (false, _, Some(body)) => format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        ),
        (false, _, None) => {
            let body = r#"{"error":"unexpected route"}"#;
            format!(
                "HTTP/1.1 404 Not Found\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
        }
        (true, false, _) => {
            let body = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"stub","choices":[{"index":0,"message":{"role":"assistant","content":"stub answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
        }
        (true, true, _) => {
            let chunk = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"stub","choices":[{"index":0,"delta":{"role":"assistant","content":"stub answer"},"finish_reason":"stop"}]}"#;
            let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            )
        }
    };
    let _ = socket.write_all(reply.as_bytes()).await;
}

/// LM Studio's state for the models the stub serves.
fn model_state(request_line: &str) -> Option<&'static str> {
    let asks = |id: &str| request_line.starts_with(&format!("GET /api/v0/models/{id} "));
    if asks("small-model") {
        Some(r#"{"id":"small-model","state":"loaded","loaded_context_length":16384}"#)
    } else if asks("big-model") {
        Some(r#"{"id":"big-model","state":"loaded","loaded_context_length":200000}"#)
    } else {
        None
    }
}

/// The models named in the chat requests a stub received.
fn chat_models(requests: &Requests) -> Vec<String> {
    requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(line, _)| line.starts_with("POST /v1/chat/completions "))
        .map(|(_, body)| body["model"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn count(requests: &Requests) -> usize {
    requests
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len()
}

/// One environment URL in the in-process credential store, for one test.
/// On drop, the earlier stored value is put back, or the name is removed.
struct EnvironmentUrl {
    name: &'static str,
    earlier: Option<String>,
}

impl EnvironmentUrl {
    fn set(name: &'static str, url: String) -> Self {
        let earlier = stored(name);
        caliche_contracts::credential_env::set(name, url);
        Self { name, earlier }
    }
}

impl Drop for EnvironmentUrl {
    fn drop(&mut self) {
        match self.earlier.take() {
            Some(value) => caliche_contracts::credential_env::set(self.name, value),
            None => caliche_contracts::credential_env::remove(self.name),
        }
    }
}

/// The value the in-process credential store holds for `name`.
fn stored(name: &str) -> Option<String> {
    caliche_contracts::credential_env::snapshot()
        .into_iter()
        .find(|(stored, _)| stored == name)
        .map(|(_, value)| value)
}

/// A test's environment URL does not outlive the test: a name that was not
/// stored is removed again, and an earlier stored value comes back.
#[test]
#[serial_test::serial(environment_urls)]
fn an_environment_url_does_not_outlive_its_test() {
    caliche_contracts::credential_env::remove("LMSTUDIO_BASE_URL");
    {
        let _url = EnvironmentUrl::set("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1".to_string());
        assert_eq!(
            stored("LMSTUDIO_BASE_URL").as_deref(),
            Some("http://127.0.0.1:9/v1")
        );
    }
    assert_eq!(stored("LMSTUDIO_BASE_URL"), None);

    {
        let _earlier =
            EnvironmentUrl::set("LMSTUDIO_BASE_URL", "http://127.0.0.1:8/v1".to_string());
        {
            let _url =
                EnvironmentUrl::set("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1".to_string());
        }
        assert_eq!(
            stored("LMSTUDIO_BASE_URL").as_deref(),
            Some("http://127.0.0.1:8/v1")
        );
    }
    assert_eq!(stored("LMSTUDIO_BASE_URL"), None);
}

/// An LM Studio service built with stub A's URL, while the environment URL
/// for LM Studio is stub B's (until the returned guard drops).
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn service_on_a() -> (AgentService, Requests, Requests, EnvironmentUrl) {
    let (url_a, stub_a) = stub_server().await;
    let (url_b, stub_b) = stub_server().await;
    let environment_url = EnvironmentUrl::set("LMSTUDIO_BASE_URL", url_b);
    let service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(
                Some(ProviderType::LMStudio),
                Some("first-model".to_string()),
            )
            .with_base_url(Some(url_a))
            .with_tool_names(Vec::new()),
    )
    .await
    .expect("service");
    (service, stub_a, stub_b, environment_url)
}

#[tokio::test]
#[serial_test::serial(environment_urls)]
async fn a_model_change_keeps_the_configured_base_url() {
    let (mut service, stub_a, stub_b, _environment_url) = service_on_a().await;

    service
        .change_model(None, Some("other-model".to_string()))
        .await
        .expect("change model");
    service.chat("hello").await.expect("chat");

    assert_eq!(chat_models(&stub_a), vec!["other-model".to_string()]);
    assert_eq!(count(&stub_b), 0);
}

#[tokio::test]
#[serial_test::serial(environment_urls)]
async fn a_model_change_through_settings_keeps_the_configured_base_url() {
    let (mut service, stub_a, stub_b, _environment_url) = service_on_a().await;

    let mut settings = service.snapshot_settings();
    settings.model = "other-model".to_string();
    service.apply_settings(settings).await.expect("apply");
    service.chat("hello").await.expect("chat");

    assert_eq!(chat_models(&stub_a), vec!["other-model".to_string()]);
    assert_eq!(count(&stub_b), 0);
}

/// The embedded server switches models on a per-request copy.
#[tokio::test]
#[serial_test::serial(environment_urls)]
async fn a_model_change_on_a_per_request_copy_keeps_the_configured_base_url() {
    let (service, stub_a, stub_b, _environment_url) = service_on_a().await;

    let mut copy = service.clone_for_request();
    copy.change_model(None, Some("other-model".to_string()))
        .await
        .expect("change model");
    copy.chat("hello").await.expect("chat");

    assert_eq!(chat_models(&stub_a), vec!["other-model".to_string()]);
    assert_eq!(count(&stub_b), 0);
}

#[tokio::test]
#[serial_test::serial(environment_urls)]
async fn another_provider_uses_its_own_url_and_switching_back_restores_the_configured_one() {
    let (mut service, stub_a, _stub_b, _lm_studio_url) = service_on_a().await;
    let (url_c, stub_c) = stub_server().await;
    let _ollama_url = EnvironmentUrl::set("OLLAMA_BASE_URL", url_c);

    service
        .change_model(Some(ProviderType::Ollama), Some("ollama-model".to_string()))
        .await
        .expect("to ollama");
    service.chat("hello").await.expect("chat on ollama");

    assert_eq!(chat_models(&stub_c), vec!["ollama-model".to_string()]);
    assert!(chat_models(&stub_a).is_empty());

    service
        .change_model(Some(ProviderType::LMStudio), Some("back-model".to_string()))
        .await
        .expect("back to lm studio");
    service
        .chat("hello again")
        .await
        .expect("chat on lm studio");

    assert_eq!(chat_models(&stub_a), vec!["back-model".to_string()]);
}

/// An LM Studio service on `model` at a stub that reports windows.
#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
async fn service_reporting(model: &str, explicit: Option<usize>) -> (AgentService, Requests) {
    let (url, stub) = stub_server().await;
    let service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(Some(ProviderType::LMStudio), Some(model.to_string()))
            .with_base_url(Some(url))
            .with_context_window(explicit)
            .with_tool_names(Vec::new()),
    )
    .await
    .expect("service");
    (service, stub)
}

/// Without an explicit window, the window is the one the server reports,
/// not the fixed 32768 fallback, and a switch follows the new model.
#[tokio::test]
async fn the_context_window_is_the_reported_one_and_follows_a_switch() {
    let (mut service, _stub) = service_reporting("small-model", None).await;
    assert_eq!(service.context_window(), Some(16_384));
    assert_eq!(service.project_context_window(), Some(16_384));

    service
        .change_model(None, Some("big-model".to_string()))
        .await
        .expect("to big");
    assert_eq!(service.context_window(), Some(200_000));

    // A model the server does not report falls back to the fixed window.
    service
        .change_model(None, Some("unknown-model".to_string()))
        .await
        .expect("to unknown");
    assert_eq!(service.context_window(), Some(32_768));
    assert_eq!(service.project_context_window(), None);
}

/// An explicit window wins over the reported one.
#[tokio::test]
async fn an_explicit_window_wins_over_the_reported_one_for_the_context_window() {
    let (mut service, _stub) = service_reporting("small-model", Some(24_000)).await;
    assert_eq!(service.context_window(), Some(24_000));

    service
        .change_model(None, Some("big-model".to_string()))
        .await
        .expect("to big");
    assert_eq!(service.context_window(), Some(24_000));
}

/// Compaction uses the reported window. On `small-model` (16384 tokens,
/// 2000 reserved) the guard compacts past 90% of 14384 tokens; with the
/// 32768 fallback it would wait until 90% of 30768.
#[tokio::test]
async fn compaction_triggers_at_the_reported_window() {
    let (mut service, stub) = service_reporting("small-model", None).await;
    service.agent_mut().set_compaction_strategy(Arc::new(
        caliche::agent::core::compaction::SlidingWindowCompaction::builder()
            .preserve_recent(1)
            .build(),
    ));
    // Context management starts at 10 messages: the system prompt, four
    // earlier turns and the new message.
    let long_turn = "word ".repeat(2_700);
    let history = (0..4)
        .flat_map(|_| [Message::user(long_turn.clone()), Message::assistant("ok")])
        .collect();

    service
        .chat_with_history(&long_turn, history)
        .await
        .expect("chat");

    let sent = stub
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(line, _)| line.starts_with("POST /v1/chat/completions "))
        .map(|(_, body)| body["messages"].as_array().map_or(0, Vec::len))
        .collect::<Vec<_>>();
    // The two oldest turns' 13,500-character messages and one reply are
    // dropped: 7 of 10 messages are sent.
    assert_eq!(sent, vec![7]);
    let report = service.agent().last_compaction_report().expect("compacted");
    assert_eq!(report.changes.len(), 3);
}
