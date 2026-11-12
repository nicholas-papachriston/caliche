//! Push a saved transcript into the agent that will answer the next prompt.

use anyhow::{Context, Result};
use caliche::AgentService;
use caliche::harness::{
    RESUME_CONTEXT_TOKEN_BUDGET, fit_resume_messages, seed_service_resume_context,
};
use caliche::messages::Message;

use crate::client::AgentClient;

/// Outcome of loading a saved transcript into model context.
pub(super) struct ResumeInstall {
    pub model_messages: usize,
    pub summarized: bool,
}

/// Seed in-process memory and, when a thin client is attached, the remote session.
///
/// Empty transcripts are a no-op so a new session keeps its configured memory.
pub(super) async fn install_resumed_context(
    service: &mut AgentService,
    stored: &[Message],
    thin: Option<(&AgentClient, &str)>,
) -> Result<Option<ResumeInstall>> {
    if stored.is_empty() {
        return Ok(None);
    }
    let fitted = fit_resume_messages(stored, RESUME_CONTEXT_TOKEN_BUDGET);
    seed_service_resume_context(service, fitted.messages.clone())
        .await
        .context("seed resumed conversation")?;
    if let Some((client, session_id)) = thin {
        client
            .restore_session_history(session_id, &fitted.messages)
            .await?;
    }
    Ok(Some(ResumeInstall {
        model_messages: fitted.messages.len(),
        summarized: fitted.summarized,
    }))
}

pub(super) fn resume_status_line(installed: &ResumeInstall) -> String {
    if installed.summarized {
        format!(
            "Context: restored {} messages (older turns summarized)",
            installed.model_messages
        )
    } else {
        format!(
            "Context: restored {} prior messages",
            installed.model_messages
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::slash_commands::SlashCommandResult;
    use crate::core::traits::{InputSource, OutputRenderer};
    use caliche::llm::ProviderType;
    use caliche::memory::MemoryStrategy;
    use caliche::testing::MockProvider;
    use caliche::{Agent, AgentService, ToolRegistry};
    use std::sync::Arc;

    #[tokio::test]
    async fn install_seeds_service_memory_from_saved_turns() {
        let provider = Arc::new(MockProvider::new(vec!["ok".to_string()]));
        let agent = Agent::new(provider, Arc::new(ToolRegistry::new()));
        let mut service =
            AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock service");
        let stored = vec![
            Message::user("Please rename parser.rs"),
            Message::assistant("Renamed parser.rs with the fs tool"),
        ];
        let installed = install_resumed_context(&mut service, &stored, None)
            .await
            .expect("install")
            .expect("history present");
        assert!(!installed.summarized);
        assert_eq!(installed.model_messages, 2);
        let memory = service.memory().expect("memory");
        let context = memory.read().await.get_context().await.expect("context");
        let blob = context
            .iter()
            .map(Message::extract_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(blob.contains("Renamed parser.rs with the fs tool"));
    }

    const DROPPED_MARKER: &str = "DROPPED-FILLER-MARKER";

    fn oversized_stored_transcript() -> Vec<Message> {
        let mut stored = vec![Message::user(format!(
            "{DROPPED_MARKER} {}",
            "x".repeat(800)
        ))];
        for index in 0..40 {
            stored.push(Message::user(format!(
                "filler question {index} {}",
                "q".repeat(800)
            )));
            stored.push(Message::assistant(format!(
                "filler answer {index} {}",
                "a".repeat(800)
            )));
        }
        stored.push(Message::user("what did you just do"));
        stored.push(Message::assistant("I renamed parser.rs"));
        stored
    }

    async fn session_memory_text(adapter: &caliche_server::ApiAdapter, session_id: &str) -> String {
        let session = adapter.get_session(session_id).await.expect("live session");
        let memory = {
            let state = session.read().await;
            let harness = state.harness.read().await;
            harness.service().memory().expect("resume memory")
        };
        let context = memory.read().await.get_context().await.expect("context");
        context
            .iter()
            .map(Message::extract_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    async fn capture_history_body(
        axum::extract::State(slot): axum::extract::State<
            Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<Vec<u8>>>>>,
        >,
        request: axum::extract::Request,
        next: axum::middleware::Next,
    ) -> axum::response::Response {
        let (parts, body) = request.into_parts();
        let bytes = axum::body::to_bytes(body, 2 * 1024 * 1024)
            .await
            .expect("read history body");
        if let Some(sender) = slot.lock().expect("history capture lock").take() {
            let _ = sender.send(bytes.to_vec());
        }
        let request = axum::extract::Request::from_parts(parts, axum::body::Body::from(bytes));
        next.run(request).await
    }

    /// The stored transcript must exceed the resume budget so the posted body
    /// cannot equal the raw slice. Omitting the POST, posting `[]`, or posting
    /// `stored` fails this test.
    #[tokio::test]
    #[serial_test::serial]
    async fn install_posts_fitted_transcript_to_session_history() {
        let stored = oversized_stored_transcript();
        let fitted = fit_resume_messages(&stored, RESUME_CONTEXT_TOKEN_BUDGET);
        assert!(
            fitted.summarized,
            "fixture must exceed the resume budget so fitted differs from stored"
        );
        let expected = serde_json::to_value(&fitted.messages).expect("fitted json");
        let stored_json = serde_json::to_value(&stored).expect("stored json");
        assert!(stored_json.to_string().contains(DROPPED_MARKER));
        assert!(
            !expected.to_string().contains(DROPPED_MARKER),
            "fitted body still contains the dropped marker"
        );

        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("OPENROUTER_API_KEY", "test-key-for-resume-post");
        let adapter = Arc::new(caliche_server::ApiAdapter::with_session_ttl(
            caliche::harness::HarnessConfig::default(),
            std::time::Duration::from_secs(60),
            None,
        ));
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session");

        let (body_tx, body_rx) = tokio::sync::oneshot::channel();
        let slot = Arc::new(std::sync::Mutex::new(Some(body_tx)));
        let app = axum::Router::new()
            .route(
                "/api/agent/sessions/{id}/history",
                axum::routing::post(caliche_server::handlers::sessions::restore_history_handler),
            )
            .with_state(Arc::clone(&adapter))
            .layer(axum::middleware::from_fn_with_state(
                slot,
                capture_history_body,
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind history listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve history");
        });

        let client = AgentClient::new(format!("http://{addr}"));
        let provider = Arc::new(MockProvider::new(vec!["ok".to_string()]));
        let agent = Agent::new(provider, Arc::new(ToolRegistry::new()));
        let mut service =
            AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock service");
        let installed =
            install_resumed_context(&mut service, &stored, Some((&client, &session_id)))
                .await
                .expect("install")
                .expect("history present");
        assert!(installed.summarized);
        assert_eq!(installed.model_messages, fitted.messages.len());

        let raw = tokio::time::timeout(std::time::Duration::from_secs(2), body_rx)
            .await
            .expect("history POST was not sent")
            .expect("history body sender dropped");
        let posted: serde_json::Value = serde_json::from_slice(&raw).expect("history json");
        assert_eq!(
            posted.get("messages"),
            Some(&expected),
            "history POST must be the fitted transcript"
        );
        assert_ne!(
            posted.get("messages"),
            Some(&stored_json),
            "history POST must not be the raw stored transcript"
        );

        let memory = session_memory_text(&adapter, &session_id).await;
        assert!(
            memory.contains("I renamed parser.rs"),
            "session memory missing the restored turn: {memory}"
        );
        assert!(
            !memory.contains(DROPPED_MARKER),
            "session memory kept a dropped turn: {memory}"
        );
    }

    #[derive(Clone)]
    struct ScriptedInput(Arc<std::sync::Mutex<std::collections::VecDeque<String>>>);

    #[async_trait::async_trait]
    impl InputSource for ScriptedInput {
        async fn read_line(&mut self, _prompt: &str) -> anyhow::Result<Option<String>> {
            Ok(self.0.lock().expect("scripted input").pop_front())
        }

        fn add_history(&mut self, _line: &str) {}

        fn save_history(&mut self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    struct StatusCapture(Arc<std::sync::Mutex<Vec<String>>>);

    impl OutputRenderer for StatusCapture {
        fn handle_command_result(&mut self, _result: SlashCommandResult) {}
        fn render_chunk(&mut self, _text: &str) {}
        fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
        fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
        fn render_tool_error(&mut self, _name: &str, _error: &str) {}
        fn render_thinking(&mut self, _content: &str) {}
        fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
        fn show_error(&mut self, message: &str) {
            self.0
                .lock()
                .expect("status capture")
                .push(message.to_string());
        }
        fn show_status(&mut self, message: &str) {
            self.0
                .lock()
                .expect("status capture")
                .push(message.to_string());
        }
        fn clear(&mut self) {}
    }

    async fn record_history(
        axum::extract::State(posted): axum::extract::State<
            Arc<std::sync::Mutex<Vec<(String, serde_json::Value)>>>,
        >,
        axum::extract::Path(session_id): axum::extract::Path<String>,
        axum::Json(body): axum::Json<serde_json::Value>,
    ) -> axum::http::StatusCode {
        posted
            .lock()
            .expect("posted history")
            .push((session_id, body));
        axum::http::StatusCode::NO_CONTENT
    }

    /// `/resume <id>` typed into the REPL must push the fitted transcript to
    /// the thin-client session and report it. Dropping the run-loop branch
    /// that matches the session resume command fails both assertions.
    #[tokio::test]
    async fn repl_resume_command_pushes_fitted_history_to_thin_session() {
        let workspace = tempfile::tempdir().expect("tempdir");
        let stored = oversized_stored_transcript();
        let fitted = fit_resume_messages(&stored, RESUME_CONTEXT_TOKEN_BUDGET);
        assert!(fitted.summarized, "fixture must exceed the resume budget");
        let mut snapshot = caliche::harness::SuspendedSession::new("snap-resume");
        snapshot.messages = stored
            .iter()
            .map(caliche::harness::ExportedMessage::from)
            .collect();
        caliche::harness::SuspendStore::for_workspace(workspace.path())
            .save(&snapshot)
            .expect("save snapshot");

        let posted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let app = axum::Router::new()
            .route(
                "/api/agent/sessions/{id}/history",
                axum::routing::post(record_history),
            )
            .with_state(Arc::clone(&posted));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind history listener");
        let addr = listener.local_addr().expect("listener addr");
        tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve history");
        });

        let state = crate::core::state::InteractiveState::new(
            crate::session::CliSession::new("local".to_string(), workspace.path().to_path_buf()),
            "openai",
            "gpt-4o-mini",
            "You are Caliche.".to_string(),
        );
        let input = ScriptedInput(Arc::new(std::sync::Mutex::new(
            ["/resume snap-resume".to_string()].into(),
        )));
        let lines = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut engine =
            crate::repl::ReplEngine::new(input, StatusCapture(Arc::clone(&lines)), state)
                .with_thin_client(AgentClient::new(format!("http://{addr}")), "sess-live");

        let provider = Arc::new(MockProvider::new(vec!["ok".to_string()]));
        let agent = Agent::new(provider, Arc::new(ToolRegistry::new()));
        let mut service =
            AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock service");
        let conversation = caliche::persistence::SqliteConversationStore::from_path(
            &workspace.path().join("sessions.db"),
        )
        .await
        .expect("conversation store");
        let session_store = crate::storage::SessionStore::from_pool(conversation.pool())
            .await
            .expect("session store");
        Box::pin(engine.run(
            &mut service,
            session_store,
            Some((Vec::new(), None, None, false, false, false)),
            None,
            None,
        ))
        .await
        .expect("repl run");

        let lines = lines.lock().expect("status capture").clone();
        let expected_status = resume_status_line(&ResumeInstall {
            model_messages: fitted.messages.len(),
            summarized: true,
        });
        assert!(
            lines.contains(&expected_status),
            "missing resume context status {expected_status:?} in {lines:?}"
        );
        let posted = posted.lock().expect("posted history");
        assert_eq!(posted.len(), 1, "one history POST per resume");
        assert_eq!(posted[0].0, "sess-live");
        assert_eq!(
            posted[0].1.get("messages"),
            Some(&serde_json::to_value(&fitted.messages).expect("fitted json")),
            "history POST must be the fitted transcript"
        );
    }
}
