//! Terminal failure path for `POST /api/agent/sessions/{id}/turn`.

use crate::adapter::ApiAdapter;
use crate::handlers::sessions::map_adapter_error;
use axum::http::StatusCode;
use caliche::harness::AgentEvent;

/// Log a failed turn, tell session subscribers, and return the HTTP error.
///
/// The message is the redacted error chain. The same text is the HTTP body and
/// the `turn.error` event so a client blocked on the event stream can return.
pub(crate) async fn reject_failed_turn(
    adapter: &ApiAdapter,
    session_id: &str,
    error: anyhow::Error,
) -> (StatusCode, String) {
    let (status, message) = map_adapter_error(error);
    tracing::error!(
        session_id,
        status = status.as_u16(),
        error = %message,
        "session turn failed"
    );
    publish_turn_error(adapter, session_id, &message).await;
    (status, message)
}

async fn publish_turn_error(adapter: &ApiAdapter, session_id: &str, message: &str) {
    let Some(session) = adapter.get_session(session_id).await else {
        return;
    };
    let sender = session.read().await.event_sender.clone();
    drop(session);
    let _ = sender.send(AgentEvent::turn_error(message.to_string()));
}

#[cfg(test)]
mod tests {
    use super::reject_failed_turn;
    use crate::adapter::ApiAdapter;
    use axum::http::StatusCode;
    use caliche::harness::{AgentEvent, HarnessConfig, public_agent_sse_event_name};
    use std::time::Duration;

    #[tokio::test]
    #[serial_test::serial]
    async fn failed_turn_publishes_redacted_terminal_event() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("OPENROUTER_API_KEY", "test-key-for-turn-error");

        let adapter =
            ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("session");
        let session = adapter.get_session(&session_id).await.expect("state");
        let mut events = session.read().await.event_sender.subscribe();
        drop(session);

        let error = anyhow::Error::msg(format!(
            "Stream error: anthropic API error (400 Bad Request): tool_result without tool_use {}-ant-api03-SUPERSECRETVALUE",
            "sk"
        ));
        let (status, body) = reject_failed_turn(&adapter, &session_id, error).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(body.contains("anthropic API error (400"));
        assert!(!body.contains("SUPERSECRETVALUE"));
        assert!(body.contains("[redacted]"));

        let event = events.try_recv().expect("turn error event");
        match event {
            AgentEvent::TurnError { message } => {
                assert_eq!(message, body);
                assert_eq!(
                    public_agent_sse_event_name(&AgentEvent::TurnError {
                        message: message.clone(),
                    }),
                    "turn.error"
                );
            }
            other => panic!("expected turn.error, got {other:?}"),
        }
    }
}
