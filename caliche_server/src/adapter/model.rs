//! Apply a thin-client `/model` or `/provider` selection to a live session.
//!
//! The REPL label is local. The next `POST /turn` carries that provider and
//! model, and this helper retargets the same harness before the turn runs so
//! history stays on the session.

use anyhow::Result;
use caliche::harness::AgentHarness;
use caliche::llm::ProviderType;

/// Point `harness` at `provider` / `model` when either differs from the session.
///
/// Empty values are ignored. A matching identity does not rebuild the provider.
/// Conversation history on the harness is left in place.
pub(super) async fn apply_turn_model(
    harness: &mut AgentHarness,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<()> {
    let provider_raw = provider.map(str::trim).filter(|value| !value.is_empty());
    let model_raw = model.map(str::trim).filter(|value| !value.is_empty());
    if provider_raw.is_none() && model_raw.is_none() {
        return Ok(());
    }

    let current_provider = harness.service().provider();
    let requested_provider = match provider_raw {
        Some(raw) => raw
            .parse::<ProviderType>()
            .map_err(|err| anyhow::anyhow!("invalid provider '{raw}': {err}"))?,
        None => current_provider,
    };
    let provider_changed = requested_provider != current_provider;
    let model_changed = model_raw.is_some_and(|next| next != harness.service().model());
    if !provider_changed && !model_changed {
        return Ok(());
    }

    harness
        .service_mut()
        .change_model(
            provider_changed.then_some(requested_provider),
            model_raw.map(str::to_owned),
        )
        .await?;
    Ok(())
}

/// Set the session's explicit context window when the turn carries one that
/// differs from it. The project context and compaction follow it.
pub(super) fn apply_turn_context_window(
    harness: &mut AgentHarness,
    context_window: Option<usize>,
) -> Result<()> {
    let Some(window) = context_window else {
        return Ok(());
    };
    if harness.service().context_window_override() != Some(window) {
        harness.service_mut().set_context_window(window)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::ApiAdapter;
    use crate::handlers::sessions::{ExecuteTurnRequest, execute_turn_handler};
    use axum::Json;
    use axum::extract::{Path, State};
    use axum::http::StatusCode;
    use caliche::harness::HarnessConfig;
    use caliche::llm::ProviderType;
    use caliche::messages::Message;
    use std::sync::Arc;

    const STARTED_MODEL: &str = "claude-sonnet-4";
    const SWITCHED_MODEL: &str = "z-ai/glm-5.3";

    /// Placeholder provider keys, so a model retarget never dials a live provider.
    fn install_keys(env: &mut caliche_test_env::EnvGuard) {
        env.set("ANTHROPIC_API_KEY", "test-key-for-model-switch");
        env.set("OPENROUTER_API_KEY", "test-key-for-model-switch");
    }

    async fn session_on_claude() -> (Arc<ApiAdapter>, String) {
        let adapter = Arc::new(ApiAdapter::with_session_ttl(
            HarnessConfig::default()
                .with_provider(ProviderType::Anthropic)
                .with_model(STARTED_MODEL),
            std::time::Duration::from_secs(60),
            None,
        ));
        let session_id = adapter
            .create_session(None, None, None)
            .await
            .expect("anthropic session");
        let session = adapter
            .get_session(&session_id)
            .await
            .expect("live session");
        {
            let state = session.write().await;
            let mut harness = state.harness.write().await;
            harness.set_history(vec![
                Message::user("summarize the repo"),
                Message::assistant("Caliche is a Rust agent runtime."),
            ]);
        }
        adapter
            .turn_cancel_handle(&session_id)
            .expect("cancel handle")
            .signal();
        (adapter, session_id)
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn turn_body_model_retargets_the_same_session() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        install_keys(&mut env);
        let (adapter, session_id) = session_on_claude().await;
        let request: ExecuteTurnRequest = serde_json::from_str(
            r#"{"input":"who are you?","provider":"openrouter","model":"z-ai/glm-5.3"}"#,
        )
        .expect("turn json");
        assert_eq!(request.provider.as_deref(), Some("openrouter"));
        assert_eq!(request.model.as_deref(), Some(SWITCHED_MODEL));
        let error = execute_turn_handler(
            State(Arc::clone(&adapter)),
            Path(session_id.clone()),
            Json(request),
        )
        .await
        .expect_err("pre-cancelled turn does not call the provider");
        assert_eq!(error.0, StatusCode::CONFLICT, "{}", error.1);

        let session = adapter
            .get_session(&session_id)
            .await
            .expect("same session remains");
        let state = session.read().await;
        let harness = state.harness.read().await;
        assert_eq!(harness.session_id(), session_id);
        assert_eq!(harness.service().provider(), ProviderType::OpenRouter);
        assert_eq!(harness.service().model(), SWITCHED_MODEL);
        assert_eq!(harness.service().llm_provider().name(), "openrouter");
        assert_eq!(harness.history().len(), 2);
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn turn_body_without_model_keeps_the_remote_model() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        install_keys(&mut env);
        let (adapter, session_id) = session_on_claude().await;
        let request: ExecuteTurnRequest =
            serde_json::from_str(r#"{"input":"who are you?"}"#).expect("bare turn json");
        assert!(request.provider.is_none());
        assert!(request.model.is_none());
        let error = execute_turn_handler(
            State(Arc::clone(&adapter)),
            Path(session_id.clone()),
            Json(request),
        )
        .await
        .expect_err("pre-cancelled turn");
        assert_eq!(error.0, StatusCode::CONFLICT, "{}", error.1);

        let session = adapter
            .get_session(&session_id)
            .await
            .expect("same session remains");
        let state = session.read().await;
        let harness = state.harness.read().await;
        assert_eq!(harness.service().provider(), ProviderType::Anthropic);
        assert_eq!(harness.service().model(), STARTED_MODEL);
        assert_eq!(harness.service().llm_provider().name(), "anthropic");
        assert_eq!(harness.history().len(), 2);
    }

    /// `/context-window 16384` in the thin client reaches the session: the
    /// next turn runs with that window, without a model switch.
    #[tokio::test]
    #[serial_test::serial]
    async fn turn_body_context_window_sets_the_session_window() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        install_keys(&mut env);
        let (adapter, session_id) = session_on_claude().await;
        let request: ExecuteTurnRequest =
            serde_json::from_str(r#"{"input":"who are you?","context_window":16384}"#)
                .expect("turn json");
        let error = execute_turn_handler(
            State(Arc::clone(&adapter)),
            Path(session_id.clone()),
            Json(request),
        )
        .await
        .expect_err("pre-cancelled turn");
        assert_eq!(error.0, StatusCode::CONFLICT, "{}", error.1);

        let session = adapter
            .get_session(&session_id)
            .await
            .expect("same session remains");
        let state = session.read().await;
        let harness = state.harness.read().await;
        assert_eq!(harness.service().model(), STARTED_MODEL);
        assert_eq!(harness.service().context_window(), Some(16_384));
        assert_eq!(harness.service().context_window_override(), Some(16_384));
    }
}
