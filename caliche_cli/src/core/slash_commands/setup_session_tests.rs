//! Setup choices reach the running session and the next launch, and a
//! provider switch that cannot complete leaves the session as it was.
//!
//! Setup prompts need a terminal, so these drive the step that takes the
//! entered value. Unit tests use the in-memory mock keychain and a temp HOME.
//! The credential store is process-wide, so each test uses a provider no
//! other test stores a key for.

use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use crate::session::CliSession;
use crate::storage::{Database, SecureCredentialManager, session_credential_provider};
use crate::test_env::EnvGuard;
use caliche::llm::{ProviderType, defaults};
use caliche::{AgentService, AgentServiceConfig};
use std::collections::HashSet;

/// Point the databases, `settings.json`, and the (mock) keychain at `home`.
fn isolated_home(env: &mut EnvGuard, home: &std::path::Path) {
    env.set("HOME", home);
    env.set(caliche_server::paths::CONFIG_DIR_ENV, home.join("config"));
    env.remove("CALICHE_NO_KEYRING");
    env.set("CALICHE_USE_KEYRING", "1");
    for key in ["AGENT_PROVIDER", "DEFAULT_PROVIDER", "API_KEY"] {
        env.remove(key);
    }
}

/// A session on ollama, built the way startup builds it: the session's
/// credential manager loads the keychain first.
async fn ollama_session() -> (InteractiveState, AgentService) {
    let startup = SecureCredentialManager::new_with_scope(None)
        .await
        .expect("credential manager");
    startup
        .apply_to_credential_store_with_priority(&HashSet::new())
        .await;
    let service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(Some(ProviderType::Ollama), Some("gpt-oss:20b".to_string()))
            .with_base_url(Some("http://127.0.0.1:9/v1".to_string()))
            .with_tool_names(Vec::new())
            .with_credential_provider(session_credential_provider(Some(&startup)).await),
    )
    .await
    .expect("ollama session");
    let state = InteractiveState::new(
        CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
        "ollama",
        "gpt-oss:20b",
        "You are Caliche.".to_string(),
    );
    (state, service)
}

#[tokio::test]
#[serial_test::serial(env)]
async fn a_default_chosen_in_setup_is_the_selection_after_a_restart() {
    let home = tempfile::tempdir().expect("tempdir");
    let mut env = EnvGuard::acquire();
    isolated_home(&mut env, home.path());
    let database = Database::open().await.expect("database");
    // The last session ended on an auto-detected ollama.
    crate::boot_provider::save_last_selection(Some(&database), "ollama", "gpt-oss:20b", None, None)
        .await;
    let manager = SecureCredentialManager::new_with_scope(None)
        .await
        .expect("credential manager");
    manager
        .set_default_model("openrouter", "anthropic/claude-haiku-5.5")
        .await
        .expect("default model");

    let model = crate::credentials::make_default_provider(&manager, "openrouter")
        .await
        .expect("make default");

    // A restart reads the selection the way startup does.
    let boot = crate::boot_provider::load_persisted_boot(
        database.load_setting("last_provider").await.expect("read"),
        database.load_setting("last_model").await.expect("read"),
    )
    .await;
    let choice = crate::boot_provider::select_server_boot_provider(
        crate::boot_provider::ServerBootProviderInput {
            explicit_provider: None,
            explicit_model: false,
            agent_provider: None,
            default_provider_env: None,
            last_provider: boot.provider.as_deref(),
            last_model: boot.model.as_deref(),
        },
    );
    assert_eq!(model, "anthropic/claude-haiku-5.5");
    assert_eq!(choice.provider, "openrouter");
    assert_eq!(boot.model.as_deref(), Some("anthropic/claude-haiku-5.5"));
}

#[tokio::test]
#[serial_test::serial(env)]
async fn a_key_entered_in_setup_is_used_by_provider_in_the_same_session() {
    let home = tempfile::tempdir().expect("tempdir");
    let mut env = EnvGuard::acquire();
    isolated_home(&mut env, home.path());
    env.remove("OPENROUTER_API_KEY");
    env.remove("OPENROUTER_MODEL");
    let (mut state, mut service) = ollama_session().await;

    // Setup saves the key through its own manager, as `interactive_setup_for` does.
    SecureCredentialManager::new_with_scope(None)
        .await
        .expect("setup manager")
        .set_api_key("openrouter", "sk-or-session-key")
        .await
        .expect("save key");
    let result =
        SlashCommandExecutor::handle_provider(&mut state, &mut service, Some("openrouter"))
            .await
            .expect("/provider");

    match result {
        SlashCommandResult::StatusMessage(message) => assert_eq!(
            message,
            format!(
                "Switched to provider: openrouter with model: {model}. Preserved 0 messages \
                 and active mode. Model resolved to {model}.",
                model = defaults::OPENROUTER
            )
        ),
        other => panic!("unexpected /provider result: {other:?}"),
    }
    assert_eq!(service.provider(), ProviderType::OpenRouter);
}

#[tokio::test]
#[serial_test::serial(env)]
async fn a_failed_provider_switch_keeps_the_old_provider() {
    let home = tempfile::tempdir().expect("tempdir");
    let mut env = EnvGuard::acquire();
    isolated_home(&mut env, home.path());
    env.remove("ANTHROPIC_API_KEY");
    let (mut state, mut service) = ollama_session().await;

    let result = SlashCommandExecutor::handle_provider(&mut state, &mut service, Some("anthropic"))
        .await
        .expect("/provider");

    match result {
        SlashCommandResult::Error(message) => assert_eq!(
            message,
            "anthropic has no API key. Run /creds setup providers or set ANTHROPIC_API_KEY. \
             Still using ollama/gpt-oss:20b."
        ),
        other => panic!("unexpected /provider result: {other:?}"),
    }
    assert_eq!(service.provider(), ProviderType::Ollama);
    assert_eq!(service.model(), "gpt-oss:20b");
}

/// A provider caliche knows but this build left out is named as such, and
/// the session keeps its provider.
#[cfg(not(feature = "local-inference-base"))]
#[tokio::test]
#[serial_test::serial(env)]
async fn a_provider_left_out_of_the_build_says_how_to_rebuild() {
    let home = tempfile::tempdir().expect("tempdir");
    let mut env = EnvGuard::acquire();
    isolated_home(&mut env, home.path());
    let (mut state, mut service) = ollama_session().await;

    let result =
        SlashCommandExecutor::handle_provider(&mut state, &mut service, Some("local-inference"))
            .await
            .expect("/provider");

    match result {
        SlashCommandResult::Error(message) => assert_eq!(
            message,
            "local-inference is not in this build. Rebuild with --features local-inference."
        ),
        other => panic!("unexpected /provider result: {other:?}"),
    }
    assert_eq!(service.provider(), ProviderType::Ollama);
}

/// `/provider`, `/subagent-provider` and `/embedding` name the feature for
/// a cloud provider left out of the default build, and the session keeps
/// its provider.
#[cfg(not(any(feature = "bedrock", feature = "azure-openai")))]
#[tokio::test]
#[serial_test::serial(env)]
async fn a_cloud_provider_left_out_of_the_build_says_how_to_rebuild() {
    let home = tempfile::tempdir().expect("tempdir");
    let mut env = EnvGuard::acquire();
    isolated_home(&mut env, home.path());
    let (mut state, mut service) = ollama_session().await;

    let mut messages = Vec::new();
    for name in ["bedrock", "azure-openai"] {
        let result = SlashCommandExecutor::handle_provider(&mut state, &mut service, Some(name))
            .await
            .expect("/provider");
        messages.push(format!("/provider {result:?}"));
    }
    let result = SlashCommandExecutor::handle_subagent_provider(&mut state, Some("bedrock"))
        .expect("/subagent-provider");
    messages.push(format!("/subagent-provider {result:?}"));
    let result = SlashCommandExecutor::handle_embedding(&mut state, &mut service, Some("bedrock"))
        .await
        .expect("/embedding");
    messages.push(format!("/embedding {result:?}"));

    assert_eq!(
        messages,
        vec![
            "/provider Error(\"bedrock is not in this build. Rebuild with --features bedrock.\")",
            "/provider Error(\"azure-openai is not in this build. Rebuild with --features azure-openai.\")",
            "/subagent-provider Error(\"bedrock is not in this build. Rebuild with --features bedrock.\")",
            "/embedding Error(\"bedrock is not in this build. Rebuild with --features bedrock.\")",
        ]
    );
    assert_eq!(service.provider(), ProviderType::Ollama);
    assert_eq!(state.subagent_provider, None);
}
