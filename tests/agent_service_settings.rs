//! AgentService apply_settings tests (mode overlay and tool filters).

use caliche::agent::modes::{InteractionMode, readonly_mode_prompt};
use caliche::agent::tools::names::{TOOL_BASH, TOOL_FILESYSTEM};
use caliche::traits::credentials::{Credentials, StaticCredentialProvider};
use caliche::{AgentService, AgentServiceConfig, AgentSettings, ProviderType};
use std::collections::HashSet;
use std::sync::Arc;

fn test_config() -> AgentServiceConfig {
    AgentServiceConfig::new()
        .with_provider(
            Some(ProviderType::Anthropic),
            Some("claude-3-5-sonnet-20241022".to_string()),
        )
        .with_context_window(Some(8_192))
        .with_credential_provider(Arc::new(StaticCredentialProvider::new(
            Credentials::ApiKey("test-key".to_string()),
        )))
        .with_all_default_tools()
}

#[tokio::test]
async fn apply_settings_empty_tools_keeps_default_tool_access() {
    let mut service = AgentService::from_config(test_config())
        .await
        .expect("agent service should build with default tools");

    service
        .apply_settings(AgentSettings {
            provider: service.provider(),
            model: service.model().to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: None,
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
        })
        .await
        .expect("settings should apply");

    assert_eq!(
        service.agent().allowed_tools(),
        None,
        "empty settings tools should mean all registered tools, not zero allowed tools"
    );
}

#[tokio::test]
async fn apply_settings_readonly_mode_overlays_base_system_prompt() {
    let base_prompt = "You are a helpful assistant for project Alpha.";
    let mut service = AgentService::from_config(test_config())
        .await
        .expect("agent service should build");

    service
        .apply_settings(AgentSettings {
            provider: service.provider(),
            model: service.model().to_string(),
            system_prompt: Some(base_prompt.to_string()),
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: Some(InteractionMode::Plan),
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
        })
        .await
        .expect("plan mode settings should apply");

    let mode_prefix =
        readonly_mode_prompt(InteractionMode::Plan).expect("plan mode should have readonly prefix");
    let effective = service.agent().system_prompt();
    assert!(
        effective.starts_with(mode_prefix),
        "effective prompt should start with mode overlay"
    );
    assert!(
        effective.contains(base_prompt),
        "effective prompt should retain base_system_prompt"
    );

    service
        .apply_settings(AgentSettings {
            provider: service.provider(),
            model: service.model().to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: Some(InteractionMode::Agent),
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
        })
        .await
        .expect("agent mode settings should apply");

    assert_eq!(
        service.agent().system_prompt(),
        base_prompt,
        "leaving readonly mode should restore base prompt without overlay"
    );
}

#[tokio::test]
async fn apply_interaction_mode_filters_write_tools_via_registry() {
    let mut service = AgentService::from_config(test_config())
        .await
        .expect("agent service should build with default tools");

    service
        .apply_interaction_mode(InteractionMode::Ask)
        .await
        .expect("ask mode should apply");

    let allowed = service
        .agent()
        .allowed_tools()
        .expect("ask mode should set an allowlist");
    let allowed_set: HashSet<_> = allowed.iter().cloned().collect();
    assert!(
        !allowed_set.contains(TOOL_BASH),
        "bash must be blocked in ask mode"
    );
    assert!(
        !allowed_set.contains(TOOL_FILESYSTEM),
        "fs must be blocked in ask mode (side-effecting)"
    );
    for name in &allowed {
        assert!(
            service.tool_registry().is_readonly(name),
            "ask allowlist should only include readonly tools, found {name}"
        );
    }

    service
        .apply_interaction_mode(InteractionMode::Agent)
        .await
        .expect("agent mode should restore full access");
    assert_eq!(
        service.agent().allowed_tools(),
        None,
        "leaving readonly mode should clear the allowlist"
    );
}

#[tokio::test]
async fn apply_interaction_mode_plan_keeps_write_plan_exception() {
    let mut service = AgentService::from_config(test_config())
        .await
        .expect("agent service should build");

    // Simulate CLI registering write_plan; registry marks it non-readonly.
    #[derive(Debug)]
    struct FakeWritePlan;
    impl caliche::agent::tools::tool::ToolMeta for FakeWritePlan {
        fn name(&self) -> &str {
            "write_plan"
        }
        fn description(&self) -> &str {
            "write plan"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type":"object","properties":{}})
        }
        fn is_readonly(&self) -> bool {
            false
        }
    }
    #[async_trait::async_trait]
    impl caliche::agent::tools::Tool for FakeWritePlan {
        async fn execute(
            &self,
            _arguments: serde_json::Value,
        ) -> caliche::agent::tools::ToolResult {
            caliche::agent::tools::ToolResult::success("ok")
        }
    }
    service.add_tool(Arc::new(FakeWritePlan));

    service
        .apply_interaction_mode(InteractionMode::Plan)
        .await
        .expect("plan mode should apply");

    let allowed = service
        .agent()
        .allowed_tools()
        .expect("plan mode should set an allowlist");
    assert!(
        allowed.iter().any(|n| n == "write_plan"),
        "plan mode must keep write_plan even though it is not readonly"
    );
    assert!(
        !allowed.iter().any(|n| n == TOOL_BASH),
        "plan mode must still block bash"
    );
}

#[tokio::test]
async fn apply_settings_preserves_explicit_tools_when_leaving_readonly() {
    let mut service = AgentService::from_config(test_config())
        .await
        .expect("agent service should build");

    service
        .apply_interaction_mode(InteractionMode::Ask)
        .await
        .expect("enter ask");

    service
        .apply_settings(AgentSettings {
            provider: service.provider(),
            model: service.model().to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: vec!["calculator".to_string()],
            mode: Some(InteractionMode::Agent),
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
        })
        .await
        .expect("leave ask with explicit tools");

    assert_eq!(
        service.agent().allowed_tools(),
        Some(vec!["calculator".to_string()]),
        "explicit tool allowlist must survive leaving readonly mode"
    );
}

#[test]
fn default_embedding_model_covers_supported_providers() {
    assert_eq!(
        caliche::agent::service::default_embedding_model(ProviderType::OpenAI).as_deref(),
        Some("text-embedding-3-small")
    );
    assert_eq!(
        caliche::agent::service::default_embedding_model(ProviderType::Ollama).as_deref(),
        Some("nomic-embed-text")
    );
    assert!(caliche::agent::service::default_embedding_model(ProviderType::Anthropic).is_none());
}

#[tokio::test]
async fn apply_settings_toggles_show_sources() {
    let mut service = AgentService::from_config(test_config())
        .await
        .expect("agent service should build");

    assert!(service.show_sources());

    service
        .apply_settings(AgentSettings {
            provider: service.provider(),
            model: service.model().to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: None,
            embedding_provider: None,
            embedding_model: None,
            show_sources: Some(false),
        })
        .await
        .expect("settings should apply");

    assert!(!service.show_sources());
    assert_eq!(service.snapshot_settings().show_sources, Some(false));
}

#[tokio::test]
async fn create_embedding_provider_builds_ollama_without_api_key() {
    use caliche::agent::service::ProviderFactory;
    use caliche::traits::credentials::EnvironmentCredentialProvider;

    let provider = ProviderFactory::create_embedding_provider(
        ProviderType::Ollama,
        "nomic-embed-text".to_string(),
        Arc::new(EnvironmentCredentialProvider::new()),
    )
    .await
    .expect("ollama embedding provider should construct");

    assert!(
        provider.is_some(),
        "saved ollama embedding selections must restore without an API key"
    );
}

fn openai_config(model: &str) -> AgentServiceConfig {
    AgentServiceConfig::new()
        .with_provider(Some(ProviderType::OpenAI), Some(model.to_string()))
        .with_credential_provider(Arc::new(StaticCredentialProvider::new(
            Credentials::ApiKey("test-key".to_string()),
        )))
}

/// A host switches the model by sending back a snapshot with a new model.
/// The snapshot's window is the old model's; it must not pin it.
#[tokio::test]
async fn a_model_switch_through_a_snapshot_takes_the_new_models_window() {
    let mut service = AgentService::from_config(openai_config("gpt-4o"))
        .await
        .expect("service");
    assert_eq!(service.context_window(), Some(128_000));

    let mut settings = service.snapshot_settings();
    settings.model = "gpt-4".to_string();
    service.apply_settings(settings).await.expect("switch");

    assert_eq!(service.context_window(), Some(8_192));
    assert_eq!(service.context_window_override(), None);
}

/// A window the host changes in the snapshot is still applied.
#[tokio::test]
async fn a_changed_window_in_a_snapshot_is_applied() {
    let mut service = AgentService::from_config(openai_config("gpt-4o"))
        .await
        .expect("service");

    let mut settings = service.snapshot_settings();
    settings.context_window = Some(64_000);
    service.apply_settings(settings).await.expect("window");

    assert_eq!(service.context_window(), Some(64_000));
    assert_eq!(service.context_window_override(), Some(64_000));
}

/// An AGENTS.md of 2000 lines (52,000 characters, about 17K tokens).
fn large_agents_md() -> caliche::ProjectContext {
    caliche::ProjectContext {
        path: std::path::PathBuf::from("/repo/AGENTS.md"),
        source_filename: "AGENTS.md".to_string(),
        content: (0..2000)
            .map(|i| format!("rule {i:04}: keep this line\n"))
            .collect(),
        share_percent: caliche::DEFAULT_PROJECT_CONTEXT_SHARE,
    }
}

const CUT_32K_END: &str = "rule 0937: keep this line\n\n[AGENTS.md is cut to fit this model's 32768-token context: these are its first 938 of 2000 lines. Read /repo/AGENTS.md for the rest.]";

/// The project context follows the window: a provider that reports none
/// gets the whole file; an explicit 32K window cuts it in the prompt, with
/// the note; 200K restores it. The text around it is kept.
#[tokio::test]
async fn the_project_context_follows_the_explicit_window() {
    let mut service = AgentService::from_config(openai_config("gpt-4o"))
        .await
        .expect("service");
    let section = service.budget_project_context(large_agents_md());
    assert!(section.ends_with("rule 1999: keep this line"), "{section}");
    service
        .agent_mut()
        .set_system_prompt(format!("base\n\n{section}\n\nafter"));

    service.set_context_window(32_768).expect("32K");
    let prompt = service.agent().system_prompt().to_string();
    assert!(prompt.starts_with("base\n\n# Project Context (from AGENTS.md)\n\nrule 0000"));
    assert!(
        prompt.ends_with(&format!("{CUT_32K_END}\n\nafter")),
        "{prompt}"
    );
    assert!(!prompt.contains("rule 0938"));

    service.set_context_window(200_000).expect("200K");
    assert_eq!(
        service.agent().system_prompt(),
        format!("base\n\n{section}\n\nafter")
    );
}

/// A host that sends back the prompt it had before a window change gets
/// the section sized for the new window, not the stale one.
#[tokio::test]
async fn apply_settings_swaps_a_stale_project_section() {
    let mut service = AgentService::from_config(openai_config("gpt-4o"))
        .await
        .expect("service");
    let section = service.budget_project_context(large_agents_md());
    let host_prompt = format!("base\n\n{section}");

    let mut settings = service.snapshot_settings();
    settings.system_prompt = Some(host_prompt);
    settings.context_window = Some(32_768);
    service.apply_settings(settings).await.expect("settings");

    let prompt = service.agent().system_prompt();
    assert!(prompt.ends_with(CUT_32K_END), "{prompt}");
    assert_eq!(
        service
            .project_context_section()
            .map(|s| s.ends_with(CUT_32K_END)),
        Some(true)
    );
}
