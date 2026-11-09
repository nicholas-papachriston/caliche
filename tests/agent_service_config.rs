//! AgentServiceConfig and tool selection builder tests.

use caliche::agent::tools::BuiltInTool;
use caliche::traits::credentials::{Credentials, StaticCredentialProvider};
use caliche::{AgentService, AgentServiceConfig, ProviderType, ToolSelection};
use std::sync::Arc;

#[test]
fn tool_selection_can_be_built_from_builtins_and_names() {
    let selection = ToolSelection::builtins(vec![BuiltInTool::Calculator])
        .with_builtin(BuiltInTool::Fs)
        .with_name("http");

    assert_eq!(
        selection.builtins,
        vec![BuiltInTool::Calculator, BuiltInTool::Fs]
    );
    assert_eq!(selection.names, Some(vec!["http".to_string()]));
    assert!(!selection.use_all_default_tools);
}

#[test]
fn agent_service_config_has_direct_tool_helpers() {
    let config = AgentServiceConfig::new()
        .with_built_in_tools(vec![BuiltInTool::Calculator, BuiltInTool::Fs]);
    assert_eq!(
        config.tool_selection.builtins,
        vec![BuiltInTool::Calculator, BuiltInTool::Fs]
    );
    assert_eq!(config.tool_selection.names, None);

    let config = AgentServiceConfig::new().with_tool_names(vec!["http".to_string()]);
    assert_eq!(config.tool_selection.names, Some(vec!["http".to_string()]));
    assert!(config.tool_selection.builtins.is_empty());

    let config = AgentServiceConfig::new().with_tool_names_or_defaults(None);
    assert!(config.tool_selection.use_all_default_tools);

    let config = AgentServiceConfig::new().with_all_default_tools();
    assert!(config.tool_selection.use_all_default_tools);
}

#[test]
fn session_autosave_dir_builder() {
    let config = AgentServiceConfig::new();
    assert!(config.session_autosave_dir.is_none());
    let dir = std::path::PathBuf::from("/tmp/caliche_sessions");
    let config = AgentServiceConfig::new().with_session_autosave_dir(dir.clone());
    assert_eq!(config.session_autosave_dir, Some(dir));
}

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
}

#[tokio::test]
async fn from_config_supports_typed_and_named_tool_selection() {
    let service = AgentService::from_config(test_config().with_tool_selection(
        ToolSelection::builtins(vec![BuiltInTool::Calculator]).with_name("http"),
    ))
    .await
    .expect("agent service should build from typed tool config");

    let tool_names: Vec<String> = service
        .list_tools()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();

    assert!(tool_names.iter().any(|name| name == "calculator"));
    assert!(tool_names.iter().any(|name| name == "http"));
    assert_eq!(service.provider(), ProviderType::Anthropic);
    assert_eq!(service.model(), "claude-3-5-sonnet-20241022");
}

#[tokio::test]
async fn from_config_uses_default_tools_when_requested() {
    let service = AgentService::from_config(test_config().with_all_default_tools())
        .await
        .expect("agent service should build with default tools");

    let tool_names: Vec<String> = service
        .list_tools()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();

    assert!(tool_names.iter().any(|name| name == "fs"));
    assert!(tool_names.iter().any(|name| name == "bash"));
    assert!(
        tool_names.iter().any(|name| name == "web_fetch"),
        "coding defaults should include web_fetch"
    );
    assert!(
        !tool_names.iter().any(|name| name == "web_search"),
        "coding defaults should not include web_search"
    );
    assert!(
        !tool_names.iter().any(|name| name == "http"),
        "coding defaults should not include http"
    );
    assert!(
        !tool_names.iter().any(|name| name == "workspace"),
        "coding defaults should not include workspace"
    );
    assert!(
        !tool_names.iter().any(|name| name == "execute_python"),
        "coding defaults should not include execute_python"
    );
}
