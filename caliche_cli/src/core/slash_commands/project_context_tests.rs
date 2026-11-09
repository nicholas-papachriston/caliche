//! A window change sizes the project context again in the REPL's base
//! prompt too, so later turns (mode changes, skills) keep the new section.

use super::SlashCommandExecutor;
use crate::core::state::InteractiveState;
use crate::session::CliSession;
use caliche::llm::ProviderType;
use caliche::traits::credentials::{Credentials, StaticCredentialProvider};
use caliche::{AgentService, AgentServiceConfig};
use std::sync::Arc;

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

#[tokio::test]
async fn context_window_resizes_the_project_context_in_the_base_prompt() {
    let mut service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(Some(ProviderType::OpenAI), Some("gpt-4o".to_string()))
            .with_tool_names(Vec::new())
            .with_credential_provider(Arc::new(StaticCredentialProvider::new(
                Credentials::ApiKey("test-key".to_string()),
            ))),
    )
    .await
    .expect("service");
    let section = service.budget_project_context(large_agents_md());
    let prompt = format!("You are Caliche.\n\n{section}");
    service.agent_mut().set_system_prompt(prompt.clone());
    let mut state = InteractiveState::new(
        CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
        "openai",
        "gpt-4o",
        prompt,
    );

    SlashCommandExecutor::handle_context_window(&mut state, &mut service, Some("32768"))
        .expect("context window");
    // A mode change rebuilds the prompt from the REPL's base prompt.
    SlashCommandExecutor::handle_plan(&mut state, &mut service)
        .await
        .expect("plan mode");

    assert!(
        state.original_system_prompt.ends_with(CUT_32K_END),
        "{}",
        state.original_system_prompt
    );
    let prompt = service.agent().system_prompt();
    assert!(prompt.contains(CUT_32K_END), "{prompt}");
    assert!(!prompt.contains("rule 1999"), "{prompt}");
}
