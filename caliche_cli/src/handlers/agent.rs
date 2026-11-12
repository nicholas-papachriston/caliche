//! Agent profile command handlers (list, show).

use crate::commands::AgentAction;
use crate::config::ConfigManager;
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use crate::utils::find_git_root;
use anyhow::Result;
use async_trait::async_trait;
use caliche::agent::profile::{
    AgentProfile, ProfileDir, default_search_path, load_profiles_from_search_path,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Handler for agent profile commands.
pub struct AgentHandler;

#[async_trait]
impl CommandHandler for AgentHandler {
    type Args = AgentAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_agent_command(args)
            .await
            .map_err(|e| CliError::Config(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "agent"
    }

    fn description(&self) -> &'static str {
        "List or show named agent profiles"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// Build profile search path: default dirs + config.agents_dirs.
async fn profile_search_path() -> Result<Vec<ProfileDir>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let git_root = find_git_root(&cwd);
    let mut paths = default_search_path(git_root.as_ref());
    let mut manager = ConfigManager::with_standard_sources();
    if let Ok(config) = manager.load().await {
        if let Some(ref dirs) = config.agents_dirs {
            paths.extend(dirs.iter().cloned().map(ProfileDir::user));
        }
    }
    Ok(paths)
}

/// Load the profile registry (search path).
async fn load_registry() -> Result<HashMap<String, AgentProfile>> {
    let path = profile_search_path().await?;
    load_profiles_from_search_path(&path).map_err(Into::into)
}

/// Load a single profile by name (for use when --agent <name> is set).
pub async fn load_profile_by_name(name: &str) -> Result<Option<AgentProfile>> {
    let registry = load_registry().await?;
    Ok(registry.get(name).cloned())
}

/// Return sorted names of profiles with mode "primary" (for /agent next cycle).
pub async fn list_primary_profile_names() -> Result<Vec<String>> {
    let registry = load_registry().await?;
    let mut names: Vec<String> = registry
        .iter()
        .filter(|(_, p)| p.is_primary())
        .map(|(n, _)| n.clone())
        .collect();
    names.sort();
    Ok(names)
}

pub async fn handle_agent_command(action: AgentAction) -> Result<()> {
    match action {
        AgentAction::List => handle_agent_list().await,
        AgentAction::Show { name } => handle_agent_show(&name).await,
    }
}

async fn handle_agent_list() -> Result<()> {
    let registry = load_registry().await?;
    if registry.is_empty() {
        println!("No agent profiles found.");
        println!("Add .md files to .caliche/agents/ or ~/.config/caliche/agents/");
        return Ok(());
    }
    println!("Agent profiles:\n");
    let mut names: Vec<_> = registry.keys().collect();
    names.sort();
    for name in names {
        let Some(p) = registry.get(name) else {
            continue;
        };
        let mode = p.mode.as_deref().unwrap_or("primary");
        let desc = p.description.as_deref().unwrap_or("(no description)");
        println!("  {}  [{}]  {}", name, mode, desc);
    }
    Ok(())
}

async fn handle_agent_show(name: &str) -> Result<()> {
    let registry = load_registry().await?;
    let p = registry
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("Profile not found: {}", name))?;
    println!("Profile: {}\n", name);
    if let Some(d) = p.description.as_deref() {
        println!("Description: {}", d);
    }
    if let Some(m) = p.mode.as_deref() {
        println!("Mode: {}", m);
    }
    if let Some(m) = p.model.as_deref() {
        println!("Model: {}", m);
    }
    if let Some(t) = p.temperature {
        println!("Temperature: {}", t);
    }
    if let Some(s) = p.steps {
        println!("Steps: {}", s);
    }
    if let Some(c) = p.color.as_deref() {
        println!("Color: {}", c);
    }
    if let Some(tools) = p.enabled_tools() {
        println!("Tools: {}", tools.join(", "));
    }
    if let Some(body) = p.system_prompt() {
        println!("\nSystem prompt:\n---\n{}\n---", body);
    }
    Ok(())
}
