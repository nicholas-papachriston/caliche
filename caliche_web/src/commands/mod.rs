//! Slash command parsing and execution for the UI.
//!
//! This module provides slash command support similar to the CLI,
//! routing commands to appropriate API endpoints.

mod executor;

pub use executor::execute_command;

// Re-export InteractionMode from types to avoid duplication
pub use crate::api::types::InteractionMode;
use caliche_contracts::slash::SlashSurface;
use caliche_contracts::slash::{UI_SLASH_COMMAND_GROUPS, UI_SLASH_COMMANDS, find_slash_command};
use caliche_contracts::slash_intent::{SlashIntent, parse_slash_intent};

/// Available slash commands with descriptions
pub const SLASH_COMMANDS: &[(&str, &str)] = UI_SLASH_COMMANDS;

const COMMAND_GROUPS: &[(&str, &[&str])] = UI_SLASH_COMMAND_GROUPS;

/// Parsed slash command
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Show help for available commands
    Help,
    /// Set interaction mode
    SetMode(InteractionMode),
    /// Show current mode
    ShowMode,
    /// Undo last file change
    Undo,
    /// Redo most recently undone file change
    Redo,
    /// Create a checkpoint with a label
    Checkpoint(String),
    /// Rollback to a checkpoint ID
    Rollback(String),
    /// Clear conversation history
    Clear,
    /// Show current context
    ShowContext,
    /// Track a file in context
    TrackFile(String),
    /// List available providers
    ListProviders,
    /// Switch to a provider
    SwitchProvider(String),
    /// Set model
    SetModel(String),
    /// List available tools
    ListTools,
    /// Show session cost
    ShowCost,
    /// Show session stats
    ShowStats,
    // Memory
    /// Show memory statistics
    MemoryStats,
    /// Search session memory
    MemorySearch(String),
    /// Clear session memory
    MemoryClear,
    // Session state
    /// Show diff of changed files
    ShowDiff,
    /// View plan document
    ShowPlan,
    /// Revert the most recent recorded change set
    RevertLast,
    /// Open local saved-conversation history (browser tab)
    ShowHistory,
    /// Explain CLI `doctor` vs Settings in the UI
    DoctorHelp,
    /// Explain CLI `init` for AGENTS.md
    InitHelp,
    /// Point users to History tab to resume; optional args ignored in web UI
    ResumeHelp,
    /// Explain exporting transcripts (CLI); optional path arg ignored in web UI
    ExportHelp,
    /// CLI command that this console does not run.
    CliOnly(&'static str),
    // Reasoning
    /// List reasoning patterns
    ListPatterns,
    /// Use ReAct reasoning
    ReactReasoning(String),
    /// Use Plan-Execute reasoning
    PlanExecuteReasoning(String),
    /// Use Tree of Thoughts reasoning
    TreeOfThoughts(String),
    // Sub-agents
    /// Spawn a sub-agent with a task description
    SpawnAgent(String),
    /// Switch to the fleet tab
    ShowAgents,
    /// Switch to the settings tab
    ShowSettings,
    /// Switch to the chat tab
    ShowChat,
    /// Switch to the documents tab
    ShowDocuments,
    /// Switch to the plugins tab
    ShowPlugins,
    /// Switch to the reasoning tab
    ShowReasoning,
}

/// Result of executing a command
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// Whether the command was successful
    pub success: bool,
    /// Message to display to the user
    pub message: String,
}

impl CommandResult {
    /// Create a successful result
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
        }
    }
}

/// Check if input starts with a slash command
pub fn is_command(input: &str) -> bool {
    input.trim().starts_with('/')
}

/// Parse a slash command from input
pub fn parse_command(input: &str) -> Result<Command, String> {
    let input = input.trim();

    if !input.starts_with('/') {
        return Err("Not a command".to_string());
    }

    if let Some(intent) = parse_slash_intent(input) {
        match intent {
            SlashIntent::Plan => return Ok(Command::SetMode(InteractionMode::Plan)),
            SlashIntent::Ask => return Ok(Command::SetMode(InteractionMode::Ask)),
            SlashIntent::Agent { .. } => return Ok(Command::SetMode(InteractionMode::Agent)),
            SlashIntent::Mode => return Ok(Command::ShowMode),
            SlashIntent::ShowPlan => return Ok(Command::ShowPlan),
            SlashIntent::Resume { .. } => return Ok(Command::ResumeHelp),
            SlashIntent::Attach { .. } => {
                // Attach is CLI-only; fall through to surface-specific matching.
            }
        }
    }

    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        return Err("Empty command".to_string());
    }

    let raw_cmd = parts[0].to_lowercase();
    let cmd = find_slash_command(&raw_cmd, SlashSurface::Ui)
        .map_or(raw_cmd.as_str(), |spec| spec.primary_name);
    let args = &parts[1..];

    match cmd {
        "/help" | "/?" => Ok(Command::Help),

        // Session commands
        "/undo" => Ok(Command::Undo),
        "/redo" => Ok(Command::Redo),
        "/checkpoint" => {
            if args.is_empty() {
                Err("Usage: /checkpoint <label>".to_string())
            } else {
                Ok(Command::Checkpoint(args.join(" ")))
            }
        }
        "/rollback" => {
            if args.is_empty() {
                Err("Usage: /rollback <checkpoint_id>".to_string())
            } else {
                Ok(Command::Rollback(args[0].to_string()))
            }
        }
        "/clear" => Ok(Command::Clear),
        "/context" => Ok(Command::ShowContext),
        "/track" => {
            if args.is_empty() {
                Err("Usage: /track <file_path>".to_string())
            } else {
                Ok(Command::TrackFile(args[0].to_string()))
            }
        }

        // Provider commands
        "/providers" => Ok(Command::ListProviders),
        "/provider" => {
            if args.is_empty() {
                Err("Usage: /provider <provider_name>".to_string())
            } else {
                Ok(Command::SwitchProvider(args[0].to_string()))
            }
        }
        "/model" => {
            if args.is_empty() {
                Err("Usage: /model <model_name>".to_string())
            } else {
                Ok(Command::SetModel(args.join(" ")))
            }
        }
        "/tools" => Ok(Command::ListTools),

        // Cost and stats
        "/cost" => Ok(Command::ShowCost),
        "/stats" => Ok(Command::ShowStats),

        // Memory commands
        "/memory" => Ok(Command::MemoryStats),
        "/memory-search" => {
            if args.is_empty() {
                Err("Usage: /memory-search <query>".to_string())
            } else {
                Ok(Command::MemorySearch(args.join(" ")))
            }
        }
        "/memory-clear" => Ok(Command::MemoryClear),

        // Session state
        "/diff" => Ok(Command::ShowDiff),

        "/revert" => Ok(Command::RevertLast),
        "/sessions" | "/history" => Ok(Command::ShowHistory),
        "/doctor" => Ok(Command::DoctorHelp),
        "/init" => Ok(Command::InitHelp),

        "/export" => Ok(Command::ExportHelp),
        "/share" => Ok(Command::CliOnly("/share")),
        "/unshare" => Ok(Command::CliOnly("/unshare")),
        "/suspend" => Ok(Command::CliOnly("/suspend")),

        // Sub-agent commands
        "/spawn" => {
            if args.is_empty() {
                Err("Usage: /spawn <task description>".to_string())
            } else {
                Ok(Command::SpawnAgent(args.join(" ")))
            }
        }
        "/agents" => Ok(Command::ShowAgents),
        "/settings" => Ok(Command::ShowSettings),
        "/chat" => Ok(Command::ShowChat),
        "/documents" | "/docs" => Ok(Command::ShowDocuments),
        "/plugins" => Ok(Command::ShowPlugins),
        "/reasoning" => Ok(Command::ShowReasoning),

        // Reasoning patterns
        "/patterns" => Ok(Command::ListPatterns),
        "/react" => {
            if args.is_empty() {
                Err("Usage: /react <prompt>".to_string())
            } else {
                Ok(Command::ReactReasoning(args.join(" ")))
            }
        }
        "/plan-execute" => {
            if args.is_empty() {
                Err("Usage: /plan-execute <prompt>".to_string())
            } else {
                Ok(Command::PlanExecuteReasoning(args.join(" ")))
            }
        }
        "/tree-of-thoughts" => {
            if args.is_empty() {
                Err("Usage: /tree-of-thoughts <prompt>".to_string())
            } else {
                Ok(Command::TreeOfThoughts(args.join(" ")))
            }
        }

        _ => Err(match suggest_command(cmd) {
            Some(suggested) => format!(
                "Unknown command: {}. Did you mean {}? Type /help for available commands.",
                cmd, suggested
            ),
            None => format!(
                "Unknown command: {}. Type /help for available commands.",
                cmd
            ),
        }),
    }
}

fn normalize_command_name(command: &str) -> &str {
    find_slash_command(command, SlashSurface::Ui).map_or(command, |spec| spec.primary_name)
}

fn suggest_command(unknown: &str) -> Option<&'static str> {
    let unknown = normalize_command_name(unknown);
    let candidates: std::collections::HashSet<&str> = SLASH_COMMANDS
        .iter()
        .map(|(cmd, _)| normalize_command_name(cmd))
        .collect();

    let mut matches: Vec<(&str, usize)> = candidates
        .into_iter()
        .filter_map(|candidate| {
            let distance = edit_distance(unknown, candidate);
            let prefix_match = unknown.starts_with(candidate) || candidate.starts_with(unknown);
            let score = if prefix_match {
                distance.saturating_sub(10)
            } else {
                distance
            };
            (score <= 3 && candidate != unknown).then_some((candidate, score))
        })
        .collect();

    matches.sort_by_key(|(_, score)| *score);
    let best_score = matches.first().map(|(_, score)| *score)?;
    let best_matches: Vec<_> = matches
        .into_iter()
        .filter(|(_, score)| *score == best_score)
        .collect();

    (best_matches.len() == 1).then_some(best_matches[0].0)
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];

    for (index, value) in dp.iter_mut().enumerate() {
        value[0] = index;
    }
    for index in 0..=b.len() {
        dp[0][index] = index;
    }

    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[a.len()][b.len()]
}

/// Generate help text for all commands
pub fn generate_help_text() -> String {
    let mut help = String::from("### Available Commands\n\n");
    for (index, (group_name, commands)) in COMMAND_GROUPS.iter().enumerate() {
        if index > 0 {
            help.push('\n');
        }
        help.push_str(&format!("#### {}\n\n", group_name));
        for (cmd, desc) in SLASH_COMMANDS
            .iter()
            .filter(|(cmd, _)| commands.contains(cmd))
        {
            help.push_str(&format!("- `{}` - {}\n", cmd, desc));
        }
    }

    help
}
