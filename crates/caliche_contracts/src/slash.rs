use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlashCommandIdDto {
    Help,
    Welcome,
    Plan,
    Ask,
    Agent,
    Mode,
    ShowPlan,
    Undo,
    Redo,
    Share,
    Unshare,
    Checkpoint,
    Rollback,
    Clear,
    Save,
    Load,
    Sessions,
    Resume,
    Suspend,
    Status,
    Context,
    Track,
    Add,
    Providers,
    Provider,
    Model,
    Models,
    ContextWindow,
    LocalModel,
    ImageGeneration,
    ImageGenerators,
    ArtifactDir,
    Tools,
    Cost,
    Budget,
    Stats,
    Memory,
    Learning,
    Diff,
    Export,
    Doctor,
    Init,
    Reindex,
    Check,
    Spawn,
    Agents,
    Settings,
    Chat,
    Documents,
    Plugins,
    Reasoning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashSurface {
    Cli,
    Ui,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashArgumentKind {
    None,
    Optional,
    Required,
    Freeform,
}

pub struct SlashCommandSpec {
    pub id: SlashCommandIdDto,
    pub primary_name: &'static str,
    pub aliases: &'static [&'static str],
    pub group: &'static str,
    pub description: &'static str,
    pub cli_supported: bool,
    pub ui_supported: bool,
    pub argument_kind: SlashArgumentKind,
}

pub const SLASH_COMMAND_SPECS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        id: SlashCommandIdDto::Help,
        primary_name: "/help",
        aliases: &["/?"],
        group: "General",
        description: "Show available commands",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Welcome,
        primary_name: "/welcome",
        aliases: &[],
        group: "General",
        description: "Show the welcome message again",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Plan,
        primary_name: "/plan",
        aliases: &[],
        group: "Mode",
        description: "Enter planning mode - interview and document requirements",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Ask,
        primary_name: "/ask",
        aliases: &[],
        group: "Mode",
        description: "Enter ask mode - answer questions (read-only)",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Agent,
        primary_name: "/agent",
        aliases: &[],
        group: "Mode",
        description: "Enter agent mode - full capabilities",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Mode,
        primary_name: "/mode",
        aliases: &[],
        group: "Mode",
        description: "Show current interaction mode",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::ShowPlan,
        primary_name: "/show-plan",
        aliases: &["/showplan"],
        group: "Session State",
        description: "View the current plan document",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Undo,
        primary_name: "/undo",
        aliases: &[],
        group: "Session",
        description: "Undo last file change",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Redo,
        primary_name: "/redo",
        aliases: &[],
        group: "Session",
        description: "Redo the most recently undone file change",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Share,
        primary_name: "/share",
        aliases: &[],
        group: "Session",
        description: "Create a redacted share link for this session",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Unshare,
        primary_name: "/unshare",
        aliases: &[],
        group: "Session",
        description: "Remove share links for this session",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Checkpoint,
        primary_name: "/checkpoint",
        aliases: &[],
        group: "Session",
        description: "Create a named checkpoint",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Required,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Rollback,
        primary_name: "/rollback",
        aliases: &[],
        group: "Session",
        description: "Rollback to a checkpoint",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Required,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Suspend,
        primary_name: "/suspend",
        aliases: &[],
        group: "Session State",
        description: "Persist the session and drop in-memory state (resume after restart)",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Resume,
        primary_name: "/resume",
        aliases: &[],
        group: "Session State",
        description: "Restore a suspended or saved session by id",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Clear,
        primary_name: "/clear",
        aliases: &[],
        group: "Session",
        description: "Clear conversation history",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Providers,
        primary_name: "/providers",
        aliases: &[],
        group: "Provider Commands",
        description: "List available providers",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Provider,
        primary_name: "/provider",
        aliases: &[],
        group: "Provider Commands",
        description: "Switch to a provider",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Required,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Model,
        primary_name: "/model",
        aliases: &[],
        group: "Provider Commands",
        description: "Set model for current provider",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Required,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::ContextWindow,
        primary_name: "/context-window",
        aliases: &["/contextwindow"],
        group: "Provider Commands",
        description: "Set or show context window override",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::LocalModel,
        primary_name: "/local-model",
        aliases: &[],
        group: "Provider Commands",
        description: "List, fit, download, or inspect local inference models",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Freeform,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::ImageGeneration,
        primary_name: "/image-generation",
        aliases: &["/image"],
        group: "Provider Commands",
        description: "Set or show image generation provider/model",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::ImageGenerators,
        primary_name: "/image-generators",
        aliases: &["/images"],
        group: "Provider Commands",
        description: "List image generation providers/models",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::ArtifactDir,
        primary_name: "/artifact-dir",
        aliases: &["/artifacts"],
        group: "Provider Commands",
        description: "Set or show generated artifact directory",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Tools,
        primary_name: "/tools",
        aliases: &[],
        group: "Provider Commands",
        description: "List available tools",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Learning,
        primary_name: "/learning",
        aliases: &["/learning-loop"],
        group: "Context Management",
        description: "Show or configure learning loop settings",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Freeform,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Sessions,
        primary_name: "/sessions",
        aliases: &["/history"],
        group: "Session State",
        description: "Browse sessions and saved conversations",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Optional,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Doctor,
        primary_name: "/doctor",
        aliases: &[],
        group: "Project",
        description: "Run or explain environment diagnostics",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Init,
        primary_name: "/init",
        aliases: &[],
        group: "Project",
        description: "Initialize project context",
        cli_supported: true,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Reindex,
        primary_name: "/reindex",
        aliases: &["/re-index"],
        group: "Project",
        description: "Clear workspace embeddings and re-index the current directory",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::None,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Check,
        primary_name: "/check",
        aliases: &["/verify"],
        group: "Project",
        description: "Run an explicit project check command",
        cli_supported: true,
        ui_supported: false,
        argument_kind: SlashArgumentKind::Freeform,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Spawn,
        primary_name: "/spawn",
        aliases: &[],
        group: "Sub-Agents",
        description: "Spawn a sub-agent to run a task in the background",
        cli_supported: false,
        ui_supported: true,
        argument_kind: SlashArgumentKind::Freeform,
    },
    SlashCommandSpec {
        id: SlashCommandIdDto::Settings,
        primary_name: "/settings",
        aliases: &[],
        group: "Navigation",
        description: "Open the settings tab",
        cli_supported: false,
        ui_supported: true,
        argument_kind: SlashArgumentKind::None,
    },
];

pub const UI_SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "Show available commands"),
    (
        "/spawn",
        "Spawn a sub-agent to run a task in the background",
    ),
    ("/agents", "Switch to the fleet tab"),
    ("/settings", "Open the settings tab (API URL, connection)"),
    ("/chat", "Switch to the chat tab"),
    ("/documents", "Open the documents tab"),
    ("/docs", "Same as /documents"),
    ("/plugins", "Open the plugins tab"),
    ("/reasoning", "Open the reasoning tab"),
    (
        "/plan",
        "Enter planning mode - interview and document requirements",
    ),
    ("/ask", "Enter ask mode - answer questions (read-only)"),
    ("/agent", "Enter agent mode - full capabilities"),
    ("/mode", "Show current interaction mode"),
    ("/undo", "Undo last file change"),
    ("/redo", "Redo the most recently undone file change"),
    ("/checkpoint", "Create a named checkpoint"),
    ("/rollback", "Rollback to a checkpoint"),
    ("/clear", "Clear conversation history"),
    ("/context", "Show current context"),
    ("/track", "Track a file in context"),
    ("/providers", "List available providers"),
    ("/provider", "Switch to a provider"),
    ("/model", "Set model for current provider"),
    ("/tools", "List available tools"),
    ("/cost", "Show session cost breakdown"),
    ("/stats", "Show usage statistics for current session"),
    ("/memory", "Show memory stats for current session"),
    ("/memory-search", "Search session memory"),
    ("/memory-clear", "Clear session memory"),
    ("/diff", "Show files changed in this session"),
    ("/show-plan", "View the current plan document (plan mode)"),
    (
        "/sessions",
        "Open the history tab (live sessions on this server)",
    ),
    ("/history", "Same as /sessions"),
    (
        "/resume",
        "Open the history tab and pick a live server session",
    ),
    ("/patterns", "List available reasoning patterns"),
    ("/react", "Use ReAct reasoning (reason + act loop)"),
    ("/plan-execute", "Use Plan-Execute reasoning"),
    ("/tree-of-thoughts", "Use Tree of Thoughts reasoning"),
];

pub const UI_SLASH_COMMAND_GROUPS: &[(&str, &[&str])] = &[
    ("General", &["/help", "/settings"]),
    (
        "Navigation",
        &["/chat", "/documents", "/docs", "/plugins", "/reasoning"],
    ),
    ("Mode Commands", &["/plan", "/ask", "/agent", "/mode"]),
    (
        "Session Commands",
        &[
            "/undo",
            "/redo",
            "/checkpoint",
            "/rollback",
            "/clear",
            "/context",
            "/track",
        ],
    ),
    (
        "Provider Commands",
        &["/providers", "/provider", "/model", "/tools"],
    ),
    ("Statistics", &["/cost", "/stats"]),
    ("Memory", &["/memory", "/memory-search", "/memory-clear"]),
    (
        "Session State",
        &["/diff", "/show-plan", "/sessions", "/history", "/resume"],
    ),
    (
        "Reasoning Patterns",
        &["/patterns", "/react", "/plan-execute", "/tree-of-thoughts"],
    ),
    ("Sub-Agents", &["/spawn", "/agents"]),
];

#[must_use]
pub fn find_slash_command(name: &str, surface: SlashSurface) -> Option<&'static SlashCommandSpec> {
    let name = name.trim();
    SLASH_COMMAND_SPECS.iter().find(|spec| {
        let supported = match surface {
            SlashSurface::Cli => spec.cli_supported,
            SlashSurface::Ui => spec.ui_supported,
        };
        supported && (spec.primary_name == name || spec.aliases.contains(&name))
    })
}

#[must_use]
pub fn all_slash_commands(surface: SlashSurface) -> Vec<&'static SlashCommandSpec> {
    SLASH_COMMAND_SPECS
        .iter()
        .filter(|spec| match surface {
            SlashSurface::Cli => spec.cli_supported,
            SlashSurface::Ui => spec.ui_supported,
        })
        .collect()
}

#[must_use]
pub fn slash_command_groups(
    surface: SlashSurface,
) -> Vec<(&'static str, Vec<&'static SlashCommandSpec>)> {
    let mut groups: Vec<(&'static str, Vec<&'static SlashCommandSpec>)> = Vec::new();
    for spec in all_slash_commands(surface) {
        if let Some((_, commands)) = groups.iter_mut().find(|(group, _)| *group == spec.group) {
            commands.push(spec);
        } else {
            groups.push((spec.group, vec![spec]));
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn showplan_alias_resolves_on_cli_and_ui() {
        let cli = find_slash_command("/showplan", SlashSurface::Cli)
            .expect("cli should resolve /showplan");
        let ui =
            find_slash_command("/showplan", SlashSurface::Ui).expect("ui should resolve /showplan");
        assert_eq!(cli.primary_name, "/show-plan");
        assert_eq!(ui.primary_name, "/show-plan");
        assert_eq!(cli.id, SlashCommandIdDto::ShowPlan);
    }

    #[test]
    fn ui_only_commands_are_absent_on_cli_surface() {
        assert!(find_slash_command("/spawn", SlashSurface::Cli).is_none());
        assert!(find_slash_command("/spawn", SlashSurface::Ui).is_some());
        assert!(find_slash_command("/settings", SlashSurface::Cli).is_none());
        assert!(find_slash_command("/settings", SlashSurface::Ui).is_some());
    }

    #[test]
    fn suspend_and_resume_are_registered() {
        let suspend =
            find_slash_command("/suspend", SlashSurface::Cli).expect("cli should resolve /suspend");
        assert_eq!(suspend.id, SlashCommandIdDto::Suspend);
        let resume =
            find_slash_command("/resume", SlashSurface::Cli).expect("cli should resolve /resume");
        assert_eq!(resume.id, SlashCommandIdDto::Resume);
    }

    #[test]
    fn find_slash_command_trims_whitespace() {
        let found = find_slash_command(" /help ", SlashSurface::Cli).expect("trim help");
        assert_eq!(found.primary_name, "/help");
    }
}
