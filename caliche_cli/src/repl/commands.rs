//! Slash command parsing and routing

use caliche_contracts::slash::{SlashSurface, find_slash_command};
use caliche_contracts::slash_intent::{SlashIntent, parse_slash_intent};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum ModeCommand {
    Plan,
    Ask,
    Agent(Option<String>),
    Mode,
    ShowPlan,
}

#[derive(Debug, Clone)]
pub enum HarnessCommand {
    Undo,
    Redo,
    Share,
    Unshare,
    Checkpoint(Option<String>),
    Rollback(Option<String>),
}

#[derive(Debug, Clone)]
pub enum SessionCommand {
    Clear,
    Save(Option<PathBuf>),
    Load(Option<PathBuf>),
    Sessions(Option<String>),
    Resume(Option<String>),
    Suspend(Option<String>),
    Status,
    Context,
    Track(Option<String>),
    Add(Option<String>),
    Recent,
    Related(Option<String>),
}

#[derive(Debug, Clone)]
pub enum ProviderCommand {
    Providers,
    Provider(Option<String>),
    Model(Option<String>),
    Models(Option<String>),
    ContextWindow(Option<String>),
    LocalModel(Vec<String>),
    SubagentProvider(Option<String>),
    SubagentModel(Option<String>),
    Embedding(Option<String>),
    Embeddings,
    ImageGeneration(Option<String>),
    ImageGenerators(Option<String>),
    ArtifactDir(Option<PathBuf>),
}

#[derive(Debug, Clone)]
pub enum UsageCommand {
    Tools,
    Cost,
    Budget,
    Stats,
}

#[derive(Debug, Clone)]
pub enum KnowledgeCommand {
    Skill(Vec<String>),
    Diff(Option<String>),
    Export {
        path: Option<String>,
        include_thinking: bool,
    },
    Compress,
    AutoCompact(Option<String>),
    Memory(Option<String>),
    Learning(Vec<String>),
    Remember(Option<String>),
    Recall(Option<String>),
    Knowledge,
    Forget(Option<String>),
}

#[derive(Debug, Clone)]
pub enum ProjectCommand {
    Changes,
    Init,
    Doctor,
    Reindex,
    Check(Vec<String>),
    Pr,
    Commit,
    Review,
    Lsp,
}

#[derive(Debug, Clone)]
pub enum SystemCommand {
    Compact,
    Verbose(Option<String>),
    Editor,
    Copy,
    Commands,
    Creds(Vec<String>),
    Attach(Vec<PathBuf>),
    Help(Option<String>),
    Welcome,
    Exit,
    Quit,
    Unknown(String),
}

/// Parsed slash command.
///
/// Variants are grouped so each group stays small enough for an exhaustive match.
#[derive(Debug, Clone)]
pub enum SlashCommand {
    Mode(ModeCommand),
    Harness(HarnessCommand),
    Session(SessionCommand),
    Provider(ProviderCommand),
    Usage(UsageCommand),
    Knowledge(KnowledgeCommand),
    Project(ProjectCommand),
    System(SystemCommand),
}

impl SlashCommand {
    /// Parse a slash command from input string
    pub fn parse(input: &str) -> Self {
        let input = input.trim();
        if !input.starts_with('/') {
            return SlashCommand::System(SystemCommand::Unknown(input.to_owned()));
        }

        if let Some(intent) = parse_slash_intent(input) {
            return match intent {
                SlashIntent::Plan => SlashCommand::Mode(ModeCommand::Plan),
                SlashIntent::Ask => SlashCommand::Mode(ModeCommand::Ask),
                SlashIntent::Agent { arg } => SlashCommand::Mode(ModeCommand::Agent(arg)),
                SlashIntent::Mode => SlashCommand::Mode(ModeCommand::Mode),
                SlashIntent::ShowPlan => SlashCommand::Mode(ModeCommand::ShowPlan),
                SlashIntent::Resume { id } => SlashCommand::Session(SessionCommand::Resume(id)),
                SlashIntent::Attach { paths } => SlashCommand::System(SystemCommand::Attach(
                    paths.into_iter().map(PathBuf::from).collect(),
                )),
            };
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts.first().copied().unwrap_or("");
        let command = find_slash_command(command, SlashSurface::Cli)
            .map_or(command, |spec| spec.primary_name);
        let args = parts.get(1..).unwrap_or(&[]);

        if let Some(parsed) = command_from_parts(command, args) {
            return parsed;
        }
        if let Some(corrected) = crate::input::resolve_slash_command(command) {
            let rebuilt = if args.is_empty() {
                corrected.to_owned()
            } else {
                format!("{corrected} {}", args.join(" "))
            };
            return Self::parse(&rebuilt);
        }
        SlashCommand::System(SystemCommand::Unknown(command.to_owned()))
    }

    /// Check if this command should exit the REPL
    pub fn is_exit(&self) -> bool {
        matches!(
            self,
            SlashCommand::System(SystemCommand::Exit | SystemCommand::Quit)
        )
    }
}

#[derive(Clone, Copy)]
enum Built {
    HarnessFirst(fn(Option<String>) -> HarnessCommand),
    SessionFirst(fn(Option<String>) -> SessionCommand),
    SessionPath(fn(Option<PathBuf>) -> SessionCommand),
    ProviderFirst(fn(Option<String>) -> ProviderCommand),
    ProviderJoined(fn(Option<String>) -> ProviderCommand),
    ProviderOwned(fn(Vec<String>) -> ProviderCommand),
    ProviderPath(fn(Option<PathBuf>) -> ProviderCommand),
    KnowledgeFirst(fn(Option<String>) -> KnowledgeCommand),
    KnowledgeOwned(fn(Vec<String>) -> KnowledgeCommand),
    ProjectOwned(fn(Vec<String>) -> ProjectCommand),
    SystemFirst(fn(Option<String>) -> SystemCommand),
    SystemOwned(fn(Vec<String>) -> SystemCommand),
}

fn command_from_parts(command: &str, args: &[&str]) -> Option<SlashCommand> {
    if let Some(parsed) = unit_command(command) {
        return Some(parsed);
    }
    if let Some(parsed) = special_command(command, args) {
        return Some(parsed);
    }
    let built = lookup(VALUED, command)?;
    Some(apply_built(*built, args))
}

fn lookup<'a, T>(table: &'a [(&str, T)], command: &str) -> Option<&'a T> {
    table
        .iter()
        .find_map(|(name, value)| (*name == command).then_some(value))
}

fn apply_built(built: Built, args: &[&str]) -> SlashCommand {
    match built {
        Built::HarnessFirst(build) => SlashCommand::Harness(build(first_string(args))),
        Built::SessionFirst(build) => SlashCommand::Session(build(first_string(args))),
        Built::SessionPath(build) => SlashCommand::Session(build(args.first().map(PathBuf::from))),
        Built::ProviderFirst(build) => SlashCommand::Provider(build(first_string(args))),
        Built::ProviderJoined(build) => SlashCommand::Provider(build(joined_arg(args))),
        Built::ProviderOwned(build) => SlashCommand::Provider(build(owned_args(args))),
        Built::ProviderPath(build) => {
            SlashCommand::Provider(build(args.first().map(PathBuf::from)))
        }
        Built::KnowledgeFirst(build) => SlashCommand::Knowledge(build(first_string(args))),
        Built::KnowledgeOwned(build) => SlashCommand::Knowledge(build(owned_args(args))),
        Built::ProjectOwned(build) => SlashCommand::Project(build(owned_args(args))),
        Built::SystemFirst(build) => SlashCommand::System(build(first_string(args))),
        Built::SystemOwned(build) => SlashCommand::System(build(owned_args(args))),
    }
}

fn first_string(args: &[&str]) -> Option<String> {
    args.first().map(|value| (*value).to_owned())
}

fn owned_args(args: &[&str]) -> Vec<String> {
    args.iter().copied().map(str::to_owned).collect()
}

fn joined_arg(args: &[&str]) -> Option<String> {
    if args.is_empty() {
        None
    } else {
        Some(args.join(" "))
    }
}

fn special_command(command: &str, args: &[&str]) -> Option<SlashCommand> {
    Some(match command {
        "/help" | "/?" => help_command(args),
        "/remember" => SlashCommand::Knowledge(KnowledgeCommand::Remember(Some(args.join(" ")))),
        "/recall" => SlashCommand::Knowledge(KnowledgeCommand::Recall(Some(args.join(" ")))),
        "/gateway" => gateway_command(args),
        "/export" => export_command(args),
        "/thinking" => SlashCommand::System(SystemCommand::Verbose(Some("thinking".to_owned()))),
        _ => return None,
    })
}

fn help_command(args: &[&str]) -> SlashCommand {
    let arg = args.first().map(|value| value.to_lowercase());
    SlashCommand::System(SystemCommand::Help(match arg.as_deref() {
        Some("all") => Some("all".to_owned()),
        Some("quick") => Some("quick".to_owned()),
        _ => None,
    }))
}

fn export_command(args: &[&str]) -> SlashCommand {
    let include_thinking = args.contains(&"--include-thinking");
    let path = args
        .iter()
        .find(|arg| **arg != "--include-thinking")
        .map(|arg| (*arg).to_owned());
    SlashCommand::Knowledge(KnowledgeCommand::Export {
        path,
        include_thinking,
    })
}

fn gateway_command(args: &[&str]) -> SlashCommand {
    let mut gateway_args = Vec::with_capacity(args.len() + 1);
    gateway_args.push("gateway".to_owned());
    gateway_args.extend(args.iter().copied().map(str::to_owned));
    SlashCommand::System(SystemCommand::Creds(gateway_args))
}

fn unit_command(command: &str) -> Option<SlashCommand> {
    lookup(UNITS, command).cloned()
}

const UNITS: &[(&str, SlashCommand)] = &[
    ("/undo", SlashCommand::Harness(HarnessCommand::Undo)),
    ("/revert", SlashCommand::Harness(HarnessCommand::Undo)),
    ("/redo", SlashCommand::Harness(HarnessCommand::Redo)),
    ("/share", SlashCommand::Harness(HarnessCommand::Share)),
    ("/unshare", SlashCommand::Harness(HarnessCommand::Unshare)),
    ("/clear", SlashCommand::Session(SessionCommand::Clear)),
    ("/status", SlashCommand::Session(SessionCommand::Status)),
    ("/context", SlashCommand::Session(SessionCommand::Context)),
    ("/recent", SlashCommand::Session(SessionCommand::Recent)),
    (
        "/providers",
        SlashCommand::Provider(ProviderCommand::Providers),
    ),
    (
        "/embeddings",
        SlashCommand::Provider(ProviderCommand::Embeddings),
    ),
    ("/tools", SlashCommand::Usage(UsageCommand::Tools)),
    ("/cost", SlashCommand::Usage(UsageCommand::Cost)),
    ("/budget", SlashCommand::Usage(UsageCommand::Budget)),
    ("/stats", SlashCommand::Usage(UsageCommand::Stats)),
    (
        "/compress",
        SlashCommand::Knowledge(KnowledgeCommand::Compress),
    ),
    (
        "/knowledge",
        SlashCommand::Knowledge(KnowledgeCommand::Knowledge),
    ),
    ("/changes", SlashCommand::Project(ProjectCommand::Changes)),
    ("/init", SlashCommand::Project(ProjectCommand::Init)),
    ("/doctor", SlashCommand::Project(ProjectCommand::Doctor)),
    ("/reindex", SlashCommand::Project(ProjectCommand::Reindex)),
    ("/re-index", SlashCommand::Project(ProjectCommand::Reindex)),
    ("/pr", SlashCommand::Project(ProjectCommand::Pr)),
    ("/commit", SlashCommand::Project(ProjectCommand::Commit)),
    ("/review", SlashCommand::Project(ProjectCommand::Review)),
    ("/lsp", SlashCommand::Project(ProjectCommand::Lsp)),
    ("/compact", SlashCommand::System(SystemCommand::Compact)),
    ("/editor", SlashCommand::System(SystemCommand::Editor)),
    ("/copy", SlashCommand::System(SystemCommand::Copy)),
    ("/commands", SlashCommand::System(SystemCommand::Commands)),
    ("/welcome", SlashCommand::System(SystemCommand::Welcome)),
    ("/exit", SlashCommand::System(SystemCommand::Exit)),
    ("/quit", SlashCommand::System(SystemCommand::Quit)),
];

const VALUED: &[(&str, Built)] = &[
    (
        "/checkpoint",
        Built::HarnessFirst(HarnessCommand::Checkpoint),
    ),
    ("/rollback", Built::HarnessFirst(HarnessCommand::Rollback)),
    ("/sessions", Built::SessionFirst(SessionCommand::Sessions)),
    ("/suspend", Built::SessionFirst(SessionCommand::Suspend)),
    ("/track", Built::SessionFirst(SessionCommand::Track)),
    ("/add", Built::SessionFirst(SessionCommand::Add)),
    ("/related", Built::SessionFirst(SessionCommand::Related)),
    ("/save", Built::SessionPath(SessionCommand::Save)),
    ("/load", Built::SessionPath(SessionCommand::Load)),
    ("/provider", Built::ProviderFirst(ProviderCommand::Provider)),
    ("/model", Built::ProviderFirst(ProviderCommand::Model)),
    ("/models", Built::ProviderFirst(ProviderCommand::Models)),
    (
        "/context-window",
        Built::ProviderFirst(ProviderCommand::ContextWindow),
    ),
    (
        "/contextwindow",
        Built::ProviderFirst(ProviderCommand::ContextWindow),
    ),
    (
        "/subagent-provider",
        Built::ProviderFirst(ProviderCommand::SubagentProvider),
    ),
    (
        "/subagent-model",
        Built::ProviderFirst(ProviderCommand::SubagentModel),
    ),
    (
        "/image-generators",
        Built::ProviderFirst(ProviderCommand::ImageGenerators),
    ),
    (
        "/images",
        Built::ProviderFirst(ProviderCommand::ImageGenerators),
    ),
    (
        "/embedding",
        Built::ProviderJoined(ProviderCommand::Embedding),
    ),
    (
        "/image-generation",
        Built::ProviderJoined(ProviderCommand::ImageGeneration),
    ),
    (
        "/image",
        Built::ProviderJoined(ProviderCommand::ImageGeneration),
    ),
    (
        "/local-model",
        Built::ProviderOwned(ProviderCommand::LocalModel),
    ),
    (
        "/local-models",
        Built::ProviderOwned(ProviderCommand::LocalModel),
    ),
    (
        "/artifact-dir",
        Built::ProviderPath(ProviderCommand::ArtifactDir),
    ),
    (
        "/artifacts",
        Built::ProviderPath(ProviderCommand::ArtifactDir),
    ),
    ("/diff", Built::KnowledgeFirst(KnowledgeCommand::Diff)),
    (
        "/autocompact",
        Built::KnowledgeFirst(KnowledgeCommand::AutoCompact),
    ),
    ("/memory", Built::KnowledgeFirst(KnowledgeCommand::Memory)),
    ("/forget", Built::KnowledgeFirst(KnowledgeCommand::Forget)),
    ("/skill", Built::KnowledgeOwned(KnowledgeCommand::Skill)),
    (
        "/learning",
        Built::KnowledgeOwned(KnowledgeCommand::Learning),
    ),
    (
        "/learning-loop",
        Built::KnowledgeOwned(KnowledgeCommand::Learning),
    ),
    ("/check", Built::ProjectOwned(ProjectCommand::Check)),
    ("/verify", Built::ProjectOwned(ProjectCommand::Check)),
    ("/verbose", Built::SystemFirst(SystemCommand::Verbose)),
    ("/verbosity", Built::SystemFirst(SystemCommand::Verbose)),
    ("/creds", Built::SystemOwned(SystemCommand::Creds)),
];

#[cfg(test)]
mod tests {
    use super::{
        HarnessCommand, KnowledgeCommand, ModeCommand, ProjectCommand, ProviderCommand,
        SessionCommand, SlashCommand, SystemCommand,
    };
    use crate::commands::{Args, Commands};
    use clap::Parser;

    #[test]
    fn parse_resume_variants() {
        assert!(matches!(
            SlashCommand::parse("/resume"),
            SlashCommand::Session(SessionCommand::Resume(None))
        ));
        assert!(matches!(
            SlashCommand::parse("/resume last"),
            SlashCommand::Session(SessionCommand::Resume(Some(value))) if value == "last"
        ));
        assert!(matches!(
            SlashCommand::parse("/resume abc123"),
            SlashCommand::Session(SessionCommand::Resume(Some(value))) if value == "abc123"
        ));
        assert!(matches!(
            SlashCommand::parse("/suspend"),
            SlashCommand::Session(SessionCommand::Suspend(None))
        ));
        assert!(matches!(
            SlashCommand::parse("/suspend abc123"),
            SlashCommand::Session(SessionCommand::Suspend(Some(value))) if value == "abc123"
        ));
    }

    #[test]
    fn parse_mode_commands() {
        assert!(matches!(
            SlashCommand::parse("/plan"),
            SlashCommand::Mode(ModeCommand::Plan)
        ));
        assert!(matches!(
            SlashCommand::parse("/ask"),
            SlashCommand::Mode(ModeCommand::Ask)
        ));
        assert!(matches!(
            SlashCommand::parse("/mode"),
            SlashCommand::Mode(ModeCommand::Mode)
        ));
        assert!(matches!(
            SlashCommand::parse("/show-plan"),
            SlashCommand::Mode(ModeCommand::ShowPlan)
        ));
        assert!(matches!(
            SlashCommand::parse("/agent"),
            SlashCommand::Mode(ModeCommand::Agent(None))
        ));
    }

    #[test]
    fn parse_attach_via_shared_intent() {
        match SlashCommand::parse("/attach ./a.png ./notes.pdf") {
            SlashCommand::System(SystemCommand::Attach(paths)) => {
                assert_eq!(paths.len(), 2);
                assert!(paths[0].ends_with("a.png"));
                assert!(paths[1].ends_with("notes.pdf"));
            }
            other => panic!("expected Attach, got {other:?}"),
        }
    }

    #[test]
    fn parse_export_include_thinking_flag() {
        match SlashCommand::parse("/export notes.md --include-thinking") {
            SlashCommand::Knowledge(KnowledgeCommand::Export {
                path,
                include_thinking,
            }) => {
                assert_eq!(path.as_deref(), Some("notes.md"));
                assert!(include_thinking);
            }
            other => panic!("expected Export, got {other:?}"),
        }
        match SlashCommand::parse("/export --include-thinking") {
            SlashCommand::Knowledge(KnowledgeCommand::Export {
                path,
                include_thinking,
            }) => {
                assert!(path.is_none());
                assert!(include_thinking);
            }
            other => panic!("expected Export, got {other:?}"),
        }
        match SlashCommand::parse("/export notes.md") {
            SlashCommand::Knowledge(KnowledgeCommand::Export {
                path,
                include_thinking,
            }) => {
                assert_eq!(path.as_deref(), Some("notes.md"));
                assert!(!include_thinking);
            }
            other => panic!("expected Export, got {other:?}"),
        }
    }

    #[test]
    fn parse_undo_aliases() {
        assert!(matches!(
            SlashCommand::parse("/undo"),
            SlashCommand::Harness(HarnessCommand::Undo)
        ));
        assert!(matches!(
            SlashCommand::parse("/revert"),
            SlashCommand::Harness(HarnessCommand::Undo)
        ));
    }

    #[test]
    fn parse_reindex_aliases() {
        assert!(matches!(
            SlashCommand::parse("/reindex"),
            SlashCommand::Project(ProjectCommand::Reindex)
        ));
        assert!(matches!(
            SlashCommand::parse("/re-index"),
            SlashCommand::Project(ProjectCommand::Reindex)
        ));
    }

    #[test]
    fn parse_check_command() {
        assert!(matches!(
            SlashCommand::parse("/check cargo check -p caliche"),
            SlashCommand::Project(ProjectCommand::Check(args))
                if args.iter().map(String::as_str).collect::<Vec<_>>()
                    == ["cargo", "check", "-p", "caliche"]
        ));
        assert!(matches!(
            SlashCommand::parse("/verify"),
            SlashCommand::Project(ProjectCommand::Check(args)) if args.is_empty()
        ));
    }

    #[test]
    fn shared_cli_slash_specs_resolve() {
        for spec in caliche_contracts::slash::all_slash_commands(
            caliche_contracts::slash::SlashSurface::Cli,
        ) {
            assert!(
                !matches!(
                    SlashCommand::parse(spec.primary_name),
                    SlashCommand::System(SystemCommand::Unknown(_))
                ),
                "{} should parse",
                spec.primary_name
            );
            for alias in spec.aliases {
                assert!(
                    !matches!(
                        SlashCommand::parse(alias),
                        SlashCommand::System(SystemCommand::Unknown(_))
                    ),
                    "{alias} should parse"
                );
            }
        }
    }

    #[test]
    fn project_and_git_slash_commands_have_clap_counterparts() {
        let pairs = [
            ("/doctor", ["caliche", "doctor"].as_slice()),
            ("/init", ["caliche", "init", "--no-ai"].as_slice()),
            ("/lsp", ["caliche", "lsp", "status"].as_slice()),
        ];

        for (slash, argv) in pairs {
            let parsed = Args::try_parse_from(argv).expect("clap command should parse");
            match (SlashCommand::parse(slash), parsed.command) {
                (SlashCommand::Project(ProjectCommand::Doctor), Some(Commands::Doctor))
                | (SlashCommand::Project(ProjectCommand::Init), Some(Commands::Init { .. }))
                | (SlashCommand::Project(ProjectCommand::Lsp), Some(Commands::Lsp { .. })) => {}
                (slash_command, command) => {
                    panic!("unexpected parity result: {slash_command:?} vs {command:?}");
                }
            }
        }
    }

    #[test]
    fn parse_verbose_levels() {
        assert!(matches!(
            SlashCommand::parse("/verbose"),
            SlashCommand::System(SystemCommand::Verbose(None))
        ));
        assert!(matches!(
            SlashCommand::parse("/verbose debug"),
            SlashCommand::System(SystemCommand::Verbose(Some(value))) if value == "debug"
        ));
        assert!(matches!(
            SlashCommand::parse("/verbosity quiet"),
            SlashCommand::System(SystemCommand::Verbose(Some(value))) if value == "quiet"
        ));
        assert!(matches!(
            SlashCommand::parse("/thinking"),
            SlashCommand::System(SystemCommand::Verbose(Some(value))) if value == "thinking"
        ));
        assert!(matches!(
            SlashCommand::parse("/verbose thinking"),
            SlashCommand::System(SystemCommand::Verbose(Some(value))) if value == "thinking"
        ));
    }

    #[test]
    fn valued_commands_keep_their_argument_shape() {
        assert!(matches!(
            SlashCommand::parse("/checkpoint alpha"),
            SlashCommand::Harness(HarnessCommand::Checkpoint(Some(name))) if name == "alpha"
        ));
        assert!(matches!(
            SlashCommand::parse("/save notes.json"),
            SlashCommand::Session(SessionCommand::Save(Some(path))) if path.ends_with("notes.json")
        ));
        assert!(matches!(
            SlashCommand::parse("/gateway status"),
            SlashCommand::System(SystemCommand::Creds(args)) if args == ["gateway", "status"]
        ));
        assert!(matches!(
            SlashCommand::parse("/remember a fact"),
            SlashCommand::Knowledge(KnowledgeCommand::Remember(Some(text))) if text == "a fact"
        ));
        assert!(matches!(
            SlashCommand::parse("/remember"),
            SlashCommand::Knowledge(KnowledgeCommand::Remember(Some(text))) if text.is_empty()
        ));
        assert!(matches!(
            SlashCommand::parse("/recall"),
            SlashCommand::Knowledge(KnowledgeCommand::Recall(Some(text))) if text.is_empty()
        ));
        assert!(matches!(
            SlashCommand::parse("/embedding openai small"),
            SlashCommand::Provider(ProviderCommand::Embedding(Some(spec))) if spec == "openai small"
        ));
    }
}
