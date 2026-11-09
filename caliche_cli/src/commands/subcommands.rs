use clap::Subcommand;

use super::config::ConfigAction;
use super::output::CompletionShell;

#[derive(Clone, Subcommand, Debug)]
pub enum Commands {
    /// Run the agent on a coding task until it is done
    #[command(display_order = 1)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche code \"Add error handling to the entry point\"\n    \
        caliche code --dry-run \"Refactor the auth module\"\n    \
        caliche code --accept-edits \"Fix all linter warnings\"\n    \
        caliche code --verify \"cargo test --lib command_ledger\" \"Harden verification reuse\"")]
    Code {
        /// Task description for the agent to complete
        task: String,
        /// Working directory (default: current directory)
        #[arg(long)]
        cwd: Option<String>,
        /// Tools to enable (comma-separated)
        #[arg(long)]
        tools: Option<String>,
        /// Maximum agent turns (default: 20)
        #[arg(long, default_value = "20")]
        max_turns: usize,
        /// Auto-accept file modifications
        #[arg(long)]
        accept_edits: bool,
        /// Show changes without applying
        #[arg(long)]
        dry_run: bool,
        /// Stream output as it happens
        #[arg(long)]
        stream: bool,
        /// Disable RepoMap codebase overview for this run
        #[arg(long)]
        no_repomap: bool,
        /// Override auto verification with an exact command (repeatable)
        #[arg(long = "verify", value_name = "COMMAND")]
        verify: Vec<String>,
        /// Skip post-task verification
        #[arg(long, conflicts_with = "verify")]
        no_verify: bool,
    },
    /// List or show named agent profiles (Markdown + YAML frontmatter).
    #[command(display_order = 2)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche agent list                 # List discovered profiles\n    \
        caliche agent show reviewer       # Show profile details")]
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },
    /// Configure provider credentials (API keys)
    #[command(display_order = 3)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche config setup              # Interactive setup wizard\n    \
        caliche config set openai --key <openai-api-key>\n    \
        caliche config list --configured  # Show configured providers\n    \
        caliche config default openai     # Set default provider")]
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// List, resume, export and delete saved sessions
    #[command(display_order = 4)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche session list              # List all sessions\n    \
        caliche session resume <id>       # Resume a session and reload its transcript\n    \
        caliche session export <id> --format markdown\n    \
        caliche session inspect-file ./snap.cls1   # Binary rkyv v1 snapshot (CLXSSN03)\n\n\
resume loads the saved transcript into the agent before the first prompt.\n\
When that transcript is too large, older turns become a summary and the latest\n\
messages plus the last tool actions stay in the model context.")]
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
    /// Run health diagnostics and check installation
    #[command(display_order = 5)]
    #[command(after_help = "\
Prints one key=value line for each setting.\n\
build_id is the git commit that the binary was built from.\n\
jwt_enabled and require_auth are false until you turn them on.\n\
listen_bind is the address that --bind and --port set, 127.0.0.1:3000 by default.\n\
open_bind_warn is true when that address listens on every interface and authentication is not required.\n\
default_provider_source names the step that chose the provider: flag, last, env or default.\n\
last means the provider and model saved by the REPL.\n\
The security defaults are described in SECURITY.md.")]
    Doctor,
    /// Show or set the deployment policy file.
    #[command(display_order = 6)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche policy show\n    \
        caliche policy show --json\n    \
        caliche policy set preset locked")]
    Policy {
        #[command(subcommand)]
        action: PolicyAction,
    },
    /// Create a redacted share URL for a remote session
    #[command(display_order = 5)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche share --session <id>\n    \
        caliche share --session <id> --visibility authenticated")]
    Share {
        /// Remote session id (defaults to the last local session id)
        #[arg(long)]
        session: Option<String>,
        /// Share visibility: public | authenticated
        #[arg(long)]
        visibility: Option<String>,
    },
    /// Revoke share links for a remote session
    #[command(display_order = 5)]
    Unshare {
        /// Remote session id (defaults to the last local session id)
        #[arg(long)]
        session: Option<String>,
    },
    /// Install and control the HTTP server as a user service (launchd or systemd)
    #[command(display_order = 5)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche daemon install --port 3000\n    \
        caliche daemon status\n    \
        caliche daemon stop\n    \
        caliche daemon uninstall")]
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// List and stop Caliche HTTP servers recorded by this user
    #[command(display_order = 5)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche server list\n    \
        caliche server status --port 3000\n    \
        caliche server stop --port 3060\n    \
        caliche server stop <id>\n\n\
Stop signals a process only when the record's pid is still caliche and is still listening on the recorded port. A dead pid is removed as a stale record and is not signaled. A pid or port that no longer matches is left running.")]
    Server {
        #[command(subcommand)]
        action: ServerAction,
    },
    /// Write or update the agent instructions file for this project
    #[command(display_order = 6)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche init                  # Model analysis and interactive questions (default)\n    \
        caliche init --force          # Skip confirmation prompts, overwrite existing\n    \
        caliche init --output docs/agent-context.md\n    \
        caliche init --no-ai          # Static template only, no API key required")]
    Init {
        /// Skip model analysis and write a static template instead (works offline)
        #[arg(long, default_value_t = false)]
        no_ai: bool,

        /// Output file path (default: AGENTS.md in current directory)
        #[arg(long)]
        output: Option<String>,

        /// Skip all confirmation prompts and overwrite existing files
        #[arg(long, short = 'f', default_value_t = false)]
        force: bool,
    },
    /// Language Server Protocol (LSP) management
    #[command(display_order = 7)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche lsp status                # Show LSP status\n    \
        caliche lsp install rust          # Install rust-analyzer\n    \
        caliche lsp list                  # List supported languages")]
    Lsp {
        #[command(subcommand)]
        action: LspAction,
    },
    /// Template management commands
    #[command(display_order = 7)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche template list             # List available templates\n    \
        caliche template show rag_agent   # Show template details\n    \
        caliche template init             # Interactive template creation")]
    Template {
        #[command(subcommand)]
        action: TemplateAction,
    },
    /// Testing and debugging commands
    #[command(display_order = 8)]
    Test {
        #[command(subcommand)]
        action: TestAction,
    },
    /// Evaluate agent outputs against a dataset
    #[command(display_order = 9)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche evaluate --dataset test.jsonl\n    \
        caliche evaluate --dataset test.jsonl --evaluator llm --criteria \"accuracy\" --provider ollama")]
    Evaluate {
        /// Path to dataset file (JSONL or JSON format)
        #[arg(short, long)]
        dataset: String,
        /// Evaluator type (exact_match, contains, regex, llm)
        #[arg(short, long, default_value = "exact_match")]
        evaluator: String,
        /// Regex pattern (required for regex evaluator)
        #[arg(long)]
        pattern: Option<String>,
        /// Evaluation criteria for the model evaluator
        #[arg(long)]
        criteria: Option<String>,
        /// Provider for the model evaluator (optional, uses the default if not set)
        #[arg(long)]
        provider: Option<String>,
        /// Model for the model evaluator (optional)
        #[arg(long)]
        model: Option<String>,
        /// Output format (json, yaml, text)
        #[arg(short, long, default_value = "text")]
        output_format: String,
    },
    /// Prompt/template management commands (alias for template commands)
    #[command(display_order = 10)]
    Prompt {
        #[command(subcommand)]
        action: PromptAction,
    },
    /// Generate shell completion scripts
    #[command(display_order = 11)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche completions bash > ~/.bash_completion.d/caliche\n    \
        caliche completions zsh > ~/.zfunc/_caliche\n    \
        caliche completions fish > ~/.config/fish/completions/caliche.fish\n    \
        caliche completions powershell -o caliche.ps1\n\n\
INSTALLATION:\n    \
    Bash:       Add to ~/.bashrc: source ~/.bash_completion.d/caliche\n    \
    Zsh:        Add ~/.zfunc to fpath, then compinit\n    \
    Fish:       Completions auto-loaded from ~/.config/fish/completions/\n    \
    PowerShell: Add to $PROFILE: . ./caliche.ps1")]
    Completions {
        /// Shell to generate completions for (bash, zsh, fish, powershell, elvish)
        #[arg(value_enum)]
        shell: CompletionShell,
        /// Output file path (default: stdout)
        #[arg(long, short = 'o')]
        output: Option<String>,
    },
    /// Generate Unix man pages
    #[command(display_order = 12)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche man generate                    # Print main man page to stdout\n    \
        caliche man generate --all              # Generate all man pages\n    \
        caliche man generate -o ./man           # Generate to directory\n    \
        caliche man install                     # Install to default location\n\n\
USAGE:\n    \
    # View man page directly\n    \
    caliche man generate | man -l -\n    \
    \n    \
    # After installation\n    \
    man caliche\n    \
    man caliche-config")]
    Man {
        #[command(subcommand)]
        action: ManAction,
    },
    /// Scheduled agent tasks (cron jobs)
    #[cfg(feature = "scheduling")]
    #[command(display_order = 13)]
    #[command(after_help = "EXAMPLES:\n    \
        caliche cron list                        # List scheduled jobs\n    \
        caliche cron add --name daily --schedule '0 9 * * *' --message 'Report'\n    \
        caliche cron remove <job-id>\n    \
        caliche cron run <job-id>               # Run once now\n    \
        caliche cron start                      # Start cron daemon")]
    Cron {
        #[command(subcommand)]
        action: CronAction,
    },
    /// Install and update skills from git
    #[command(display_order = 14)]
    Skills {
        #[command(subcommand)]
        action: SkillsAction,
    },
}

/// Install, update, and inspect downloaded skills.
#[derive(Clone, Subcommand, Debug)]
pub enum SkillsAction {
    /// Copy a skill from `owner/repo`, a GitHub URL, or a local directory.
    Install {
        source: String,
        /// Install under the user skill directory instead of this project.
        #[arg(long)]
        global: bool,
    },
    /// Show skill updates. Add `--apply` to write the ones you accept.
    Update {
        /// Installed skill names. Omit the list to review every installed skill.
        names: Vec<String>,
        /// Write the reviewed updates.
        #[arg(long)]
        apply: bool,
        /// Use the user skill directory.
        #[arg(long)]
        global: bool,
    },
    /// List installed skills and their lock hashes.
    List {
        /// Use the user skill directory.
        #[arg(long)]
        global: bool,
    },
}

/// `policy show` and `policy set preset`.
#[derive(Clone, Subcommand, Debug)]
pub enum PolicyAction {
    /// Print the deployment policy and the host capability panel.
    Show {
        /// Print the effective policy as JSON.
        #[arg(long)]
        json: bool,
        /// Host label for the capability panel. Default is this operating system.
        #[arg(long)]
        host: Option<String>,
        /// Policy file. Default is `CALICHE_POLICY_PATH` or the user config path.
        #[arg(long)]
        path: Option<String>,
    },
    /// Change the deployment file.
    Set {
        #[command(subcommand)]
        action: PolicySetAction,
    },
}

/// Values `policy set` can write.
#[derive(Clone, Subcommand, Debug)]
pub enum PolicySetAction {
    /// Set the preset and write `policy.toml`.
    Preset {
        /// `locked`, `review`, `standard`, `trusted`, or `danger`.
        name: String,
        /// Policy file. Default is `CALICHE_POLICY_PATH` or the user config path.
        #[arg(long)]
        path: Option<String>,
    },
}

impl Commands {
    /// Session resume is the only subcommand that should continue into the REPL.
    #[must_use]
    pub const fn continues_to_repl(&self) -> bool {
        matches!(
            self,
            Self::Session {
                action: SessionAction::Resume { .. }
            }
        )
    }
}

#[cfg(feature = "scheduling")]
#[derive(Clone, Subcommand, Debug)]
pub enum CronAction {
    /// List all scheduled jobs
    List,
    /// Add a new job
    Add {
        #[arg(long)]
        name: String,
        #[arg(long, help = "Cron expression (e.g. 0 9 * * *) or interval like 30m")]
        schedule: String,
        #[arg(long)]
        message: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        delivery_file: Option<String>,
    },
    /// Remove a job
    Remove { job_id: String },
    /// Show job status and last run
    Status { job_id: String },
    /// Show execution history
    Logs {
        job_id: String,
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Run a job once now
    Run { job_id: String },
    /// Start the cron service daemon
    Start,
}

#[derive(Clone, Subcommand, Debug)]
pub enum TemplateAction {
    /// List all available templates
    List {
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Show details for a specific template
    Show {
        /// Template ID (e.g., "rag_agent:1.0.0")
        template_id: String,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Create an agent from a template
    Use {
        /// Template ID (e.g., "rag_agent:1.0.0")
        template_id: String,
        /// Agent name (optional)
        #[arg(long)]
        name: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Validate a template file
    Validate {
        /// Path to template file
        template_path: String,
    },
    /// Create/register a new template
    Create {
        /// Path to template file
        template_path: String,
    },
    /// Interactive template creation wizard
    Init {
        /// Output path for the template file (optional)
        #[arg(long)]
        output: Option<String>,
    },
    /// Test a template with example inputs
    Test {
        /// Template ID (e.g., "rag_agent:1.0.0")
        template_id: String,
        /// Test inputs (comma-separated, optional - uses template examples if not provided)
        #[arg(long)]
        test_inputs: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Get a specific template (supports versioned lookups)
    Get {
        /// Template name or ID (e.g., "rag_agent" or "rag_agent:1.0.0")
        template_id: String,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Benchmark a template's performance
    Benchmark {
        /// Template ID (e.g., "rag_agent:1.0.0")
        template_id: String,
        /// Number of iterations
        #[arg(long, default_value = "10")]
        iterations: usize,
        /// Test input message (optional)
        #[arg(long)]
        test_input: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
}

#[derive(Clone, Subcommand, Debug)]
pub enum PromptAction {
    /// List all available prompts/templates
    List {
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Get a specific prompt/template (supports versioned lookups)
    Get {
        /// Template name or ID (e.g., "rag_agent" or "rag_agent:1.0.0")
        template_id: String,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
    /// Test a prompt/template with input
    Test {
        /// Template name or ID (e.g., "rag_agent" or "rag_agent:1.0.0")
        template_id: String,
        /// Input JSON for testing (e.g., '{"input": "Hello"}')
        #[arg(long)]
        input: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        dir: Option<String>,
    },
}

#[derive(Clone, Subcommand, Debug)]
pub enum TestAction {
    /// Run test suite against agent
    Run {
        /// Template ID to test (optional)
        #[arg(long)]
        template: Option<String>,
        /// Test file with scenarios (optional)
        #[arg(long)]
        test_file: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        template_dir: Option<String>,
    },
    /// Benchmark agent performance
    Benchmark {
        /// Template ID (required)
        #[arg(long)]
        template: String,
        /// Number of iterations
        #[arg(long, default_value = "10")]
        iterations: usize,
        /// Test input message (optional)
        #[arg(long)]
        test_input: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        template_dir: Option<String>,
    },
    /// Record interaction for replay
    Record {
        /// Output file path
        #[arg(long)]
        output: String,
        /// Template ID (optional)
        #[arg(long)]
        template: Option<String>,
        /// Template directory path (default: templates/)
        #[arg(long)]
        template_dir: Option<String>,
    },
    /// Replay recorded interaction
    Replay {
        /// Input file path
        #[arg(long)]
        input: String,
        /// Compare outputs with original
        #[arg(long)]
        compare: bool,
    },
}

/// Agent profile subcommands.
#[derive(Clone, Subcommand, Debug)]
pub enum AgentAction {
    /// List discovered agent profiles (name, description, mode).
    List,
    /// Show details for a named profile.
    Show {
        /// Profile name (filename stem, e.g. reviewer).
        name: String,
    },
}

/// Recorded HTTP server actions.
///
/// `daemon` installs the OS unit. `server` lists and stops every recorded
/// `caliche --server`, including ones the unit started.
#[derive(Clone, Subcommand, Debug)]
pub enum ServerAction {
    /// Show every recorded server and whether it is running or stale
    List,
    /// Show one recorded server, or every record when no id or port is given
    Status {
        /// Record id printed by `server list`
        id: Option<String>,
        /// Listen port
        #[arg(long)]
        port: Option<u16>,
    },
    /// Stop a recorded server by id or port, or remove a stale record
    Stop {
        /// Record id printed by `server list`
        id: Option<String>,
        /// Listen port
        #[arg(long)]
        port: Option<u16>,
    },
}

/// User daemon lifecycle actions (launchd on macOS, systemd --user on Linux).
#[derive(Clone, Subcommand, Debug)]
pub enum DaemonAction {
    /// Write and enable a user unit that runs `caliche --server`
    Install {
        /// HTTP listen port
        #[arg(long, default_value_t = 3000)]
        port: u16,
        /// Unit label (macOS LaunchAgent label or systemd unit basename)
        #[arg(long)]
        label: Option<String>,
        /// Working directory for the daemon process (default: current directory)
        #[arg(long)]
        cwd: Option<String>,
    },
    /// Remove the user unit
    Uninstall {
        #[arg(long)]
        label: Option<String>,
    },
    /// Start / load the installed unit
    Start {
        #[arg(long)]
        label: Option<String>,
        /// Port recorded for status text only when the unit was installed with this port
        #[arg(long)]
        port: Option<u16>,
    },
    /// Stop / unload the installed unit
    Stop {
        #[arg(long)]
        label: Option<String>,
    },
    /// Show install path and running state
    Status {
        #[arg(long)]
        label: Option<String>,
    },
}

/// LSP management actions
#[derive(Clone, Subcommand, Debug)]
pub enum LspAction {
    /// Show LSP status and available language servers
    Status,
    /// Install a language server
    Install {
        /// Language to install server for (rust, typescript, python, go)
        language: String,
    },
    /// List supported languages
    List,
    /// Enable LSP for the current session
    Enable,
    /// Disable LSP for the current session
    Disable,
}

/// Session management actions
#[derive(Clone, Subcommand, Debug)]
pub enum SessionAction {
    /// List all saved sessions
    List {
        /// Show detailed information for each session
        #[arg(long)]
        verbose: bool,
    },
    /// Resume a saved session and reload its transcript into the agent
    Resume {
        /// Session id, unique prefix, or "last"
        session_id: String,
    },
    /// Delete a saved session
    Delete {
        /// Session ID to delete
        session_id: String,
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },
    /// Rename a session
    Rename {
        /// Current session ID
        session_id: String,
        /// New name for the session
        new_name: String,
    },
    /// Show details for a specific session
    Show {
        /// Session ID to show
        session_id: String,
    },
    /// Export a session to a file
    Export {
        /// Session ID to export
        session_id: String,
        /// Output file path (defaults to session_id.json)
        #[arg(long)]
        output: Option<String>,
        /// Output format (json, markdown)
        #[arg(long, default_value = "json")]
        format: String,
        /// Keep provider chain-of-thought (`reasoning_content`) in the file.
        ///
        /// Omitted by default. Public `share` links stay text-only either way.
        /// Hiding thinking in the REPL does not drop it from the saved session;
        /// this flag is how a later review includes it in an export.
        #[arg(long)]
        include_thinking: bool,
    },
    /// Show aggregate usage statistics across all sessions
    Stats {
        /// Filter by provider
        #[arg(long)]
        provider: Option<String>,
        /// Filter by model
        #[arg(long)]
        model: Option<String>,
        /// Number of days to include (default: all)
        #[arg(long)]
        days: Option<u32>,
    },
    /// Print metadata for a binary session snapshot file (CLXSSN03, `.cls1`)
    InspectFile {
        /// Path to `.cls1` snapshot (rkyv v1)
        path: std::path::PathBuf,
        /// Include shared_state keys and longer conversation preview
        #[arg(long)]
        verbose: bool,
    },
}

/// Man page generation actions
#[derive(Clone, Subcommand, Debug)]
pub enum ManAction {
    /// Generate man page(s) to stdout or directory
    Generate {
        /// Output directory (default: stdout for single page)
        #[arg(long, short = 'o')]
        output: Option<String>,
        /// Generate man pages for all subcommands
        #[arg(long)]
        all: bool,
    },
    /// Install man pages to system location
    Install {
        /// Installation directory (default: /usr/local/share/man/man1 or platform equivalent)
        #[arg(long)]
        dir: Option<String>,
    },
}
