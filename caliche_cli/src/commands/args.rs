use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Opt-in runtime profile for the default agent (REPL, one-shot, `--server`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum RuntimeProfile {
    /// `fs` and `bash`, short system prompt, no MCP or skill discovery.
    Lean,
}

use super::output::{ChainType, OutputFormat, RagCliFusionMode, RagCliSearchMode};
use super::subcommands::Commands;

#[derive(Clone, Parser, Debug)]
#[command(name = "caliche")]
#[command(version, long_version = crate::build_info::LONG_VERSION)]
#[command(about = "Run an agent from the terminal or over HTTP")]
#[command(long_about = "Caliche runs an agent. You choose the model provider. \
    The default tools are fs, bash, and web_fetch.\n\n\
    Run without arguments to enter interactive mode.\n\
    Pass a bare prompt for a one-shot turn.\n\
    Use --server to start the HTTP server.\n\
    JWT stays off until you enable it; see SECURITY.md.\n\
    Subcommands cover config, sessions, diagnostics, and coding tasks.")]
#[command(after_help = "START HERE:\n  \
    caliche                              # Interactive session\n  \
    caliche \"explain this repo\"          # One-shot prompt\n  \
    caliche --server --bind 127.0.0.1    # HTTP server on loopback\n  \
    caliche config setup                 # First-time provider setup\n  \
    caliche --document README.md --doc-format markdown --query-collection docs --message \"memory safety\" --search-mode full_text\n\n\
COMMON FLAGS:\n  \
    --provider <name>     Select provider\n  \
    --model <id>          Select model\n  \
    --attach <path>       Attach files like @path references\n  \
    --output-format json  Structured automation output\n  \
    --resume-last         Continue the latest saved session")]
#[command(propagate_version = true)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Commands>,

    // === Input/Output Options ===
    /// Message to send to the agent (if not provided, enters interactive mode)
    #[arg(short, long, help_heading = "Input/Output")]
    pub message: Option<String>,

    /// Prompt to send (alias for --message, for non-interactive mode)
    #[arg(short = 'p', long, help_heading = "Input/Output")]
    pub prompt: Option<String>,

    /// One-shot prompt without --message (captures all remaining words)
    #[arg(
        value_name = "PROMPT",
        trailing_var_arg = true,
        allow_hyphen_values = true,
        help_heading = "Input/Output"
    )]
    pub prompt_args: Vec<String>,

    /// Attach file(s) to the prompt (repeatable). Equivalent to adding @path references.
    #[arg(short = 'a', long = "attach", help_heading = "Input/Output")]
    pub attach: Vec<String>,

    /// Process piped stdin content with a prompt (e.g., cat file.txt | caliche --pipe "explain this")
    #[arg(long, help_heading = "Input/Output")]
    pub pipe: Option<String>,

    /// Output format (text, json, jsonl, stream)
    #[arg(
        long,
        short = 'f',
        default_value = "text",
        help_heading = "Input/Output"
    )]
    pub output_format: OutputFormat,

    /// Quiet mode - suppress spinner and progress indicators
    #[arg(short = 'q', long, help_heading = "Input/Output")]
    pub quiet: bool,

    /// REPL output verbosity: quiet, normal, thinking, verbose, or debug.
    ///
    /// `thinking`, `verbose`, and `debug` stream model thought chains.
    /// This is separate from `--verbose`, which only raises log level.
    #[arg(
        long = "output-verbosity",
        value_name = "LEVEL",
        help_heading = "Input/Output"
    )]
    pub output_verbosity: Option<String>,

    /// REPL opening banner: off, compact, or full.
    ///
    /// `compact` is the default. Set `full` to also show tools, the session id,
    /// and startup notes. Set `off` to show warnings only.
    #[arg(long, value_name = "MODE", help_heading = "Input/Output")]
    pub banner: Option<String>,

    /// Auto-approve all permissions (use with caution)
    #[arg(long, help_heading = "Input/Output")]
    pub yolo: bool,

    // === Provider Options ===
    /// Model provider to use, e.g. openai, anthropic, openrouter, ollama,
    /// lmstudio, or local-inference.
    #[arg(long, help_heading = "Provider Options")]
    pub provider: Option<String>,

    /// Model to use (provider-specific model ID)
    #[arg(long, help_heading = "Provider Options")]
    pub model: Option<String>,

    /// Hugging Face repository to download for local-inference (overrides model source for downloads)
    #[arg(long, help_heading = "Provider Options")]
    pub local_inference_download_repo: Option<String>,

    /// Hugging Face revision to download for local-inference
    #[arg(long, help_heading = "Provider Options")]
    pub local_inference_download_revision: Option<String>,

    /// Backend for local-inference (candle, ort, mlx, gguf/llamacpp)
    #[arg(long, help_heading = "Provider Options")]
    pub local_inference_backend: Option<String>,

    /// Directory where local-inference model artifacts are stored
    #[arg(long, help_heading = "Provider Options")]
    pub local_inference_model_dir: Option<PathBuf>,

    /// Hugging Face cache directory used for local-inference bootstrap
    #[arg(long, help_heading = "Provider Options")]
    pub local_inference_hf_cache_dir: Option<PathBuf>,

    /// Disable automatic detection of local providers (Ollama, LM Studio)
    #[arg(long, help_heading = "Provider Options")]
    pub no_auto_detect: bool,

    /// Prefer local providers over cloud when available
    #[arg(long, help_heading = "Provider Options")]
    pub prefer_local: bool,

    /// List available models for a provider and exit
    #[arg(long, help_heading = "Provider Options")]
    pub list_models: Option<String>,

    /// Image generation provider to expose as the generate_image tool.
    #[arg(
        long,
        env = "CALICHE_IMAGE_PROVIDER",
        help_heading = "Provider Options"
    )]
    pub image_provider: Option<String>,

    /// Image generation model ID.
    #[arg(long, env = "CALICHE_IMAGE_MODEL", help_heading = "Provider Options")]
    pub image_model: Option<String>,

    /// Directory for generated artifacts. Relative paths resolve against the workspace.
    #[arg(long, env = "CALICHE_ARTIFACT_DIR", help_heading = "Agent Options")]
    pub artifact_dir: Option<PathBuf>,

    // === Session Options ===
    /// Session ID to resume a previous conversation
    #[arg(long, help_heading = "Session Options")]
    pub session: Option<String>,

    /// Resume the most recent conversation session
    #[arg(long, help_heading = "Session Options")]
    pub resume_last: bool,

    /// Continue the most recent session (combine with --message for multi-turn scripting).
    /// When used with --session, that explicit ID takes precedence over "last".
    #[arg(short = 'C', long = "continue", help_heading = "Session Options")]
    pub continue_session: bool,

    /// Save session on exit
    #[arg(long, help_heading = "Session Options")]
    pub save_session: bool,

    /// Disable automatic session saving after each response
    #[arg(long, help_heading = "Session Options")]
    pub no_autosave: bool,

    // === Agent Options ===
    /// Tools to enable: comma-separated names, or presets `coding`/`default`
    /// (fs, bash, web_fetch), `readonly` (fs), `none` (no tools), or `extended`
    /// (coding plus http, workspace, execute_python). `web_search` stays opt-in.
    #[arg(long, help_heading = "Agent Options")]
    pub tools: Option<String>,

    /// Opt-in runtime profile. `lean` is the Lean profile: `fs` and `bash` only
    /// (no `web_fetch`), plus a short
    /// system prompt, and no MCP discovery, skill index, workspace embed, research
    /// skill, plugin store, or project-context injection. `--tools` still overrides
    /// the tool list. `--workspace-embed` still turns workspace embed on.
    #[arg(long, value_enum, help_heading = "Agent Options")]
    pub profile: Option<RuntimeProfile>,

    /// Comma-separated tool names to exclude after `--tools` / defaults are applied
    #[arg(long = "exclude-tools", help_heading = "Agent Options")]
    pub exclude_tools: Option<String>,

    /// List available tools and exit
    #[arg(short = 'l', long, help_heading = "Agent Options")]
    pub list_tools: bool,

    /// Provider for spawned subagents. Persists like `/subagent-provider`.
    #[arg(long, help_heading = "Agent Options")]
    pub subagent_provider: Option<String>,

    /// Model for spawned subagents. Accepts `provider/model`. Persists like `/subagent-model`.
    #[arg(long, help_heading = "Agent Options")]
    pub subagent_model: Option<String>,

    /// Run a methodical investigate/verify loop for long tasks.
    #[arg(long, help_heading = "Agent Options")]
    pub complex: bool,

    /// Maximum investigate/verify rounds for `--complex` (default 8).
    #[arg(long, help_heading = "Agent Options")]
    pub complex_max_rounds: Option<usize>,

    /// Sandbox policy for bash/fs (read-only, workspace-write, network-restricted, danger-full-access). Default when omitted: workspace-write. Use danger-full-access to run without sandbox.
    #[arg(long, help_heading = "Agent Options")]
    pub sandbox: Option<String>,

    /// Sandbox backend override (native, wasm, noop, docker). Default: native.
    #[arg(long, help_heading = "Agent Options")]
    pub sandbox_backend: Option<String>,

    /// Permission mode (default, accept-edits, accept-all, deny-all, interactive)
    #[arg(long, help_heading = "Agent Options")]
    pub permission_mode: Option<String>,

    /// Index the current directory with embeddings and use RAG in chat.
    /// One-shot `-p` prompts skip indexing unless this flag is set.
    /// Interactive REPL and `code` still index by default; use `--no-workspace-embed` to skip.
    #[arg(long, help_heading = "Agent Options")]
    pub workspace_embed: bool,

    /// Named agent profile to use (from .caliche/agents/ or ~/.config/caliche/agents/).
    #[arg(long, help_heading = "Agent Options")]
    pub agent: Option<String>,

    /// Disable workspace RAG (no embedding index; use when embedding provider is unavailable or to reduce startup cost).
    #[arg(long = "no-workspace-embed", help_heading = "Agent Options")]
    pub no_workspace_embed: bool,

    /// Disable automatic skill detection from message intent.
    #[arg(long = "no-auto-skill", help_heading = "Agent Options")]
    pub no_auto_skill: bool,

    /// Enable line-level BM25 for workspace embedding (default: true in the CLI; hybrid vector+keyword search).
    #[arg(long, default_value_t = true, help_heading = "Agent Options")]
    pub line_level_bm25: bool,

    /// Build code graph during workspace indexing (used by AST/graph exploration tools)
    #[arg(long, help_heading = "Agent Options")]
    pub build_code_graph: bool,

    /// Force workspace hashline tagging on/off (overrides automatic edit-tool behavior)
    #[arg(long, help_heading = "Agent Options")]
    pub workspace_hashline: Option<bool>,

    /// Force workspace hashline result formatting on/off (overrides automatic edit-tool behavior)
    #[arg(long, help_heading = "Agent Options")]
    pub workspace_hashline_format: Option<bool>,

    /// Enable MMR diversity reranking for RAG workspace queries (reduces similar chunks)
    #[arg(long, help_heading = "Agent Options")]
    pub rag_mmr: bool,

    /// MMR lambda 0.0-1.0 for RAG (0 = max diversity, 1 = max relevance; default 0.7 when --rag-mmr)
    #[arg(long, help_heading = "Agent Options")]
    pub rag_mmr_lambda: Option<f64>,

    /// Enable temporal decay for RAG (down-weight older chunks; chunks need created_at metadata)
    #[arg(long, help_heading = "Agent Options")]
    pub rag_temporal_decay: bool,

    /// Half-life in days for RAG temporal decay (default 30 when --rag-temporal-decay)
    #[arg(long, help_heading = "Agent Options")]
    pub rag_temporal_decay_half_life: Option<f64>,

    /// Enable context-window compaction when approaching token limit (sliding window)
    #[arg(long, help_heading = "Agent Options")]
    pub compaction: bool,

    /// Compaction strategy (currently only sliding_window is supported)
    #[arg(long, default_value = "sliding_window", help_heading = "Agent Options")]
    pub compaction_strategy: String,

    /// Fraction of context window that triggers compaction (e.g. 0.90)
    #[arg(long, help_heading = "Agent Options")]
    pub compaction_trigger: Option<f64>,

    /// Target fraction after compaction (e.g. 0.70)
    #[arg(long, help_heading = "Agent Options")]
    pub compaction_target: Option<f64>,

    /// Max compaction attempts per turn
    #[arg(long, help_heading = "Agent Options")]
    pub compaction_max_attempts: Option<u32>,

    /// Override model context window size in tokens
    #[arg(long, env = "CALICHE_CONTEXT_WINDOW", help_heading = "Agent Options")]
    pub context_window: Option<usize>,

    /// Failover fallback provider:model (repeatable, e.g. --failover-fallback anthropic:claude-sonnet-4-5)
    #[arg(long, help_heading = "Agent Options")]
    pub failover_fallback: Vec<String>,

    /// Enable session tools (sessions_list, sessions_history, sessions_send, sessions_spawn)
    #[arg(long, help_heading = "Agent Options")]
    pub session_tools: bool,

    /// Enable distillation-based learning: observe tool trajectories and persist reusable skill plugins.
    #[arg(long, env = "CALICHE_LEARNING_LOOP", help_heading = "Agent Options")]
    pub learning_loop: bool,

    /// Tool-call interval between learning checkpoints.
    #[arg(
        long,
        env = "CALICHE_LEARNING_INTERVAL",
        help_heading = "Agent Options"
    )]
    pub learning_interval: Option<usize>,

    /// Minimum tool calls before a trajectory can be distilled into a skill.
    #[arg(
        long,
        env = "CALICHE_LEARNING_MIN_TOOL_CALLS",
        help_heading = "Agent Options"
    )]
    pub learning_min_tool_calls: Option<usize>,

    /// Maximum auto-created skills per session.
    #[arg(
        long,
        env = "CALICHE_LEARNING_MAX_SKILLS",
        help_heading = "Agent Options"
    )]
    pub learning_max_skills: Option<usize>,

    /// MEMORY.md path for durable learning nudges. Defaults to .caliche/MEMORY.md when learning is enabled.
    #[arg(
        long,
        env = "CALICHE_LEARNING_MEMORY_FILE",
        help_heading = "Agent Options"
    )]
    pub learning_memory_file: Option<PathBuf>,

    // === Server Options ===
    /// Start the HTTP server instead of the CLI
    #[arg(short = 's', long, help_heading = "Server Options")]
    pub server: bool,

    /// HTTP server port
    #[arg(long, default_value = "3000", help_heading = "Server Options")]
    pub port: u16,

    /// HTTP server listen address
    #[arg(long, default_value = "127.0.0.1", help_heading = "Server Options")]
    pub bind: String,

    /// Enable MCP server in stdio mode (for CLI integration)
    #[arg(long, help_heading = "Server Options")]
    pub mcp_stdio: bool,

    /// Enable ACP server in stdio mode (newline-delimited JSON-RPC).
    #[arg(long, help_heading = "Server Options")]
    pub acp_stdio: bool,

    /// Enable MCP transports in server mode (comma-separated: streamable-http,all)
    #[arg(long, help_heading = "Server Options")]
    pub mcp_transport: Option<String>,

    // === Model Parameters ===
    /// Sampling temperature for the model (0.0-2.0)
    #[arg(long, help_heading = "Model Parameters")]
    pub temperature: Option<f32>,

    /// Top-p sampling parameter (0.0-1.0)
    #[arg(long, help_heading = "Model Parameters")]
    pub top_p: Option<f32>,

    /// Maximum tokens in response
    #[arg(long, help_heading = "Model Parameters")]
    pub max_tokens: Option<u32>,

    /// Request timeout in seconds
    #[arg(long, help_heading = "Model Parameters")]
    pub timeout: Option<u64>,

    // === Reasoning Options ===
    /// Reasoning pattern to use (react, plan-execute, self-ask, reflexion, tree-of-thoughts)
    #[arg(long, help_heading = "Reasoning Options")]
    pub reasoning_pattern: Option<String>,

    /// Maximum reasoning iterations/steps
    #[arg(long, help_heading = "Reasoning Options")]
    pub max_iterations: Option<usize>,

    /// Maximum tree depth for Tree of Thoughts
    #[arg(long, help_heading = "Reasoning Options")]
    pub tree_depth: Option<usize>,

    /// Number of branches for Tree of Thoughts
    #[arg(long, help_heading = "Reasoning Options")]
    pub tree_branches: Option<usize>,

    // === Chain Options ===
    /// Chain type to use (llm, sequential, map-reduce)
    #[arg(long, help_heading = "Chain Options")]
    pub chain_type: Option<ChainType>,

    /// Chain template: for `llm`, the prompt string (default `{input}`). For `sequential`, JSON or
    /// path to JSON `{"sequential":{"stages":[{"template":"..."},...]}}`. For `map-reduce`, JSON or
    /// path to `{"map_reduce":{"map_template":"...","reduce_template":"...","input_key":"input"}}`.
    #[arg(long, help_heading = "Chain Options")]
    pub template: Option<String>,

    /// Template variables as JSON. For `map-reduce`, must be a JSON array of objects (per map item)
    /// unless omitted (single item from `--message` using `input_key`).
    #[arg(long, help_heading = "Chain Options")]
    pub template_vars: Option<String>,

    // === Document Options ===
    /// Document file to load
    #[arg(long, help_heading = "Document Options")]
    pub document: Option<String>,

    /// Document format (csv, json, markdown, html)
    #[arg(long, help_heading = "Document Options")]
    pub doc_format: Option<String>,

    /// Index documents into a RAG collection. Persistent only when VECTOR_BACKEND is configured; otherwise process-local.
    #[arg(long, help_heading = "Document Options")]
    pub index_to: Option<String>,

    /// Query a RAG collection or the documents loaded in this invocation
    #[arg(long, help_heading = "Document Options")]
    pub query_collection: Option<String>,

    /// Chunk size for text splitting
    #[arg(long, default_value = "500", help_heading = "Document Options")]
    pub chunk_size: usize,

    /// Chunk overlap
    #[arg(long, default_value = "50", help_heading = "Document Options")]
    pub chunk_overlap: usize,

    /// Enable hashline tagging on indexed chunks (line:hash format for fs op=edit compatibility)
    #[arg(long, help_heading = "Document Options")]
    pub hashline: bool,

    /// Format search results with hashline tags (requires --hashline on ingest)
    #[arg(long, help_heading = "Document Options")]
    pub hashline_format: bool,

    /// Retrieval mode for CLI document queries
    #[arg(long, value_enum, help_heading = "Document Options")]
    pub search_mode: Option<RagCliSearchMode>,

    /// Fusion mode for CLI document queries
    #[arg(long, value_enum, help_heading = "Document Options")]
    pub fusion_mode: Option<RagCliFusionMode>,

    /// Dense weight for weighted hybrid CLI queries
    #[arg(long, help_heading = "Document Options")]
    pub dense_weight: Option<f32>,

    /// Sparse/full-text weight for weighted hybrid CLI queries
    #[arg(long, help_heading = "Document Options")]
    pub sparse_weight: Option<f32>,

    /// Number of retrieval results to return for CLI document queries
    #[arg(long, help_heading = "Document Options")]
    pub rag_top_k: Option<usize>,

    /// Minimum similarity threshold for CLI document queries
    #[arg(long, help_heading = "Document Options")]
    pub rag_min_similarity: Option<f32>,

    // === Advanced Options ===
    /// Workflow definition file (JSON/YAML)
    #[arg(long, help_heading = "Advanced Options")]
    pub workflow: Option<String>,

    /// Multi-agent pattern (pipeline, debate, hierarchical, supervisor)
    #[arg(long, help_heading = "Advanced Options")]
    pub multi_agent_pattern: Option<String>,

    /// Config file path (TOML or YAML)
    #[arg(short = 'c', long, help_heading = "Advanced Options")]
    pub config: Option<String>,

    /// Enable verbose logging
    #[arg(short, long, help_heading = "Advanced Options")]
    pub verbose: bool,

    /// Enable debug logging to file (writes to ~/.local/share/caliche/logs/)
    #[arg(long, help_heading = "Advanced Options")]
    pub debug: bool,

    /// Turn off performance timing (on by default with --debug and in debug builds)
    #[arg(long, help_heading = "Advanced Options")]
    pub no_perf: bool,

    /// Load credentials from a specific .env file instead of the OS keychain.
    /// When set, the keychain is bypassed entirely and all provider API keys
    /// are sourced exclusively from the given file (plus any shell-level env vars).
    /// Without this flag no `.env` file is read, not even one in the working directory.
    #[arg(long, help_heading = "Advanced Options", value_name = "PATH")]
    pub env_file: Option<String>,
}

impl Args {
    /// True when `--profile lean` is selected.
    #[must_use]
    pub const fn is_lean_profile(&self) -> bool {
        matches!(self.profile, Some(RuntimeProfile::Lean))
    }

    /// Effective single-message input, preferring explicit flags over positional text.
    pub fn effective_message(&self) -> Option<String> {
        self.message
            .clone()
            .or_else(|| self.prompt.clone())
            .or_else(|| {
                let joined = self.prompt_args.join(" ").trim().to_string();
                (!joined.is_empty()).then_some(joined)
            })
    }

    /// Whether the current invocation includes single-message input.
    pub fn has_message_input(&self) -> bool {
        self.effective_message().is_some()
    }

    /// Whether this invocation should open the interactive REPL.
    ///
    /// `tty` is stdin+stdout terminal state so unit tests can cover both lanes.
    #[must_use]
    pub fn wants_repl(&self, tty: bool) -> bool {
        if !tty
            || self.server
            || self.acp_stdio
            || self.has_message_input()
            || self.list_models.is_some()
            || self.list_tools
        {
            return false;
        }
        if self.pipe.is_some() {
            return false;
        }
        match &self.command {
            None => true,
            Some(command) => command.continues_to_repl(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Commands;
    use super::*;
    use clap::Parser;

    #[test]
    fn wants_repl_for_bare_tty_and_not_for_oneshot_or_code() {
        let bare = Args::parse_from(["caliche"]);
        assert!(bare.wants_repl(true));
        assert!(!bare.wants_repl(false));

        let prompt = Args::parse_from(["caliche", "-p", "hi"]);
        assert!(!prompt.wants_repl(true));

        let acp = Args::parse_from(["caliche", "--acp-stdio"]);
        assert!(acp.acp_stdio);
        assert!(!acp.wants_repl(true));
        assert!(!acp.wants_repl(false));

        let code = Args::parse_from(["caliche", "code", "fix the parser"]);
        assert!(!code.wants_repl(true));
        assert!(matches!(code.command, Some(Commands::Code { .. })));

        let doctor = Args::parse_from(["caliche", "doctor"]);
        assert!(!doctor.wants_repl(true));

        for argv in [
            ["caliche", "config", "path"].as_slice(),
            ["caliche", "-q", "config", "path"].as_slice(),
            ["caliche", "config", "list"].as_slice(),
            ["caliche", "config", "show"].as_slice(),
            ["caliche", "config", "schema"].as_slice(),
        ] {
            let args = Args::parse_from(argv);
            assert!(!args.wants_repl(true), "{argv:?} must not open the REPL");
            assert!(
                matches!(args.command, Some(Commands::Config { .. })),
                "{argv:?} must parse as config, got {:?}",
                args.command
            );
        }
        let defaults = crate::handlers::DoctorArgs::server_defaults();
        assert_eq!(doctor.bind, defaults.bind);
        assert_eq!(doctor.port, defaults.port);
        assert_eq!(doctor.bind, "127.0.0.1");
        assert_eq!(doctor.port, 3000);
    }

    #[test]
    fn explicit_unspecified_bind_is_accepted() {
        let args = Args::parse_from(["caliche", "--server", "--bind", "0.0.0.0"]);
        assert_eq!(args.bind, "0.0.0.0");
        assert_eq!(args.port, 3000);
    }

    #[test]
    fn doctor_bind_flags_are_the_server_listen() {
        let args = Args::parse_from(["caliche", "--bind", "127.0.0.1", "--port", "8080", "doctor"]);
        assert_eq!(args.bind, "127.0.0.1");
        assert_eq!(args.port, 8080);
    }

    #[test]
    fn version_text_includes_build_id_line() {
        use clap::CommandFactory;
        let rendered = Args::command().render_long_version();
        assert!(
            rendered.contains("\nbuild_id="),
            "version text missing build_id line: {rendered}"
        );
        assert!(rendered.contains(crate::build_info::BUILD_ID));
    }

    #[test]
    fn help_frames_a_general_purpose_harness() {
        use clap::CommandFactory;
        let long_help = Args::command().render_long_help().to_string();
        assert!(long_help.contains("Caliche runs an agent"), "{long_help}");
        assert!(
            !long_help.contains("AI-powered CLI for interactive coding"),
            "{long_help}"
        );
        assert!(
            long_help.contains("JWT stays off until you enable it"),
            "{long_help}"
        );
        assert!(
            long_help.contains("--server --bind 127.0.0.1"),
            "{long_help}"
        );
        assert!(
            long_help.contains("default: 127.0.0.1"),
            "help must show the loopback bind default, got: {long_help}"
        );
        assert!(
            !long_help.contains("default: 0.0.0.0"),
            "help must not claim an open bind default, got: {long_help}"
        );
        let short_help = Args::command().render_help().to_string();
        assert!(
            short_help.contains("Run an agent from the terminal or over HTTP"),
            "{short_help}"
        );
    }

    #[test]
    fn flags_before_code_still_parse_as_subcommand() {
        let args = Args::parse_from([
            "caliche",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--yolo",
            "--no-workspace-embed",
            "--save-session",
            "code",
            "implement mean and median",
            "--cwd",
            "/tmp/caliche-complex-wf",
            "--accept-edits",
            "--max-turns",
            "12",
            "--no-repomap",
            "--stream",
            "--verify",
            "cargo test --lib command_ledger",
        ]);
        assert!(args.yolo);
        assert!(args.save_session);
        assert!(!args.wants_repl(true));
        match args.command {
            Some(Commands::Code {
                task,
                cwd,
                accept_edits,
                stream,
                max_turns,
                no_repomap,
                verify,
                no_verify,
                ..
            }) => {
                assert_eq!(task, "implement mean and median");
                assert_eq!(cwd.as_deref(), Some("/tmp/caliche-complex-wf"));
                assert!(accept_edits);
                assert!(stream);
                assert_eq!(max_turns, 12);
                assert!(no_repomap);
                assert_eq!(verify, vec!["cargo test --lib command_ledger".to_string()]);
                assert!(!no_verify);
            }
            other => panic!("expected Code subcommand, got {other:?}"),
        }
    }

    #[test]
    fn code_no_verify_parses() {
        let args = Args::parse_from(["caliche", "code", "task", "--no-verify"]);
        match args.command {
            Some(Commands::Code {
                no_verify, verify, ..
            }) => {
                assert!(no_verify);
                assert!(verify.is_empty());
            }
            other => panic!("expected Code subcommand, got {other:?}"),
        }
    }

    #[test]
    fn trailing_prompt_feeds_reasoning_pattern() {
        let args = Args::parse_from([
            "caliche",
            "--reasoning-pattern",
            "plan-execute",
            "--max-iterations",
            "2",
            "What is 17*19?",
        ]);
        assert_eq!(args.reasoning_pattern.as_deref(), Some("plan-execute"));
        assert_eq!(args.max_iterations, Some(2));
        assert_eq!(args.effective_message().as_deref(), Some("What is 17*19?"));
        assert!(args.message.is_none());
    }

    #[test]
    fn lean_profile_parses_as_opt_in() {
        let bare = Args::parse_from(["caliche"]);
        assert!(!bare.is_lean_profile());
        let lean = Args::parse_from(["caliche", "--profile", "lean"]);
        assert!(lean.is_lean_profile());
        assert!(lean.tools.is_none());
        assert!(!crate::bootstrap::effective_workspace_embed_with_tty(
            &lean, true
        ));
        let embedded = Args::parse_from(["caliche", "--profile", "lean", "--workspace-embed"]);
        assert!(crate::bootstrap::effective_workspace_embed_with_tty(
            &embedded, true
        ));
    }

    #[test]
    fn workspace_embed_flag_is_off_unless_passed() {
        let bare = Args::parse_from(["caliche"]);
        assert!(!bare.workspace_embed);
        assert!(!bare.no_workspace_embed);

        let explicit = Args::parse_from(["caliche", "--workspace-embed"]);
        assert!(explicit.workspace_embed);

        let disabled = Args::parse_from(["caliche", "--no-workspace-embed"]);
        assert!(disabled.no_workspace_embed);
    }

    #[test]
    fn subagent_and_complex_flags_parse() {
        let args = Args::parse_from([
            "caliche",
            "--subagent-provider",
            "anthropic",
            "--subagent-model",
            "claude-sonnet-4-5",
            "--complex",
            "--complex-max-rounds",
            "6",
        ]);
        assert_eq!(args.subagent_provider.as_deref(), Some("anthropic"));
        assert_eq!(args.subagent_model.as_deref(), Some("claude-sonnet-4-5"));
        assert!(args.complex);
        assert_eq!(args.complex_max_rounds, Some(6));
    }

    #[test]
    fn debug_and_no_perf_parse() {
        let debug = Args::parse_from(["caliche", "--debug"]);
        assert!(debug.debug);
        assert!(!debug.no_perf);

        let both = Args::parse_from(["caliche", "--debug", "--no-perf"]);
        assert!(both.debug);
        assert!(both.no_perf);
    }
}
