//! Code execution and agent loop handlers.
//!
//! Handles the `code` subcommand for running agent tasks on codebases.

use crate::bootstrap::{self, WorkspaceEmbedOptions};
use crate::commands::Args;
use crate::errors::{CliError, CliResult};
use crate::output_mode::OutputMode;
use crate::prompts::PromptBuilder;
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use caliche::agent::core::ToolExecutionDriver;
use caliche::harness::{
    AgentEvent, BroadcastRecv, CodingContextPolicy, CodingMode, CodingTaskExecutor,
    CodingTaskRequest, HarnessConfig, VerificationPolicy, classify_recv,
    config::PermissionModeConfig,
};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;

impl CodeArgs {
    /// Build handler args from the `code` subcommand fields and top-level provider/model flags.
    pub fn from_command(
        task: &str,
        cwd: &Option<String>,
        tools: &Option<String>,
        max_turns: usize,
        accept_edits: bool,
        dry_run: bool,
        stream: bool,
        no_repomap: bool,
        verify: Vec<String>,
        no_verify: bool,
        ctx: &CommandContext<'_>,
        cli_args: &Args,
    ) -> Self {
        Self {
            task: task.to_owned(),
            cwd: cwd.clone(),
            tools: tools.clone(),
            max_turns,
            accept_edits: accept_edits || ctx.yolo,
            dry_run,
            stream,
            no_repomap,
            verify,
            no_verify,
            provider: ctx.provider.clone(),
            model: ctx.model.clone(),
            yolo: ctx.yolo,
            workspace_embed: bootstrap::effective_workspace_embed(cli_args),
            workspace_options: WorkspaceEmbedOptions::from_args(cli_args),
            complex: cli_args.complex,
            context_window: cli_args.context_window,
        }
    }
}

/// Arguments for the code command.
#[derive(Debug, Clone)]
pub struct CodeArgs {
    pub task: String,
    pub cwd: Option<String>,
    pub tools: Option<String>,
    pub max_turns: usize,
    pub accept_edits: bool,
    pub dry_run: bool,
    pub stream: bool,
    pub no_repomap: bool,
    /// Exact verification commands; empty means infer from the workspace.
    pub verify: Vec<String>,
    /// Skip post-task verification.
    pub no_verify: bool,
    /// Provider from top-level --provider (e.g. openai, anthropic).
    pub provider: Option<String>,
    /// Model from top-level --model (e.g. gpt-4o, claude-3-5-sonnet).
    pub model: Option<String>,
    /// Top-level `--yolo` auto-approves all tool permissions.
    pub yolo: bool,
    /// Attach workspace RAG on the coding harness (default on; `--no-workspace-embed` disables).
    pub workspace_embed: bool,
    /// Indexer flags copied from top-level CLI args.
    pub workspace_options: WorkspaceEmbedOptions,
    /// Use the methodical complex-task prompt for this coding run.
    pub complex: bool,
    /// Top-level `--context-window`: the model's window, set by the user.
    pub context_window: Option<usize>,
}

/// Handler for the `code` subcommand.
pub struct CodeHandler;

#[async_trait]
impl CommandHandler for CodeHandler {
    type Args = CodeArgs;

    async fn execute(
        &self,
        args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let output_mode = if args.stream {
            OutputMode::Stream
        } else {
            OutputMode::Text
        };

        Box::pin(run_code_command_harness(
            args,
            output_mode,
            ctx.cancellation_token,
        ))
        .await
        .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "code"
    }

    fn description(&self) -> &'static str {
        "Run agent on a code task with tool access"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Execute an agent task with access to file system, bash, and other tools.\n\
             The agent will work iteratively until the task is complete or max turns is reached.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "caliche code \"Add error handling to the entry point\"",
            "caliche code \"Refactor the auth module\" --max-turns 20",
            "caliche code \"Fix the bug in the parser module\" --accept-edits",
            "caliche code \"Harden command ledger\" --verify \"cargo test --lib command_ledger\"",
        ]
    }
}

fn coding_tool_execution_driver(configured: Option<&str>) -> Option<ToolExecutionDriver> {
    match configured.map(str::trim) {
        None | Some("") | Some("auto") => Some(ToolExecutionDriver::DedupOnly),
        Some(value) => ToolExecutionDriver::parse(value),
    }
}

/// Permission mode for unattended `code` runs.
///
/// `--yolo` accepts every tool. `--accept-edits` or a non-TTY host accepts
/// file edits without a prompt so piped agents do not hang on stdin.
fn coding_permission_mode(accept_edits: bool, yolo: bool, tty: bool) -> PermissionModeConfig {
    if yolo {
        PermissionModeConfig::AcceptAll
    } else if accept_edits || !tty {
        PermissionModeConfig::AcceptEdits
    } else {
        PermissionModeConfig::Default
    }
}

const fn coding_task_mode(dry_run: bool) -> CodingMode {
    if dry_run {
        CodingMode::Plan
    } else {
        CodingMode::FullAuto
    }
}

fn coding_verification_policy(verify: &[String], no_verify: bool) -> VerificationPolicy {
    if no_verify {
        return VerificationPolicy::None;
    }
    let commands: Vec<String> = verify
        .iter()
        .map(|command| command.trim())
        .filter(|command| !command.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if commands.is_empty() {
        VerificationPolicy::Auto
    } else {
        VerificationPolicy::Commands(commands)
    }
}

fn format_code_event(event: &AgentEvent, stream_text: bool) -> Option<String> {
    match event {
        AgentEvent::Info { message } => Some(format!("[info] {message}\n")),
        AgentEvent::Warning { message } => Some(format!("[warn] {message}\n")),
        AgentEvent::Error { error, .. } => Some(format!("[error] {error}\n")),
        AgentEvent::TurnStart { turn_id, .. } => Some(format!("[turn {turn_id}]\n")),
        AgentEvent::ToolExecuting { tool_name, .. } => Some(format!("[{tool_name}]\n")),
        AgentEvent::ToolComplete {
            tool_name,
            success,
            duration_ms,
            ..
        } => {
            let status = if *success { "ok" } else { "failed" };
            Some(format!("[{tool_name} {status} {duration_ms}ms]\n"))
        }
        AgentEvent::ToolError {
            tool_name, error, ..
        } => Some(format!("[{tool_name} error] {error}\n")),
        AgentEvent::FileWrite { path, .. } => Some(format!("[write] {}\n", path.display())),
        AgentEvent::FileRead { path, .. } => Some(format!("[read] {}\n", path.display())),
        AgentEvent::TextChunk { content } if stream_text => Some(content.clone()),
        _ => None,
    }
}

fn print_code_line(text: &str) {
    let mut out = std::io::stdout();
    let _ = write!(out, "{text}");
    let _ = out.flush();
}

enum CodePrinterStep {
    Line(String),
    Skip,
    Stop,
}

fn code_printer_step(
    result: Result<AgentEvent, tokio::sync::broadcast::error::RecvError>,
    stream_text: bool,
) -> CodePrinterStep {
    match classify_recv(result) {
        BroadcastRecv::Event(event) => match format_code_event(&event, stream_text) {
            Some(text) => CodePrinterStep::Line(text),
            None => CodePrinterStep::Skip,
        },
        BroadcastRecv::Resync(resync) => CodePrinterStep::Line(resync.cli_line()),
        BroadcastRecv::Closed => CodePrinterStep::Stop,
    }
}

fn spawn_code_event_printer(
    mut rx: tokio::sync::broadcast::Receiver<AgentEvent>,
    stream_text: bool,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match code_printer_step(rx.recv().await, stream_text) {
                CodePrinterStep::Line(text) => print_code_line(&text),
                CodePrinterStep::Skip => {}
                CodePrinterStep::Stop => break,
            }
        }
    })
}

/// Run the code command using the AgentHarness infrastructure.
/// When `cancellation_token` is `Some`, Ctrl+C will abort the task.
async fn run_code_command_harness(
    args: CodeArgs,
    output_mode: OutputMode,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<()> {
    let CodeArgs {
        task,
        cwd,
        tools,
        max_turns,
        accept_edits,
        yolo,
        dry_run,
        no_repomap,
        verify,
        no_verify,
        provider,
        model,
        workspace_embed,
        workspace_options,
        complex,
        context_window,
        ..
    } = args;

    let workspace = resolve_code_workspace(cwd.as_deref())?;
    let tool_names = parse_code_tool_names(tools.as_deref());
    let runtime_config = Some(crate::config::load_standard_config_strict().await?);
    let formatter_rules = runtime_config
        .as_ref()
        .map(bootstrap::formatter_rules_from_config)
        .unwrap_or_default();

    if dry_run {
        println!("[DRY RUN] Would execute task: {}", task);
        println!("[DRY RUN] Max turns: {}", max_turns);
        println!("[DRY RUN] Accept edits: {}", accept_edits);
        let _ = std::io::stdout().flush();
        return Ok(());
    }

    let embed_discovery = workspace_embed.then(spawn_code_embed_discovery);
    let context_prefix =
        build_code_repomap_context(&workspace, &task, no_repomap, runtime_config.as_ref());
    let provider_type = match provider.as_deref() {
        Some(name) => Some(
            name.parse::<caliche::llm::ProviderType>()
                .map_err(|err| anyhow::anyhow!("Invalid provider: {err}"))?,
        ),
        None => None,
    };
    let tty = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let stream_text = matches!(output_mode, OutputMode::Stream);
    let permission_mode = coding_permission_mode(accept_edits, yolo, tty);
    let config = HarnessConfig {
        provider: provider_type,
        model: model.clone(),
        tools: tool_names.clone(),
        workspace_root: Some(workspace.clone()),
        max_turns: if complex {
            max_turns.max(80)
        } else {
            max_turns
        },
        permission_mode,
        formatter_rules,
        system_prompt: Some(coding_system_prompt(workspace_embed, complex, None)),
        tool_execution_driver: coding_tool_execution_driver(
            runtime_config
                .as_ref()
                .and_then(|cfg| cfg.tool_execution_driver.as_deref()),
        ),
        ..Default::default()
    };

    let mut executor = CodingTaskExecutor::new(config).await?;
    apply_code_project_context(
        executor.harness_mut().service_mut(),
        workspace_embed,
        complex,
        context_window,
        runtime_config.as_ref(),
    )
    .await?;
    attach_code_workspace_embedding(
        &mut executor,
        embed_discovery,
        workspace.clone(),
        &workspace_options,
    )
    .await;
    install_code_memory(&mut executor, runtime_config.as_ref()).await;

    let request = CodingTaskRequest {
        task: task.clone(),
        workspace_root: Some(workspace.clone()),
        provider: provider_type,
        model,
        mode: coding_task_mode(dry_run),
        permission_mode,
        max_turns,
        tools: tool_names,
        verify: coding_verification_policy(&verify, no_verify),
        context: CodingContextPolicy { context_prefix },
    };

    print_code_start_banner(
        &workspace,
        max_turns,
        provider.as_deref(),
        request.model.as_deref(),
    );
    let events = spawn_code_event_printer(executor.harness().subscribe(), stream_text);
    let result = run_coding_executor(executor, request, cancellation_token, &events).await?;
    events.abort();
    print_code_result(&result);
    Ok(())
}

fn resolve_code_workspace(cwd: Option<&str>) -> Result<PathBuf> {
    if let Some(dir) = cwd {
        std::env::set_current_dir(dir)
            .map_err(|e| anyhow::anyhow!("Failed to change to directory '{dir}': {e}"))?;
        return Ok(PathBuf::from(dir));
    }
    Ok(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn parse_code_tool_names(tools: Option<&str>) -> Option<Vec<String>> {
    tools.map(|tools_str| {
        tools_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<String>>()
    })
}

fn spawn_code_embed_discovery()
-> tokio::task::JoinHandle<Option<caliche::discovery::embedding::EmbeddingProviderDiscovery>> {
    tokio::spawn(async {
        Box::pin(caliche::discovery::embedding::discover_best_embedding_provider()).await
    })
}

fn build_code_repomap_context(
    workspace: &PathBuf,
    task: &str,
    no_repomap: bool,
    runtime_config: Option<&crate::config::Config>,
) -> Option<String> {
    let repomap_enabled = !no_repomap && repomap_enabled_from_env_or_config(runtime_config);
    let max_tokens = repomap_max_tokens(runtime_config);
    if !repomap_enabled {
        return None;
    }
    let map_str = crate::repomap::build(workspace, Some(task), max_tokens);
    if map_str.is_empty() {
        None
    } else {
        Some(map_str)
    }
}

fn repomap_enabled_from_env_or_config(runtime_config: Option<&crate::config::Config>) -> bool {
    std::env::var("CALICHE_REPOMAP_ENABLED")
        .ok()
        .and_then(|v| match v.to_lowercase().as_str() {
            "0" | "false" | "no" => Some(false),
            "1" | "true" | "yes" => Some(true),
            _ => None,
        })
        .unwrap_or_else(|| {
            runtime_config
                .and_then(|c| c.repomap.as_ref().and_then(|r| r.enabled))
                .unwrap_or(true)
        })
}

fn repomap_max_tokens(runtime_config: Option<&crate::config::Config>) -> usize {
    std::env::var("CALICHE_REPOMAP_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| runtime_config.and_then(|c| c.repomap.as_ref().and_then(|r| r.max_tokens)))
        .unwrap_or(2000)
}

async fn attach_code_workspace_embedding(
    executor: &mut CodingTaskExecutor,
    embed_discovery: Option<
        tokio::task::JoinHandle<Option<caliche::discovery::embedding::EmbeddingProviderDiscovery>>,
    >,
    workspace: PathBuf,
    workspace_options: &WorkspaceEmbedOptions,
) {
    let Some(handle) = embed_discovery else {
        return;
    };
    match handle.await {
        Ok(Some(discovery)) => {
            let provider = executor
                .harness()
                .service()
                .embedding_provider()
                .unwrap_or_else(|| Arc::clone(&discovery.provider));
            if let Err(error) = bootstrap::attach_workspace_embedding(
                executor.harness_mut().service_mut(),
                workspace,
                workspace_options,
                provider,
            )
            .await
            {
                tracing::warn!(error = %error, "failed to attach workspace RAG for code");
            }
        }
        Ok(None) => {
            tracing::debug!("code embedding discovery found no provider");
        }
        Err(error) => {
            tracing::warn!(error = %error, "code embedding discovery task failed");
        }
    }
}

async fn install_code_memory(
    executor: &mut CodingTaskExecutor,
    runtime_config: Option<&crate::config::Config>,
) {
    let max_tokens = runtime_config
        .and_then(|c| c.memory_max_tokens)
        .unwrap_or(16_000);
    let Ok(memory) = crate::core::memory_manager::create_memory_strategy(
        crate::core::memory_manager::MemoryStrategyType::Composite,
        max_tokens,
        executor.harness().service().llm_provider(),
        vec![],
        executor.harness().service().embedding_provider(),
    )
    .await
    else {
        return;
    };
    executor
        .harness_mut()
        .service_mut()
        .set_memory_strategy(Some(memory));
}

fn print_code_start_banner(
    workspace: &PathBuf,
    max_turns: usize,
    provider: Option<&str>,
    model: Option<&str>,
) {
    println!(
        "Running coding task in {} (max {max_turns} turns)",
        workspace.display()
    );
    if let Some(provider_name) = provider {
        println!(
            "Provider: {provider_name}  Model: {}",
            model.unwrap_or("default")
        );
    }
    let _ = std::io::stdout().flush();
}

async fn run_coding_executor(
    mut executor: CodingTaskExecutor,
    request: CodingTaskRequest,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
    events: &tokio::task::JoinHandle<()>,
) -> Result<caliche::harness::CodingTaskResult> {
    if let Some(token) = cancellation_token {
        let handle = tokio::spawn(async move { executor.run(&request).await });
        let abort = handle.abort_handle();
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                abort.abort();
                events.abort();
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
            r = handle => return r?,
        }
    }
    executor.run(&request).await
}

fn print_code_result(result: &caliche::harness::CodingTaskResult) {
    println!(
        "\nTask completed in {} turns ({} ms)",
        result.turns, result.duration_ms
    );
    let trimmed = result.final_response.trim();
    if !trimmed.is_empty() {
        println!("\n{trimmed}");
    }
    if !result.changed_files.is_empty() {
        println!("Files modified: {}", result.changed_files.len());
        for file in &result.changed_files {
            println!("  - {}", file.display());
        }
    }
    print_code_verification(result);
    if let Some(checkpoint) = result.checkpoint_id.as_deref() {
        println!("Checkpoint: {checkpoint} (use --rollback to undo)");
    }
}

fn print_code_verification(result: &caliche::harness::CodingTaskResult) {
    if result.verification.commands.is_empty() {
        return;
    }
    println!(
        "Verification: {}",
        if result.verification.passed {
            "passed"
        } else {
            "failed"
        }
    );
    for run in &result.verification.commands {
        println!(
            "  - {} ({})",
            run.command,
            run.exit_code
                .map(|code| code.to_string())
                .unwrap_or_else(|| "no exit code".to_string())
        );
    }
}

fn coding_system_prompt(
    workspace_embed: bool,
    complex: bool,
    project_section: Option<&str>,
) -> String {
    let mut builder = PromptBuilder::new().with_default_base(workspace_embed);
    if complex {
        builder = builder.add_skill(caliche::harness::COMPLEX_TASK_PROMPT.to_owned());
    }
    builder.project_context(project_section).build()
}

/// Add the project context, sized to the model's window: the explicit
/// window first, else the one the provider reports.
async fn apply_code_project_context(
    service: &mut caliche::AgentService,
    workspace_embed: bool,
    complex: bool,
    context_window: Option<usize>,
    runtime_config: Option<&crate::config::Config>,
) -> Result<()> {
    if let Some(window) =
        context_window.or_else(|| runtime_config.and_then(|cfg| cfg.context_window))
    {
        service.set_context_window(window)?;
    }
    let Some(project_config) = super::load_project_config().await else {
        return Ok(());
    };
    let share = runtime_config
        .and_then(|cfg| cfg.project_context_share)
        .unwrap_or(caliche::DEFAULT_PROJECT_CONTEXT_SHARE);
    let section = service.budget_project_context(project_config.into_context(share));
    service.agent_mut().set_system_prompt(coding_system_prompt(
        workspace_embed,
        complex,
        Some(&section),
    ));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_handler_metadata() {
        let handler = CodeHandler;
        assert_eq!(handler.name(), "code");
        assert!(!handler.description().is_empty());
        assert!(handler.help().is_some());
        assert!(!handler.examples().is_empty());
    }

    #[test]
    fn test_code_args() {
        let args = CodeArgs {
            task: "test task".to_string(),
            cwd: None,
            tools: Some("bash,file".to_string()),
            max_turns: 10,
            no_repomap: false,
            verify: Vec::new(),
            no_verify: false,
            accept_edits: false,
            dry_run: true,
            stream: false,
            provider: None,
            model: None,
            yolo: false,
            workspace_embed: true,
            workspace_options: WorkspaceEmbedOptions::default(),
            complex: false,
            context_window: None,
        };
        assert_eq!(args.task, "test task");
        assert!(args.dry_run);
    }

    #[test]
    fn coding_task_mode_uses_full_auto_so_failed_tests_repair() {
        assert_eq!(coding_task_mode(false), CodingMode::FullAuto);
        assert_eq!(coding_task_mode(true), CodingMode::Plan);
    }

    #[test]
    fn coding_verification_policy_maps_flags() {
        assert_eq!(
            coding_verification_policy(&[], false),
            VerificationPolicy::Auto
        );
        assert_eq!(
            coding_verification_policy(&["  ".to_string()], false),
            VerificationPolicy::Auto
        );
        assert_eq!(
            coding_verification_policy(&["cargo test --lib command_ledger".to_string()], false),
            VerificationPolicy::Commands(vec!["cargo test --lib command_ledger".to_string()])
        );
        assert_eq!(
            coding_verification_policy(&["cargo test".to_string()], true),
            VerificationPolicy::None
        );
    }

    #[test]
    fn coding_auto_driver_keeps_all_tools_available() {
        assert_eq!(
            coding_tool_execution_driver(None),
            Some(ToolExecutionDriver::DedupOnly)
        );
        assert_eq!(
            coding_tool_execution_driver(Some("auto")),
            Some(ToolExecutionDriver::DedupOnly)
        );
        assert_eq!(
            coding_tool_execution_driver(Some("phased")),
            Some(ToolExecutionDriver::PlannerFirstPhased)
        );
        assert_eq!(
            coding_tool_execution_driver(Some("off")),
            Some(ToolExecutionDriver::Off)
        );
    }

    #[test]
    fn coding_permission_mode_defaults_to_accept_edits_off_tty() {
        assert_eq!(
            coding_permission_mode(false, false, false),
            PermissionModeConfig::AcceptEdits
        );
        assert_eq!(
            coding_permission_mode(false, true, true),
            PermissionModeConfig::AcceptAll
        );
        assert_eq!(
            coding_permission_mode(false, false, true),
            PermissionModeConfig::Default
        );
        assert_eq!(
            coding_permission_mode(true, false, true),
            PermissionModeConfig::AcceptEdits
        );
    }

    #[test]
    fn format_code_event_prints_fs_and_bash_progress() {
        let executing = AgentEvent::ToolExecuting {
            tool_id: "t1".into(),
            tool_name: "fs".into(),
            progress: Some(0.0),
            action_id: "act-1".into(),
        };
        assert_eq!(
            format_code_event(&executing, false).as_deref(),
            Some("[fs]\n")
        );

        let bash = AgentEvent::ToolExecuting {
            tool_id: "t2".into(),
            tool_name: "bash".into(),
            progress: Some(0.0),
            action_id: "act-2".into(),
        };
        assert_eq!(format_code_event(&bash, false).as_deref(), Some("[bash]\n"));

        let complete = AgentEvent::tool_complete(
            "t1",
            "fs",
            &caliche::agent::tools::ToolResult::success("ok"),
            12,
            "act-1",
        );
        assert_eq!(
            format_code_event(&complete, false).as_deref(),
            Some("[fs ok 12ms]\n")
        );

        let read = AgentEvent::FileRead {
            path: PathBuf::from("src/lib.rs"),
            lines: None,
            size_bytes: 10,
        };
        assert_eq!(
            format_code_event(&read, false).as_deref(),
            Some("[read] src/lib.rs\n")
        );
    }

    #[tokio::test]
    async fn coding_system_prompt_uses_cli_discipline_sections() {
        let prompt = coding_system_prompt(false, false, None);
        assert!(prompt.contains("IDENTITY:"));
        assert!(prompt.contains("EXECUTION:"));
        assert!(prompt.contains("SELF-VALIDATION:"));
        assert!(prompt.contains("FILE DISCIPLINE:"));
        assert!(prompt.contains("NEVER re-read a file you already read"));
        assert!(prompt.contains("cargo test"));
        assert!(prompt.contains("unreachable"));
        assert!(!prompt.contains("WORKSPACE CONTEXT (indexed embeddings)"));
    }

    #[tokio::test]
    async fn coding_system_prompt_includes_workspace_usage_when_embed_on() {
        let prompt = coding_system_prompt(true, false, None);
        assert!(prompt.contains("WORKSPACE CONTEXT (indexed embeddings)"));
        assert!(prompt.contains("Relevant workspace context"));
    }

    #[tokio::test]
    async fn code_event_printer_reports_lag_with_the_skipped_count() {
        use tokio::sync::broadcast;

        let (tx, mut rx) = broadcast::channel(1);
        tx.send(AgentEvent::Info {
            message: "m0".into(),
        })
        .expect("send");
        tx.send(AgentEvent::Info {
            message: "m1".into(),
        })
        .expect("send");
        tx.send(AgentEvent::Info {
            message: "m2".into(),
        })
        .expect("send");
        drop(tx);

        let mut lines = Vec::new();
        loop {
            match code_printer_step(rx.recv().await, false) {
                CodePrinterStep::Line(text) => lines.push(text),
                CodePrinterStep::Skip => {}
                CodePrinterStep::Stop => break,
            }
        }
        assert_eq!(
            lines,
            vec![
                "[resync] skipped 2 events; re-read session state before trusting later events\n"
                    .to_owned(),
                "[info] m2\n".to_owned(),
            ]
        );
    }
}
