//! REPL main loop (slash commands, @mentions, message dispatch).

use super::{InputResult, ReplEngine};
use crate::cancellation::CancellationToken;
use crate::context_refs::{ContextResolver, ResolvedContext};
use crate::core::session::SessionManager;
use crate::core::slash_commands::{SlashCommandExecutor, SlashCommandResult};
use crate::core::traits::{InputSource, OutputRenderer};
use crate::handlers;
use crate::repl::commands::{
    HarnessCommand, KnowledgeCommand, ProviderCommand, SessionCommand, SlashCommand, SystemCommand,
};
use crate::repl::profile_harness;
use crate::storage::Database;
use anyhow::{Context, Result};
use caliche::AgentService;
use caliche::agent::profile::AgentProfile;
use caliche::harness::formatter::FormatterRule;
use caliche::harness::{AgentHarness, HarnessConfig};
use std::path::PathBuf;
use std::sync::Arc;

impl<I: InputSource + Clone + 'static, R: OutputRenderer> ReplEngine<I, R> {
    /// Check whether the current caller may start the subagent named in an `@mention`.
    ///
    /// While a primary agent override is active that profile is the caller, so its own
    /// `task_permissions` decide. Otherwise the workspace `orchestration.task_acl` applies.
    async fn authorize_mention(&self, target: &str) -> Result<()> {
        let caller = self.state.current_agent_override.as_deref();
        if let Some(caller) = caller {
            match handlers::load_profile_by_name(caller).await {
                Ok(Some(profile)) => {
                    let policy = profile
                        .task_policy()
                        .with_context(|| format!("Read task_permissions of profile '{caller}'"))?;
                    if !policy.is_empty() {
                        return policy.authorize(caller, target);
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        profile = caller,
                        error = %e,
                        "Could not read the active profile; applying the workspace task ACL"
                    );
                }
            }
        }
        self.mention_policy
            .authorize(caller.unwrap_or("user"), target)
    }

    async fn remote_session_slash_result(
        &self,
        cmd: &SlashCommand,
    ) -> Option<Result<SlashCommandResult>> {
        let client = self.thin_client.as_ref()?;
        let session_id = self.thin_session_id.as_deref()?;
        let result = match cmd {
            SlashCommand::Harness(HarnessCommand::Undo) => client
                .undo_last(session_id)
                .await
                .map(|response| SlashCommandResult::StatusMessage(response.status_message())),
            SlashCommand::Harness(HarnessCommand::Redo) => client
                .redo_last(session_id)
                .await
                .map(|response| SlashCommandResult::StatusMessage(response.status_message())),
            SlashCommand::Knowledge(KnowledgeCommand::Diff(None)) => client
                .session_diff(session_id)
                .await
                .map(|response| SlashCommandResult::ShowPopup {
                    title: "Remote session diff (per-message undo)".to_string(),
                    content: format_remote_session_diff(&response),
                }),
            SlashCommand::Harness(HarnessCommand::Share) => {
                client.share_session(session_id).await.map(|response| {
                    SlashCommandResult::StatusMessage(format!(
                        "Share created: {} ({} redaction(s))",
                        response.url, response.redaction_count
                    ))
                })
            }
            SlashCommand::Harness(HarnessCommand::Unshare) => client
                .unshare_session(session_id)
                .await
                .map(|()| SlashCommandResult::StatusMessage("Share links removed".to_string())),
            _ => return None,
        };
        Some(result)
    }

    pub async fn run(
        &mut self,
        service: &mut AgentService,
        session_store: crate::storage::SessionStore,
        preloaded_config: Option<PreloadedConfig>,
        cancellation_token: Option<&CancellationToken>,
        database: Option<&Database>,
    ) -> Result<()> {
        let (
            formatter_rules,
            notify_command,
            tool_execution_driver,
            auto_skill,
            use_harness_turns,
            auto_decompose,
        ) = preloaded_config.unwrap_or_else(|| self.default_loop_config());
        let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        let harness = if use_harness_turns {
            let harness_config = HarnessConfig::default()
                .with_provider(service.provider())
                .with_model(service.model().to_string())
                .with_workspace(workspace.clone())
                .with_auto_checkpoint(false)
                .with_formatters(formatter_rules.clone())
                .with_system_prompt(service.agent().system_prompt().to_string());
            let harness_config = if let Some(ref cmd) = notify_command {
                harness_config.with_notify_command(cmd.clone())
            } else {
                harness_config
            };
            Some(
                AgentHarness::new(harness_config)
                    .await
                    .context("Failed to create harness for REPL turns")?,
            )
        } else {
            None
        };

        let mut ctx = LoopContext {
            harness,
            session_manager: SessionManager::from_store(session_store),
            workspace,
            formatter_rules,
            notify_command,
            tool_execution_driver,
            auto_skill,
            use_harness_turns,
            auto_decompose,
            cancellation_token,
            database,
        };

        // Preload custom commands in background so first slash command is responsive
        let custom_commands = Arc::clone(&self.custom_commands);
        let commands_loaded = Arc::clone(&self.commands_loaded);
        tokio::spawn(async move {
            Self::ensure_commands_loaded(custom_commands, commands_loaded).await;
        });

        let mut cancel_hint_shown = false;
        loop {
            if let Step::Exit = self.poll_cancellation(cancellation_token, &mut cancel_hint_shown) {
                break;
            }

            if let Some(ref mut deferred) = self.deferred_workspace {
                if let Some(msg) = deferred.poll_apply(service).await {
                    self.renderer.show_status(&msg);
                }
            }

            let prompt = self.build_prompt();
            match Self::read_logical_message(&mut self.input, &prompt).await {
                Ok(Some(line)) => {
                    if let Step::Exit = Box::pin(self.handle_line(&mut ctx, service, line)).await? {
                        break;
                    }
                }
                Ok(None) => {
                    if let Err(e) = self.input.save_history() {
                        tracing::warn!(error = %e, "failed to save REPL history on EOF");
                    }
                    break;
                }
                Err(err) => {
                    let err = err.to_string();
                    self.renderer.show_error(&format!("Prompt error: {}", err));
                    if let Err(e) = self.input.save_history() {
                        tracing::warn!(
                            error = %e,
                            "failed to save REPL history after prompt error"
                        );
                    }
                    break;
                }
            }
        }

        // Save session if requested
        if self.state.save_session {
            if let Err(e) = ctx.session_manager.save(&self.state).await {
                tracing::warn!(error = %e, "failed to save REPL session on exit");
            }
        }

        Ok(())
    }

    /// The loop settings when the caller did not preload them.
    fn default_loop_config(&self) -> PreloadedConfig {
        // `preloaded_config` should always be `Some` when called via
        // `run_repl` (which pre-loads config before constructing the
        // engine). Reaching this branch means a caller bypassed the
        // standard entry point; fall back to safe defaults rather than
        // performing a synchronous disk read inside the REPL loop.
        tracing::warn!(
            "ReplEngine::run called without preloaded_config; \
                 using safe defaults (no disk I/O). \
                 Call run_repl() to avoid this."
        );
        (
            vec![],
            self.notify_command.clone(),
            None,
            true,
            false,
            false,
        )
    }

    /// Show the Ctrl+C hint once per interrupt, and stop after a second one.
    fn poll_cancellation(
        &mut self,
        cancellation_token: Option<&CancellationToken>,
        cancel_hint_shown: &mut bool,
    ) -> Step {
        let Some(token) = cancellation_token else {
            return Step::Continue;
        };
        if token.exit_requested() {
            self.renderer.show_status("Interrupted twice. Exiting.");
            self.save_history_on_exit();
            return Step::Exit;
        }
        if !token.is_cancelled() {
            *cancel_hint_shown = false;
        } else if !*cancel_hint_shown {
            self.renderer.show_status("Press Ctrl+C again to exit.");
            *cancel_hint_shown = true;
        }
        Step::Continue
    }

    fn save_history_on_exit(&mut self) {
        if let Err(e) = self.input.save_history() {
            tracing::warn!(error = %e, "failed to save REPL history on exit");
        }
    }

    async fn handle_line(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        line: String,
    ) -> Result<Step> {
        if !line.is_empty() {
            self.input.add_history(&line);
        }
        match self.process_input(&line) {
            InputResult::Continue => Ok(Step::Continue),
            InputResult::Exit => {
                self.save_history_on_exit();
                Ok(Step::Exit)
            }
            InputResult::Message(msg) => Box::pin(self.handle_message(ctx, service, &msg)).await,
        }
    }

    async fn handle_message(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        msg: &str,
    ) -> Result<Step> {
        if super::is_slash_command(msg) {
            return Box::pin(self.handle_slash_command(ctx, service, msg)).await;
        }

        // @mention for subagent (e.g. @reviewer <task>)
        if let Some((name, task_rest)) = subagent_mention(msg) {
            Box::pin(self.run_mention(ctx, service, name, task_rest)).await;
            return Ok(Step::Continue);
        }

        let resolved = Self::resolve_context_for_message(service, msg).await;
        let session_id = self.state.session_id.clone();
        let prompt_preview: String = resolved.message.chars().take(100).collect();

        // Primary agent override: run this turn with the named profile
        if let Some(name) = self.state.current_agent_override.clone() {
            Box::pin(self.run_override_turn(
                ctx,
                service,
                &name,
                resolved,
                (&session_id, &prompt_preview),
            ))
            .await;
            return Ok(Step::Continue);
        }

        self.run_message_turn(
            ctx.harness.as_mut(),
            service,
            &resolved.message,
            &ctx.tool_execution_driver,
            ctx.auto_skill,
            ctx.use_harness_turns,
            ctx.auto_decompose,
            ctx.cancellation_token,
            &mut ctx.session_manager,
            Some((&session_id, &prompt_preview)),
            resolved.attachments,
            resolved.allowed_paths,
        )
        .await?;
        Ok(Step::Continue)
    }

    /// Run a prompt whose `@` references are already resolved as a turn,
    /// without decomposition or a turn notification (custom commands and
    /// editor content).
    async fn run_resolved_context(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        resolved: ResolvedContext,
    ) -> Result<()> {
        self.run_message_turn(
            ctx.harness.as_mut(),
            service,
            &resolved.message,
            &ctx.tool_execution_driver,
            ctx.auto_skill,
            ctx.use_harness_turns,
            false,
            ctx.cancellation_token,
            &mut ctx.session_manager,
            None,
            resolved.attachments,
            resolved.allowed_paths,
        )
        .await
    }

    /// The custom command `msg` names, rendered with its arguments and its
    /// `@` references resolved.
    async fn custom_command_context(
        &self,
        service: &AgentService,
        msg: &str,
    ) -> Option<ResolvedContext> {
        let cmd_parts: Vec<&str> = msg.split_whitespace().collect();
        let cmd_name = cmd_parts.first().unwrap_or(&"");
        let custom_cmd = self.custom_commands.read().await.get(cmd_name)?.clone();
        let args: Vec<&str> = cmd_parts.get(1..).unwrap_or(&[]).to_vec();
        Some(
            custom_cmd
                .resolve_prompt(&args, ContextResolver::from_service(service))
                .await,
        )
    }

    async fn handle_slash_command(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        msg: &str,
    ) -> Result<Step> {
        Self::ensure_commands_loaded(
            Arc::clone(&self.custom_commands),
            Arc::clone(&self.commands_loaded),
        )
        .await;

        if let Some(resolved) = self.custom_command_context(service, msg).await {
            Box::pin(self.run_resolved_context(ctx, service, resolved)).await?;
            return Ok(Step::Continue);
        }

        // Not a custom command, try built-in commands
        let cmd = SlashCommand::parse(msg);
        if cmd.is_exit() {
            self.save_history_on_exit();
            return Ok(Step::Exit);
        }

        if let Some(result) = self.remote_session_slash_result(&cmd).await {
            match result {
                Ok(result) => self.renderer.handle_command_result(result),
                Err(e) => self
                    .renderer
                    .show_error(&format!("Remote command error: {:#}", e)),
            }
            return Ok(Step::Continue);
        }

        ensure_checkpoint_harness(ctx, service, &cmd).await?;

        let selection_before = (service.provider(), service.model().to_string());
        match SlashCommandExecutor::execute(
            &cmd,
            &mut self.state,
            service,
            ctx.harness.as_mut(),
            Some(&ctx.session_manager),
        )
        .await
        {
            Ok(result) => {
                Box::pin(self.after_slash_command(ctx, service, &cmd, result, selection_before))
                    .await
            }
            Err(e) => {
                self.renderer.show_error(&format!("Command error: {:#}", e));
                Ok(Step::Continue)
            }
        }
    }

    async fn after_slash_command(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        cmd: &SlashCommand,
        result: SlashCommandResult,
        selection_before: (caliche::llm::ProviderType, String),
    ) -> Result<Step> {
        // `/provider`, `/model`, and a setup that picks a new default all
        // change the selection; the next launch starts with it.
        if selection_before != (service.provider(), service.model().to_string()) {
            super::super::startup::persist_last_provider_model(service, ctx.database).await;
        }
        self.persist_command_settings(cmd, service, ctx.database)
            .await;
        let svc = ctx
            .harness
            .as_mut()
            .map(|h| h.service_mut())
            .unwrap_or(service);
        self.maybe_refresh_completions(svc);
        self.refresh_context_ref_completions(svc);

        // Handle editor content specially (parse @ refs like main flow)
        if let SlashCommandResult::EditorContent(content) = result {
            let resolved = Self::resolve_context_for_message(service, &content).await;
            Box::pin(self.run_resolved_context(ctx, service, resolved)).await?;
            return Ok(Step::Continue);
        }

        if matches!(cmd, SlashCommand::Session(SessionCommand::Resume(_)))
            && matches!(
                &result,
                SlashCommandResult::StatusMessage(msg) if msg.starts_with("Resumed")
            )
        {
            self.install_resumed_context(service).await;
        }

        // Sync display mode flags to renderer after command.
        if matches!(
            cmd,
            SlashCommand::System(SystemCommand::Verbose(_) | SystemCommand::Compact)
        ) {
            self.renderer
                .set_output_verbosity(self.state.output_verbosity);
        }

        let should_exit = matches!(result, SlashCommandResult::Exit);
        self.renderer.handle_command_result(result);
        if should_exit {
            self.save_history_on_exit();
            return Ok(Step::Exit);
        }
        Ok(Step::Continue)
    }

    /// Save the settings a successful command changed, so the next launch keeps them.
    async fn persist_command_settings(
        &self,
        cmd: &SlashCommand,
        service: &mut AgentService,
        database: Option<&Database>,
    ) {
        let SlashCommand::Provider(provider_cmd) = cmd else {
            return;
        };
        match provider_cmd {
            ProviderCommand::SubagentProvider(_) | ProviderCommand::SubagentModel(_) => {
                service.set_subagent_selection(profile_harness::selection_from_state(
                    self.state.subagent_provider.as_deref(),
                    self.state.subagent_model.as_deref(),
                ));
                save_settings(
                    database,
                    &[
                        (
                            "subagent_provider",
                            self.state.subagent_provider.as_deref().unwrap_or(""),
                        ),
                        (
                            "subagent_model",
                            self.state.subagent_model.as_deref().unwrap_or(""),
                        ),
                    ],
                )
                .await;
            }
            ProviderCommand::Embedding(Some(_)) => {
                save_settings(
                    database,
                    &[
                        (
                            "embedding_provider",
                            self.state.embedding_provider.as_deref().unwrap_or(""),
                        ),
                        (
                            "embedding_model",
                            self.state.embedding_model.as_deref().unwrap_or(""),
                        ),
                    ],
                )
                .await;
            }
            ProviderCommand::ImageGeneration(_) | ProviderCommand::ArtifactDir(_) => {
                save_settings(
                    database,
                    &[
                        (
                            "image_generation_provider",
                            self.state
                                .image_generation_provider
                                .as_deref()
                                .unwrap_or(""),
                        ),
                        (
                            "image_generation_model",
                            self.state.image_generation_model.as_deref().unwrap_or(""),
                        ),
                        (
                            "artifact_dir",
                            self.state
                                .artifact_dir
                                .as_ref()
                                .and_then(|path| path.to_str())
                                .unwrap_or(""),
                        ),
                    ],
                )
                .await;
            }
            _ => {}
        }
    }

    async fn install_resumed_context(&mut self, service: &mut AgentService) {
        let thin = self
            .thin_client
            .as_ref()
            .zip(self.thin_session_id.as_deref());
        match crate::repl::resume::install_resumed_context(service, &self.state.messages, thin)
            .await
        {
            Ok(Some(installed)) => {
                self.renderer
                    .show_status(&crate::repl::resume::resume_status_line(&installed));
            }
            Ok(None) => {}
            Err(error) => {
                self.renderer.show_error(&format!(
                    "Saved transcript did not reload into the agent: {error:#}"
                ));
            }
        }
    }

    fn profile_harness_config(
        &self,
        ctx: &LoopContext<'_>,
        service: &AgentService,
        profile: &AgentProfile,
    ) -> HarnessConfig {
        profile_harness::harness_config_from_profile(
            profile,
            service.provider(),
            ctx.workspace.clone(),
            ctx.formatter_rules.clone(),
            ctx.notify_command.clone(),
            service.model().to_string(),
            &profile_harness::selection_from_state(
                self.state.subagent_provider.as_deref(),
                self.state.subagent_model.as_deref(),
            ),
        )
    }

    /// Run one ephemeral turn with the subagent profile `name`.
    async fn run_mention(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        name: &str,
        task_rest: &str,
    ) {
        if let Err(e) = self.authorize_mention(name).await {
            self.renderer.show_error(&format!("{e:#}"));
            return;
        }
        match handlers::load_profile_by_name(name).await {
            Ok(Some(profile)) if !profile.is_subagent() => {
                self.renderer.show_error(&format!(
                    "Profile '{}' is not a subagent (mode: {}). Use @mention only with subagent profiles.",
                    name,
                    profile.mode.as_deref().unwrap_or("primary")
                ));
            }
            Ok(Some(profile)) => {
                Box::pin(self.run_subagent(ctx, service, &profile, task_rest)).await;
            }
            Ok(None) => {
                self.renderer.show_error(&format!(
                    "Unknown profile: '{}'. Use 'caliche agent list' to see profiles.",
                    name
                ));
            }
            Err(e) => {
                self.renderer
                    .show_error(&format!("Failed to load profile '{}': {}", name, e));
            }
        }
    }

    async fn run_subagent(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        profile: &AgentProfile,
        task_rest: &str,
    ) {
        let task_context = if task_rest.is_empty() {
            ResolvedContext {
                message: "Respond to the user.".to_string(),
                attachments: Vec::new(),
                allowed_paths: Vec::new(),
            }
        } else {
            Self::resolve_context_for_message(service, task_rest).await
        };
        let config = self.profile_harness_config(ctx, service, profile);
        let mut sub_harness = match caliche::AgentHarness::new(config)
            .await
            .context("Create subagent harness")
        {
            Ok(sub_harness) => sub_harness,
            Err(e) => {
                self.renderer
                    .show_error(&format!("Failed to create subagent: {}", e));
                return;
            }
        };
        if let Err(e) =
            profile_harness::apply_profile_bash_rules_to_service(profile, sub_harness.service_mut())
        {
            self.renderer
                .show_error(&format!("Profile bash rules: {}", e));
            return;
        }
        profile_harness::apply_profile_temperature_to_service(profile, sub_harness.service_mut());
        let len_before = self.state.messages.len();
        if let Err(e) = self
            .process_message(
                sub_harness.service_mut(),
                &task_context.message,
                ctx.cancellation_token,
                task_context.attachments,
                task_context.allowed_paths,
                true,
            )
            .await
        {
            self.renderer.show_error(&format!("Subagent error: {}", e));
        } else {
            // Ephemeral: do not merge into main session
            self.state.messages.truncate(len_before);
        }
        self.input.request_stdin_drain();
    }

    /// Run this turn with the primary agent override profile `name`.
    async fn run_override_turn(
        &mut self,
        ctx: &mut LoopContext<'_>,
        service: &mut AgentService,
        name: &str,
        resolved: ResolvedContext,
        notify_ctx: (&str, &str),
    ) {
        let profile = match handlers::load_profile_by_name(name).await {
            Ok(Some(profile)) => profile,
            Ok(None) => {
                self.renderer
                    .show_error(&format!("Override profile '{}' not found. Cleared.", name));
                self.state.current_agent_override = None;
                return;
            }
            Err(e) => {
                self.renderer
                    .show_error(&format!("Failed to load override profile: {}", e));
                return;
            }
        };
        let config = self.profile_harness_config(ctx, service, &profile);
        let mut sub_harness = match caliche::AgentHarness::new(config)
            .await
            .context("Create override agent harness")
        {
            Ok(sub_harness) => sub_harness,
            Err(e) => {
                self.renderer
                    .show_error(&format!("Failed to create override agent: {}", e));
                return;
            }
        };
        if let Err(e) = profile_harness::apply_profile_bash_rules_to_service(
            &profile,
            sub_harness.service_mut(),
        ) {
            self.renderer
                .show_error(&format!("Profile bash rules: {}", e));
            return;
        }
        profile_harness::apply_profile_temperature_to_service(&profile, sub_harness.service_mut());
        if let Err(e) = self
            .run_message_turn(
                None,
                sub_harness.service_mut(),
                &resolved.message,
                &ctx.tool_execution_driver,
                ctx.auto_skill,
                false,
                false,
                ctx.cancellation_token,
                &mut ctx.session_manager,
                Some(notify_ctx),
                resolved.attachments,
                resolved.allowed_paths,
            )
            .await
        {
            self.renderer
                .show_error(&format!("Override agent error: {}", e));
        } else if self.state.should_autosave(30) {
            if let Err(e) = ctx.session_manager.maybe_autosave(&self.state).await {
                tracing::warn!(error = %e, "failed to autosave REPL session");
            }
            self.state.mark_autosaved();
        }
    }
}

/// Settings `run_repl` loads before the loop starts: formatter rules, the notify
/// command, the tool execution driver, auto-skill, harness turns, and auto-decompose.
type PreloadedConfig = (
    Vec<FormatterRule>,
    Option<String>,
    Option<String>,
    bool,
    bool,
    bool,
);

/// Whether the loop reads another line or stops.
enum Step {
    Continue,
    Exit,
}

/// What the loop keeps between lines.
struct LoopContext<'a> {
    harness: Option<AgentHarness>,
    session_manager: SessionManager,
    workspace: PathBuf,
    formatter_rules: Vec<FormatterRule>,
    notify_command: Option<String>,
    tool_execution_driver: Option<String>,
    auto_skill: bool,
    use_harness_turns: bool,
    auto_decompose: bool,
    cancellation_token: Option<&'a CancellationToken>,
    database: Option<&'a Database>,
}

/// The profile name and task of an `@name task` subagent mention.
///
/// A name holds only letters, digits, `-` and `_`; any other `@` word (for
/// example `@src/main.rs`) is a context reference, so the line is a message.
fn subagent_mention(msg: &str) -> Option<(&str, &str)> {
    let after_at = msg.trim_start().strip_prefix('@')?.trim_start();
    let name = after_at.split_whitespace().next().unwrap_or("");
    let is_name = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_');
    is_name.then(|| (name, after_at[name.len()..].trim()))
}

/// Create the harness on the first `/checkpoint` or `/rollback`, from the
/// current service so `/provider` and `/model` are reflected.
async fn ensure_checkpoint_harness(
    ctx: &mut LoopContext<'_>,
    service: &AgentService,
    cmd: &SlashCommand,
) -> Result<()> {
    let wants_checkpoints = matches!(
        cmd,
        SlashCommand::Harness(HarnessCommand::Checkpoint(_) | HarnessCommand::Rollback(_))
    );
    if !wants_checkpoints || ctx.harness.is_some() {
        return Ok(());
    }
    let mut harness_config = HarnessConfig::default()
        .with_provider(service.provider())
        .with_model(service.model().to_string())
        .with_workspace(ctx.workspace.clone())
        .with_auto_checkpoint(false)
        .with_formatters(ctx.formatter_rules.clone());
    if let Some(ref cmd) = ctx.notify_command {
        harness_config = harness_config.with_notify_command(cmd.clone());
    }
    ctx.harness = Some(
        AgentHarness::new(harness_config)
            .await
            .context("Failed to create harness for checkpoint support")?,
    );
    Ok(())
}

/// Save each setting in order; a failure is logged and the rest are still saved.
async fn save_settings(database: Option<&Database>, settings: &[(&str, &str)]) {
    let Some(db) = database else {
        return;
    };
    for (key, value) in settings {
        if let Err(e) = db.save_setting(key, value).await {
            tracing::warn!(error = %e, "failed to persist {key} setting");
        }
    }
}

fn format_remote_session_diff(summary: &caliche_contracts::sessions::DiffResponseDto) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Active change sets: {} | Reverted change sets: {}",
        summary.active_change_sets, summary.reverted_change_sets
    ));
    lines.push(format!(
        "Can undo: {} | Can redo: {}",
        summary.can_undo, summary.can_redo
    ));

    if summary.messages.is_empty() && summary.reverted_messages.is_empty() {
        lines.push("No per-message file changes recorded.".to_string());
        return lines.join("\n");
    }

    if !summary.messages.is_empty() {
        lines.push("\nActive:".to_string());
        for message in &summary.messages {
            lines.push(format!(
                "- {} ({} file change(s))",
                message.message_id, message.changed_files
            ));
            for snapshot in &message.snapshots {
                lines.push(format!(
                    "  - {:?} {} ({} -> {})",
                    snapshot.change,
                    snapshot.path,
                    snapshot
                        .before_len
                        .map(|len| len.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    snapshot
                        .after_len
                        .map(|len| len.to_string())
                        .unwrap_or_else(|| "-".to_string())
                ));
            }
        }
    }

    if !summary.reverted_messages.is_empty() {
        lines.push("\nReverted:".to_string());
        for message in &summary.reverted_messages {
            lines.push(format!(
                "- {} ({} file change(s))",
                message.message_id, message.changed_files
            ));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
#[path = "run_loop_tests.rs"]
mod tests;
