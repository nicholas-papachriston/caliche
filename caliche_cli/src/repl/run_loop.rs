//! REPL main loop (slash commands, @mentions, message dispatch).

use super::{InputResult, ReplEngine};
use crate::context_refs::ResolvedContext;
use crate::core::slash_commands::{SlashCommandExecutor, SlashCommandResult};
use crate::core::traits::{InputSource, OutputRenderer};
use crate::handlers;
use crate::repl::commands::{
    HarnessCommand, KnowledgeCommand, ProviderCommand, SessionCommand, SlashCommand, SystemCommand,
};
use crate::repl::profile_harness;
use anyhow::{Context, Result};
use caliche::AgentService;
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
        preloaded_config: Option<(
            Vec<caliche::harness::formatter::FormatterRule>,
            Option<String>,
            Option<String>,
            bool,
            bool,
            bool,
        )>,
        cancellation_token: Option<&crate::cancellation::CancellationToken>,
        database: Option<&crate::storage::Database>,
    ) -> Result<()> {
        use crate::core::session::SessionManager;
        use caliche::harness::{AgentHarness, HarnessConfig};

        let workspace_root = std::env::current_dir().ok();
        let workspace = workspace_root.clone().unwrap_or_else(|| PathBuf::from("."));

        let (
            formatter_rules,
            notify_command,
            tool_execution_driver,
            auto_skill,
            use_harness_turns,
            auto_decompose,
        ) = match preloaded_config {
            Some((rules, cmd, driver, auto, use_harness, auto_decomp)) => {
                (rules, cmd, driver, auto, use_harness, auto_decomp)
            }
            None => {
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
        };

        let mut harness: Option<AgentHarness> = None;

        if use_harness_turns {
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
            harness = Some(
                AgentHarness::new(harness_config)
                    .await
                    .context("Failed to create harness for REPL turns")?,
            );
        }

        let mut session_manager = SessionManager::from_store(session_store);

        // Preload custom commands in background so first slash command is responsive
        let custom_commands = Arc::clone(&self.custom_commands);
        let commands_loaded = Arc::clone(&self.commands_loaded);
        tokio::spawn(async move {
            Self::ensure_commands_loaded(custom_commands, commands_loaded).await;
        });

        let mut cancel_hint_shown = false;
        loop {
            if let Some(token) = cancellation_token {
                if token.exit_requested() {
                    self.renderer.show_status("Interrupted twice. Exiting.");
                    if let Err(e) = self.input.save_history() {
                        tracing::warn!(error = %e, "failed to save REPL history on exit");
                    }
                    break;
                }
                if token.is_cancelled() {
                    if !cancel_hint_shown {
                        self.renderer.show_status("Press Ctrl+C again to exit.");
                        cancel_hint_shown = true;
                    }
                } else {
                    cancel_hint_shown = false;
                }
            }

            if let Some(ref mut deferred) = self.deferred_workspace {
                if let Some(msg) = deferred.poll_apply(service).await {
                    self.renderer.show_status(&msg);
                }
            }

            let prompt = self.build_prompt();
            match Self::read_logical_message(&mut self.input, &prompt).await {
                Ok(Some(line)) => {
                    if !line.is_empty() {
                        self.input.add_history(&line);
                    }
                    match self.process_input(&line) {
                        InputResult::Continue => continue,
                        InputResult::Exit => {
                            if let Err(e) = self.input.save_history() {
                                tracing::warn!(error = %e, "failed to save REPL history on exit");
                            }
                            break;
                        }
                        InputResult::Message(msg) => {
                            // Check if it's a slash command
                            if super::is_slash_command(&msg) {
                                Self::ensure_commands_loaded(
                                    Arc::clone(&self.custom_commands),
                                    Arc::clone(&self.commands_loaded),
                                )
                                .await;

                                // First check for custom commands
                                let cmd_parts: Vec<&str> = msg.split_whitespace().collect();
                                let cmd_name = cmd_parts.first().unwrap_or(&"");

                                // Check custom commands
                                let custom_prompt = {
                                    let registry = self.custom_commands.read().await;
                                    if let Some(custom_cmd) = registry.get(cmd_name) {
                                        // Extract arguments
                                        let args: Vec<&str> = cmd_parts
                                            .get(1..)
                                            .unwrap_or(&[])
                                            .iter()
                                            .copied()
                                            .collect();

                                        // Render the prompt with arguments
                                        Some(custom_cmd.render_prompt(&args))
                                    } else {
                                        None
                                    }
                                };

                                if let Some(prompt) = custom_prompt {
                                    let resolved =
                                        Self::resolve_context_for_message(service, &prompt).await;
                                    self.run_message_turn(
                                        harness.as_mut(),
                                        service,
                                        &resolved.message,
                                        &tool_execution_driver,
                                        auto_skill,
                                        use_harness_turns,
                                        false,
                                        cancellation_token,
                                        &mut session_manager,
                                        None,
                                        resolved.attachments,
                                        resolved.allowed_paths,
                                    )
                                    .await?;
                                    continue;
                                }

                                // Not a custom command, try built-in commands
                                let cmd = SlashCommand::parse(&msg);
                                if cmd.is_exit() {
                                    if let Err(e) = self.input.save_history() {
                                        tracing::warn!(
                                            error = %e,
                                            "failed to save REPL history on exit"
                                        );
                                    }
                                    break;
                                }

                                if let Some(result) = self.remote_session_slash_result(&cmd).await {
                                    match result {
                                        Ok(result) => self.renderer.handle_command_result(result),
                                        Err(e) => self
                                            .renderer
                                            .show_error(&format!("Remote command error: {:#}", e)),
                                    }
                                    continue;
                                }

                                // Lazy-create harness only when /checkpoint or /rollback is first used.
                                // Build config from current service so /provider and /model are reflected.
                                if matches!(
                                    cmd,
                                    SlashCommand::Harness(
                                        HarnessCommand::Checkpoint(_) | HarnessCommand::Rollback(_),
                                    )
                                ) && harness.is_none()
                                {
                                    let mut harness_config = HarnessConfig::default()
                                        .with_provider(service.provider())
                                        .with_model(service.model().to_string())
                                        .with_workspace(workspace.clone())
                                        .with_auto_checkpoint(false)
                                        .with_formatters(formatter_rules.clone());
                                    if let Some(ref cmd) = notify_command {
                                        harness_config =
                                            harness_config.with_notify_command(cmd.clone());
                                    }
                                    harness =
                                        Some(AgentHarness::new(harness_config).await.context(
                                            "Failed to create harness for checkpoint support",
                                        )?);
                                }

                                let selection_before =
                                    (service.provider(), service.model().to_string());
                                match SlashCommandExecutor::execute(
                                    &cmd,
                                    &mut self.state,
                                    service,
                                    harness.as_mut(),
                                    Some(&session_manager),
                                )
                                .await
                                {
                                    Ok(result) => {
                                        // `/provider`, `/model`, and a setup that picks a
                                        // new default all change the selection; the next
                                        // launch starts with it.
                                        if selection_before
                                            != (service.provider(), service.model().to_string())
                                        {
                                            super::super::startup::persist_last_provider_model(
                                                service, database,
                                            )
                                            .await;
                                        }
                                        if matches!(
                                            cmd,
                                            SlashCommand::Provider(
                                                ProviderCommand::SubagentProvider(_)
                                                    | ProviderCommand::SubagentModel(_),
                                            )
                                        ) {
                                            service.set_subagent_selection(
                                                profile_harness::selection_from_state(
                                                    self.state.subagent_provider.as_deref(),
                                                    self.state.subagent_model.as_deref(),
                                                ),
                                            );
                                            if let Some(db) = database {
                                                let p = self
                                                    .state
                                                    .subagent_provider
                                                    .as_deref()
                                                    .unwrap_or("");
                                                let m = self
                                                    .state
                                                    .subagent_model
                                                    .as_deref()
                                                    .unwrap_or("");
                                                if let Err(e) =
                                                    db.save_setting("subagent_provider", p).await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist subagent_provider setting"
                                                    );
                                                }
                                                if let Err(e) =
                                                    db.save_setting("subagent_model", m).await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist subagent_model setting"
                                                    );
                                                }
                                            }
                                        }
                                        if matches!(
                                            cmd,
                                            SlashCommand::Provider(ProviderCommand::Embedding(
                                                Some(_)
                                            ))
                                        ) {
                                            if let Some(db) = database {
                                                let provider = self
                                                    .state
                                                    .embedding_provider
                                                    .as_deref()
                                                    .unwrap_or("");
                                                let model = self
                                                    .state
                                                    .embedding_model
                                                    .as_deref()
                                                    .unwrap_or("");
                                                if let Err(e) = db
                                                    .save_setting("embedding_provider", provider)
                                                    .await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist embedding_provider setting"
                                                    );
                                                }
                                                if let Err(e) =
                                                    db.save_setting("embedding_model", model).await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist embedding_model setting"
                                                    );
                                                }
                                            }
                                        }
                                        if matches!(
                                            cmd,
                                            SlashCommand::Provider(
                                                ProviderCommand::ImageGeneration(_)
                                                    | ProviderCommand::ArtifactDir(_),
                                            )
                                        ) {
                                            if let Some(db) = database {
                                                let provider = self
                                                    .state
                                                    .image_generation_provider
                                                    .as_deref()
                                                    .unwrap_or("");
                                                let model = self
                                                    .state
                                                    .image_generation_model
                                                    .as_deref()
                                                    .unwrap_or("");
                                                let artifact_dir = self
                                                    .state
                                                    .artifact_dir
                                                    .as_ref()
                                                    .and_then(|path| path.to_str())
                                                    .unwrap_or("");
                                                if let Err(e) = db
                                                    .save_setting(
                                                        "image_generation_provider",
                                                        provider,
                                                    )
                                                    .await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist image_generation_provider setting"
                                                    );
                                                }
                                                if let Err(e) = db
                                                    .save_setting("image_generation_model", model)
                                                    .await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist image_generation_model setting"
                                                    );
                                                }
                                                if let Err(e) = db
                                                    .save_setting("artifact_dir", artifact_dir)
                                                    .await
                                                {
                                                    tracing::warn!(
                                                        error = %e,
                                                        "failed to persist artifact_dir setting"
                                                    );
                                                }
                                            }
                                        }
                                        let svc = harness
                                            .as_mut()
                                            .map(|h| h.service_mut())
                                            .unwrap_or(service);
                                        self.maybe_refresh_completions(svc);
                                        self.refresh_context_ref_completions(svc);

                                        // Handle editor content specially (parse @ refs like main flow)
                                        if let SlashCommandResult::EditorContent(content) = result {
                                            let resolved = Self::resolve_context_for_message(
                                                service, &content,
                                            )
                                            .await;
                                            self.run_message_turn(
                                                harness.as_mut(),
                                                service,
                                                &resolved.message,
                                                &tool_execution_driver,
                                                auto_skill,
                                                use_harness_turns,
                                                false,
                                                cancellation_token,
                                                &mut session_manager,
                                                None,
                                                resolved.attachments,
                                                resolved.allowed_paths,
                                            )
                                            .await?;
                                            continue;
                                        }

                                        if matches!(
                                            cmd,
                                            SlashCommand::Session(SessionCommand::Resume(_))
                                        ) && matches!(
                                            &result,
                                            SlashCommandResult::StatusMessage(msg)
                                                if msg.starts_with("Resumed")
                                        ) {
                                            let thin = self
                                                .thin_client
                                                .as_ref()
                                                .zip(self.thin_session_id.as_deref());
                                            match crate::repl::resume::install_resumed_context(
                                                service,
                                                &self.state.messages,
                                                thin,
                                            )
                                            .await
                                            {
                                                Ok(Some(installed)) => {
                                                    self.renderer.show_status(
                                                        &crate::repl::resume::resume_status_line(
                                                            &installed,
                                                        ),
                                                    );
                                                }
                                                Ok(None) => {}
                                                Err(error) => {
                                                    self.renderer.show_error(&format!(
                                                        "Saved transcript did not reload into the agent: {error:#}"
                                                    ));
                                                }
                                            }
                                        }

                                        // Sync display mode flags to renderer after command.
                                        if matches!(
                                            cmd,
                                            SlashCommand::System(
                                                SystemCommand::Verbose(_) | SystemCommand::Compact,
                                            )
                                        ) {
                                            self.renderer
                                                .set_output_verbosity(self.state.output_verbosity);
                                        }

                                        let should_exit =
                                            matches!(result, SlashCommandResult::Exit);
                                        self.renderer.handle_command_result(result);

                                        // Handle exit
                                        if should_exit {
                                            if let Err(e) = self.input.save_history() {
                                                tracing::warn!(
                                                    error = %e,
                                                    "failed to save REPL history on exit"
                                                );
                                            }
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        self.renderer
                                            .show_error(&format!("Command error: {:#}", e));
                                    }
                                }
                                continue;
                            }

                            // @mention for subagent (e.g. @reviewer <task>)
                            if msg.trim_start().starts_with('@') {
                                let after_at = msg
                                    .trim_start()
                                    .strip_prefix('@')
                                    .unwrap_or("")
                                    .trim_start();
                                let name = after_at.split_whitespace().next().unwrap_or("");
                                if !name.is_empty()
                                    && name
                                        .chars()
                                        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
                                {
                                    let task_rest = after_at[name.len()..].trim();
                                    if let Err(e) = self.authorize_mention(name).await {
                                        self.renderer.show_error(&format!("{e:#}"));
                                        continue;
                                    }
                                    match handlers::load_profile_by_name(name).await {
                                        Ok(Some(profile)) => {
                                            if !profile.is_subagent() {
                                                self.renderer.show_error(&format!(
                                                    "Profile '{}' is not a subagent (mode: {}). Use @mention only with subagent profiles.",
                                                    name,
                                                    profile.mode.as_deref().unwrap_or("primary")
                                                ));
                                                continue;
                                            }
                                            let task_context = if task_rest.is_empty() {
                                                ResolvedContext {
                                                    message: "Respond to the user.".to_string(),
                                                    attachments: Vec::new(),
                                                    allowed_paths: Vec::new(),
                                                }
                                            } else {
                                                Self::resolve_context_for_message(
                                                    service, task_rest,
                                                )
                                                .await
                                            };
                                            let workspace = workspace_root
                                                .clone()
                                                .unwrap_or_else(|| PathBuf::from("."));
                                            let config =
                                                profile_harness::harness_config_from_profile(
                                                    &profile,
                                                    service.provider(),
                                                    workspace.clone(),
                                                    formatter_rules.clone(),
                                                    notify_command.clone(),
                                                    service.model().to_string(),
                                                    &profile_harness::selection_from_state(
                                                        self.state.subagent_provider.as_deref(),
                                                        self.state.subagent_model.as_deref(),
                                                    ),
                                                );
                                            match caliche::AgentHarness::new(config)
                                                .await
                                                .context("Create subagent harness")
                                            {
                                                Ok(mut sub_harness) => {
                                                    if let Err(e) =
                                                        profile_harness::apply_profile_bash_rules_to_service(
                                                            &profile,
                                                            sub_harness.service_mut(),
                                                        )
                                                    {
                                                        self.renderer.show_error(&format!(
                                                            "Profile bash rules: {}",
                                                            e
                                                        ));
                                                        continue;
                                                    }
                                                    profile_harness::apply_profile_temperature_to_service(
                                                        &profile,
                                                        sub_harness.service_mut(),
                                                    );
                                                    let len_before = self.state.messages.len();
                                                    if let Err(e) = self
                                                        .process_message(
                                                            sub_harness.service_mut(),
                                                            &task_context.message,
                                                            cancellation_token,
                                                            task_context.attachments,
                                                            task_context.allowed_paths,
                                                            true,
                                                        )
                                                        .await
                                                    {
                                                        self.renderer.show_error(&format!(
                                                            "Subagent error: {}",
                                                            e
                                                        ));
                                                    } else {
                                                        // Ephemeral: do not merge into main session
                                                        self.state.messages.truncate(len_before);
                                                    }
                                                    self.input.request_stdin_drain();
                                                    continue;
                                                }
                                                Err(e) => {
                                                    self.renderer.show_error(&format!(
                                                        "Failed to create subagent: {}",
                                                        e
                                                    ));
                                                }
                                            }
                                            continue;
                                        }
                                        Ok(None) => {
                                            self.renderer.show_error(&format!(
                                                "Unknown profile: '{}'. Use 'caliche agent list' to see profiles.",
                                                name
                                            ));
                                            continue;
                                        }
                                        Err(e) => {
                                            self.renderer.show_error(&format!(
                                                "Failed to load profile '{}': {}",
                                                name, e
                                            ));
                                            continue;
                                        }
                                    }
                                }
                            }

                            let resolved = Self::resolve_context_for_message(service, &msg).await;
                            let session_id = self.state.session_id.clone();
                            let prompt_preview: String =
                                resolved.message.chars().take(100).collect();

                            // Primary agent override: run this turn with the named profile
                            if let Some(ref name) = self.state.current_agent_override {
                                match handlers::load_profile_by_name(name).await {
                                    Ok(Some(profile)) => {
                                        let workspace = workspace_root
                                            .clone()
                                            .unwrap_or_else(|| PathBuf::from("."));
                                        let config = profile_harness::harness_config_from_profile(
                                            &profile,
                                            service.provider(),
                                            workspace,
                                            formatter_rules.clone(),
                                            notify_command.clone(),
                                            service.model().to_string(),
                                            &profile_harness::selection_from_state(
                                                self.state.subagent_provider.as_deref(),
                                                self.state.subagent_model.as_deref(),
                                            ),
                                        );
                                        match caliche::AgentHarness::new(config)
                                            .await
                                            .context("Create override agent harness")
                                        {
                                            Ok(mut sub_harness) => {
                                                if let Err(e) =
                                                    profile_harness::apply_profile_bash_rules_to_service(
                                                        &profile,
                                                        sub_harness.service_mut(),
                                                    )
                                                {
                                                    self.renderer.show_error(&format!(
                                                        "Profile bash rules: {}",
                                                        e
                                                    ));
                                                } else {
                                                    profile_harness::apply_profile_temperature_to_service(
                                                        &profile,
                                                        sub_harness.service_mut(),
                                                    );
                                                    if let Err(e) = self
                                                        .run_message_turn(
                                                            None,
                                                            sub_harness.service_mut(),
                                                            &resolved.message,
                                                            &tool_execution_driver,
                                                            auto_skill,
                                                            false,
                                                            false,
                                                            cancellation_token,
                                                            &mut session_manager,
                                                            Some((&session_id, &prompt_preview)),
                                                            resolved.attachments.clone(),
                                                            resolved.allowed_paths.clone(),
                                                        )
                                                        .await
                                                    {
                                                        self.renderer.show_error(&format!(
                                                            "Override agent error: {}",
                                                            e
                                                        ));
                                                    } else if self.state.should_autosave(30) {
                                                        if let Err(e) = session_manager
                                                            .maybe_autosave(&self.state)
                                                            .await
                                                        {
                                                            tracing::warn!(
                                                                error = %e,
                                                                "failed to autosave REPL session"
                                                            );
                                                        }
                                                        self.state.mark_autosaved();
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                self.renderer.show_error(&format!(
                                                    "Failed to create override agent: {}",
                                                    e
                                                ));
                                            }
                                        }
                                        continue;
                                    }
                                    Ok(None) => {
                                        self.renderer.show_error(&format!(
                                            "Override profile '{}' not found. Cleared.",
                                            name
                                        ));
                                        self.state.current_agent_override = None;
                                        continue;
                                    }
                                    Err(e) => {
                                        self.renderer.show_error(&format!(
                                            "Failed to load override profile: {}",
                                            e
                                        ));
                                        continue;
                                    }
                                }
                            }

                            self.run_message_turn(
                                harness.as_mut(),
                                service,
                                &resolved.message,
                                &tool_execution_driver,
                                auto_skill,
                                use_harness_turns,
                                auto_decompose,
                                cancellation_token,
                                &mut session_manager,
                                Some((&session_id, &prompt_preview)),
                                resolved.attachments,
                                resolved.allowed_paths,
                            )
                            .await?;
                        }
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
            if let Err(e) = session_manager.save(&self.state).await {
                tracing::warn!(error = %e, "failed to save REPL session on exit");
            }
        }

        Ok(())
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
