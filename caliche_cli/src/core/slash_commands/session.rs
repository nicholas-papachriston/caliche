use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::session::SessionManager;
use crate::core::state::InteractiveState;
use crate::cost::format_tokens;
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::modes::InteractionMode;
use caliche::harness::{ExportedMessage, SessionPhase, SuspendStore, SuspendedSession};
use caliche::llm::ProviderType;
use caliche::messages::TokenUsage;
use std::path::PathBuf;

impl SlashCommandExecutor {
    pub(crate) fn handle_tools(service: &AgentService) -> Result<SlashCommandResult> {
        let tools = service.list_tools();
        let (items, descriptions): (Vec<String>, Vec<Option<String>>) = tools
            .iter()
            .map(|(name, desc, _)| (name.to_string(), Some(desc.to_string())))
            .unzip();
        Ok(SlashCommandResult::ShowList {
            title: "Available Tools".to_string(),
            items,
            descriptions,
        })
    }

    pub(crate) fn handle_cost(state: &InteractiveState) -> Result<SlashCommandResult> {
        Ok(SlashCommandResult::ShowTokenUsage {
            session_tokens: state.total_tokens,
            context_tokens: state.context_tokens.unwrap_or(0),
            cache_creation_tokens: state.cache_creation_tokens,
            cache_read_tokens: state.cache_read_tokens,
            session_prompt_tokens: state.session_prompt_tokens,
        })
    }

    pub(crate) fn handle_budget(state: &InteractiveState) -> Result<SlashCommandResult> {
        Ok(SlashCommandResult::ShowBudget {
            session_tokens: state.total_tokens,
            context_tokens: state.context_tokens.unwrap_or(0),
        })
    }

    pub(crate) fn handle_stats(
        state: &InteractiveState,
        service: &AgentService,
    ) -> Result<SlashCommandResult> {
        Ok(SlashCommandResult::ShowStats {
            total_tokens: state.total_tokens,
            message_count: state.messages.len(),
            provider: service.provider().to_string(),
            model: service.model().to_string(),
            quality_summary: format!(
                "Tool success: {:.1}% ({} ok / {} total)\nNarration-only: {}\nRetries: {} attempted, {} recovered\nLoop detections: text={}, tool={}\nCancellations: {}\nIteration limits: {}",
                state.quality_metrics.tool_success_rate(),
                state.quality_metrics.tool_calls_succeeded,
                state.quality_metrics.tool_calls_started,
                state.quality_metrics.narration_only_responses,
                state.quality_metrics.tool_retries_attempted,
                state.quality_metrics.tool_retries_succeeded,
                state.quality_metrics.text_loop_detections,
                state.quality_metrics.tool_loop_detections,
                state.quality_metrics.cancellations,
                state.quality_metrics.iteration_limits_reached
            ),
        })
    }

    pub(crate) async fn handle_clear(
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        state.clear_messages();
        replace_service_memory_with_history(service, Vec::new()).await?;

        if state.interaction_mode == InteractionMode::Plan {
            state.plan_document = None;
        }

        Ok(SlashCommandResult::StatusMessage(
            "Conversation cleared.".to_string(),
        ))
    }

    pub(crate) fn handle_compact(state: &mut InteractiveState) -> Result<SlashCommandResult> {
        state.toggle_compact();
        Ok(SlashCommandResult::StatusMessage(format!(
            "Output verbosity {}",
            state.output_verbosity
        )))
    }

    pub(crate) fn handle_verbose(
        state: &mut InteractiveState,
        level: Option<&str>,
    ) -> Result<SlashCommandResult> {
        match level {
            None => state.toggle_verbose(),
            Some(value) => match value.parse::<caliche::observability::OutputVerbosity>() {
                Ok(verbosity) => state.set_output_verbosity(verbosity),
                Err(error) => return Ok(SlashCommandResult::Error(error.to_string())),
            },
        }
        Ok(SlashCommandResult::StatusMessage(format!(
            "Output verbosity {} (thinking {}, cache detail {})",
            state.output_verbosity,
            if state.output_verbosity.show_thinking() {
                "on"
            } else {
                "off"
            },
            if state.output_verbosity.show_cache_detail() {
                "on"
            } else {
                "off"
            }
        )))
    }

    pub(crate) async fn handle_save(
        state: &InteractiveState,
        path: Option<&PathBuf>,
    ) -> Result<SlashCommandResult> {
        use crate::export::ExportFormat;
        let default_path = PathBuf::from("conversation.json");
        let path = path.unwrap_or(&default_path);
        let format = ExportFormat::from_extension(path);

        let mut export = state.conversation_export.clone();

        // Add messages to export
        for msg in &state.messages {
            let role_str = match msg.role {
                caliche::messages::MessageRole::User => "user",
                caliche::messages::MessageRole::Assistant => "assistant",
                caliche::messages::MessageRole::System => "system",
                caliche::messages::MessageRole::Tool => "tool",
            };
            export.add_message(role_str, &msg.extract_text(), None);
        }

        if let Some(ref plan) = state.plan_document {
            if !plan.is_empty() {
                export.plan = Some(plan.to_markdown());
            }
        }

        export.save(path, format).await?;
        Ok(SlashCommandResult::StatusMessage(format!(
            "Conversation saved to {}",
            path.display()
        )))
    }

    pub(crate) async fn handle_load(
        state: &mut InteractiveState,
        service: &mut AgentService,
        path: Option<&PathBuf>,
    ) -> Result<SlashCommandResult> {
        let path = match path {
            Some(p) => p,
            None => return Ok(SlashCommandResult::Error("Usage: /load <file>".to_string())),
        };

        use crate::export::ConversationExport;
        use caliche::messages::Message;

        let export = ConversationExport::load(path).await?;

        // Rebuild messages
        let mut history: Vec<Message> = Vec::new();
        for msg in &export.messages {
            match msg.role.as_str() {
                "user" => history.push(Message::user(&msg.content)),
                "assistant" => history.push(match msg.reasoning_content.as_deref() {
                    Some(reasoning) => Message::assistant_with_reasoning(&msg.content, reasoning),
                    None => Message::assistant(&msg.content),
                }),
                "system" => history.push(Message::system(&msg.content)),
                _ => {}
            }
        }

        let model_history: Vec<Message> = history
            .iter()
            .map(Message::without_reasoning_content)
            .collect();
        if !export.provider.is_empty() {
            let provider: ProviderType = export
                .provider
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid provider in import: {}", e))?;
            let model = (!export.model.is_empty()).then(|| export.model.clone());
            apply_runtime_settings(
                state,
                service,
                Some(provider),
                model,
                state.interaction_mode,
            )
            .await?;
        }
        replace_service_memory_with_history(service, model_history).await?;

        state.messages = history;
        state.total_tokens = export.total_tokens;
        state.context_tokens = Some(export.total_tokens);
        state.conversation_export = export;
        sync_runtime_metadata(state, service);
        state.token_budget.set_used(state.total_tokens);

        Ok(SlashCommandResult::StatusMessage(format!(
            "Conversation loaded from {} ({} messages) using {}/{}",
            path.display(),
            state.messages.len(),
            service.provider(),
            service.model()
        )))
    }

    pub(crate) async fn handle_sessions(
        session_manager: Option<&SessionManager>,
        search: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let mgr = session_manager.ok_or_else(|| anyhow::anyhow!("Session store not available"))?;
        let summaries = mgr.list_sessions(search).await?;
        let items: Vec<String> = summaries
            .iter()
            .map(|s| {
                let session_name = s.name.as_deref().unwrap_or("-");
                format!(
                    "{}  {:<20}  {:>4} msgs  {}",
                    short_session_id(&s.id),
                    session_name,
                    s.message_count,
                    format_updated(s.updated_at)
                )
            })
            .collect();
        let descriptions: Vec<Option<String>> = summaries
            .iter()
            .map(|s| {
                let provider_model = match (&s.provider, &s.model) {
                    (Some(provider), Some(model)) => format!("{provider}/{model}"),
                    (Some(provider), None) => provider.to_string(),
                    _ => "-".to_string(),
                };
                Some(format!(
                    "cwd={}  provider/model={}",
                    cwd_tail(&s.working_directory),
                    provider_model
                ))
            })
            .collect();
        let title = match search {
            Some(q) => format!("Sessions matching \"{}\" ({})", q, items.len()),
            None => format!(
                "Previous sessions ({}) - use /resume <id-prefix> or /resume last",
                items.len()
            ),
        };
        if items.is_empty() {
            Ok(SlashCommandResult::StatusMessage(match search {
                Some(q) => format!("No sessions matching \"{}\"", q),
                None => "No previous sessions. Start a conversation and sessions will appear here."
                    .to_string(),
            }))
        } else {
            Ok(SlashCommandResult::ShowList {
                title,
                items,
                descriptions,
            })
        }
    }

    fn suspend_dir(state: &InteractiveState) -> PathBuf {
        std::env::var("CALICHE_SUSPEND_DIR")
            .ok()
            .map(|raw| raw.trim().to_string())
            .filter(|raw| !raw.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| state.working_directory.join(".caliche").join("suspended"))
    }

    pub(crate) fn handle_suspend(
        state: &InteractiveState,
        id: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let session_id = id.unwrap_or(state.session_id.as_str()).to_string();
        let mut snapshot = SuspendedSession::new(&session_id);
        snapshot.phase = SessionPhase::Idle;
        snapshot.turn_count = usize::try_from(state.turn_count).unwrap_or(0);
        snapshot.workspace = Some(state.working_directory.clone());
        snapshot.messages = state.messages.iter().map(ExportedMessage::from).collect();
        snapshot.total_usage = TokenUsage {
            total_tokens: u32::try_from(state.total_tokens).ok(),
            ..TokenUsage::default()
        };
        let path = SuspendStore::new(Self::suspend_dir(state)).save(&snapshot)?;
        Ok(SlashCommandResult::StatusMessage(format!(
            "Session suspended to {} (resume with /resume {session_id})",
            path.display()
        )))
    }

    pub(crate) async fn handle_resume(
        state: &mut InteractiveState,
        service: &mut AgentService,
        session_manager: Option<&SessionManager>,
        id: Option<&str>,
    ) -> Result<SlashCommandResult> {
        if let Some(raw_id) = id.filter(|value| !value.is_empty() && *value != "last") {
            let store = SuspendStore::new(Self::suspend_dir(state));
            if store.contains(raw_id) {
                let snapshot = store.load(raw_id)?;
                let history = snapshot.to_messages();
                replace_service_memory_with_history(service, history.clone()).await?;
                state.session_id = snapshot.session_id.clone();
                state.messages = history;
                state.turn_count = u32::try_from(snapshot.turn_count).unwrap_or(0);
                state.total_tokens = u64::from(snapshot.total_usage.total_tokens.unwrap_or(0));
                sync_runtime_metadata(state, service);
                return Ok(SlashCommandResult::StatusMessage(format!(
                    "Resumed suspended session {} ({} messages, phase {:?})",
                    snapshot.session_id,
                    state.messages.len(),
                    snapshot.phase
                )));
            }
        }

        let mgr = session_manager.ok_or_else(|| anyhow::anyhow!("Session store not available"))?;
        let resolved_id = match parse_resume_request(id) {
            ResumeRequest::Last => {
                let sessions = mgr.list_sessions(None).await?;
                let last = sessions.first().ok_or_else(|| {
                    anyhow::anyhow!(
                        "No previous sessions found. Start a conversation to create one."
                    )
                })?;
                last.id.clone()
            }
            ResumeRequest::ById(raw) => mgr.resolve_session_id(&raw).await?,
            ResumeRequest::PromptSelection => {
                let sessions = mgr.list_sessions(None).await?;
                if sessions.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(
                        "No previous sessions found. Start a conversation to create one."
                            .to_string(),
                    ));
                }
                let items: Vec<String> = sessions
                    .iter()
                    .take(8)
                    .map(|s| {
                        format!(
                            "{}  {:<20}  {:>4} msgs  {}",
                            short_session_id(&s.id),
                            s.name.as_deref().unwrap_or("-"),
                            s.message_count,
                            format_updated(s.updated_at)
                        )
                    })
                    .collect();
                let descriptions: Vec<Option<String>> = sessions
                    .iter()
                    .take(8)
                    .map(|s| Some(format!("cwd={}", cwd_tail(&s.working_directory))))
                    .collect();
                return Ok(SlashCommandResult::ShowList {
                    title: "Resume a session - run /resume <id-prefix> or /resume last".to_string(),
                    items,
                    descriptions,
                });
            }
        };
        let provider = state.token_budget.provider.clone();
        let model = state.token_budget.model.clone();
        let system_prompt = state.original_system_prompt.clone();
        let new_state = mgr
            .load(&resolved_id, &provider, &model, system_prompt)
            .await?;
        let restored_provider = new_state.token_budget.provider.clone();
        let restored_model = new_state.token_budget.model.clone();
        let restored_provider_type: ProviderType = restored_provider
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid provider in saved session: {}", e))?;
        let history = new_state.messages.clone();
        *state = new_state;
        apply_runtime_settings(
            state,
            service,
            Some(restored_provider_type),
            Some(restored_model.clone()),
            state.interaction_mode,
        )
        .await?;
        replace_service_memory_with_history(service, history).await?;
        sync_runtime_metadata(state, service);
        state.token_budget.set_used(state.total_tokens);

        Ok(SlashCommandResult::StatusMessage(format!(
            "Resumed session {} ({} messages) using {}/{}. Mode: {}. Use /mode to inspect mode contract.",
            short_session_id(&resolved_id),
            state.messages.len(),
            service.provider(),
            service.model(),
            state.interaction_mode.name()
        )))
    }

    pub(crate) fn handle_status(
        state: &InteractiveState,
        service: &AgentService,
    ) -> Result<SlashCommandResult> {
        let provider = service.provider().to_string();
        let model = service.model().to_string();

        // Format context window usage
        let context_display = if let Some(context_tokens) = state.context_tokens {
            let max_tokens = state.token_budget.max_tokens;
            let usage_percent = if max_tokens > 0 {
                (context_tokens as f64 / max_tokens as f64) * 100.0
            } else {
                0.0
            };
            let usage_bar = state.token_budget.format_bar(20);
            format!(
                "Context: {} / {} ({:.1}%)\n{}",
                format_tokens(context_tokens),
                format_tokens(max_tokens),
                usage_percent,
                usage_bar
            )
        } else {
            format!(
                "Context: {} (estimated)",
                format_tokens(state.token_budget.used_tokens)
            )
        };

        let subagent_line = if state.subagent_provider.is_some() || state.subagent_model.is_some() {
            format!(
                "Subagent: {} / {}",
                state.subagent_provider.as_deref().unwrap_or("(main)"),
                state.subagent_model.as_deref().unwrap_or("(main)")
            )
        } else {
            String::new()
        };
        let override_line = state
            .current_agent_override
            .as_deref()
            .map(|n| format!("Agent override: {} (next message(s) use this profile)", n))
            .unwrap_or_default();
        let image_line = service
            .image_generation_provider()
            .map(|provider| {
                format!(
                    "Image generation: {} / {}",
                    provider.name(),
                    provider.model()
                )
            })
            .unwrap_or_default();
        let artifact_line = service
            .artifact_store()
            .map(|store| format!("Artifacts: {}", store.root().display()))
            .unwrap_or_default();
        let tool_list = service.list_tools();
        let tool_count = tool_list.len();
        let tools_line = if tool_count <= 10 {
            let names: Vec<&str> = tool_list.iter().map(|(n, _, _)| n.as_str()).collect();
            format!("Tools: {} ({})", names.join(", "), tool_count)
        } else {
            format!("Tools: {} enabled (use /tools to list)", tool_count)
        };
        let quality_line = format!(
            "Quality: {:.1}% tool success, narration-only={}, loop(text/tool)={}/{}, iteration-limits={}",
            state.quality_metrics.tool_success_rate(),
            state.quality_metrics.narration_only_responses,
            state.quality_metrics.text_loop_detections,
            state.quality_metrics.tool_loop_detections,
            state.quality_metrics.iteration_limits_reached
        );
        let content = if subagent_line.is_empty()
            && override_line.is_empty()
            && image_line.is_empty()
            && artifact_line.is_empty()
        {
            format!(
                "Session: {}\nProvider: {}\nModel: {}\n{}\n{}\nMessages: {}\nSession Tokens: {}\n\n{}",
                state.session_id,
                provider,
                model,
                tools_line,
                quality_line,
                state.messages.len(),
                format_tokens(state.total_tokens),
                context_display
            )
        } else {
            let extra = [subagent_line, override_line, image_line, artifact_line]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "Session: {}\nProvider: {}\nModel: {}\n{}\n{}\n{}\nMessages: {}\nSession Tokens: {}\n\n{}",
                state.session_id,
                provider,
                model,
                extra,
                tools_line,
                quality_line,
                state.messages.len(),
                format_tokens(state.total_tokens),
                context_display
            )
        };
        Ok(SlashCommandResult::ShowPopup {
            title: "Status".to_string(),
            content,
        })
    }

    pub(crate) fn handle_context(state: &InteractiveState) -> Result<SlashCommandResult> {
        let mut lines = vec![
            format!("Workspace: {}", state.working_directory.display()),
            String::new(),
        ];

        // Add context window information
        if let Some(context_tokens) = state.context_tokens {
            let max_tokens = state.token_budget.max_tokens;
            let usage_percent = if max_tokens > 0 {
                (context_tokens as f64 / max_tokens as f64) * 100.0
            } else {
                0.0
            };
            let warning = if usage_percent >= 90.0 {
                " (CRITICAL)"
            } else if usage_percent >= 75.0 {
                " (WARNING)"
            } else {
                ""
            };
            lines.push(format!(
                "Context Window: {} / {} ({:.1}%){}",
                format_tokens(context_tokens),
                format_tokens(max_tokens),
                usage_percent,
                warning
            ));
            lines.push(state.token_budget.format_bar(30));
            lines.push(String::new());
        }

        if state.tracked_files.is_empty() {
            lines.push("Tracked files: (none)".to_string());
            lines.push("Use /track <file> or /add <glob> to add files.".to_string());
        } else {
            lines.push(format!("Tracked files ({}):", state.tracked_files.len()));
            for f in state.tracked_files.iter().take(20) {
                lines.push(format!("  - {}", f.display()));
            }
            if state.tracked_files.len() > 20 {
                lines.push(format!("  ... and {} more", state.tracked_files.len() - 20));
            }
        }
        Ok(SlashCommandResult::ShowPopup {
            title: "Context".to_string(),
            content: lines.join("\n"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::SlashCommandExecutor;
    use crate::core::state::InteractiveState;
    use crate::export::{ConversationExport, ExportFormat};
    use crate::session::CliSession;
    use caliche::memory::MemoryStrategy;
    use caliche::messages::Message;

    #[tokio::test]
    async fn load_keeps_reasoning_for_review_and_strips_model_replay() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("conversation.json");
        let mut export = ConversationExport::new("sess", "", "");
        export.add_message_with_reasoning(
            "assistant",
            "the answer",
            Some("hidden-cot-token"),
            None,
        );
        export
            .save(&path, ExportFormat::Json)
            .await
            .expect("write export");

        let provider =
            std::sync::Arc::new(caliche::testing::MockProvider::new(vec!["ok".to_string()]));
        let agent =
            caliche::Agent::new(provider, std::sync::Arc::new(caliche::ToolRegistry::new()));
        let mut service =
            caliche::AgentService::new_with_agent(agent, caliche::llm::ProviderType::OpenAI)
                .expect("mock service");
        let mut state = InteractiveState::new(
            CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
            "openai",
            "mock-model",
            "prompt".to_string(),
        );

        SlashCommandExecutor::handle_load(&mut state, &mut service, Some(&path))
            .await
            .expect("load");

        assert_eq!(
            state.messages.last().and_then(Message::reasoning_content),
            Some("hidden-cot-token")
        );
        let memory = service.memory().expect("seeded memory");
        let context = memory.read().await.get_context().await.expect("context");
        assert!(
            context
                .iter()
                .all(|message| message.reasoning_content().is_none())
        );
        let wire = serde_json::to_string(&context).expect("memory json");
        assert!(!wire.contains("hidden-cot-token"));
        assert!(wire.contains("the answer"));
    }
}
