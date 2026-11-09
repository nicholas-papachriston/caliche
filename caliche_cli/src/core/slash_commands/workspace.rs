#[derive(Clone, Copy)]
enum LearningCommand {
    Status,
    Enable,
    Disable,
    Interval,
    MinToolCalls,
    MaxSkills,
    MemoryFile,
    AutoCreate,
    AutoPatch,
    PreferWeak,
    Holdout,
    MinHoldout,
}

fn learning_command(name: &str) -> Option<LearningCommand> {
    const COMMANDS: &[(&str, LearningCommand)] = &[
        ("status", LearningCommand::Status),
        ("show", LearningCommand::Status),
        ("on", LearningCommand::Enable),
        ("enable", LearningCommand::Enable),
        ("enabled", LearningCommand::Enable),
        ("off", LearningCommand::Disable),
        ("disable", LearningCommand::Disable),
        ("disabled", LearningCommand::Disable),
        ("interval", LearningCommand::Interval),
        ("min", LearningCommand::MinToolCalls),
        ("min-tool-calls", LearningCommand::MinToolCalls),
        ("min-tool-calls-for-skill", LearningCommand::MinToolCalls),
        ("max", LearningCommand::MaxSkills),
        ("max-skills", LearningCommand::MaxSkills),
        ("max-skills-per-session", LearningCommand::MaxSkills),
        ("memory", LearningCommand::MemoryFile),
        ("memory-file", LearningCommand::MemoryFile),
        ("auto-create", LearningCommand::AutoCreate),
        ("create-skills", LearningCommand::AutoCreate),
        ("auto-patch", LearningCommand::AutoPatch),
        ("patch-skills", LearningCommand::AutoPatch),
        ("prefer-weak", LearningCommand::PreferWeak),
        ("weak", LearningCommand::PreferWeak),
        ("weakness", LearningCommand::PreferWeak),
        ("holdout", LearningCommand::Holdout),
        ("require-holdout", LearningCommand::Holdout),
        ("min-holdout", LearningCommand::MinHoldout),
        ("min-holdout-extent", LearningCommand::MinHoldout),
    ];
    COMMANDS
        .iter()
        .find_map(|(key, command)| (*key == name).then_some(*command))
}

fn learning_memory_path(state: &InteractiveState, value: &str) -> Option<PathBuf> {
    if value.eq_ignore_ascii_case("clear") || value.eq_ignore_ascii_case("none") {
        return None;
    }
    let path = PathBuf::from(value);
    Some(if path.is_absolute() {
        path
    } else {
        state.working_directory.join(path)
    })
}

use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::learning::CheckpointConfig;
use std::path::PathBuf;

impl SlashCommandExecutor {
    pub(crate) fn handle_skill(
        service: &mut AgentService,
        args: &[String],
    ) -> Result<SlashCommandResult> {
        if args.is_empty() {
            let catalog = crate::handlers::interactive::skills::skill_catalog();
            let items: Vec<String> = catalog.iter().map(|skill| skill.name.clone()).collect();
            let descriptions: Vec<Option<String>> = catalog
                .iter()
                .map(|skill| Some(skill.description.clone()))
                .collect();
            Ok(SlashCommandResult::ShowList {
                title: "Skills".to_string(),
                items,
                descriptions,
            })
        } else {
            let name = &args[0];
            match crate::handlers::interactive::skills::apply_skill_by_name(name, service) {
                Ok(_) => Ok(SlashCommandResult::StatusMessage(format!(
                    "Skill '{}' applied.",
                    name
                ))),
                Err(e) => Ok(SlashCommandResult::Error(format!("Skill failed: {}", e))),
            }
        }
    }

    pub(crate) async fn handle_export(
        state: &InteractiveState,
        path: Option<&PathBuf>,
        include_thinking: bool,
    ) -> Result<SlashCommandResult> {
        use crate::export::ExportFormat;
        let default_path = PathBuf::from("conversation.json");
        let path = path.unwrap_or(&default_path);
        let format = ExportFormat::from_extension(path);

        let mut export = state.conversation_export.clone();
        for msg in &state.messages {
            let role_str = match msg.role {
                caliche::messages::MessageRole::User => "user",
                caliche::messages::MessageRole::Assistant => "assistant",
                caliche::messages::MessageRole::System => "system",
                _ => "tool",
            };
            let reasoning = if include_thinking {
                msg.reasoning_content()
            } else {
                None
            };
            export.add_message_with_reasoning(role_str, &msg.extract_text(), reasoning, None);
        }

        if let Some(ref plan) = state.plan_document {
            if !plan.is_empty() {
                export.plan = Some(plan.to_markdown());
            }
        }

        export.save(path, format).await?;
        Ok(SlashCommandResult::StatusMessage(format!(
            "Exported to {} ({} messages)",
            path.display(),
            export.messages.len()
        )))
    }

    pub(crate) async fn handle_compress(
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        let before = state
            .context_tokens
            .unwrap_or(state.token_budget.used_tokens);
        match crate::core::autocompact::maybe_auto_compact(state, service).await {
            Ok(true) => {
                let after = state.context_tokens.unwrap_or(before);
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Compressed conversation. Context reduced by {} tokens.",
                    before.saturating_sub(after)
                )))
            }
            Ok(false) => Ok(SlashCommandResult::StatusMessage(
                "Nothing to compress (context is minimal).".to_string(),
            )),
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Compression failed: {}",
                e
            ))),
        }
    }

    pub(crate) fn handle_autocompact(
        state: &mut InteractiveState,
        arg: Option<&str>,
    ) -> Result<SlashCommandResult> {
        match arg {
            None => {
                if supports_interactive_selection() {
                    let items = vec![
                        "on - enable auto-compact".to_string(),
                        "off - disable auto-compact".to_string(),
                        "set threshold - choose a percentage".to_string(),
                    ];
                    if let Some(index) = prompt_select_index("Configure auto-compact", &items, 0)? {
                        match index {
                            0 => {
                                state.auto_compact_enabled = true;
                                return Ok(SlashCommandResult::StatusMessage(
                                    "Auto-compact enabled.".to_string(),
                                ));
                            }
                            1 => {
                                state.auto_compact_enabled = false;
                                return Ok(SlashCommandResult::StatusMessage(
                                    "Auto-compact disabled.".to_string(),
                                ));
                            }
                            2 => {
                                let threshold = dialoguer::Input::<f64>::new()
                                    .with_prompt("Auto-compact threshold (%)")
                                    .default(state.auto_compact_threshold)
                                    .interact_text()?;
                                state.set_auto_compact_threshold(threshold);
                                return Ok(SlashCommandResult::StatusMessage(format!(
                                    "Auto-compact threshold set to {:.1}%",
                                    threshold
                                )));
                            }
                            _ => {}
                        }
                    }
                }

                // Show current status
                let status = if state.auto_compact_enabled {
                    "enabled"
                } else {
                    "disabled"
                };
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Auto-compact: {} (threshold: {:.1}%)",
                    status, state.auto_compact_threshold
                )))
            }
            Some("on") => {
                state.auto_compact_enabled = true;
                Ok(SlashCommandResult::StatusMessage(
                    "Auto-compact enabled.".to_string(),
                ))
            }
            Some("off") => {
                state.auto_compact_enabled = false;
                Ok(SlashCommandResult::StatusMessage(
                    "Auto-compact disabled.".to_string(),
                ))
            }
            Some(threshold_str) => {
                // Try to parse as threshold percentage
                match threshold_str.parse::<f64>() {
                    Ok(threshold) => {
                        state.set_auto_compact_threshold(threshold);
                        Ok(SlashCommandResult::StatusMessage(format!(
                            "Auto-compact threshold set to {:.1}%",
                            threshold
                        )))
                    }
                    Err(_) => Ok(SlashCommandResult::Error(
                        "Usage: /autocompact [on|off|<threshold>]".to_string(),
                    )),
                }
            }
        }
    }

    pub(crate) fn handle_learning(
        state: &InteractiveState,
        service: &mut AgentService,
        args: &[String],
    ) -> Result<SlashCommandResult> {
        let subcommand = args.first().map(String::as_str).unwrap_or("status");
        match learning_command(subcommand) {
            Some(LearningCommand::Status) => Ok(SlashCommandResult::ShowPopup {
                title: "Learning Loop".to_owned(),
                content: Self::format_learning_loop_status(service),
            }),
            Some(LearningCommand::Enable) => {
                let config = service.learning_loop_config().unwrap_or_default();
                service.set_learning_loop_config(config);
                Ok(SlashCommandResult::StatusMessage(
                    "Learning loop enabled for this session.".to_owned(),
                ))
            }
            Some(LearningCommand::Disable) => {
                service.clear_learning_loop();
                Ok(SlashCommandResult::StatusMessage(
                    "Learning loop disabled for this session.".to_owned(),
                ))
            }
            Some(LearningCommand::Interval) => Self::set_learning_usize(
                service,
                args,
                "interval",
                true,
                "Learning checkpoint interval set to {value} tool calls.",
                |config, value| config.interval_tool_calls = value,
            ),
            Some(LearningCommand::MinToolCalls) => Self::set_learning_usize(
                service,
                args,
                "min-tool-calls",
                true,
                "Learning minimum tool calls for skill creation set to {value}.",
                |config, value| config.min_tool_calls_for_skill = value,
            ),
            Some(LearningCommand::MaxSkills) => Self::set_learning_usize(
                service,
                args,
                "max-skills",
                false,
                "Learning maximum auto-created skills per session set to {value}.",
                |config, value| config.max_skills_per_session = value,
            ),
            Some(LearningCommand::MemoryFile) => Self::set_learning_memory(state, service, args),
            Some(LearningCommand::AutoCreate) => Self::set_learning_flag(
                service,
                args,
                "auto-create",
                "Learning auto-create skills",
                |config, value| config.auto_create_skills = value,
            ),
            Some(LearningCommand::AutoPatch) => Self::set_learning_flag(
                service,
                args,
                "auto-patch",
                "Learning auto-patch skills",
                |config, value| config.auto_patch_skills = value,
            ),
            Some(LearningCommand::PreferWeak) => Self::set_learning_flag(
                service,
                args,
                "prefer-weak",
                "Learning prefer-weak skills",
                |config, value| config.prefer_weak_skills = value,
            ),
            Some(LearningCommand::Holdout) => Self::set_learning_flag(
                service,
                args,
                "holdout",
                "Learning holdout gate",
                |config, value| config.require_holdout_pass = value,
            ),
            Some(LearningCommand::MinHoldout) => Self::set_learning_holdout(service, args),
            None => Ok(SlashCommandResult::Error(
                "Usage: /learning [status|on|off|interval <n>|min-tool-calls <n>|max-skills <n>|memory-file <path|clear>|auto-create <on|off>|auto-patch <on|off>|prefer-weak <on|off>|holdout <on|off>|min-holdout <0-1>]".to_owned(),
            )),
        }
    }

    fn set_learning_usize(
        service: &mut AgentService,
        args: &[String],
        setting: &str,
        at_least_one: bool,
        message: &str,
        write: fn(&mut CheckpointConfig, usize),
    ) -> Result<SlashCommandResult> {
        let mut value = Self::parse_learning_usize(args.get(1), setting)?;
        if at_least_one {
            value = value.max(1);
        }
        let mut config = service.learning_loop_config().unwrap_or_default();
        write(&mut config, value);
        service.set_learning_loop_config(config);
        Ok(SlashCommandResult::StatusMessage(
            message.replace("{value}", &value.to_string()),
        ))
    }

    fn set_learning_flag(
        service: &mut AgentService,
        args: &[String],
        setting: &str,
        label: &str,
        write: fn(&mut CheckpointConfig, bool),
    ) -> Result<SlashCommandResult> {
        let value = Self::parse_learning_bool(args.get(1), setting)?;
        let mut config = service.learning_loop_config().unwrap_or_default();
        write(&mut config, value);
        service.set_learning_loop_config(config);
        let state = if value { "enabled" } else { "disabled" };
        Ok(SlashCommandResult::StatusMessage(format!(
            "{label} {state}."
        )))
    }

    fn set_learning_memory(
        state: &InteractiveState,
        service: &mut AgentService,
        args: &[String],
    ) -> Result<SlashCommandResult> {
        let Some(value) = args.get(1).map(String::as_str) else {
            return Ok(SlashCommandResult::Error(
                "Usage: /learning memory-file <path|clear>".to_owned(),
            ));
        };
        let mut config = service.learning_loop_config().unwrap_or_default();
        config.memory_file_path = learning_memory_path(state, value);
        let memory_file = config
            .memory_file_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(disabled)".to_owned());
        service.set_learning_loop_config(config);
        Ok(SlashCommandResult::StatusMessage(format!(
            "Learning memory file set to {memory_file}."
        )))
    }

    fn set_learning_holdout(
        service: &mut AgentService,
        args: &[String],
    ) -> Result<SlashCommandResult> {
        let Some(raw) = args.get(1) else {
            return Ok(SlashCommandResult::Error(
                "Usage: /learning min-holdout <0.0-1.0>".to_owned(),
            ));
        };
        let value: f64 = raw.parse().map_err(|_| {
            anyhow::anyhow!("Learning setting 'min-holdout' must be a number in [0, 1].")
        })?;
        let value = value.clamp(0.0, 1.0);
        let mut config = service.learning_loop_config().unwrap_or_default();
        config.min_holdout_extent = value;
        service.set_learning_loop_config(config);
        Ok(SlashCommandResult::StatusMessage(format!(
            "Learning minimum holdout extent set to {value}."
        )))
    }

    fn parse_learning_usize(value: Option<&String>, setting: &str) -> Result<usize> {
        let Some(value) = value else {
            anyhow::bail!("Usage: /learning {setting} <number>");
        };
        value
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("Learning setting '{setting}' must be a number."))
    }

    fn parse_learning_bool(value: Option<&String>, setting: &str) -> Result<bool> {
        match value.map(|value| value.as_str()) {
            Some("on" | "true" | "yes" | "1" | "enable" | "enabled") => Ok(true),
            Some("off" | "false" | "no" | "0" | "disable" | "disabled") => Ok(false),
            _ => anyhow::bail!("Usage: /learning {setting} <on|off>"),
        }
    }

    fn format_learning_loop_status(service: &AgentService) -> String {
        let Some(config) = service.learning_loop_config() else {
            let defaults = CheckpointConfig::default();
            return format!(
                "Learning Loop\n=============\n\nStatus: disabled\n\nDefault settings when enabled:\n{}\n\nUsage: /learning on, /learning interval <n>, /learning memory-file <path>",
                learning_settings_lines(&defaults, "  ")
            );
        };
        format!(
            "Learning Loop\n=============\n\nStatus: enabled\n{}\n\nUsage: /learning off, /learning interval <n>, /learning min-tool-calls <n>, /learning max-skills <n>, /learning memory-file <path|clear>, /learning auto-create <on|off>, /learning auto-patch <on|off>, /learning prefer-weak <on|off>, /learning holdout <on|off>, /learning min-holdout <0-1>",
            learning_settings_lines(&config, "")
        )
    }

    pub(crate) async fn handle_init(_service: &mut AgentService) -> Result<SlashCommandResult> {
        use crate::handlers::run_init_ai_for_repl;

        match run_init_ai_for_repl(None, false).await {
            Ok(_) => Ok(SlashCommandResult::StatusMessage(
                "AGENTS.md updated with AI analysis.".to_string(),
            )),
            Err(e) => Ok(SlashCommandResult::Error(format!("Init failed: {}", e))),
        }
    }

    pub(crate) async fn handle_doctor() -> Result<SlashCommandResult> {
        use crate::handlers::doctor::doctor_report_to_string;

        match doctor_report_to_string().await {
            Ok(content) => Ok(SlashCommandResult::ShowPopup {
                title: "Health Check".to_string(),
                content,
            }),
            Err(e) => Ok(SlashCommandResult::Error(format!("Doctor failed: {}", e))),
        }
    }

    pub(crate) async fn handle_reindex(service: &mut AgentService) -> Result<SlashCommandResult> {
        match service.reindex_current_workspace_background().await {
            Ok(root) => Ok(SlashCommandResult::StatusMessage(format!(
                "Cleared workspace index and started background re-index for {}",
                root.display()
            ))),
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Unable to re-index workspace: {e}"
            ))),
        }
    }
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

fn learning_settings_lines(config: &CheckpointConfig, indent: &str) -> String {
    let memory_file = config
        .memory_file_path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "(disabled)".to_owned());
    format!(
        "{indent}checkpoint interval: {} tool calls\n{indent}min tool calls for skill: {}\n{indent}max auto-created skills: {}\n{indent}memory file: {memory_file}\n{indent}auto-create skills: {}\n{indent}auto-patch skills: {}\n{indent}prefer weak skills: {}\n{indent}require holdout: {}\n{indent}min holdout extent: {}",
        config.interval_tool_calls,
        config.min_tool_calls_for_skill,
        config.max_skills_per_session,
        on_off(config.auto_create_skills),
        on_off(config.auto_patch_skills),
        on_off(config.prefer_weak_skills),
        on_off(config.require_holdout_pass),
        config.min_holdout_extent,
    )
}

#[cfg(test)]
mod learning_status_tests {
    use super::learning_settings_lines;
    use caliche::agent::learning::CheckpointConfig;

    #[test]
    fn settings_lines_share_one_field_list() {
        let config = CheckpointConfig::default();
        let enabled = learning_settings_lines(&config, "");
        let disabled = learning_settings_lines(&config, "  ");
        assert!(enabled.starts_with("checkpoint interval:"));
        assert!(disabled.starts_with("  checkpoint interval:"));
        assert!(
            enabled.contains("auto-create skills: on")
                || enabled.contains("auto-create skills: off")
        );
        assert_eq!(
            enabled.matches("checkpoint interval:").count(),
            disabled.matches("checkpoint interval:").count()
        );
    }
}

#[cfg(test)]
mod export_tests {
    use super::SlashCommandExecutor;
    use crate::core::state::InteractiveState;
    use crate::session::CliSession;
    use caliche::messages::Message;

    #[tokio::test]
    async fn export_flag_keeps_reasoning_and_default_drops_it() {
        let mut state = InteractiveState::new(
            CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
            "openai",
            "mock-model",
            "prompt".to_string(),
        );
        state.add_message(Message::assistant_with_reasoning(
            "the answer",
            "hidden-cot-token",
        ));
        let dir = tempfile::tempdir().expect("tempdir");
        let with_flag = dir.path().join("with.json");
        let without_flag = dir.path().join("without.json");

        SlashCommandExecutor::handle_export(&state, Some(&with_flag), true)
            .await
            .expect("export with thinking");
        SlashCommandExecutor::handle_export(&state, Some(&without_flag), false)
            .await
            .expect("export without thinking");

        let with_body = std::fs::read_to_string(&with_flag).expect("read with");
        let without_body = std::fs::read_to_string(&without_flag).expect("read without");
        assert!(with_body.contains("hidden-cot-token"));
        assert!(with_body.contains("the answer"));
        assert!(
            !without_body.contains("hidden-cot-token"),
            "omitting --include-thinking must drop the trace"
        );
        assert!(!without_body.contains("reasoning_content"));
        assert!(without_body.contains("the answer"));
        assert_eq!(
            state.messages[0].reasoning_content(),
            Some("hidden-cot-token"),
            "export must not strip the live transcript"
        );
    }
}
