use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use crate::handlers;
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::modes::{InteractionMode, PlanDocument};

impl SlashCommandExecutor {
    pub(crate) fn handle_welcome(
        state: &InteractiveState,
        service: &AgentService,
    ) -> Result<SlashCommandResult> {
        let provider = service.provider().to_string();
        let model = service.model().to_string();
        let tool_count = service.list_tools().len();
        let mode_contract = format_mode_contract(state.interaction_mode);
        let content = format!(
            "Caliche\n\nProvider: {} | Model: {}\nSession: {}...\n\n{}\n\nCapabilities: {} tools\n\nQuick keys: / commands  @ context  Tab complete  exit quit\n/sessions list  /resume continue  Ctrl+C cancel",
            provider,
            model,
            caliche_contracts::text::truncate_utf8(&state.session_id, 8),
            mode_contract,
            tool_count
        );
        Ok(SlashCommandResult::ShowPopup {
            title: "Welcome".to_string(),
            content,
        })
    }

    pub(crate) async fn handle_plan(
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        if state.interaction_mode != InteractionMode::Plan {
            state.interaction_mode = InteractionMode::Plan;
            if state.plan_document.is_none() {
                state.plan_document = Some(PlanDocument::new());
            }
            apply_runtime_settings(state, service, None, None, InteractionMode::Plan).await?;
            Ok(SlashCommandResult::StatusMessage(
                mode_switch_message(InteractionMode::Plan).to_string(),
            ))
        } else {
            Ok(SlashCommandResult::StatusMessage(
                "Already in Plan mode. Use /showplan to review or /agent to execute.".to_string(),
            ))
        }
    }

    pub(crate) async fn handle_ask(
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        if state.interaction_mode != InteractionMode::Ask {
            state.interaction_mode = InteractionMode::Ask;
            apply_runtime_settings(state, service, None, None, InteractionMode::Ask).await?;
            Ok(SlashCommandResult::StatusMessage(
                mode_switch_message(InteractionMode::Ask).to_string(),
            ))
        } else {
            Ok(SlashCommandResult::StatusMessage(
                "Already in Ask mode. Use /agent to resume execution mode.".to_string(),
            ))
        }
    }

    pub(crate) async fn handle_agent(
        state: &mut InteractiveState,
        service: &mut AgentService,
        arg: Option<&str>,
    ) -> Result<SlashCommandResult> {
        // Primary agent override: /agent <name>, /agent next, /agent clear
        if let Some(a) = arg {
            let a_lower = a.to_lowercase();
            if a_lower == "clear" {
                state.current_agent_override = None;
                return Ok(SlashCommandResult::StatusMessage(
                    "Agent override cleared. Using default agent.".to_string(),
                ));
            }
            if a_lower == "next" {
                let primaries = handlers::list_primary_profile_names().await?;
                if primaries.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(
                        "No primary agent profiles found. Add .md files to .caliche/agents/ or ~/.config/caliche/agents/ with mode: primary.".to_string(),
                    ));
                }
                let current = state.current_agent_override.as_deref();
                let idx = current
                    .and_then(|c| primaries.iter().position(|p| *p == c))
                    .map(|i| (i + 1) % primaries.len())
                    .unwrap_or(0);
                let next_name = primaries[idx].clone();
                state.current_agent_override = Some(next_name.clone());
                return Ok(SlashCommandResult::StatusMessage(format!(
                    "Agent: {} (primary profile). Next message(s) will use this agent.",
                    next_name
                )));
            }
            // Set override to named profile (must be primary or we allow any for flexibility)
            return match handlers::load_profile_by_name(a).await {
                Ok(Some(profile)) => {
                    state.current_agent_override = Some(a.to_string());
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "Agent: {} ({}). Next message(s) will use this profile.",
                        a,
                        profile.description.as_deref().unwrap_or("profile")
                    )))
                }
                Ok(None) => Ok(SlashCommandResult::Error(format!(
                    "Unknown profile: '{}'. Use 'caliche agent list' to see profiles.",
                    a
                ))),
                Err(e) => Err(e.into()),
            };
        }
        // No arg: switch to Agent mode or show override
        if state.interaction_mode != InteractionMode::Agent {
            state.interaction_mode = InteractionMode::Agent;
            apply_runtime_settings(state, service, None, None, InteractionMode::Agent).await?;
            Ok(SlashCommandResult::StatusMessage(
                mode_switch_message(InteractionMode::Agent).to_string(),
            ))
        } else if let Some(ref name) = state.current_agent_override {
            Ok(SlashCommandResult::StatusMessage(format!(
                "Agent mode. Override: {} (next message(s) use this profile). Use /agent clear to reset.",
                name
            )))
        } else {
            Ok(SlashCommandResult::StatusMessage(
                "Already in Agent mode.".to_string(),
            ))
        }
    }

    pub(crate) fn handle_mode(state: &InteractiveState) -> Result<SlashCommandResult> {
        let m = state.interaction_mode;
        let content = format!(
            "{}\n\n{}\n\n{}",
            format_mode_contract(m),
            m.description(),
            if m.is_readonly() {
                "Write tools are disabled in this mode."
            } else {
                "Write tools are enabled in this mode."
            }
        );
        Ok(SlashCommandResult::ShowPopup {
            title: "Interaction Mode".to_string(),
            content,
        })
    }

    pub(crate) fn handle_showplan(state: &InteractiveState) -> Result<SlashCommandResult> {
        if let Some(ref plan) = state.plan_document {
            if plan.is_empty() {
                Ok(SlashCommandResult::StatusMessage(
                    "Plan document is empty. Start a conversation in Plan mode to build a plan."
                        .to_string(),
                ))
            } else {
                let md = plan.to_markdown();
                Ok(SlashCommandResult::ShowPopup {
                    title: "Plan".to_string(),
                    content: md,
                })
            }
        } else {
            Ok(SlashCommandResult::StatusMessage(
                "No plan document. Use /plan to start planning.".to_string(),
            ))
        }
    }
}
