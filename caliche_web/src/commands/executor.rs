//! Slash command execution against the API and local UI state.

use crate::api::{
    ApiClient, DiffChangeKind, DiffMessageSummary, DiffResponse, ModelInfo, ProviderCatalogEntry,
};
use crate::commands::{self, Command, CommandResult};
use crate::state::ConfigState;

fn format_change_set_line(entry: &DiffMessageSummary) -> String {
    let files = entry
        .snapshots
        .iter()
        .map(|snapshot| {
            let change = match snapshot.change {
                DiffChangeKind::Created => "created",
                DiffChangeKind::Modified => "modified",
                DiffChangeKind::Deleted => "deleted",
            };
            format!("- `{}` ({})", snapshot.path, change)
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "`{}` • {} file(s)\n{}",
        entry.message_id, entry.changed_files, files
    )
}

fn format_diff_command_output(diff: &DiffResponse) -> String {
    let mut body = format!(
        "**Session Diff**\n\nActive change sets: {}\nRedo queue: {}\nFiles touched: {}\nUndo available: {}\nRedo available: {}\n\n",
        diff.active_change_sets,
        diff.reverted_change_sets,
        diff.files_changed.len(),
        if diff.can_undo { "yes" } else { "no" },
        if diff.can_redo { "yes" } else { "no" }
    );

    if !diff.messages.is_empty() {
        body.push_str("**Active**\n");
        for entry in &diff.messages {
            body.push_str(&format!("{}\n\n", format_change_set_line(entry)));
        }
    }

    if !diff.reverted_messages.is_empty() {
        body.push_str("**Redo Queue**\n");
        for entry in &diff.reverted_messages {
            body.push_str(&format!("{}\n\n", format_change_set_line(entry)));
        }
    }

    body.trim_end().to_string()
}

/// Execute a slash command and return the result.
pub async fn execute_command(
    cmd: Command,
    client: &ApiClient,
    session_id: Option<&str>,
    providers: &[ProviderCatalogEntry],
    models: &[ModelInfo],
    config: &ConfigState,
) -> CommandResult {
    match cmd {
        Command::Help => CommandResult::success(commands::generate_help_text()),

        Command::SetMode(mode) => {
            if let Some(sid) = session_id {
                match client.set_mode(sid, mode).await {
                    Ok(response) => CommandResult::success(format!(
                        "**Mode changed to {}**\n\n{}",
                        mode, response.description
                    )),
                    Err(e) => CommandResult::error(format!("Failed to set mode: {}", e)),
                }
            } else {
                CommandResult::success(format!(
                    "**Mode set to {}**\n\n{}\n\n*Note: Mode will be applied when session starts.*",
                    mode,
                    mode.description()
                ))
            }
        }

        Command::ShowMode => {
            if let Some(sid) = session_id {
                match client.get_mode(sid).await {
                    Ok(response) => CommandResult::success(format!(
                        "**Current mode: {}**\n\n{}\n\nWrite tools enabled: {}",
                        response.mode, response.description, response.write_tools_enabled
                    )),
                    Err(e) => CommandResult::error(format!("Failed to get mode: {}", e)),
                }
            } else {
                CommandResult::success("**Current mode: agent** (default)\n\nNo active session.")
            }
        }

        Command::Undo => {
            if let Some(sid) = session_id {
                match client.undo(sid).await {
                    Ok(response) => CommandResult::success(response.status_message()),
                    Err(e) => CommandResult::error(format!("Failed to undo: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::Redo => {
            if let Some(sid) = session_id {
                match client.redo(sid).await {
                    Ok(response) => CommandResult::success(response.status_message()),
                    Err(e) => CommandResult::error(format!("Failed to redo: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::Checkpoint(label) => {
            if let Some(sid) = session_id {
                match client.create_checkpoint(sid, &label).await {
                    Ok(response) => CommandResult::success(format!(
                        "**Checkpoint created**\n\nLabel: {}\nID: `{}`",
                        label, response.checkpoint_id
                    )),
                    Err(e) => CommandResult::error(format!("Failed to create checkpoint: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::Rollback(checkpoint_id) => {
            if let Some(sid) = session_id {
                match client.rollback(sid, &checkpoint_id).await {
                    Ok(response) => {
                        let files = if response.files_restored.is_empty() {
                            "No files restored".to_string()
                        } else {
                            response.files_restored.join("\n- ")
                        };
                        CommandResult::success(format!(
                            "**Rolled back to checkpoint `{}`**\n\nFiles restored:\n- {}",
                            checkpoint_id, files
                        ))
                    }
                    Err(e) => CommandResult::error(format!("Failed to rollback: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::Clear => CommandResult::success("**Conversation cleared**"),

        Command::ShowContext => {
            if let Some(sid) = session_id {
                match client.get_context(sid).await {
                    Ok(ctx) => {
                        let mut msg = format!(
                            "**Workspace Context**\n\n**Root:** `{}`\n",
                            ctx.workspace_root
                        );

                        if let Some(ref pt) = ctx.project_type {
                            msg.push_str(&format!("**Project Type:** {}\n", pt));
                        }

                        if let Some(ref git) = ctx.git_context {
                            msg.push_str(&format!(
                                "\n**Git:**\n- Branch: `{}`\n- Changes: {}\n- Staged: {}, Modified: {}\n",
                                git.branch, git.has_changes, git.staged_count, git.modified_count
                            ));
                        }

                        if !ctx.tracked_files.is_empty() {
                            msg.push_str("\n**Tracked Files:**\n");
                            for file in &ctx.tracked_files {
                                msg.push_str(&format!(
                                    "- `{}` ({} bytes, reads: {})\n",
                                    file.path, file.size_bytes, file.read_count
                                ));
                            }
                        } else {
                            msg.push_str("\n*No files tracked yet.*");
                        }

                        CommandResult::success(msg)
                    }
                    Err(e) => CommandResult::error(format!("Failed to get context: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::TrackFile(path) => {
            if let Some(sid) = session_id {
                match client.track_file(sid, &path).await {
                    Ok(info) => CommandResult::success(format!(
                        "**Now tracking:** `{}`\n\nSize: {} bytes{}",
                        info.path,
                        info.size_bytes,
                        info.language
                            .map(|l| format!("\nLanguage: {}", l))
                            .unwrap_or_default()
                    )),
                    Err(e) => CommandResult::error(format!("Failed to track file: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::ListProviders => {
            if providers.is_empty() {
                CommandResult::success(
                    "**No providers available**\n\nConnect to the API server first.",
                )
            } else {
                let current = config.provider.as_deref().unwrap_or("(none)");
                let list = providers
                    .iter()
                    .map(|p| {
                        if Some(p.id.as_str()) == config.provider.as_deref() {
                            format!("- **{}** ({}, current)", p.id, p.display_name)
                        } else {
                            format!("- {} ({})", p.id, p.display_name)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                CommandResult::success(format!(
                    "**Available Providers**\n\nCurrent: {}\n\n{}",
                    current, list
                ))
            }
        }

        Command::SwitchProvider(provider) => {
            if providers
                .iter()
                .any(|p| p.id.eq_ignore_ascii_case(&provider))
            {
                CommandResult::success(format!("**Switched to provider:** {}", provider))
            } else {
                let available = providers
                    .iter()
                    .map(|provider| provider.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                CommandResult::error(format!(
                    "Unknown provider: {}\n\nAvailable: {}",
                    provider, available
                ))
            }
        }

        Command::SetModel(model) => {
            let model_exists = models
                .iter()
                .any(|m| m.id.eq_ignore_ascii_case(&model) || m.name.eq_ignore_ascii_case(&model));

            if model_exists {
                CommandResult::success(format!("**Model set to:** {}", model))
            } else {
                let available: Vec<_> = models
                    .iter()
                    .filter(|m| {
                        config
                            .provider
                            .as_ref()
                            .map(|p| m.provider_id.eq_ignore_ascii_case(p))
                            .unwrap_or(true)
                    })
                    .take(10)
                    .map(|m| format!("- {} ({})", m.name, m.provider_id))
                    .collect();

                CommandResult::error(format!(
                    "Unknown model: {}\n\nAvailable models:\n{}",
                    model,
                    available.join("\n")
                ))
            }
        }

        Command::ListTools => match client.get_tools().await {
            Ok(tools) => {
                if tools.is_empty() {
                    CommandResult::success("**No tools available**")
                } else {
                    let list = tools
                        .iter()
                        .map(|t| format!("- **{}**: {}", t.name, t.description))
                        .collect::<Vec<_>>()
                        .join("\n");
                    CommandResult::success(format!("**Available Tools**\n\n{}", list))
                }
            }
            Err(e) => CommandResult::error(format!("Failed to get tools: {}", e)),
        },

        Command::ShowCost => {
            CommandResult::success("*Cost tracking is shown in the token usage display.*")
        }

        Command::ShowStats => {
            CommandResult::success("*Session statistics are shown in the token usage display.*")
        }

        Command::MemoryStats => {
            if let Some(sid) = session_id {
                match client.get_memory_stats(sid).await {
                    Ok(stats) => {
                        let mut msg = String::from("**Memory Stats**\n\n");
                        msg.push_str(&format!("- Entries: {}\n", stats.entry_count));
                        if let Some(tokens) = stats.total_tokens {
                            msg.push_str(&format!("- Tokens: {}\n", tokens));
                        }
                        if let Some(ref summary) = stats.summary {
                            msg.push_str(&format!("\n**Summary:**\n{}", summary));
                        }
                        CommandResult::success(msg)
                    }
                    Err(e) => CommandResult::error(format!("Failed to get memory stats: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::MemorySearch(query) => {
            if let Some(sid) = session_id {
                match client.search_memory(sid, &query).await {
                    Ok(results) => {
                        if results.results.is_empty() {
                            CommandResult::success("*No matching memories found.*")
                        } else {
                            let mut msg = format!(
                                "**Memory Search:** `{}`\n\n{} result(s)\n\n",
                                query, results.total
                            );
                            for (i, entry) in results.results.iter().enumerate().take(5) {
                                let rel = entry
                                    .relevance
                                    .map(|r| format!(" [{:.0}%]", r * 100.0))
                                    .unwrap_or_default();
                                msg.push_str(&format!("{}. {}{}\n\n", i + 1, entry.content, rel));
                            }
                            CommandResult::success(msg)
                        }
                    }
                    Err(e) => CommandResult::error(format!("Failed to search memory: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::MemoryClear => {
            if let Some(sid) = session_id {
                match client.clear_memory(sid).await {
                    Ok(()) => CommandResult::success("**Memory cleared.**"),
                    Err(e) => CommandResult::error(format!("Failed to clear memory: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::ShowDiff => {
            if let Some(sid) = session_id {
                match client.get_diff(sid).await {
                    Ok(diff) => {
                        if !diff.has_any_changes() {
                            CommandResult::success("*No changes in this session.*")
                        } else {
                            CommandResult::success(format_diff_command_output(&diff))
                        }
                    }
                    Err(e) => CommandResult::error(format!("Failed to get diff: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::ShowPlan => {
            if let Some(sid) = session_id {
                match client.get_plan(sid).await {
                    Ok(plan) => {
                        if let Some(content) = plan.content {
                            CommandResult::success(format!("**Plan Document**\n\n{}", content))
                        } else {
                            CommandResult::success(
                                "*No plan document yet. Use `/plan` mode and ask the agent to create a plan.*",
                            )
                        }
                    }
                    Err(e) => CommandResult::error(format!("Failed to get plan: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::RevertLast => {
            if let Some(sid) = session_id {
                match client.undo(sid).await {
                    Ok(resp) => CommandResult::success(resp.status_message()),
                    Err(e) => CommandResult::error(format!("Failed to revert changes: {}", e)),
                }
            } else {
                CommandResult::error("No active session. Start a conversation first.")
            }
        }

        Command::SpawnAgent(ref task) => CommandResult::success(format!(
            "**Spawning agent**\n\nTask: {}\n\n*Switching to fleet tab…*",
            task
        )),

        Command::ShowAgents => CommandResult::success("*Switching to fleet tab…*"),

        Command::ShowSettings => CommandResult::success(
            "*Switching to settings tab…*\n\nSet the API base URL, provider, and model here.",
        ),

        Command::ShowChat => CommandResult::success("*Switching to chat…*"),

        Command::ShowDocuments => CommandResult::success("*Switching to documents tab…*"),

        Command::ShowPlugins => CommandResult::success("*Switching to plugins tab…*"),

        Command::ShowReasoning => CommandResult::success(
            "*Switching to reasoning tab…*\n\nType `/patterns` in chat to list reasoning patterns from the server.",
        ),

        Command::ShowHistory => CommandResult::success(
            "**History**\n\nThe **history** tab lists live sessions on this server. \
             Open a row to resume that session in chat.",
        ),

        Command::DoctorHelp => CommandResult::success(
            "**CLI only**\n\n\
             `/doctor` is not available in this console. \
             Run `caliche doctor` in a terminal. \
             Use **settings** here to check the API base URL.",
        ),

        Command::InitHelp => CommandResult::success(
            "**CLI only**\n\n\
             `/init` is not available in this console. \
             From the project root, run `caliche init` in a terminal.",
        ),

        Command::ResumeHelp => CommandResult::success(
            "**History**\n\n\
             Open the **history** tab and choose a live server session. \
             `/resume <id>` is a REPL command, not a web action.",
        ),

        Command::CliOnly(name) => CommandResult::success(format!(
            "**CLI only**\n\n`{name}` is not available in this console."
        )),

        Command::ExportHelp => CommandResult::success(
            "**CLI only**\n\n\
             `/export` does not write a file from this console. \
             In the REPL, `/export <path>` writes the transcript.",
        ),

        Command::ListPatterns => match client.get_reasoning_patterns().await {
            Ok(patterns) => {
                if patterns.is_empty() {
                    CommandResult::success(
                        "**No reasoning patterns available** (feature may not be enabled on the server).",
                    )
                } else {
                    let list = patterns
                        .iter()
                        .map(|p| format!("- **{}**: {}", p.name, p.description))
                        .collect::<Vec<_>>()
                        .join("\n");
                    CommandResult::success(format!(
                        "**Reasoning Patterns**\n\nUse with `/react`, `/plan-execute`, `/tree-of-thoughts`:\n\n{}",
                        list
                    ))
                }
            }
            Err(_) => CommandResult::success(
                "**Reasoning Patterns** (feature may not be enabled on the server)\n\n\
                - **react** — ReAct: interleaved reasoning and tool use\n\
                - **plan-execute** — Plan first, then execute steps\n\
                - **tree-of-thoughts** — Explore multiple reasoning branches\n\
                - **reflexion** — Reflect and refine answers iteratively\n\
                - **self-ask** — Decompose questions into sub-questions",
            ),
        },

        Command::ReactReasoning(prompt) => CommandResult::success(format!(
            "*Sending with ReAct reasoning pattern…*\n\nPrompt: {}",
            prompt
        )),

        Command::PlanExecuteReasoning(prompt) => CommandResult::success(format!(
            "*Sending with Plan-Execute reasoning pattern…*\n\nPrompt: {}",
            prompt
        )),

        Command::TreeOfThoughts(prompt) => CommandResult::success(format!(
            "*Sending with Tree-of-Thoughts reasoning pattern…*\n\nPrompt: {}",
            prompt
        )),
    }
}
