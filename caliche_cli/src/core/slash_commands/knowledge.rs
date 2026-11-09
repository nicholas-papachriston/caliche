use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use anyhow::Result;
use caliche::AgentService;

impl SlashCommandExecutor {
    pub(crate) async fn handle_memory(
        state: &mut InteractiveState,
        service: &mut AgentService,
        arg: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use crate::core::memory_manager::{create_memory_strategy, format_memory_stats};

        let mut selected_arg: Option<String> = None;
        if arg.is_none() && supports_interactive_selection() {
            let items = vec![
                "show - view current memory status".to_string(),
                "config - view memory configuration".to_string(),
                "buffer - sliding window".to_string(),
                "summary - progressive summarization".to_string(),
                "entity - track entities".to_string(),
                "vector - semantic retrieval".to_string(),
                "enhanced - combined techniques".to_string(),
            ];
            if let Some(index) = prompt_select_index("Configure memory", &items, 0)? {
                let selected = match index {
                    0 => "show",
                    1 => "config",
                    2 => "buffer",
                    3 => "summary",
                    4 => "entity",
                    5 => "vector",
                    6 => "enhanced",
                    _ => "show",
                };
                selected_arg = Some(selected.to_string());
            }
        }

        let arg = arg.or(selected_arg.as_deref());

        match arg {
            None => {
                let stats = format_memory_stats(service.memory()).await;
                let content = format!(
                    "Memory Strategy\n==============\n\n{}\n\nAvailable strategies:\n  buffer   - Sliding window (default)\n  summary  - Progressive summarization\n  entity   - Entity extraction and tracking\n  vector   - Semantic retrieval (requires vector-memory feature)\n  enhanced - Combined techniques\n\nUsage: /memory <strategy>",
                    stats
                );
                Ok(SlashCommandResult::ShowPopup {
                    title: "Memory Strategy".to_string(),
                    content,
                })
            }
            Some("show") | Some("status") => {
                // Show current memory strategy and stats
                let stats = format_memory_stats(service.memory()).await;
                let content = format!(
                    "Memory Strategy\n==============\n\n{}\n\nAvailable strategies:\n  buffer   - Sliding window (default)\n  summary  - Progressive summarization\n  entity   - Entity extraction and tracking\n  vector   - Semantic retrieval (requires vector-memory feature)\n  enhanced - Combined techniques\n\nUsage: /memory <strategy>",
                    stats
                );
                Ok(SlashCommandResult::ShowPopup {
                    title: "Memory Strategy".to_string(),
                    content,
                })
            }
            Some("config") => {
                let content = format!(
                    "Memory Configuration\n===================\n\nMax tokens: {}\nAuto-compact: {} (threshold: {:.1}%)\n\nAvailable strategies:\n  buffer   - Simple sliding window\n  summary  - Summarizes older messages\n  entity   - Extracts and tracks entities\n  vector   - Semantic search (requires setup)\n  enhanced - Multiple techniques combined",
                    state.token_budget.max_tokens,
                    if state.auto_compact_enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    state.auto_compact_threshold
                );
                Ok(SlashCommandResult::ShowPopup {
                    title: "Memory Configuration".to_string(),
                    content,
                })
            }
            Some(strategy_str) => {
                // Switch memory strategy
                let strategy_type = match strategy_str.parse() {
                    Ok(s) => s,
                    Err(_error) => {
                        return Ok(SlashCommandResult::Error(format!(
                            "Unknown memory strategy: {}. Use: buffer, summary, entity, composite, vector, enhanced",
                            strategy_str
                        )));
                    }
                };

                // Extract current history
                let history = if let Some(memory) = service.memory() {
                    memory.read().await.get_context().await.unwrap_or_default()
                } else {
                    Vec::new()
                };

                // Get provider for creating new strategy
                let provider_arc = service.llm_provider();
                let embedding_provider = service.embedding_provider();
                let max_tokens = state.token_budget.max_tokens as usize;

                // Create new memory strategy
                let new_memory = create_memory_strategy(
                    strategy_type,
                    max_tokens,
                    provider_arc,
                    history,
                    embedding_provider,
                )
                .await?;

                // Update service
                service.set_memory_strategy(Some(Box::new(new_memory)));

                Ok(SlashCommandResult::StatusMessage(format!(
                    "Switched to {} memory strategy.",
                    strategy_type.as_str()
                )))
            }
        }
    }

    pub(crate) async fn handle_remember(
        state: &mut InteractiveState,
        service: &AgentService,
        text: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use crate::core::knowledge_store::{create_knowledge_store, remember};

        let text = match text {
            Some(t) if !t.trim().is_empty() => t,
            _ => {
                return Ok(SlashCommandResult::Error(
                    "Usage: /remember <text>".to_string(),
                ));
            }
        };

        // Initialize knowledge store if needed
        if state.knowledge_store.is_none() {
            let embedding_provider = service.embedding_provider();
            match create_knowledge_store(embedding_provider) {
                Ok(Some(store)) => {
                    state.knowledge_store = Some(store);
                }
                Ok(None) => {
                    return Ok(SlashCommandResult::Error(
                        "Knowledge store requires an embedding-capable provider. \
                         Your current provider does not support embeddings."
                            .to_string(),
                    ));
                }
                Err(e) => {
                    return Ok(SlashCommandResult::Error(format!(
                        "Failed to create knowledge store: {}",
                        e
                    )));
                }
            }
        }

        let store = match Self::knowledge_store_ref(state) {
            Ok(s) => s,
            Err(cmd) => return Ok(cmd),
        };
        match remember(store, text.to_string(), None, None).await {
            Ok(id) => Ok(SlashCommandResult::StatusMessage(format!(
                "Remembered: {} (ID: {})",
                text,
                caliche_contracts::text::truncate_utf8(&id, 20)
            ))),
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Failed to remember: {}",
                e
            ))),
        }
    }

    pub(crate) async fn handle_recall(
        state: &mut InteractiveState,
        service: &AgentService,
        query: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use crate::core::knowledge_store::{create_knowledge_store, recall};

        let query = match query {
            Some(q) if !q.trim().is_empty() => q,
            _ => {
                return Ok(SlashCommandResult::Error(
                    "Usage: /recall <query>".to_string(),
                ));
            }
        };

        // Initialize knowledge store if needed
        if state.knowledge_store.is_none() {
            let embedding_provider = service.embedding_provider();
            match create_knowledge_store(embedding_provider) {
                Ok(Some(store)) => {
                    state.knowledge_store = Some(store);
                }
                Ok(None) => {
                    return Ok(SlashCommandResult::Error(
                        "Knowledge store requires an embedding-capable provider.".to_string(),
                    ));
                }
                Err(e) => {
                    return Ok(SlashCommandResult::Error(format!(
                        "Failed to create knowledge store: {}",
                        e
                    )));
                }
            }
        }

        let store = match Self::knowledge_store_ref(state) {
            Ok(s) => s,
            Err(cmd) => return Ok(cmd),
        };
        match recall(store, query, 5).await {
            Ok(results) => {
                if results.is_empty() {
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "No knowledge found for: {}",
                        query
                    )))
                } else {
                    let items: Vec<String> = results
                        .iter()
                        .map(|(entry, score)| {
                            format!(
                                "{} (score: {:.2})",
                                entry.content.chars().take(60).collect::<String>(),
                                score
                            )
                        })
                        .collect();
                    let descriptions: Vec<Option<String>> = results
                        .iter()
                        .map(|(entry, _)| {
                            Some(format!(
                                "Category: {} | ID: {}",
                                entry.category,
                                caliche_contracts::text::truncate_utf8(&entry.id, 20)
                            ))
                        })
                        .collect();
                    Ok(SlashCommandResult::ShowList {
                        title: format!("Knowledge for: {}", query),
                        items,
                        descriptions,
                    })
                }
            }
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Failed to recall: {}",
                e
            ))),
        }
    }

    pub(crate) async fn handle_knowledge(
        state: &mut InteractiveState,
        service: &AgentService,
    ) -> Result<SlashCommandResult> {
        use crate::core::knowledge_store::{create_knowledge_store, knowledge_summary};

        // Initialize knowledge store if needed
        if state.knowledge_store.is_none() {
            let embedding_provider = service.embedding_provider();
            match create_knowledge_store(embedding_provider) {
                Ok(Some(store)) => {
                    state.knowledge_store = Some(store);
                }
                Ok(None) => {
                    return Ok(SlashCommandResult::ShowPopup {
                        title: "Knowledge Store".to_string(),
                        content: "Knowledge store requires an embedding-capable provider.\nYour current provider does not support embeddings.".to_string(),
                    });
                }
                Err(e) => {
                    return Ok(SlashCommandResult::Error(format!(
                        "Failed to create knowledge store: {}",
                        e
                    )));
                }
            }
        }

        let store = match Self::knowledge_store_ref(state) {
            Ok(s) => s,
            Err(cmd) => return Ok(cmd),
        };
        let summary = knowledge_summary(store).await;
        Ok(SlashCommandResult::ShowPopup {
            title: "Knowledge Store".to_string(),
            content: summary,
        })
    }

    pub(crate) async fn handle_forget(
        state: &mut InteractiveState,
        service: &AgentService,
        id: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use crate::core::knowledge_store::{create_knowledge_store, forget};

        let id = match id {
            Some(i) if !i.trim().is_empty() => i,
            _ => {
                return Ok(SlashCommandResult::Error("Usage: /forget <id>".to_string()));
            }
        };

        // Initialize knowledge store if needed
        if state.knowledge_store.is_none() {
            let embedding_provider = service.embedding_provider();
            match create_knowledge_store(embedding_provider) {
                Ok(Some(store)) => {
                    state.knowledge_store = Some(store);
                }
                Ok(None) => {
                    return Ok(SlashCommandResult::Error(
                        "Knowledge store requires an embedding-capable provider.".to_string(),
                    ));
                }
                Err(e) => {
                    return Ok(SlashCommandResult::Error(format!(
                        "Failed to create knowledge store: {}",
                        e
                    )));
                }
            }
        }

        let store = match Self::knowledge_store_ref(state) {
            Ok(s) => s,
            Err(cmd) => return Ok(cmd),
        };
        match forget(store, id).await {
            Ok(true) => Ok(SlashCommandResult::StatusMessage(format!(
                "Forgot knowledge entry: {}",
                id
            ))),
            Ok(false) => Ok(SlashCommandResult::Error(format!(
                "Knowledge entry not found: {}",
                id
            ))),
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Failed to forget: {}",
                e
            ))),
        }
    }

    pub(crate) fn handle_changes(state: &InteractiveState) -> Result<SlashCommandResult> {
        let changes = state.get_modified_files();

        if changes.is_empty() {
            Ok(SlashCommandResult::StatusMessage(
                "No files modified in this session.".to_string(),
            ))
        } else {
            let mut lines = vec![format!("Files modified: {}", changes.len()), String::new()];

            for change in changes {
                let change_type_str = match change.change_type {
                    crate::core::state::FileChangeType::Created => "Created",
                    crate::core::state::FileChangeType::Modified => "Modified",
                    crate::core::state::FileChangeType::Deleted => "Deleted",
                };

                lines.push(format!("  {} - {}", change_type_str, change.path.display()));
            }

            Ok(SlashCommandResult::ShowPopup {
                title: "File Changes".to_string(),
                content: lines.join("\n"),
            })
        }
    }
}
