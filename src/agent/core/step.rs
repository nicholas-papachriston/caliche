use super::ToolCallingConfig;
use super::chat::{StepResult, extract_step_usage};
use super::chat_helpers::{
    IndexedToolCall, TokenUsageAccumulator, advance_phase_if_needed,
    append_invalid_tool_messages_native, append_owned_tool_results_to_history,
    append_tool_results_to_history, current_phase_allowed_tools, ensure_phase_plan_for_query,
    execute_tools_parallel, is_length_finish_reason, refuse_length_truncated_calls,
    validate_and_filter_tool_calls,
};
use super::{message_builder, tool_extraction, tool_output};
use crate::agent::ToolExecutionDriver;
use crate::agent::core::tool_ordering::{TOOL_ORDERING_THRESHOLD, order_tools};
use crate::agent::hooks::TurnContext;
use crate::agent::tools::control::HANDOFF_TOOL_NAME;
use crate::agent::tools::tool::ToolCall;
use crate::errors::AgentError;
use crate::messages::{Message, MessageRole};
use std::collections::HashSet;
use std::sync::Arc;

impl super::Agent {
    /// Extract the last user query from a message list (for tool ordering and phase planning).
    #[must_use]
    pub fn extract_last_user_query(messages: &[Message]) -> beef::Cow<'_, str> {
        messages
            .iter()
            .rev()
            .find_map(|m| {
                if matches!(m.role, MessageRole::User) {
                    m.content.as_text().map(beef::Cow::borrowed)
                } else {
                    None
                }
            })
            .unwrap_or_else(|| beef::Cow::borrowed(""))
    }

    /// Execute a single step: one LLM call plus immediate tool executions.
    ///
    /// Returns a [`StepResult`] containing the response, tool calls, and updated history.
    /// The caller controls whether to continue looping or return the final response.
    #[tracing::instrument(
        name = "agent.step",
        skip_all,
        fields(
            agent.step.iteration = iteration,
            agent.step.session_id = session_id,
            agent.step.message_count = messages.len(),
            agent.step.llm_ms = tracing::field::Empty,
            agent.step.tool_ms = tracing::field::Empty,
        )
    )]
    pub async fn step(
        &self,
        messages: Arc<Vec<Message>>,
        tool_cfg: &ToolCallingConfig,
        pre_ordered_schemas: Option<&Arc<Vec<crate::messages::ToolSchema>>>,
        query: &str,
        iteration: usize,
        session_id: &str,
    ) -> Result<StepResult, AgentError> {
        let use_native = tool_cfg.use_native;
        let phased_enabled = is_phased_enabled(self.tool_execution_driver, tool_cfg);
        let phase_allowed_tools = resolve_phase_allowed_tools(self, phased_enabled, query).await?;
        let tool_schemas = resolve_step_tool_schemas(
            self,
            tool_cfg,
            pre_ordered_schemas,
            query,
            use_native,
            phase_allowed_tools.as_ref(),
        )
        .await;
        let tool_schema_tokens = if use_native {
            self.tools
                .estimate_schema_tokens(self.token_counter.as_ref())
        } else {
            0
        };
        let messages =
            apply_step_context_budget(self, messages, session_id, tool_schema_tokens).await?;
        let messages = rewrite_phased_tool_descriptions(
            self,
            messages,
            use_native,
            tool_cfg.include_tool_descriptions,
            phased_enabled,
            phase_allowed_tools,
        );

        let tools_option = if use_native { Some(tool_schemas) } else { None };
        let tool_choice_option = self.tool_choice_for_iteration(iteration, use_native);
        let turn_ctx = TurnContext {
            session_id,
            turn_count: iteration,
            workspace_root: self.workspace_root(),
        };
        let messages = self.run_pre_llm_hooks(messages, &turn_ctx).await;
        self.ensure_context_window_available(messages.as_ref(), tool_schema_tokens)?;

        let (response, llm_duration_ms) = complete_step_llm_call(
            self,
            &messages,
            &tools_option,
            tool_choice_option,
            use_native,
        )
        .await?;

        let step_usage = extract_step_usage(&response);
        let response_text = Arc::new(response.content.as_text().to_string());
        let (tool_calls, recovered_from_text) =
            collect_step_tool_calls(use_native, &response, &response_text);

        let mut history_arc = messages;
        push_assistant_history(
            Arc::make_mut(&mut history_arc),
            use_native,
            &tool_calls,
            recovered_from_text,
            &response,
            &response_text,
        );

        let (tool_results, tool_duration_ms, step_usage) = execute_step_tools(
            self,
            Arc::make_mut(&mut history_arc),
            use_native,
            recovered_from_text,
            &tool_calls,
            &response,
            step_usage,
        )
        .await?;

        let any_tools_executed_this_step = !tool_results.is_empty();
        let mut has_pending_tool_calls = !tool_calls.is_empty();
        if !has_pending_tool_calls
            && any_tools_executed_this_step
            && phased_enabled
            && advance_phase_if_needed(phased_enabled, &self.tool_intelligence).await
        {
            has_pending_tool_calls = true;
        }

        tracing::Span::current().record("agent.step.llm_ms", llm_duration_ms);
        tracing::Span::current().record("agent.step.tool_ms", tool_duration_ms);

        Ok(StepResult {
            response_text,
            tool_calls,
            tool_results,
            updated_history: history_arc,
            usage: step_usage,
            has_pending_tool_calls,
            llm_duration_ms,
            tool_duration_ms,
        })
    }

    /// Legacy context management: compaction guard or plain binary-search pruning.
    /// Extracted so the context-engine path can delegate to it as a fallback.
    async fn legacy_context_management(
        &self,
        messages: &mut Arc<Vec<Message>>,
        context_budget: usize,
        tool_schema_tokens: usize,
    ) -> Result<(), AgentError> {
        #[cfg(feature = "compaction")]
        {
            if let Some(ref guard) = self.compaction_guard {
                match guard
                    .check_and_compact(
                        messages.as_ref(),
                        self.max_context_tokens,
                        tool_schema_tokens,
                        self.token_counter.as_ref(),
                    )
                    .await
                {
                    Ok(super::context_window_guard::GuardResult::WithinBudget) => {}
                    Ok(super::context_window_guard::GuardResult::Compacted(r)) => {
                        self.store_compaction_report(r.report);
                        *messages = Arc::new(r.messages);
                    }
                    Ok(super::context_window_guard::GuardResult::OverBudget {
                        estimated_tokens,
                        budget,
                    }) => {
                        tracing::warn!(
                            "Context over budget: {} > {}; falling back to prune",
                            estimated_tokens,
                            budget
                        );
                        let vec = Arc::make_mut(messages);
                        Self::prune_history_for_budget(self.token_counter.as_ref(), vec, budget);
                        self.ensure_context_window_available(
                            messages.as_ref(),
                            tool_schema_tokens,
                        )?;
                    }
                    Err(e) => return Err(AgentError::CompletionFailed(e.to_string())),
                }
            } else {
                let estimated_tokens =
                    Self::estimate_context_tokens(self.token_counter.as_ref(), messages.as_ref());
                if estimated_tokens > context_budget {
                    let vec = Arc::make_mut(messages);
                    Self::prune_history_for_budget(
                        self.token_counter.as_ref(),
                        vec,
                        context_budget,
                    );
                    self.ensure_context_window_available(messages.as_ref(), tool_schema_tokens)?;
                }
            }
        }
        #[cfg(not(feature = "compaction"))]
        {
            let estimated_tokens = super::Agent::estimate_context_tokens(
                self.token_counter.as_ref(),
                messages.as_ref(),
            );
            if estimated_tokens > context_budget {
                tracing::warn!(
                    "Context exceeds budget: {} tokens > {} max. Pruning history.",
                    estimated_tokens,
                    context_budget
                );
                let vec = Arc::make_mut(messages);
                super::Agent::prune_history_for_budget(
                    self.token_counter.as_ref(),
                    vec,
                    context_budget,
                );
                self.ensure_context_window_available(messages.as_ref(), tool_schema_tokens)?;
            }
        }
        Ok(())
    }
}

fn is_phased_enabled(driver: ToolExecutionDriver, tool_cfg: &ToolCallingConfig) -> bool {
    matches!(driver, ToolExecutionDriver::PlannerFirstPhased)
        && (tool_cfg.use_native
            || tool_cfg.include_tool_descriptions
            || tool_cfg.supports_text_based_tools)
}

async fn resolve_phase_allowed_tools(
    agent: &super::Agent,
    phased_enabled: bool,
    query: &str,
) -> Result<Option<HashSet<String>>, AgentError> {
    if !phased_enabled {
        return Ok(None);
    }
    let tool_schemas_all = agent.tools.get_tool_schemas();
    ensure_phase_plan_for_query(
        phased_enabled,
        query,
        &tool_schemas_all,
        agent.allowed_tools.as_ref(),
        &agent.tool_intelligence,
    )
    .await?;
    Ok(current_phase_allowed_tools(phased_enabled, &agent.tool_intelligence).await)
}

async fn resolve_step_tool_schemas(
    agent: &super::Agent,
    tool_cfg: &ToolCallingConfig,
    pre_ordered_schemas: Option<&Arc<Vec<crate::messages::ToolSchema>>>,
    query: &str,
    use_native: bool,
    phase_allowed_tools: Option<&HashSet<String>>,
) -> Arc<Vec<crate::messages::ToolSchema>> {
    if !use_native {
        return Arc::clone(&tool_cfg.tool_schemas);
    }
    let ordered = if let Some(pre) = pre_ordered_schemas {
        Arc::clone(pre)
    } else if tool_cfg.tool_schemas.len() <= TOOL_ORDERING_THRESHOLD {
        Arc::clone(&tool_cfg.tool_schemas)
    } else {
        order_tools(
            &tool_cfg.tool_schemas,
            query,
            agent.tool_ordering_embedding_provider.as_ref(),
        )
        .await
    };
    filter_schemas_by_phase(ordered, phase_allowed_tools)
}

fn filter_schemas_by_phase(
    ordered: Arc<Vec<crate::messages::ToolSchema>>,
    phase_allowed_tools: Option<&HashSet<String>>,
) -> Arc<Vec<crate::messages::ToolSchema>> {
    let Some(allowed) = phase_allowed_tools else {
        return ordered;
    };
    let filtered: Vec<_> = ordered
        .iter()
        .filter(|s| allowed.contains(s.name.as_ref()))
        .cloned()
        .collect();
    if filtered.is_empty() {
        ordered
    } else {
        Arc::new(filtered)
    }
}

const BUDGET_PRUNING_MESSAGE_THRESHOLD: usize = 10;

async fn apply_step_context_budget(
    agent: &super::Agent,
    mut messages: Arc<Vec<Message>>,
    session_id: &str,
    tool_schema_tokens: usize,
) -> Result<Arc<Vec<Message>>, AgentError> {
    if messages.len() < BUDGET_PRUNING_MESSAGE_THRESHOLD {
        return Ok(messages);
    }
    let context_budget = agent.max_context_tokens.saturating_sub(tool_schema_tokens);
    #[cfg(feature = "context-engine")]
    {
        if let Some(ref engine) = agent.context_engine {
            let budget = super::context_engine::TokenBudget::new(
                agent.max_context_tokens,
                agent.max_tokens.unwrap_or(4096) as usize,
                tool_schema_tokens,
            );
            let assembled = engine
                .assemble(
                    session_id,
                    messages.as_ref(),
                    &budget,
                    agent.token_counter.as_ref(),
                )
                .await
                .map_err(|e| AgentError::CompletionFailed(e.to_string()))?;
            return Ok(Arc::new(assembled.messages));
        }
        agent
            .legacy_context_management(&mut messages, context_budget, tool_schema_tokens)
            .await?;
        return Ok(messages);
    }
    #[cfg(not(feature = "context-engine"))]
    {
        let _ = session_id;
        agent
            .legacy_context_management(&mut messages, context_budget, tool_schema_tokens)
            .await?;
        Ok(messages)
    }
}

fn rewrite_phased_tool_descriptions(
    agent: &super::Agent,
    messages: Arc<Vec<Message>>,
    use_native: bool,
    include_tool_descriptions: bool,
    phased_enabled: bool,
    phase_allowed_tools: Option<HashSet<String>>,
) -> Arc<Vec<Message>> {
    if use_native || !include_tool_descriptions || !phased_enabled {
        return messages;
    }
    let selected = phase_allowed_tools.unwrap_or_else(|| {
        let mut s = HashSet::new();
        s.insert(HANDOFF_TOOL_NAME.to_string());
        s
    });
    let mut out = (*messages).clone();
    out.retain(|m| {
        !(matches!(m.role, MessageRole::System)
            && m.content
                .as_text()
                .is_some_and(|t| t.starts_with("Available tools:\n")))
    });
    let tools_description = message_builder::describe_tools_filtered(&agent.tools, &selected);
    let tool_msg = Message {
        role: MessageRole::System,
        content: crate::messages::MessageContent::text(format!(
            "Available tools:\n{tools_description}"
        )),
        metadata: None,
    };
    if out.is_empty() {
        out.push(tool_msg);
    } else {
        out.insert(1, tool_msg);
    }
    Arc::new(out)
}

async fn complete_step_llm_call(
    agent: &super::Agent,
    messages: &Arc<Vec<Message>>,
    tools_option: &Option<Arc<Vec<crate::messages::ToolSchema>>>,
    tool_choice_option: Option<crate::messages::ToolChoice>,
    use_native: bool,
) -> Result<(crate::messages::CompletionResponse, u64), AgentError> {
    let provider_name = agent.llm_provider.name().to_string();
    let tool_count = tools_option.as_ref().map_or(0, |t| t.len());
    let complete_request = crate::messages::CompletionRequest {
        messages: Arc::clone(messages),
        max_tokens: agent.max_tokens.or(Some(4096)),
        temperature: agent.temperature.or(Some(0.7)),
        reasoning_effort: agent
            .reasoning_effort_slot
            .as_ref()
            .and_then(|slot| slot.current()),
        tools: tools_option.clone(),
        tool_choice: tool_choice_option,
        response_format: None,
    };
    tracing::info!(
        provider = %provider_name,
        tool_count,
        native_tools = use_native,
        message_count = messages.len(),
        "llm.call.start"
    );
    let start_time = std::time::Instant::now();
    let response = agent
        .llm_provider
        .complete(complete_request)
        .await
        .map_err(|e| AgentError::CompletionFailed(e.to_string()))?;
    let duration_ms = start_time.elapsed().as_millis() as u64;
    let response_chars = response.content.as_text().len();
    tracing::info!(
        provider = %provider_name,
        duration_ms,
        response_chars,
        "llm.call.end"
    );
    Ok((response, duration_ms))
}

fn collect_step_tool_calls(
    use_native: bool,
    response: &crate::messages::CompletionResponse,
    response_text: &str,
) -> (Vec<ToolCall>, bool) {
    let mut tool_calls: Vec<ToolCall> = if use_native {
        response
            .content
            .as_tool_calls()
            .iter()
            .map(|tc| ToolCall {
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let recovered_from_text = tool_calls.is_empty();
    if recovered_from_text {
        tool_calls = tool_extraction::extract_tool_calls_from_text(response_text);
        if !tool_calls.is_empty() {
            tracing::info!(
                extracted_count = tool_calls.len(),
                use_native,
                "Extracted tool calls from assistant text"
            );
        }
    }
    (tool_calls, recovered_from_text)
}

fn push_assistant_history(
    history: &mut Vec<Message>,
    use_native: bool,
    tool_calls: &[ToolCall],
    recovered_from_text: bool,
    response: &crate::messages::CompletionResponse,
    response_text: &Arc<String>,
) {
    let reasoning_for_history = response.reasoning_content.clone().unwrap_or_default();
    if use_native && !tool_calls.is_empty() {
        let tool_call_infos: Vec<crate::messages::ToolCallInfo> = if recovered_from_text {
            tool_extraction::tool_calls_as_infos(tool_calls)
        } else {
            response.content.extract_tool_calls()
        };
        if tool_call_infos.is_empty() {
            history.push(
                Message {
                    role: MessageRole::Assistant,
                    content: crate::messages::MessageContent::Text(Arc::clone(response_text)),
                    metadata: None,
                }
                .with_reasoning_content(reasoning_for_history),
            );
            return;
        }
        history.push(
            Message {
                role: MessageRole::Assistant,
                content: crate::messages::MessageContent::Structured {
                    text: if response_text.is_empty() {
                        None
                    } else {
                        Some(Arc::clone(response_text))
                    },
                    tool_calls: Some(tool_call_infos),
                    tool_call_id: None,
                },
                metadata: None,
            }
            .with_reasoning_content(reasoning_for_history),
        );
        return;
    }
    history.push(
        Message {
            role: MessageRole::Assistant,
            content: crate::messages::MessageContent::Text(Arc::clone(response_text)),
            metadata: None,
        }
        .with_reasoning_content(reasoning_for_history),
    );
}

type StepToolResults = Vec<(String, crate::agent::tools::tool::ToolResult)>;

async fn execute_step_tools(
    agent: &super::Agent,
    history: &mut Vec<Message>,
    use_native: bool,
    recovered_from_text: bool,
    tool_calls: &[ToolCall],
    response: &crate::messages::CompletionResponse,
    step_usage: crate::messages::TokenUsage,
) -> Result<(StepToolResults, u64, crate::messages::TokenUsage), AgentError> {
    if tool_calls.is_empty() {
        return Ok((Vec::new(), 0, step_usage));
    }
    let indexed = index_step_tool_calls(use_native, recovered_from_text, tool_calls, response);
    if is_length_finish_reason(response.finish_reason.as_deref()) {
        return Ok(refuse_truncated_step_tools(
            history, use_native, &indexed, step_usage,
        ));
    }
    execute_validated_step_tools(
        agent,
        history,
        use_native,
        tool_calls.len(),
        indexed,
        step_usage,
    )
    .await
}

fn index_step_tool_calls(
    use_native: bool,
    recovered_from_text: bool,
    tool_calls: &[ToolCall],
    response: &crate::messages::CompletionResponse,
) -> Vec<IndexedToolCall> {
    let tool_call_ids: Vec<String> = if use_native {
        if recovered_from_text {
            (0..tool_calls.len())
                .map(|idx| format!("call_{idx}"))
                .collect()
        } else {
            response
                .content
                .extract_tool_calls()
                .iter()
                .map(crate::messages::ToolCallInfo::id_string)
                .collect()
        }
    } else {
        Vec::new()
    };
    tool_calls
        .iter()
        .enumerate()
        .map(|(idx, tc)| {
            let id = if use_native {
                tool_call_ids
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("call_{idx}"))
            } else {
                format!("call_{idx}")
            };
            IndexedToolCall {
                idx,
                id,
                name: tc.name.as_ref().to_string(),
                arguments: tc.arguments.clone(),
                action_id: None,
            }
        })
        .collect()
}

fn refuse_truncated_step_tools(
    history: &mut Vec<Message>,
    use_native: bool,
    indexed: &[IndexedToolCall],
    step_usage: crate::messages::TokenUsage,
) -> (StepToolResults, u64, crate::messages::TokenUsage) {
    tracing::warn!(
        "Refusing {} tool call(s) in step path: finish_reason indicates length truncation",
        indexed.len()
    );
    let refusals = refuse_length_truncated_calls(indexed);
    append_owned_tool_results_to_history(history, use_native, &refusals);
    let tool_results = refusals
        .into_iter()
        .map(|(_idx, _id, name, result)| (name, result))
        .collect();
    (tool_results, 0, step_usage)
}

async fn execute_validated_step_tools(
    agent: &super::Agent,
    history: &mut Vec<Message>,
    use_native: bool,
    requested_count: usize,
    indexed: Vec<IndexedToolCall>,
    mut step_usage: crate::messages::TokenUsage,
) -> Result<(StepToolResults, u64, crate::messages::TokenUsage), AgentError> {
    let validation = validate_and_filter_tool_calls(
        indexed,
        &agent.tools,
        agent.allowed_tools.as_ref(),
        agent.permission_mode,
        agent.tool_execution_driver,
        &agent.tool_intelligence,
    )
    .await?;
    agent.record_readonly_filter_hits(validation.allowed_tools_denials);
    if use_native && !validation.invalid.is_empty() {
        append_invalid_tool_messages_native(history, &validation.invalid);
    }
    if validation.valid.is_empty() {
        tracing::warn!(
            "All {} tool call(s) were invalid or denied and have been filtered out",
            requested_count
        );
        return Ok((Vec::new(), 0, step_usage));
    }
    tracing::info!(
        batch_size = validation.valid.len(),
        "tool.execute_batch.start"
    );
    let exec = execute_tools_parallel(
        crate::agent::core::chat_helpers::tool_pipeline::ExecuteToolsParallelParams {
            valid_tool_calls: validation.valid,
            tools: &agent.tools,
            hooks: &agent.hooks,
            cancellation_flag: None,
            user_context: agent.user_context.as_ref(),
            metrics: agent.metrics.as_ref(),
            tool_intelligence: &agent.tool_intelligence,
            approval_resolver: agent.approval_resolver.clone(),
            amendment_persister: agent.amendment_persister.clone(),
            permission_mode: agent.permission_mode,
            workspace_root: agent.workspace_root_buf(),
            #[cfg(feature = "tools-rag")]
            workspace_rag: agent.workspace_rag.as_deref(),
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: agent.workspace_rag_top_k,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: agent.workspace_rag_min_similarity,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: agent.agentfs_tool_logger.as_deref(),
            #[cfg(feature = "sandbox")]
            sandbox: agent
                .sandbox
                .as_deref()
                .map(|h| h as &dyn caliche_sandbox::SandboxHandleOps),
            operation_manager: agent.operation_manager.as_deref(),
            operation_translators: agent.operation_translators.as_deref(),
        },
    )
    .await;
    if use_native && !exec.denied.is_empty() {
        append_invalid_tool_messages_native(history, &exec.denied);
    }
    let executed = exec.executed;
    let tool_duration_ms = executed
        .iter()
        .map(|(_idx, _id, _name, _args, dur, _res)| *dur)
        .sum();
    let results_for_history: Vec<(usize, &str, &str, &crate::agent::tools::tool::ToolResult)> =
        executed
            .iter()
            .map(|(idx, id, name, _args, _dur, res)| (*idx, id.as_str(), name.as_str(), res))
            .collect();
    let max_chars = agent.max_tool_output_in_history;
    let annotate = agent.annotate_tool_output_truncation;
    append_tool_results_to_history(
        history,
        use_native,
        &results_for_history,
        |content, tool_name| {
            tool_output::summarize_tool_output(
                content,
                tool_name,
                tool_output::ToolOutputLimits {
                    max_chars,
                    summarize_threshold: max_chars,
                    head_chars: max_chars * 4 / 5,
                    tail_chars: max_chars / 10,
                    annotate,
                },
            )
        },
    );
    let mut usage_acc = TokenUsageAccumulator::default();
    usage_acc.add_usage(&step_usage);
    for (_idx, _id, _name, _args, _dur, tool_result) in &executed {
        if let Some(tool_usage) = tool_result.usage.as_ref() {
            usage_acc.add_usage(tool_usage);
        }
    }
    step_usage = usage_acc.into_token_usage();
    let tool_results = executed
        .into_iter()
        .map(|(_idx, _id, name, _args, _dur, result)| (name, result))
        .collect();
    Ok((tool_results, tool_duration_ms, step_usage))
}
