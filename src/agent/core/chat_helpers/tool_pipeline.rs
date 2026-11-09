use crate::agent::ToolExecutionDriver;
use crate::agent::core::ToolIntelligenceState;
use crate::agent::hooks::HookRegistry;
use crate::agent::hooks::permission::PermissionMode;
use crate::agent::hooks::{AmendmentPersister, ApprovalResolver};
use crate::agent::tools::ToolRegistry;
use crate::agent::tools::execution::{
    HookExecutionMiddleware, ToolExecutionEngine, ToolExecutionMiddleware, ToolExecutionRequest,
};
use crate::agent::tools::tool::{ToolCall, ToolResult as ToolResultType};
use crate::errors::AgentError;
use crate::messages::{Message, MessageRole};
use crate::observability::ToolMetrics;
use crate::utils::user_context::UserContext;
use beef::lean::Cow;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct IndexedToolCall {
    pub idx: usize,
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    /// UUID v7 shared with the tool's start and result stream events.
    pub action_id: Option<String>,
}

#[derive(Debug, Default)]
pub struct ToolValidationOutcome {
    pub valid: Vec<(usize, String, Arc<ToolCall>)>, // (idx, id, tool_call)
    pub invalid: Vec<(usize, String, String, Cow<'static, str>)>, // (idx, id, name, reason)
    /// Tool calls denied because they were outside `allowed_tools` (readonly modes).
    pub allowed_tools_denials: usize,
}

#[derive(Debug, Default)]
pub struct ToolExecutionOutcome {
    pub executed: Vec<(
        usize,
        String,
        String,
        serde_json::Value,
        u64,
        ToolResultType,
    )>, // (idx,id,name,args,dur,result)
    pub denied: Vec<(usize, String, String, Cow<'static, str>)>, // (idx,id,name,reason)
}

fn extract_file_scheme_url(arguments: &serde_json::Value) -> Option<String> {
    let starts_with_file_scheme = |value: &str| value.trim_start().starts_with("file://");

    if let Some(url) = arguments.get("url").and_then(|v| v.as_str())
        && starts_with_file_scheme(url)
    {
        return Some(url.to_string());
    }
    if let Some(endpoint) = arguments.get("endpoint").and_then(|v| v.as_str())
        && starts_with_file_scheme(endpoint)
    {
        return Some(endpoint.to_string());
    }
    if let Some(base_url) = arguments.get("base_url").and_then(|v| v.as_str())
        && starts_with_file_scheme(base_url)
    {
        return Some(base_url.to_string());
    }
    if let Some(urls) = arguments.get("urls") {
        if let Some(url) = urls.as_str()
            && starts_with_file_scheme(url)
        {
            return Some(url.to_string());
        }
        if let Some(url_list) = urls.as_array() {
            for url in url_list {
                if let Some(url_str) = url.as_str()
                    && starts_with_file_scheme(url_str)
                {
                    return Some(url_str.to_string());
                }
            }
        }
    }
    None
}

pub async fn validate_and_filter_tool_calls(
    tool_calls: Vec<IndexedToolCall>,
    tools: &ToolRegistry,
    allowed_tools: Option<&Arc<HashSet<String>>>,
    permission_mode: PermissionMode,
    tool_execution_driver: ToolExecutionDriver,
    tool_intelligence: &Arc<Mutex<ToolIntelligenceState>>,
) -> Result<ToolValidationOutcome, AgentError> {
    let mut outcome = ToolValidationOutcome::default();
    let mut seen_calls: Vec<(String, serde_json::Value)> = Vec::new();

    let mut intel_guard = if matches!(
        tool_execution_driver,
        ToolExecutionDriver::DedupOnly | ToolExecutionDriver::PlannerFirstPhased
    ) {
        Some(tool_intelligence.lock().await)
    } else {
        None
    };

    for tc in tool_calls {
        // In-turn deduplication: identical (name, arguments) calls are skipped.
        if seen_calls
            .iter()
            .any(|(n, a)| n == &tc.name && a == &tc.arguments)
        {
            outcome.invalid.push((
                tc.idx,
                tc.id,
                tc.name,
                Cow::borrowed("Duplicate tool call (same name + arguments)"),
            ));
            continue;
        }
        seen_calls.push((tc.name.clone(), tc.arguments.clone()));

        // Allowed tools filter
        if let Some(allowed) = allowed_tools
            && !allowed.contains(&*tc.name)
        {
            outcome.allowed_tools_denials = outcome.allowed_tools_denials.saturating_add(1);
            outcome.invalid.push((
                tc.idx,
                tc.id,
                tc.name.clone(),
                Cow::owned(format!("Tool '{}' is not in allowed_tools list", tc.name)),
            ));
            continue;
        }

        // Cross-step deduplication (best-effort)
        if let Some(ref mut intel) = intel_guard {
            let report = intel
                .deduplicator
                .is_redundant(tc.name.as_ref(), &tc.arguments);
            if report.is_redundant {
                outcome.invalid.push((
                    tc.idx,
                    tc.id,
                    tc.name.clone(),
                    report
                        .suggestion
                        .map_or_else(|| Cow::borrowed("Redundant tool call"), Cow::owned),
                ));
                continue;
            }
            intel
                .deduplicator
                .record_call(tc.name.clone(), tc.arguments.clone());

            // Cross-iteration dedup for grep: skip if a prior grep covered the same or parent path
            // with a pattern that subsumes the new one.
            if tc.name == "fs"
                && let Some(op) = tc.arguments.get("op").and_then(|v| v.as_str())
                && op == "grep"
            {
                let new_path = tc
                    .arguments
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or(".");
                let new_pattern = tc
                    .arguments
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if intel
                    .execution_history
                    .has_covering_grep(new_path, new_pattern)
                {
                    outcome.invalid.push((
                                tc.idx,
                                tc.id,
                                tc.name,
                                Cow::owned(format!(
                                    "Redundant grep: a prior grep already covered path '{new_path}' with a similar pattern. Use the existing results."
                                )),
                            ));
                    continue;
                }
            }
        }

        // Permission mode (global fast-path)
        if matches!(permission_mode, PermissionMode::DenyAll) {
            outcome.invalid.push((
                tc.idx,
                tc.id,
                tc.name.clone(),
                Cow::borrowed("Permission mode is DenyAll"),
            ));
            continue;
        }

        // Normalize arguments when streaming wrapped failed JSON in {"raw": "<string>"}
        let arguments = ToolRegistry::normalize_tool_arguments(&tc.name, &tc.arguments);

        // Guardrail: http only supports network URLs. Local file access should use fs.
        if tc.name.as_str() == "http"
            && let Some(local_url) = extract_file_scheme_url(&arguments)
        {
            outcome.invalid.push((
                tc.idx,
                tc.id,
                tc.name.clone(),
                Cow::owned(format!(
                    "http does not support local file URL '{local_url}'; use fs with op=list/read/grep for local filesystem access"
                )),
            ));
            continue;
        }

        // Validate tool call
        match tools.validate_tool_call(tc.name.as_ref(), &arguments) {
            Ok(()) => {
                outcome.valid.push((
                    tc.idx,
                    tc.id,
                    Arc::new(ToolCall {
                        name: beef::lean::Cow::owned(tc.name.clone()),
                        arguments,
                    }),
                ));
            }
            Err(e) => {
                outcome
                    .invalid
                    .push((tc.idx, tc.id, tc.name.clone(), Cow::owned(e.to_string())));
            }
        }
    }

    // Release lock before any awaited operations downstream.
    drop(intel_guard);

    Ok(outcome)
}

/// Parameters for [`execute_tools_parallel`]; groups many arguments to satisfy `clippy::too_many_arguments`.
pub struct ExecuteToolsParallelParams<'a> {
    pub valid_tool_calls: Vec<(usize, String, Arc<ToolCall>)>,
    pub tools: &'a ToolRegistry,
    pub hooks: &'a HookRegistry,
    pub cancellation_flag: Option<&'a Arc<std::sync::atomic::AtomicBool>>,
    pub user_context: Option<&'a UserContext>,
    pub metrics: Option<&'a Arc<ToolMetrics>>,
    pub tool_intelligence: &'a Arc<Mutex<ToolIntelligenceState>>,
    pub approval_resolver: Option<Arc<dyn ApprovalResolver>>,
    pub amendment_persister: Option<Arc<dyn AmendmentPersister>>,
    pub permission_mode: PermissionMode,
    pub workspace_root: Option<std::path::PathBuf>,
    #[cfg(feature = "tools-rag")]
    pub workspace_rag: Option<&'a crate::rag::RagService>,
    #[cfg(feature = "tools-rag")]
    pub workspace_rag_top_k: usize,
    #[cfg(feature = "tools-rag")]
    pub workspace_rag_min_similarity: f32,
    #[cfg(feature = "agentfs")]
    pub agentfs_tool_logger: Option<&'a crate::agent::tools::AgentFsToolLogger>,
    #[cfg(feature = "sandbox")]
    pub sandbox: Option<&'a dyn caliche_sandbox::SandboxHandleOps>,
    /// When set, tool calls go through translate → commit → execute → format.
    pub operation_manager: Option<&'a dyn crate::agent::operations::OperationManager>,
    /// Translators for the operation path. Ignored when `operation_manager` is `None`.
    pub operation_translators: Option<&'a crate::agent::operations::TranslatorRegistry>,
}

/// Drop duplicate tool calls in the same batch (same tool name + identical JSON arguments).
fn dedupe_valid_tool_calls(
    calls: Vec<(usize, String, Arc<ToolCall>)>,
) -> Vec<(usize, String, Arc<ToolCall>)> {
    use std::collections::HashSet;
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut out = Vec::with_capacity(calls.len());
    for (idx, id, tc) in calls {
        let name = tc.name.to_string();
        let args_key = serde_json::to_string(&tc.arguments).unwrap_or_default();
        if seen.insert((name.clone(), args_key)) {
            out.push((idx, id, tc));
        } else {
            tracing::debug!(
                target: "caliche::tools",
                tool = %name,
                call_id = %id,
                "suppressing duplicate tool call in the same execution batch"
            );
        }
    }
    out
}

struct NoopExecutionMiddleware;

#[async_trait::async_trait]
impl ToolExecutionMiddleware for NoopExecutionMiddleware {}

struct PreparedOperationCalls {
    engine_calls: Vec<(usize, String, Arc<ToolCall>)>,
    ids_by_idx: std::collections::HashMap<usize, Vec<crate::agent::operations::OperationId>>,
    invalid: Vec<(usize, String, String, ToolResultType)>,
}

async fn commit_translated_calls(
    valid_tool_calls: Vec<(usize, String, Arc<ToolCall>)>,
    manager: &dyn crate::agent::operations::OperationManager,
    translators: &crate::agent::operations::TranslatorRegistry,
) -> PreparedOperationCalls {
    let mut engine_calls = Vec::new();
    let mut ids_by_idx = std::collections::HashMap::new();
    let mut invalid = Vec::new();
    for (idx, id, call) in valid_tool_calls {
        let translation = translators.resolve(call.name.as_ref()).translate(&call);
        match &translation.status {
            crate::agent::operations::ToolCallStatus::Invalid { message } => {
                invalid.push((
                    idx,
                    id,
                    call.name.to_string(),
                    ToolResultType::error(message),
                ));
            }
            crate::agent::operations::ToolCallStatus::Submitted { .. } => {
                match manager.commit(&translation.operations).await {
                    Ok(op_ids) => {
                        ids_by_idx.insert(idx, op_ids);
                        engine_calls.push((idx, id, call));
                    }
                    Err(err) => {
                        invalid.push((
                            idx,
                            id,
                            call.name.to_string(),
                            ToolResultType::error(format!("operation commit failed: {err}")),
                        ));
                    }
                }
            }
        }
    }
    PreparedOperationCalls {
        engine_calls,
        ids_by_idx,
        invalid,
    }
}

#[cfg(feature = "sandbox")]
fn refuse_remote_without_sandbox_isolation(
    valid_tool_calls: &[(usize, String, Arc<ToolCall>)],
) -> ToolExecutionOutcome {
    refuse_remote_sandbox_gate(
        valid_tool_calls,
        "sandbox is configured; RemoteOperationManager must use \
         SandboxOperationTransport (LocalOperationManager keeps engine isolation)",
    )
}

#[cfg(feature = "sandbox")]
fn refuse_remote_sandbox_handle_mismatch(
    valid_tool_calls: &[(usize, String, Arc<ToolCall>)],
) -> ToolExecutionOutcome {
    refuse_remote_sandbox_gate(
        valid_tool_calls,
        "sandbox is configured; isolating RemoteOperationTransport must bind \
         the agent's configured sandbox handle",
    )
}

#[cfg(feature = "sandbox")]
fn refuse_remote_sandbox_gate(
    valid_tool_calls: &[(usize, String, Arc<ToolCall>)],
    reason: &'static str,
) -> ToolExecutionOutcome {
    ToolExecutionOutcome {
        executed: Vec::new(),
        denied: valid_tool_calls
            .iter()
            .map(|(idx, id, call)| {
                (
                    *idx,
                    id.clone(),
                    call.name.to_string(),
                    beef::lean::Cow::const_str(reason),
                )
            })
            .collect(),
    }
}

async fn execute_via_operations(
    params: ExecuteToolsParallelParams<'_>,
    valid_tool_calls: Vec<(usize, String, Arc<ToolCall>)>,
    manager: &dyn crate::agent::operations::OperationManager,
    translators: &crate::agent::operations::TranslatorRegistry,
    hook_middleware: HookExecutionMiddleware,
) -> ToolExecutionOutcome {
    use crate::agent::operations::OperationExecutionMode;

    if manager.execution_mode() == OperationExecutionMode::Remote {
        #[cfg(feature = "sandbox")]
        if params.sandbox.is_some() {
            if !manager.isolates_in_sandbox() {
                return refuse_remote_without_sandbox_isolation(&valid_tool_calls);
            }
            if !manager.binds_configured_sandbox(params.sandbox) {
                return refuse_remote_sandbox_handle_mismatch(&valid_tool_calls);
            }
        }
        return execute_remote_operations(valid_tool_calls, manager, translators, &hook_middleware)
            .await;
    }

    let prepared = commit_translated_calls(valid_tool_calls, manager, translators).await;
    let valid_tool_calls = prepared.engine_calls;

    let engine = ToolExecutionEngine::new();
    let middlewares: [&dyn crate::agent::tools::execution::ToolExecutionMiddleware; 1] =
        [&hook_middleware];
    let result = engine
        .execute_parallel(
            ToolExecutionRequest {
                valid_tool_calls,
                tools: params.tools,
                user_context: params.user_context,
                metrics: params.metrics,
                tool_intelligence: params.tool_intelligence,
                cancellation_flag: params.cancellation_flag,
                workspace_root: params.workspace_root,
                #[cfg(feature = "agentfs")]
                agentfs_tool_logger: params.agentfs_tool_logger,
                #[cfg(feature = "sandbox")]
                sandbox: params.sandbox,
                #[cfg(feature = "tools-rag")]
                workspace_rag: params.workspace_rag,
                #[cfg(feature = "tools-rag")]
                workspace_rag_top_k: params.workspace_rag_top_k,
                #[cfg(feature = "tools-rag")]
                workspace_rag_min_similarity: params.workspace_rag_min_similarity,
            },
            &middlewares,
        )
        .await;
    record_engine_audit(
        manager,
        &prepared.ids_by_idx,
        &result.executed,
        &result.denied,
    );
    let mut executed = result.executed;
    executed.extend(
        prepared
            .invalid
            .into_iter()
            .map(|(idx, id, name, result)| (idx, id, name, serde_json::Value::Null, 0, result)),
    );
    executed.sort_by_key(|(idx, _, _, _, _, _)| *idx);
    ToolExecutionOutcome {
        executed,
        denied: result.denied,
    }
}

fn record_engine_audit(
    manager: &dyn crate::agent::operations::OperationManager,
    ids_by_idx: &std::collections::HashMap<usize, Vec<crate::agent::operations::OperationId>>,
    executed: &[(
        usize,
        String,
        String,
        serde_json::Value,
        u64,
        ToolResultType,
    )],
    denied: &[(usize, String, String, beef::lean::Cow<'static, str>)],
) {
    use crate::agent::operations::{OperationExecution, OperationId};
    let mut results: Vec<(OperationId, OperationExecution)> = Vec::new();
    for (idx, _, _, _, _, result) in executed {
        let Some(ids) = ids_by_idx.get(idx) else {
            continue;
        };
        for id in ids {
            let execution = if result.success {
                OperationExecution::completed(id.clone(), result.content.clone())
            } else {
                OperationExecution::failed(
                    id.clone(),
                    result
                        .error
                        .clone()
                        .unwrap_or_else(|| "tool execution failed".to_string()),
                )
            };
            results.push((id.clone(), execution));
        }
    }
    for (idx, _, _, reason) in denied {
        let Some(ids) = ids_by_idx.get(idx) else {
            continue;
        };
        for id in ids {
            results.push((
                id.clone(),
                OperationExecution::failed(id.clone(), reason.to_string()),
            ));
        }
    }
    manager.record_engine_results(&results);
}

async fn execute_remote_operations(
    valid_tool_calls: Vec<(usize, String, Arc<ToolCall>)>,
    manager: &dyn crate::agent::operations::OperationManager,
    translators: &crate::agent::operations::TranslatorRegistry,
    middleware: &dyn crate::agent::tools::execution::ToolExecutionMiddleware,
) -> ToolExecutionOutcome {
    let mut executed = Vec::new();
    let mut denied = Vec::new();
    let mut work = Vec::new();
    for (idx, id, call) in valid_tool_calls {
        let name = call.name.to_string();
        let ctx = crate::agent::tools::execution::ToolExecutionContext {
            tool_name: call.name.as_ref(),
            arguments: &call.arguments,
            call_id: &id,
        };
        if let Err(reason) = middleware.before_execute(&ctx).await {
            denied.push((idx, id, name, reason));
            continue;
        }
        if let Err(reason) = middleware.admit_execution(&ctx).await {
            denied.push((idx, id, name, reason));
            continue;
        }
        work.push((idx, id, call, name));
    }

    let dispatched =
        futures_util::future::join_all(work.into_iter().map(|(idx, id, call, name)| async move {
            let (duration_ms, result) = crate::agent::operations::execute_tool_call_via_operations(
                call.as_ref(),
                &id,
                translators,
                manager,
                &NoopExecutionMiddleware,
            )
            .await;
            (idx, id, name, call.arguments.clone(), duration_ms, result)
        }))
        .await;

    for (idx, id, name, arguments, duration_ms, mut result) in dispatched {
        let ctx = crate::agent::tools::execution::ToolExecutionContext {
            tool_name: &name,
            arguments: &arguments,
            call_id: &id,
        };
        middleware.after_execute(&ctx, &mut result).await;
        executed.push((idx, id, name, arguments, duration_ms, result));
    }
    executed.sort_by_key(|(idx, _, _, _, _, _)| *idx);
    ToolExecutionOutcome { executed, denied }
}

fn mode_overlay(mode: PermissionMode) -> crate::policy::ModeOverlay {
    match mode {
        PermissionMode::Default => crate::policy::ModeOverlay::Default,
        PermissionMode::AcceptEdits => crate::policy::ModeOverlay::AcceptEdits,
        PermissionMode::AcceptAll => crate::policy::ModeOverlay::AcceptAll,
        PermissionMode::DenyAll => crate::policy::ModeOverlay::DenyAll,
        PermissionMode::Interactive => crate::policy::ModeOverlay::Interactive,
    }
}

pub async fn execute_tools_parallel(
    mut params: ExecuteToolsParallelParams<'_>,
) -> ToolExecutionOutcome {
    let valid_tool_calls = dedupe_valid_tool_calls(std::mem::take(&mut params.valid_tool_calls));
    let mut hook_middleware = HookExecutionMiddleware::new(params.hooks.clone())
        .with_overlay(mode_overlay(params.permission_mode))
        .with_approval_resolver(params.approval_resolver.clone())
        .with_amendment_persister(params.amendment_persister.clone());
    if let Some(root) = params.workspace_root.as_ref() {
        hook_middleware = hook_middleware.with_workspace(root.to_string_lossy());
    }

    if let Some(manager) = params.operation_manager {
        let owned_translators = crate::agent::operations::TranslatorRegistry::builtin();
        let translators = params.operation_translators.unwrap_or(&owned_translators);
        return execute_via_operations(
            params,
            valid_tool_calls,
            manager,
            translators,
            hook_middleware,
        )
        .await;
    }

    let engine = ToolExecutionEngine::new();
    let middlewares: [&dyn crate::agent::tools::execution::ToolExecutionMiddleware; 1] =
        [&hook_middleware];

    let result = engine
        .execute_parallel(
            ToolExecutionRequest {
                valid_tool_calls,
                tools: params.tools,
                user_context: params.user_context,
                metrics: params.metrics,
                tool_intelligence: params.tool_intelligence,
                cancellation_flag: params.cancellation_flag,
                workspace_root: params.workspace_root,
                #[cfg(feature = "agentfs")]
                agentfs_tool_logger: params.agentfs_tool_logger,
                #[cfg(feature = "sandbox")]
                sandbox: params.sandbox,
                #[cfg(feature = "tools-rag")]
                workspace_rag: params.workspace_rag,
                #[cfg(feature = "tools-rag")]
                workspace_rag_top_k: params.workspace_rag_top_k,
                #[cfg(feature = "tools-rag")]
                workspace_rag_min_similarity: params.workspace_rag_min_similarity,
            },
            &middlewares,
        )
        .await;

    ToolExecutionOutcome {
        executed: result.executed,
        denied: result.denied,
    }
}

/// True when the provider stopped generation because the response hit the token limit.
///
/// Tool calls from such a message can carry incomplete arguments, so the agent loop refuses
/// them instead of executing them.
#[must_use]
pub fn is_length_finish_reason(reason: Option<&str>) -> bool {
    reason.is_some_and(|r| {
        let lower = r.to_ascii_lowercase();
        lower == "length" || lower == "max_tokens" || lower.contains("max_token")
    })
}

/// Error text returned to the model for a tool call refused because of length truncation.
#[must_use]
pub fn length_truncated_tool_error(tool_name: &str) -> String {
    format!(
        "Error: tool call `{tool_name}` was not executed because the assistant response \
         was truncated (finish_reason=length). Re-issue the tool call with complete \
         arguments in a new turn."
    )
}

/// Build refusal results for every tool call in a length-truncated assistant message.
#[must_use]
pub fn refuse_length_truncated_calls(
    calls: &[IndexedToolCall],
) -> Vec<(usize, String, String, ToolResultType)> {
    calls
        .iter()
        .map(|call| {
            (
                call.idx,
                call.id.clone(),
                call.name.clone(),
                ToolResultType::error(length_truncated_tool_error(&call.name)),
            )
        })
        .collect()
}

/// Append owned tool results to the history without summarizing them.
///
/// Use this for refusals and other short synthetic results whose text is already final.
pub fn append_owned_tool_results_to_history(
    history: &mut Vec<Message>,
    use_native: bool,
    results: &[(usize, String, String, ToolResultType)],
) {
    let borrowed: Vec<(usize, &str, &str, &ToolResultType)> = results
        .iter()
        .map(|(idx, id, name, result)| (*idx, id.as_str(), name.as_str(), result))
        .collect();
    append_tool_results_to_history(history, use_native, &borrowed, |content, _name| {
        content.to_string()
    });
}

pub fn append_invalid_tool_messages_native(
    history: &mut Vec<Message>,
    invalid: &[(usize, String, String, Cow<'static, str>)],
) {
    for (_idx, tool_call_id, _tool_name, error_msg) in invalid {
        let error_text = format!("Tool call validation failed: {error_msg}");
        history.push(Message {
            role: MessageRole::Tool,
            content: crate::messages::MessageContent::Structured {
                text: Some(Arc::new(error_text)),
                tool_calls: None,
                tool_call_id: Some(beef::lean::Cow::owned(tool_call_id.clone())),
            },
            metadata: None,
        });
    }
}

pub fn append_tool_results_to_history(
    history: &mut Vec<Message>,
    use_native: bool,
    results: &[(usize, &str, &str, &ToolResultType)],
    summarize: impl Fn(&str, &str) -> String,
) {
    if use_native {
        for (_idx, tool_call_id, tool_name, tool_result) in results {
            let tool_content = if tool_result.success {
                if tool_result.content.is_empty() {
                    "Tool executed successfully (no output)".to_string()
                } else {
                    summarize(&tool_result.content, tool_name)
                }
            } else {
                format!(
                    "Error: {}",
                    tool_result.error.as_deref().unwrap_or("Unknown error")
                )
            };

            history.push(Message {
                role: MessageRole::Tool,
                content: crate::messages::MessageContent::Structured {
                    text: Some(Arc::new(tool_content)),
                    tool_calls: None,
                    tool_call_id: Some(beef::lean::Cow::owned((*tool_call_id).to_string())),
                },
                metadata: None,
            });
        }
        return;
    }

    let mut combined_results = String::new();
    for (_idx, _tool_call_id, tool_name, tool_result) in results {
        if !combined_results.is_empty() {
            combined_results.push_str("\n\n");
        }
        use std::fmt::Write;
        if tool_result.success {
            let summarized_content = summarize(&tool_result.content, tool_name);
            if write!(
                combined_results,
                "Tool {tool_name} executed successfully. Result: {summarized_content}"
            )
            .is_err()
            {
                combined_results.push_str("Tool ");
                combined_results.push_str(tool_name);
                combined_results.push_str(" executed successfully (result too long to display)");
            }
        } else if write!(
            combined_results,
            "Tool {} execution failed. Error: {}",
            tool_name,
            tool_result.error.as_deref().unwrap_or("Unknown error")
        )
        .is_err()
        {
            combined_results.push_str("Tool ");
            combined_results.push_str(tool_name);
            combined_results.push_str(" execution failed");
        }
    }

    history.push(Message {
        role: MessageRole::User,
        content: crate::messages::MessageContent::text(format!(
            "Parallel tool execution completed ({} tool(s)):\n{}",
            results.len(),
            combined_results
        )),
        metadata: None,
    });
}

#[cfg(test)]
mod tests {
    use super::{
        ExecuteToolsParallelParams, IndexedToolCall, PermissionMode,
        append_owned_tool_results_to_history, execute_tools_parallel, is_length_finish_reason,
        refuse_length_truncated_calls,
    };
    use crate::agent::operations::OperationManager;

    #[test]
    fn length_finish_reasons_are_recognized_across_provider_spellings() {
        assert!(is_length_finish_reason(Some("length")));
        assert!(is_length_finish_reason(Some("LENGTH")));
        assert!(is_length_finish_reason(Some("max_tokens")));
        assert!(is_length_finish_reason(Some("MAX_TOKENS")));
        assert!(is_length_finish_reason(Some("model_max_tokens_reached")));
        assert!(!is_length_finish_reason(Some("stop")));
        assert!(!is_length_finish_reason(Some("tool_calls")));
        assert!(!is_length_finish_reason(None));
    }

    #[test]
    fn truncated_calls_are_refused_and_recorded_for_every_call() {
        let calls = vec![
            IndexedToolCall {
                idx: 0,
                id: "call_0".to_string(),
                name: "fs".to_string(),
                arguments: serde_json::json!({"op": "read"}),
                action_id: None,
            },
            IndexedToolCall {
                idx: 1,
                id: "call_1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
                action_id: None,
            },
        ];

        let refusals = refuse_length_truncated_calls(&calls);
        assert_eq!(refusals.len(), 2);
        for (_idx, _id, name, result) in &refusals {
            assert!(!result.success);
            let error = result.error.as_deref().unwrap_or_default();
            assert!(error.contains(name), "refusal must name the tool: {error}");
            assert!(error.contains("finish_reason=length"));
        }

        let mut history = Vec::new();
        append_owned_tool_results_to_history(&mut history, true, &refusals);
        assert_eq!(history.len(), 2, "native mode records one message per call");
    }

    #[derive(Debug)]
    struct EchoTool;

    impl crate::agent::tools::tool::ToolMeta for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "echo"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
    }

    #[async_trait::async_trait]
    impl crate::agent::tools::tool::Tool for EchoTool {
        async fn execute(&self, args: serde_json::Value) -> crate::agent::tools::tool::ToolResult {
            crate::agent::tools::tool::ToolResult::success(args.to_string())
        }
    }

    #[derive(Debug)]
    struct DenyEchoHook;

    #[async_trait::async_trait]
    impl crate::agent::hooks::LifecycleHook for DenyEchoHook {
        async fn on_pre_tool_use(
            &self,
            _ctx: &crate::agent::hooks::HookContext<'_>,
        ) -> crate::agent::hooks::HookResult {
            crate::agent::hooks::HookResult::deny("blocked by policy")
        }
    }

    fn echo_registry() -> crate::agent::tools::ToolRegistry {
        let registry = crate::agent::tools::ToolRegistry::new();
        registry.register(std::sync::Arc::new(EchoTool));
        registry
    }

    fn echo_call() -> (
        usize,
        String,
        std::sync::Arc<crate::agent::tools::tool::ToolCall>,
    ) {
        (
            0,
            "c1".to_string(),
            std::sync::Arc::new(crate::agent::tools::tool::ToolCall {
                name: beef::lean::Cow::const_str("echo"),
                arguments: serde_json::json!({"msg": "hi"}),
            }),
        )
    }

    fn bash_invalid_call() -> (
        usize,
        String,
        std::sync::Arc<crate::agent::tools::tool::ToolCall>,
    ) {
        (
            0,
            "c-bash".to_string(),
            std::sync::Arc::new(crate::agent::tools::tool::ToolCall {
                name: beef::lean::Cow::const_str("bash"),
                arguments: serde_json::json!({}),
            }),
        )
    }

    #[tokio::test]
    async fn local_ops_happy_path_executes_via_engine_and_finishes_audit() {
        let registry = echo_registry();
        let hooks = crate::agent::hooks::HookRegistry::new();
        let manager = crate::agent::operations::LocalOperationManager::new();
        let translators = crate::agent::operations::TranslatorRegistry::builtin();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let outcome = execute_tools_parallel(ExecuteToolsParallelParams {
            valid_tool_calls: vec![echo_call()],
            tools: &registry,
            hooks: &hooks,
            cancellation_flag: None,
            user_context: None,
            metrics: None,
            tool_intelligence: &intel,
            approval_resolver: None,
            amendment_persister: None,
            permission_mode: PermissionMode::Default,
            workspace_root: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 0,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.0,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            #[cfg(feature = "sandbox")]
            sandbox: None,
            operation_manager: Some(&manager),
            operation_translators: Some(&translators),
        })
        .await;
        assert_eq!(outcome.denied.len(), 0);
        assert_eq!(outcome.executed.len(), 1);
        assert!(outcome.executed[0].5.success);
        assert!(outcome.executed[0].5.content.contains("hi"));
        let ids = manager.committed_ids();
        assert_eq!(ids.len(), 1);
        assert_eq!(
            manager.status(&ids[0]).await.expect("audit status"),
            crate::agent::operations::OperationStatus::Completed
        );
        let err = manager
            .execute(&ids)
            .await
            .expect("already finished is resolvable");
        assert_eq!(
            err[0].status,
            crate::agent::operations::OperationStatus::Completed
        );
    }

    #[tokio::test]
    async fn default_path_without_manager_executes_via_engine() {
        let registry = echo_registry();
        let hooks = crate::agent::hooks::HookRegistry::new();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let outcome = execute_tools_parallel(ExecuteToolsParallelParams {
            valid_tool_calls: vec![echo_call()],
            tools: &registry,
            hooks: &hooks,
            cancellation_flag: None,
            user_context: None,
            metrics: None,
            tool_intelligence: &intel,
            approval_resolver: None,
            amendment_persister: None,
            permission_mode: PermissionMode::Default,
            workspace_root: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 0,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.0,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            #[cfg(feature = "sandbox")]
            sandbox: None,
            operation_manager: None,
            operation_translators: None,
        })
        .await;
        assert_eq!(outcome.denied.len(), 0);
        assert_eq!(outcome.executed.len(), 1);
        assert!(outcome.executed[0].5.success);
    }

    #[tokio::test]
    async fn local_ops_path_hook_deny_stays_in_denied_list() {
        let registry = echo_registry();
        let mut hooks = crate::agent::hooks::HookRegistry::new();
        hooks.register(
            crate::agent::hooks::HookMatcher::tool("echo"),
            std::sync::Arc::new(DenyEchoHook),
        );
        let manager = crate::agent::operations::LocalOperationManager::new();
        let translators = crate::agent::operations::TranslatorRegistry::builtin();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let outcome = execute_tools_parallel(ExecuteToolsParallelParams {
            valid_tool_calls: vec![echo_call()],
            tools: &registry,
            hooks: &hooks,
            cancellation_flag: None,
            user_context: None,
            metrics: None,
            tool_intelligence: &intel,
            approval_resolver: None,
            amendment_persister: None,
            permission_mode: PermissionMode::Default,
            workspace_root: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 0,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.0,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            #[cfg(feature = "sandbox")]
            sandbox: None,
            operation_manager: Some(&manager),
            operation_translators: Some(&translators),
        })
        .await;
        assert_eq!(outcome.executed.len(), 0);
        assert_eq!(outcome.denied.len(), 1);
        assert!(outcome.denied[0].3.contains("blocked by policy"));
    }

    #[tokio::test]
    async fn local_ops_path_invalid_translation_is_executed_error() {
        let registry = echo_registry();
        let hooks = crate::agent::hooks::HookRegistry::new();
        let manager = crate::agent::operations::LocalOperationManager::new();
        let translators = crate::agent::operations::TranslatorRegistry::builtin();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let outcome = execute_tools_parallel(ExecuteToolsParallelParams {
            valid_tool_calls: vec![bash_invalid_call()],
            tools: &registry,
            hooks: &hooks,
            cancellation_flag: None,
            user_context: None,
            metrics: None,
            tool_intelligence: &intel,
            approval_resolver: None,
            amendment_persister: None,
            permission_mode: PermissionMode::Default,
            workspace_root: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 0,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.0,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            #[cfg(feature = "sandbox")]
            sandbox: None,
            operation_manager: Some(&manager),
            operation_translators: Some(&translators),
        })
        .await;
        assert_eq!(outcome.denied.len(), 0);
        assert_eq!(outcome.executed.len(), 1);
        assert!(!outcome.executed[0].5.success);
        let error = outcome.executed[0].5.error.as_deref().unwrap_or_default();
        assert!(
            error.contains("command"),
            "invalid bash translation should name command: {error}"
        );
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn remote_ops_with_sandbox_and_recording_transport_fails_closed() {
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        #[derive(Debug)]
        struct FakeSandbox;

        #[async_trait::async_trait]
        impl caliche_sandbox::SandboxHandleOps for FakeSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> anyhow::Result<ExecResult> {
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, _path: &str) -> anyhow::Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let registry = echo_registry();
        let hooks = crate::agent::hooks::HookRegistry::new();
        let transport =
            std::sync::Arc::new(crate::agent::operations::RecordingTransport::default());
        let manager = crate::agent::operations::RemoteOperationManager::new(transport);
        let translators = crate::agent::operations::TranslatorRegistry::builtin();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let sandbox = FakeSandbox;
        let outcome = execute_tools_parallel(ExecuteToolsParallelParams {
            valid_tool_calls: vec![echo_call()],
            tools: &registry,
            hooks: &hooks,
            cancellation_flag: None,
            user_context: None,
            metrics: None,
            tool_intelligence: &intel,
            approval_resolver: None,
            amendment_persister: None,
            permission_mode: PermissionMode::Default,
            workspace_root: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 0,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.0,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            sandbox: Some(&sandbox),
            operation_manager: Some(&manager),
            operation_translators: Some(&translators),
        })
        .await;
        assert_eq!(outcome.executed.len(), 0);
        assert_eq!(outcome.denied.len(), 1);
        assert!(
            outcome.denied[0].3.contains("SandboxOperationTransport"),
            "fail-closed reason: {}",
            outcome.denied[0].3
        );
    }

    #[cfg(feature = "sandbox")]
    fn pipeline_params<'a>(
        registry: &'a crate::agent::tools::ToolRegistry,
        hooks: &'a crate::agent::hooks::HookRegistry,
        intel: &'a std::sync::Arc<tokio::sync::Mutex<crate::agent::core::ToolIntelligenceState>>,
        sandbox: Option<&'a dyn caliche_sandbox::SandboxHandleOps>,
        manager: Option<&'a dyn crate::agent::operations::OperationManager>,
        translators: Option<&'a crate::agent::operations::TranslatorRegistry>,
    ) -> ExecuteToolsParallelParams<'a> {
        ExecuteToolsParallelParams {
            valid_tool_calls: vec![echo_call()],
            tools: registry,
            hooks,
            cancellation_flag: None,
            user_context: None,
            metrics: None,
            tool_intelligence: intel,
            approval_resolver: None,
            amendment_persister: None,
            permission_mode: PermissionMode::Default,
            workspace_root: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 0,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.0,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            sandbox,
            operation_manager: manager,
            operation_translators: translators,
        }
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn remote_ops_with_sandbox_refuses_mismatched_isolating_handle() {
        use crate::agent::operations::{RemoteOperationManager, SandboxOperationTransport};
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        #[derive(Debug)]
        struct FakeSandbox;

        #[async_trait::async_trait]
        impl caliche_sandbox::SandboxHandleOps for FakeSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> anyhow::Result<ExecResult> {
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, _path: &str) -> anyhow::Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let registry = echo_registry();
        let hooks = crate::agent::hooks::HookRegistry::new();
        let transport_sandbox = std::sync::Arc::new(FakeSandbox);
        let configured = FakeSandbox;
        let manager = RemoteOperationManager::new(std::sync::Arc::new(
            SandboxOperationTransport::new(transport_sandbox),
        ));
        let translators = crate::agent::operations::TranslatorRegistry::builtin();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let outcome = execute_tools_parallel(pipeline_params(
            &registry,
            &hooks,
            &intel,
            Some(&configured),
            Some(&manager),
            Some(&translators),
        ))
        .await;
        assert_eq!(outcome.executed.len(), 0);
        assert_eq!(outcome.denied.len(), 1);
        assert!(
            outcome.denied[0].3.contains("configured sandbox handle"),
            "mismatch reason: {}",
            outcome.denied[0].3
        );
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn remote_ops_with_sandbox_accepts_matching_isolating_handle() {
        use crate::agent::operations::{RemoteOperationManager, SandboxOperationTransport};
        use caliche_sandbox::traits::ExecResult;
        use std::collections::HashMap;
        use std::time::Duration;

        #[derive(Debug)]
        struct FakeSandbox;

        #[async_trait::async_trait]
        impl caliche_sandbox::SandboxHandleOps for FakeSandbox {
            async fn exec(
                &self,
                _command: &[&str],
                _working_dir: Option<&str>,
                _env: Option<&HashMap<String, String>>,
                _timeout: Duration,
            ) -> anyhow::Result<ExecResult> {
                Ok(ExecResult {
                    stdout: String::new(),
                    stderr: String::new(),
                    stdout_truncated: false,
                    stderr_truncated: false,
                    exit_code: 0,
                    duration: Duration::from_millis(1),
                })
            }

            async fn read_file(&self, _path: &str) -> anyhow::Result<Vec<u8>> {
                Ok(Vec::new())
            }

            async fn write_file(&self, _path: &str, _content: &[u8]) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let registry = echo_registry();
        let hooks = crate::agent::hooks::HookRegistry::new();
        let sandbox: std::sync::Arc<dyn caliche_sandbox::SandboxHandleOps> =
            std::sync::Arc::new(FakeSandbox);
        let manager = RemoteOperationManager::new(std::sync::Arc::new(
            SandboxOperationTransport::new(std::sync::Arc::clone(&sandbox)),
        ));
        let translators = crate::agent::operations::TranslatorRegistry::builtin();
        let intel = std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::core::ToolIntelligenceState::new(),
        ));
        let outcome = execute_tools_parallel(pipeline_params(
            &registry,
            &hooks,
            &intel,
            Some(sandbox.as_ref()),
            Some(&manager),
            Some(&translators),
        ))
        .await;
        assert!(
            outcome.denied.is_empty(),
            "matching handle must pass the sandbox bind gate: {:?}",
            outcome.denied
        );
        assert_eq!(outcome.executed.len(), 1);
    }
}
