//! Translate → commit → execute → format, used when an [`OperationManager`] is set.

use super::{OperationManager, ToolCallStatus, TranslatorRegistry};
use crate::agent::tools::ToolPermit;
use crate::agent::tools::execution::{HookExecutionMiddleware, ToolExecutionContext};
use crate::agent::tools::tool::{ToolCall, ToolResult};
use std::sync::Arc;
use std::time::Instant;

/// Run one validated tool call through the operation layer.
///
/// The call runs only after `gate` admits it, so permission and audit
/// behavior match the engine path. MCP and other registry tools use
/// [`super::RegistryToolTranslator`].
pub async fn execute_tool_call_via_operations(
    call: &ToolCall,
    call_id: &str,
    translators: &TranslatorRegistry,
    manager: &dyn OperationManager,
    gate: &HookExecutionMiddleware,
) -> (u64, ToolResult) {
    let start = Instant::now();
    let ctx = ToolExecutionContext {
        tool_name: call.name.as_ref(),
        arguments: &call.arguments,
        call_id,
    };
    let mut result = match gate.admit(&ctx).await {
        Ok(permit) => execute_admitted_via_operations(call, translators, manager, &permit).await,
        Err(reason) => ToolResult::error(reason.into_owned()),
    };
    gate.after_execute(&ctx, &mut result).await;
    (
        u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        result,
    )
}

/// Translate, commit and execute one call the gate already admitted.
pub async fn execute_admitted_via_operations(
    call: &ToolCall,
    translators: &TranslatorRegistry,
    manager: &dyn OperationManager,
    permit: &ToolPermit<'_>,
) -> ToolResult {
    let translator = translators.resolve(call.name.as_ref());
    let translation = translator.translate(call);
    match &translation.status {
        ToolCallStatus::Invalid { message } => ToolResult::error(message.clone()),
        ToolCallStatus::Submitted { .. } => match manager.commit(&translation.operations).await {
            Ok(ids) => {
                let status = ToolCallStatus::Submitted {
                    operation_ids: ids.clone(),
                };
                match manager.execute(&ids, permit).await {
                    Ok(executions) => translator.format_result(&status, &executions),
                    Err(err) => ToolResult::error(format!("operation execute failed: {err}")),
                }
            }
            Err(err) => ToolResult::error(format!("operation commit failed: {err}")),
        },
    }
}

/// Shared translator table used when an agent opts into operation dispatch.
#[must_use]
pub fn shared_builtin_translators() -> Arc<TranslatorRegistry> {
    Arc::new(TranslatorRegistry::builtin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::HookRegistry;
    use beef::lean::Cow;

    #[tokio::test]
    async fn translate_commit_execute_format_uses_remote_execute() {
        use crate::agent::operations::{RecordingTransport, RemoteOperationManager};

        let transport = Arc::new(RecordingTransport {
            received: std::sync::Mutex::new(Vec::new()),
            canned: Some("5".to_string()),
        });
        let manager = RemoteOperationManager::new(transport);
        let translators = TranslatorRegistry::builtin();
        let call = ToolCall {
            name: Cow::const_str("add"),
            arguments: serde_json::json!({"a": 2, "b": 3}),
        };
        let gate = HookExecutionMiddleware::new(HookRegistry::new());
        let (_ms, via_ops) =
            execute_tool_call_via_operations(&call, "c1", &translators, &manager, &gate).await;
        assert!(via_ops.success);
        assert_eq!(via_ops.content, "5");
    }
}
