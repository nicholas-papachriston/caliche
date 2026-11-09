//! Translate → commit → execute → format, used when an [`OperationManager`] is set.

use super::{OperationManager, ToolCallStatus, TranslatorRegistry};
use crate::agent::tools::execution::{ToolExecutionContext, ToolExecutionMiddleware};
use crate::agent::tools::tool::{ToolCall, ToolResult};
use std::sync::Arc;
use std::time::Instant;

/// Run one validated tool call through the operation layer.
///
/// Hooks still run via `middleware` so permission / audit behavior matches the
/// engine path. MCP and other registry tools use [`super::RegistryToolTranslator`].
pub async fn execute_tool_call_via_operations(
    call: &ToolCall,
    call_id: &str,
    translators: &TranslatorRegistry,
    manager: &dyn OperationManager,
    middleware: &dyn ToolExecutionMiddleware,
) -> (u64, ToolResult) {
    let start = Instant::now();
    let ctx = ToolExecutionContext {
        tool_name: call.name.as_ref(),
        arguments: &call.arguments,
        call_id,
    };
    if let Err(reason) = middleware.before_execute(&ctx).await {
        return (
            u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
            ToolResult::error(reason.into_owned()),
        );
    }

    let translator = translators.resolve(call.name.as_ref());
    let translation = translator.translate(call);
    let mut result = match &translation.status {
        ToolCallStatus::Invalid { message } => ToolResult::error(message.clone()),
        ToolCallStatus::Submitted { .. } => match manager.commit(&translation.operations).await {
            Ok(ids) => {
                let status = ToolCallStatus::Submitted {
                    operation_ids: ids.clone(),
                };
                match manager.execute(&ids).await {
                    Ok(executions) => translator.format_result(&status, &executions),
                    Err(err) => ToolResult::error(format!("operation execute failed: {err}")),
                }
            }
            Err(err) => ToolResult::error(format!("operation commit failed: {err}")),
        },
    };

    middleware.after_execute(&ctx, &mut result).await;
    (
        u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        result,
    )
}

/// Shared translator table used when an agent opts into operation dispatch.
#[must_use]
pub fn shared_builtin_translators() -> Arc<TranslatorRegistry> {
    Arc::new(TranslatorRegistry::builtin())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use beef::lean::Cow;

    struct AllowAll;

    #[async_trait]
    impl ToolExecutionMiddleware for AllowAll {}

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
        let (_ms, via_ops) =
            execute_tool_call_via_operations(&call, "c1", &translators, &manager, &AllowAll).await;
        assert!(via_ops.success);
        assert_eq!(via_ops.content, "5");
    }
}
