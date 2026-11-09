//! Sync, I/O-free tool translators.
//!
//! A translator validates a model tool call and emits versioned [`Operation`]
//! specs. It must not perform I/O or await.

use super::codec::{encode_fs_call, encode_shell, encode_tool_invoke};
use super::{OperationExecution, OperationStatus, ToolCallStatus, Translation};
use crate::agent::tools::tool::{ToolCall, ToolResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Validates a tool call and translates it into operations. No I/O.
pub trait ToolTranslator: Send + Sync + std::fmt::Debug {
    /// Canonical tool name this translator handles (`*` = fallback).
    fn tool_name(&self) -> &str;

    /// Validate `call` and emit zero or more operations plus a call status.
    fn translate(&self, call: &ToolCall) -> Translation;

    /// Format a recorded call status and prepared operation output for the model.
    fn format_result(
        &self,
        status: &ToolCallStatus,
        executions: &[OperationExecution],
    ) -> ToolResult;
}

/// Default formatter: join successful outputs; first error wins.
#[must_use]
pub fn format_executions(status: &ToolCallStatus, executions: &[OperationExecution]) -> ToolResult {
    if let ToolCallStatus::Invalid { message } = status {
        return ToolResult::error(message.clone());
    }
    if executions.is_empty() {
        return ToolResult::error("no operations were executed for this tool call");
    }
    let mut parts = Vec::new();
    let mut payload = None;
    for execution in executions {
        if execution.status == OperationStatus::Failed {
            return ToolResult::error(
                execution
                    .error
                    .clone()
                    .unwrap_or_else(|| "operation failed".to_string()),
            );
        }
        if !execution.output.is_empty() {
            parts.push(execution.output.clone());
        }
        if payload.is_none() {
            payload = execution.typed_payload.clone();
        }
    }
    let mut result = ToolResult::success(parts.join("\n"));
    result.typed_payload = payload;
    result
}

/// Compat translator: any tool call becomes a single `tool_invoke` operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct RegistryToolTranslator;

impl ToolTranslator for RegistryToolTranslator {
    fn tool_name(&self) -> &'static str {
        "*"
    }

    fn translate(&self, call: &ToolCall) -> Translation {
        let name = call.name.as_ref();
        if name.is_empty() {
            return Translation::invalid("tool name is empty");
        }
        Translation::submitted(vec![encode_tool_invoke(name, &call.arguments)])
    }

    fn format_result(
        &self,
        status: &ToolCallStatus,
        executions: &[OperationExecution],
    ) -> ToolResult {
        format_executions(status, executions)
    }
}

/// Validates `bash` / `shell` calls into a serializable [`OperationKind::Shell`] spec.
#[derive(Debug, Default, Clone, Copy)]
pub struct BashToolTranslator;

impl ToolTranslator for BashToolTranslator {
    fn tool_name(&self) -> &'static str {
        "bash"
    }

    fn translate(&self, call: &ToolCall) -> Translation {
        let Some(command) = call.arguments.get("command").and_then(|v| v.as_str()) else {
            return Translation::invalid("bash requires a string `command` argument");
        };
        if command.trim().is_empty() {
            return Translation::invalid("bash command is empty");
        }
        let timeout = call.arguments.get("timeout").cloned();
        let working_dir = call.arguments.get("working_dir").cloned();
        Translation::submitted(vec![encode_shell(command, timeout, working_dir)])
    }

    fn format_result(
        &self,
        status: &ToolCallStatus,
        executions: &[OperationExecution],
    ) -> ToolResult {
        format_executions(status, executions)
    }
}

/// Validates `fs` calls into versioned file operations.
#[derive(Debug, Default, Clone, Copy)]
pub struct FsToolTranslator;

impl ToolTranslator for FsToolTranslator {
    fn tool_name(&self) -> &'static str {
        "fs"
    }

    fn translate(&self, call: &ToolCall) -> Translation {
        match encode_fs_call(&call.arguments) {
            Ok(operation) => Translation::submitted(vec![operation]),
            Err(message) => Translation::invalid(message),
        }
    }

    fn format_result(
        &self,
        status: &ToolCallStatus,
        executions: &[OperationExecution],
    ) -> ToolResult {
        format_executions(status, executions)
    }
}

/// Named translators plus a fallback (`RegistryToolTranslator`).
#[derive(Debug, Clone)]
pub struct TranslatorRegistry {
    named: HashMap<String, Arc<dyn ToolTranslator>>,
    fallback: Arc<dyn ToolTranslator>,
}

impl TranslatorRegistry {
    /// Built-in bash / fs translators with a registry fallback.
    #[must_use]
    pub fn builtin() -> Self {
        let mut named: HashMap<String, Arc<dyn ToolTranslator>> = HashMap::new();
        let bash: Arc<dyn ToolTranslator> = Arc::new(BashToolTranslator);
        named.insert("bash".to_string(), Arc::clone(&bash));
        named.insert("shell".to_string(), bash);
        named.insert("fs".to_string(), Arc::new(FsToolTranslator));
        Self {
            named,
            fallback: Arc::new(RegistryToolTranslator),
        }
    }

    /// Resolve a translator for `tool_name`.
    #[must_use]
    pub fn resolve(&self, tool_name: &str) -> &dyn ToolTranslator {
        self.named
            .get(tool_name)
            .map_or_else(|| self.fallback.as_ref(), Arc::as_ref)
    }
}

impl Default for TranslatorRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::operations::OperationKind;
    use beef::lean::Cow;

    fn call(name: &'static str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            name: Cow::const_str(name),
            arguments,
        }
    }

    #[test]
    fn bash_translator_rejects_empty_command() {
        let translation = BashToolTranslator.translate(&call("bash", serde_json::json!({})));
        assert!(matches!(translation.status, ToolCallStatus::Invalid { .. }));
        assert!(translation.operations.is_empty());
    }

    #[test]
    fn bash_translator_emits_shell_spec() {
        let translation = BashToolTranslator.translate(&call(
            "bash",
            serde_json::json!({"command": "echo hi", "timeout": 5}),
        ));
        assert_eq!(translation.operations.len(), 1);
        assert_eq!(translation.operations[0].kind, OperationKind::Shell);
        assert_eq!(
            translation.operations[0].spec["command"].as_str(),
            Some("echo hi")
        );
        assert!(matches!(
            translation.status,
            ToolCallStatus::Submitted { .. }
        ));
    }

    #[test]
    fn fs_translator_rejects_non_string_write_content() {
        let translation = FsToolTranslator.translate(&call(
            "fs",
            serde_json::json!({"op": "write", "path": "a.txt", "content": 1}),
        ));
        assert!(matches!(translation.status, ToolCallStatus::Invalid { .. }));
    }

    #[test]
    fn fs_translator_requires_path_for_read() {
        let translation =
            FsToolTranslator.translate(&call("fs", serde_json::json!({"op": "read"})));
        assert!(matches!(translation.status, ToolCallStatus::Invalid { .. }));
    }

    #[test]
    fn registry_translator_wraps_any_tool() {
        let translation = RegistryToolTranslator.translate(&call(
            "calculator",
            serde_json::json!({"expression": "1+1"}),
        ));
        assert_eq!(translation.operations[0].kind, OperationKind::ToolInvoke);
        assert_eq!(
            translation.operations[0].spec["name"].as_str(),
            Some("calculator")
        );
    }

    #[test]
    fn builtin_registry_routes_bash_and_fallback() {
        let registry = TranslatorRegistry::builtin();
        assert_eq!(registry.resolve("bash").tool_name(), "bash");
        assert_eq!(registry.resolve("unknown").tool_name(), "*");
    }
}
