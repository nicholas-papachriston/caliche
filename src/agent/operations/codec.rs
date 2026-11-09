//! Single Operation ↔ tool-call codec.
//!
//! Translators encode model calls into versioned specs; the sandbox transport
//! (and any registry replay) decodes those specs. Keep both directions here so
//! kind ↔ tool mapping cannot drift.

use super::{Operation, OperationKind};
use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Decoded, kind-specific payload after validating a committed spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedOperation {
    /// Registry tool invoke.
    ToolInvoke { name: String, arguments: Value },
    /// `bash` / shell command.
    Shell {
        command: String,
        timeout_secs: u64,
        working_dir: Option<String>,
    },
    /// `fs` `op=read`.
    FileRead { path: String },
    /// `fs` `op=write` with required string content.
    FileWrite { path: String, content: String },
    /// `fs` `op=edit`.
    FileEdit { path: String, spec: Value },
    /// Other `fs` ops (`list`, `grep`, `glob`, `delete`).
    FileQuery { spec: Value },
}

/// Kinds the sandbox worker is allowed to run (no MCP / provider tools).
#[must_use]
pub const fn sandbox_allows(kind: OperationKind) -> bool {
    matches!(
        kind,
        OperationKind::Shell | OperationKind::FileRead | OperationKind::FileWrite
    )
}

/// Encode a registry tool call.
#[must_use]
pub fn encode_tool_invoke(name: &str, arguments: &Value) -> Operation {
    Operation::new(
        OperationKind::ToolInvoke,
        name.to_string(),
        serde_json::json!({
            "name": name,
            "arguments": arguments,
        }),
    )
}

/// Encode a bash/shell call. `command` must already be non-empty.
#[must_use]
pub fn encode_shell(
    command: &str,
    timeout: Option<Value>,
    working_dir: Option<Value>,
) -> Operation {
    let mut spec = serde_json::Map::new();
    spec.insert("command".to_string(), Value::String(command.to_string()));
    if let Some(timeout) = timeout {
        spec.insert("timeout".to_string(), timeout);
    }
    if let Some(working_dir) = working_dir {
        spec.insert("working_dir".to_string(), working_dir);
    }
    Operation::new(OperationKind::Shell, "bash", Value::Object(spec))
}

/// Encode an `fs` tool call, validating `op` / `path` / write `content`.
pub fn encode_fs_call(arguments: &Value) -> Result<Operation, String> {
    let Some(op) = arguments.get("op").and_then(Value::as_str) else {
        return Err("fs requires an `op` argument".to_string());
    };
    let kind = match op {
        "read" => OperationKind::FileRead,
        "write" => OperationKind::FileWrite,
        "edit" => OperationKind::FileEdit,
        "list" | "grep" | "glob" | "delete" => OperationKind::FileQuery,
        other => return Err(format!("unsupported fs op '{other}'")),
    };
    if matches!(
        kind,
        OperationKind::FileRead | OperationKind::FileWrite | OperationKind::FileEdit
    ) && arguments
        .get("path")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(format!("fs op '{op}' requires a non-empty `path`"));
    }
    if kind == OperationKind::FileWrite {
        match arguments.get("content") {
            Some(Value::String(_)) => {}
            Some(_) => return Err("fs write content must be a string".to_string()),
            None => return Err("fs write requires string `content`".to_string()),
        }
    }
    Ok(Operation::new(kind, "fs", arguments.clone()))
}

/// Decode a committed spec into a typed payload.
pub fn decode_operation(operation: &Operation) -> Result<DecodedOperation> {
    match operation.kind {
        OperationKind::ToolInvoke => {
            let name = operation
                .spec
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(operation.tool_name.as_str())
                .to_string();
            let arguments = operation
                .spec
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Null);
            Ok(DecodedOperation::ToolInvoke { name, arguments })
        }
        OperationKind::Shell => {
            let command = operation
                .spec
                .get("command")
                .and_then(Value::as_str)
                .context("shell operation missing command")?
                .to_string();
            let timeout_secs = operation
                .spec
                .get("timeout")
                .and_then(Value::as_u64)
                .unwrap_or(30);
            let working_dir = operation
                .spec
                .get("working_dir")
                .and_then(Value::as_str)
                .map(str::to_string);
            Ok(DecodedOperation::Shell {
                command,
                timeout_secs,
                working_dir,
            })
        }
        OperationKind::FileRead => Ok(DecodedOperation::FileRead {
            path: require_path(operation, "file_read")?,
        }),
        OperationKind::FileWrite => {
            let path = require_path(operation, "file_write")?;
            let content = match operation.spec.get("content") {
                Some(Value::String(text)) => text.clone(),
                Some(_) => bail!("file_write content must be a string"),
                None => bail!("file_write missing content"),
            };
            Ok(DecodedOperation::FileWrite { path, content })
        }
        OperationKind::FileEdit => Ok(DecodedOperation::FileEdit {
            path: require_path(operation, "file_edit")?,
            spec: operation.spec.clone(),
        }),
        OperationKind::FileQuery => Ok(DecodedOperation::FileQuery {
            spec: operation.spec.clone(),
        }),
    }
}

fn require_path(operation: &Operation, label: &str) -> Result<String> {
    operation
        .spec
        .get("path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .map(str::to_string)
        .with_context(|| format!("{label} missing path"))
}

/// Replay a decoded spec as a registry `(tool_name, arguments)` pair.
#[cfg(test)]
#[must_use]
fn decoded_to_registry_call(decoded: &DecodedOperation) -> (String, Value) {
    match decoded {
        DecodedOperation::ToolInvoke { name, arguments } => (name.clone(), arguments.clone()),
        DecodedOperation::Shell {
            command,
            timeout_secs,
            working_dir,
        } => {
            let mut args = serde_json::Map::new();
            args.insert("command".to_string(), Value::String(command.clone()));
            args.insert("timeout".to_string(), Value::from(*timeout_secs));
            if let Some(working_dir) = working_dir {
                args.insert(
                    "working_dir".to_string(),
                    Value::String(working_dir.clone()),
                );
            }
            ("bash".to_string(), Value::Object(args))
        }
        DecodedOperation::FileRead { path } => (
            "fs".to_string(),
            serde_json::json!({"op": "read", "path": path}),
        ),
        DecodedOperation::FileWrite { path, content } => (
            "fs".to_string(),
            serde_json::json!({"op": "write", "path": path, "content": content}),
        ),
        DecodedOperation::FileEdit { spec, .. } | DecodedOperation::FileQuery { spec } => {
            ("fs".to_string(), spec.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_write_fails_closed_on_non_string_content() {
        let op = Operation::new(
            OperationKind::FileWrite,
            "fs",
            serde_json::json!({"path": "a.txt", "content": 123}),
        );
        let err = decode_operation(&op).expect_err("non-string content");
        assert!(
            err.to_string().contains("string"),
            "fail-closed reason: {err}"
        );
    }

    #[test]
    fn file_write_fails_closed_when_content_missing() {
        let op = Operation::new(
            OperationKind::FileWrite,
            "fs",
            serde_json::json!({"path": "a.txt"}),
        );
        let err = decode_operation(&op).expect_err("missing content");
        assert!(
            err.to_string().contains("content"),
            "fail-closed reason: {err}"
        );
    }

    #[test]
    fn encode_fs_write_rejects_non_string_content() {
        let err = encode_fs_call(&serde_json::json!({
            "op": "write",
            "path": "a.txt",
            "content": {"nested": true}
        }))
        .expect_err("non-string");
        assert!(err.contains("string"));
    }

    #[test]
    fn sandbox_allowlist_is_shell_and_plain_files() {
        assert!(sandbox_allows(OperationKind::Shell));
        assert!(sandbox_allows(OperationKind::FileRead));
        assert!(sandbox_allows(OperationKind::FileWrite));
        assert!(!sandbox_allows(OperationKind::ToolInvoke));
        assert!(!sandbox_allows(OperationKind::FileEdit));
        assert!(!sandbox_allows(OperationKind::FileQuery));
    }

    #[test]
    fn decode_encode_shell_roundtrip() {
        let op = encode_shell("echo hi", Some(Value::from(5)), None);
        let decoded = decode_operation(&op).expect("decode");
        assert_eq!(
            decoded,
            DecodedOperation::Shell {
                command: "echo hi".to_string(),
                timeout_secs: 5,
                working_dir: None,
            }
        );
        let (tool, args) = decoded_to_registry_call(&decoded);
        assert_eq!(tool, "bash");
        assert_eq!(args["command"], "echo hi");
    }
}
