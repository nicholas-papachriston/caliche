//! Versioned, serializable operation specs produced by [`super::ToolTranslator`].

use serde::{Deserialize, Serialize};

/// Schema version for [`Operation`] envelopes.
pub const OPERATION_SPEC_VERSION: u32 = 1;

/// Stable identifier assigned when an operation is committed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationId(pub String);

impl OperationId {
    /// Wrap an owned id string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for OperationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Kind of durable work a translator emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationKind {
    /// Invoke an existing registry tool by name (compat path).
    ToolInvoke,
    /// Local or sandboxed shell command (`bash`).
    Shell,
    /// Filesystem read (`fs` `op=read`).
    FileRead,
    /// Filesystem write (`fs` `op=write`).
    FileWrite,
    /// Filesystem edit (`fs` `op=edit`).
    FileEdit,
    /// Other `fs` ops (`list`, `grep`, `glob`, `delete`).
    FileQuery,
}

/// Versioned serializable description of work for asynchronous execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Operation {
    /// Envelope version (`OPERATION_SPEC_VERSION`).
    pub version: u32,
    /// Assigned on commit; `None` while the translator is still building the spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<OperationId>,
    /// Work classification.
    pub kind: OperationKind,
    /// Canonical tool name the spec came from (`bash`, `fs`, …).
    pub tool_name: String,
    /// Model tool-call id, when the translator saw one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    /// Kind-specific JSON payload (command, path, registry args, …).
    pub spec: serde_json::Value,
}

impl Operation {
    /// Build a versioned spec without an id (assigned at commit).
    #[must_use]
    pub fn new(kind: OperationKind, tool_name: impl Into<String>, spec: serde_json::Value) -> Self {
        Self {
            version: OPERATION_SPEC_VERSION,
            id: None,
            kind,
            tool_name: tool_name.into(),
            call_id: None,
            spec,
        }
    }

    /// Attach the originating tool-call id.
    #[must_use]
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        self.call_id = Some(call_id.into());
        self
    }
}

/// Lifecycle of a committed operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    /// Committed and eligible to run.
    Ready,
    /// Blocked on external completion (remote sandbox, approval, …).
    Awaiting,
    /// Cancel requested but not yet acknowledged.
    Canceling,
    /// Finished successfully.
    Completed,
    /// Finished with an error.
    Failed,
    /// Cancelled before or during execution.
    Canceled,
}

/// Outcome of executing one committed operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationExecution {
    /// Operation id.
    pub id: OperationId,
    /// Terminal or in-flight status.
    pub status: OperationStatus,
    /// Model-facing output (stdout, file contents, registry result).
    pub output: String,
    /// Error text when `status == Failed`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Optional structured payload from the underlying tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_payload: Option<serde_json::Value>,
}

impl OperationExecution {
    /// Successful execution.
    #[must_use]
    pub fn completed(id: OperationId, output: impl Into<String>) -> Self {
        Self {
            id,
            status: OperationStatus::Completed,
            output: output.into(),
            error: None,
            typed_payload: None,
        }
    }

    /// Failed execution.
    #[must_use]
    pub fn failed(id: OperationId, error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            id,
            status: OperationStatus::Failed,
            output: String::new(),
            error: Some(error),
            typed_payload: None,
        }
    }
}

/// Translation outcome: validation error or submitted operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallStatus {
    /// Specs were accepted; ids are filled after commit.
    Submitted {
        /// Committed operation ids (empty until the manager assigns them).
        operation_ids: Vec<OperationId>,
    },
    /// Translator rejected the call without emitting work.
    Invalid {
        /// Validation error shown to the model.
        message: String,
    },
}

/// Sync translator output: status plus zero or more operations.
#[derive(Debug, Clone)]
pub struct Translation {
    /// Validation / submission status.
    pub status: ToolCallStatus,
    /// Operations to commit (empty when `status` is `Invalid`).
    pub operations: Vec<Operation>,
}

impl Translation {
    /// Accepted translation with one or more operations.
    #[must_use]
    pub fn submitted(operations: Vec<Operation>) -> Self {
        Self {
            status: ToolCallStatus::Submitted {
                operation_ids: Vec::new(),
            },
            operations,
        }
    }

    /// Validation failure; no operations are committed.
    #[must_use]
    pub fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: ToolCallStatus::Invalid {
                message: message.into(),
            },
            operations: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_json_is_versioned() {
        let op = Operation::new(
            OperationKind::Shell,
            "bash",
            serde_json::json!({"command": "echo hi"}),
        );
        let json = serde_json::to_string(&op).expect("serialize");
        let loaded: Operation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(loaded.version, OPERATION_SPEC_VERSION);
        assert_eq!(loaded.kind, OperationKind::Shell);
        assert!(loaded.id.is_none());
    }
}
