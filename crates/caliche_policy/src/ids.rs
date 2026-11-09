//! Action identity and the tool-plus-arguments binding hash.

use crate::error::PolicyError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Kind of action a binding or approval refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// A tool call.
    Tool,
    /// Outbound network.
    Egress,
    /// MCP server connect.
    McpConnect,
    /// Provider/model selection.
    ModelUse,
    /// External harness spawn.
    HarnessSpawn,
    /// Subagent task invoke.
    TaskInvoke,
    /// Answer to a pending approval.
    ApprovalAnswer,
    /// Write of policy or config.
    ConfigWrite,
    /// Facet could not name the action.
    Unresolved,
}

impl ActionKind {
    /// Stable name used inside [`ActionId`].
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tool => "tool",
            Self::Egress => "egress",
            Self::McpConnect => "mcp_connect",
            Self::ModelUse => "model_use",
            Self::HarnessSpawn => "harness_spawn",
            Self::TaskInvoke => "task_invoke",
            Self::ApprovalAnswer => "approval_answer",
            Self::ConfigWrite => "config_write",
            Self::Unresolved => "unresolved",
        }
    }
}

/// Identity of an action, separate from the argument hash.
///
/// `bash` with `git status` and `bash` with `curl` share a tool name and differ
/// in [`ActionBinding`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActionId {
    kind: ActionKind,
    name: String,
}

impl ActionId {
    /// Build an id from a kind and a name (tool name, host, or pending id).
    #[must_use]
    pub fn new(kind: ActionKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
        }
    }

    /// Kind of this action.
    #[must_use]
    pub const fn kind(&self) -> ActionKind {
        self.kind
    }

    /// Tool name, host, or other label.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// `apr_` prefixed id for one pending approval.
///
/// Id for one pending approval. This crate defines the id. It does not mint
/// approvals across CLI, UI, HTTP, and ACP.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PendingId(String);

impl PendingId {
    /// Mint a new `apr_` id from a UUID v7.
    #[must_use]
    pub fn new() -> Self {
        Self(format!("apr_{}", uuid::Uuid::now_v7().simple()))
    }

    /// Parse an existing id.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError::InvalidPendingId`] when the value is empty or
    /// does not start with `apr_`.
    pub fn parse(value: impl AsRef<str>) -> Result<Self, PolicyError> {
        let value = value.as_ref();
        if value.starts_with("apr_") && value.len() > 4 {
            Ok(Self(value.to_owned()))
        } else {
            Err(PolicyError::InvalidPendingId(value.to_owned()))
        }
    }

    /// Borrow the id text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for PendingId {
    fn default() -> Self {
        Self::new()
    }
}

/// `blake3` of canonical `{args, cwd, dialect, policy_version, tool}`.
///
/// Computed lazily (ask, a matched content rule, or audit). The executor
/// recomputes it immediately before spawn or write. A mismatch cancels the
/// call; the stale approval is not reused.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActionBinding(pub [u8; 32]);

impl core::fmt::Debug for ActionBinding {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("ActionBinding(")?;
        for byte in &self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        formatter.write_str(")")
    }
}

impl ActionBinding {
    /// Hash the tool name, arguments, cwd, dialect, and policy version.
    #[must_use]
    pub fn compute(
        tool: &str,
        args: &Value,
        cwd: &str,
        dialect: Option<&str>,
        policy_version: u64,
    ) -> Self {
        let canonical = canonical_binding(tool, args, cwd, dialect, policy_version);
        let hash = blake3::hash(canonical.as_bytes());
        Self(*hash.as_bytes())
    }

    /// Compare a previously approved binding with a fresh hash of the args.
    #[must_use]
    pub fn recheck(self, actual: Self) -> BindingRecheck {
        if self == actual {
            BindingRecheck::Fresh
        } else {
            BindingRecheck::Stale {
                expected: self,
                actual,
            }
        }
    }
}

/// Result of checking an approval against the arguments about to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingRecheck {
    /// The arguments and policy version still match the approval.
    Fresh,
    /// Arguments or policy version changed. Cancel and call `decide` again.
    Stale {
        /// Binding stored on the approval.
        expected: ActionBinding,
        /// Binding of the arguments about to run.
        actual: ActionBinding,
    },
}

fn canonical_binding(
    tool: &str,
    args: &Value,
    cwd: &str,
    dialect: Option<&str>,
    policy_version: u64,
) -> String {
    let mut out = String::from('{');
    out.push_str("\"args\":");
    push_canonical(&mut out, args);
    out.push_str(",\"cwd\":");
    push_json_string(&mut out, cwd);
    out.push_str(",\"dialect\":");
    match dialect {
        Some(dialect) => push_json_string(&mut out, dialect),
        None => out.push_str("null"),
    }
    out.push_str(",\"policy_version\":");
    out.push_str(&policy_version.to_string());
    out.push_str(",\"tool\":");
    push_json_string(&mut out, tool);
    out.push('}');
    out
}

fn push_canonical(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => push_json_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                push_canonical(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                push_json_string(out, key);
                out.push(':');
                if let Some(child) = map.get(*key) {
                    push_canonical(out, child);
                } else {
                    out.push_str("null");
                }
            }
            out.push('}');
        }
    }
}

fn push_json_string(out: &mut String, value: &str) {
    match serde_json::to_string(value) {
        Ok(encoded) => out.push_str(&encoded),
        Err(_) => out.push_str("\"\""),
    }
}
