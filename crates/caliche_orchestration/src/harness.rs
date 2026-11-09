//! Dispatch contract for local agents and external harnesses.
//!
//! Coordinator code talks only to [`HarnessRuntime`]. The Caliche engine
//! implements this for `Agent`; Pi and Hermes use [`crate::ExternalCliRuntime`].

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// One dispatch from the orchestrator to a harness.
#[derive(Debug, Clone, Default)]
pub struct HarnessRequest {
    /// User/task message.
    pub message: String,
    /// Optional persona (system prompt) for this call only.
    pub persona: Option<String>,
    /// Optional tool-name ceiling for this call. Local agents enforce it;
    /// external CLIs receive it as advisory prompt text unless an adapter maps it.
    pub tools: Option<Vec<String>>,
    /// Optional instructions prepended to the task.
    pub instructions: Option<String>,
    /// Optional bound on this dispatch. Runtimes that spawn a process must
    /// honour it and kill the child; dropping the future is not enough.
    pub timeout: Option<Duration>,
}

impl HarnessRequest {
    /// Build a request with only a message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ..Self::default()
        }
    }
}

/// Token counts reported by a harness. CLI adapters typically leave these unset.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessUsage {
    /// Prompt tokens when the harness reports them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// Completion tokens when the harness reports them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    /// Total tokens when the harness reports them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

/// Result of one harness dispatch.
#[derive(Debug, Clone, Default)]
pub struct HarnessResponse {
    /// Model/harness output text (untrusted until the coordinator quarantines it).
    pub text: String,
    /// Optional usage. Unknown for most CLI harnesses.
    pub usage: HarnessUsage,
}

/// Something the coordinator can dispatch a task to.
///
/// Implementations must be safe to share across tasks. They must not run the
/// prompt through a shell. Process-based adapters must set `kill_on_drop` and
/// honour [`HarnessRequest::timeout`].
#[async_trait]
pub trait HarnessRuntime: Send + Sync {
    /// Stable label for logs (`caliche`, `pi`, `hermes`, `external_cli`).
    fn kind(&self) -> &'static str {
        "custom"
    }

    /// Run one task and return the harness output.
    async fn execute(&self, request: HarnessRequest) -> Result<HarnessResponse>;
}

/// Runtime that returns a fixed string. Hosts use this for dry-run graphs and tests.
#[derive(Debug, Clone)]
pub struct StaticRuntime {
    /// Text returned by every [`HarnessRuntime::execute`] call.
    pub text: String,
    /// Label for logs.
    pub kind: &'static str,
}

impl StaticRuntime {
    /// Build a static runtime that always returns `text`.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind: "static",
        }
    }
}

#[async_trait]
impl HarnessRuntime for StaticRuntime {
    fn kind(&self) -> &'static str {
        self.kind
    }

    async fn execute(&self, _request: HarnessRequest) -> Result<HarnessResponse> {
        Ok(HarnessResponse {
            text: self.text.clone(),
            usage: HarnessUsage::default(),
        })
    }
}
