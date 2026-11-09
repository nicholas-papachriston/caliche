//! Agent-specific error types for type-safe error handling.

/// Errors that can occur during agent operations.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// LLM provider error
    #[error("LLM provider error: {0}")]
    LlmProvider(String),

    /// LLM provider is unavailable or unreachable (network failure, quota exceeded, etc.)
    #[error("LLM provider unavailable: {0}")]
    ProviderUnavailable(String),

    /// Failed to get LLM completion
    #[error("Failed to get LLM completion: {0}")]
    CompletionFailed(String),

    /// Invalid tool call
    #[error("Invalid tool call '{tool_name}': {reason}")]
    InvalidToolCall { tool_name: String, reason: String },

    /// Tool execution error
    #[error("Tool '{tool_name}' execution failed: {reason}")]
    ToolExecution { tool_name: String, reason: String },

    /// Tool not found
    #[error("Tool '{0}' not found")]
    ToolNotFound(String),

    /// Tool execution timed out
    #[error("Tool '{tool_name}' timed out after {timeout_ms}ms")]
    ToolTimeout { tool_name: String, timeout_ms: u64 },

    /// Message building error
    #[error("Failed to build messages: {0}")]
    MessageBuilding(String),

    /// Token limit exceeded (soft limit for context budget management)
    #[error("Token limit exceeded: {current} > {limit}")]
    TokenLimitExceeded { current: usize, limit: usize },

    /// Context window exceeded (provider hard limit)
    #[error(
        "Context window exceeded: request needs {current} tokens but the effective context window is {limit}. Reduce the input/history/tool schemas, lower reserved output tokens, or configure a larger context window."
    )]
    ContextWindowExceeded { current: usize, limit: usize },

    /// Maximum iterations reached
    #[error("Maximum tool call iterations ({0}) reached")]
    MaxIterationsReached(usize),

    /// Handoff error
    #[error("Handoff failed: {0}")]
    HandoffFailed(String),

    /// Memory error
    #[error("Memory operation failed: {0}")]
    Memory(String),

    /// Workflow execution failed
    #[error("Workflow failed: {0}")]
    WorkflowFailed(String),

    /// Chain execution failed
    #[error("Chain failed: {0}")]
    ChainFailed(String),

    /// Other agent error — prefer adding a specific variant over using this
    #[error("Agent error: {0}")]
    Other(String),
}

impl From<std::fmt::Error> for AgentError {
    fn from(e: std::fmt::Error) -> Self {
        AgentError::Other(format!("Format error: {}", e))
    }
}

impl From<crate::errors::ToolError> for AgentError {
    fn from(e: crate::errors::ToolError) -> Self {
        match e {
            crate::errors::ToolError::NotFound(name) => AgentError::ToolNotFound(name),
            crate::errors::ToolError::ExecutionFailed { tool_name, reason } => {
                AgentError::ToolExecution { tool_name, reason }
            }
            crate::errors::ToolError::InvalidParameters { tool_name, reason } => {
                AgentError::InvalidToolCall { tool_name, reason }
            }
            other => AgentError::Other(format!("Tool error: {}", other)),
        }
    }
}

impl From<crate::errors::WorkflowError> for AgentError {
    fn from(e: crate::errors::WorkflowError) -> Self {
        AgentError::WorkflowFailed(e.to_string())
    }
}

#[cfg(feature = "chains")]
impl From<crate::chains::ChainError> for AgentError {
    fn from(e: crate::chains::ChainError) -> Self {
        AgentError::ChainFailed(e.to_string())
    }
}

// Note: anyhow::Error already implements From<E> for any E: std::error::Error + Send + Sync + 'static.
// Since AgentError derives thiserror::Error (which implements std::error::Error), the conversion
// `AgentError -> anyhow::Error` via `?` works automatically without an explicit impl here.
