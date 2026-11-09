//! Control and lifecycle error types.

/// Control and lifecycle errors that can occur during agent operations
#[derive(Debug, thiserror::Error)]
pub enum ControlError {
    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Permission denied for tool execution
    #[error("Permission denied for tool '{tool}': {reason}")]
    PermissionDenied { tool: String, reason: String },

    /// Hook execution failed
    #[error("Hook execution failed: {0}")]
    HookFailed(String),

    /// Session fork failed
    #[error("Session fork failed: {0}")]
    SessionForkFailed(String),

    /// MCP server error
    #[error("MCP server error: {0}")]
    McpServerError(String),

    /// Invalid hook configuration
    #[error("Invalid hook configuration: {0}")]
    InvalidHookConfig(String),

    /// Tool execution blocked by hook
    #[error("Tool execution blocked by hook: {0}")]
    ToolBlockedByHook(String),
}

// Allow conversion from AgentError
impl From<crate::errors::AgentError> for ControlError {
    fn from(e: crate::errors::AgentError) -> Self {
        match e {
            crate::errors::AgentError::ToolNotFound(name) => ControlError::ToolNotFound(name),
            crate::errors::AgentError::ToolExecution { tool_name, reason } => {
                ControlError::PermissionDenied {
                    tool: tool_name,
                    reason,
                }
            }
            other => ControlError::HookFailed(format!("Agent error: {}", other)),
        }
    }
}
