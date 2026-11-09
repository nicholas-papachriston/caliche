//! Tool-specific error types for type-safe error handling.

/// Errors that can occur during tool operations.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Tool not found
    #[error("Tool '{0}' not found")]
    NotFound(String),

    /// Invalid tool parameters
    #[error("Invalid parameters for tool '{tool_name}': {reason}")]
    InvalidParameters { tool_name: String, reason: String },

    /// Tool execution failed
    #[error("Tool '{tool_name}' execution failed: {reason}")]
    ExecutionFailed { tool_name: String, reason: String },

    /// Tool validation error
    #[error("Tool validation failed: {0}")]
    Validation(String),

    /// Tool timeout
    #[error("Tool '{0}' execution timed out")]
    Timeout(String),

    /// Tool permission denied
    #[error("Permission denied for tool '{tool_name}': {reason}")]
    PermissionDenied { tool_name: String, reason: String },

    /// Other tool error
    #[error("Tool error: {0}")]
    Other(String),
}
