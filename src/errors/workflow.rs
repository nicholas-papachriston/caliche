//! Workflow-specific error types for type-safe error handling.

/// Errors that can occur during workflow operations.
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    /// Workflow not found
    #[error("Workflow '{0}' not found")]
    NotFound(String),

    /// Execution not found
    #[error("Execution '{0}' not found")]
    ExecutionNotFound(String),

    /// Execution still running
    #[error("Execution '{0}' is still running")]
    ExecutionStillRunning(String),

    /// Execution failed
    #[error("Execution '{execution_id}' failed: {reason}")]
    ExecutionFailed {
        execution_id: String,
        reason: String,
    },

    /// Node not found
    #[error("Node '{0}' not found")]
    NodeNotFound(String),

    /// Invalid workflow structure
    #[error("Invalid workflow structure: {0}")]
    InvalidStructure(String),

    /// Workflow contains cycles
    #[error("Workflow contains cycles. Use entry_point() for state machine mode.")]
    ContainsCycles,

    /// Entry point not found
    #[error("Entry point '{0}' does not exist in workflow nodes")]
    EntryPointNotFound(String),

    /// Edge references non-existent node
    #[error("Edge references non-existent {node_type}: {node_id}. {details}")]
    InvalidEdge {
        node_type: String,
        node_id: String,
        details: String,
    },

    /// Maximum steps exceeded
    #[error("State machine exceeded maximum steps ({0}). Possible infinite loop.")]
    MaxStepsExceeded(usize),

    /// Interrupt not found
    #[error("Interrupt '{0}' not found")]
    InterruptNotFound(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    /// No result available
    #[error("No result available for execution '{0}'")]
    NoResultAvailable(String),

    /// No suitable agents available
    #[error("No suitable agents available for workflow execution")]
    NoAgentsAvailable,

    /// Other workflow error
    #[error("Workflow error: {0}")]
    Other(String),
}
