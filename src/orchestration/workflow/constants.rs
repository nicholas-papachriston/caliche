pub(super) const MAX_STEPS: usize = 1000;

/// Execution context variable key for node result
pub(super) const VAR_RESULT: &str = "__result";
/// Execution context variable key for error
pub(super) const VAR_ERROR: &str = "__error";
/// Execution context variable key for last output
pub(super) const VAR_LAST_OUTPUT: &str = "__last_output";
/// Metadata key for forced next node ID
pub(super) const META_FORCED_NEXT_NODE_ID: &str = "forced_next_node_id";
