pub mod iteration_control;
pub mod phase_planning;
pub mod token_tracking;
pub mod tool_pipeline;

pub use iteration_control::{
    IterationLimiter, SOFT_WRAP_UP_MESSAGE, iteration_limit_reached_suffix, should_inject_soft_wrap,
};
pub use phase_planning::{
    advance_phase_if_needed, current_phase_allowed_tools, ensure_phase_plan_for_query,
    get_current_phase_display,
};
pub use token_tracking::TokenUsageAccumulator;
pub use tool_pipeline::{
    IndexedToolCall, append_invalid_tool_messages_native, append_owned_tool_results_to_history,
    append_tool_results_to_history, execute_tools_parallel, is_length_finish_reason,
    length_truncated_tool_error, refuse_length_truncated_calls, validate_and_filter_tool_calls,
};
