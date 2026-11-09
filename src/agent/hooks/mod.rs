//! Agent lifecycle hooks for intercepting and controlling agent behavior.
//!
//! This module provides hooks that can be registered to intercept agent operations
//! at specific points in the execution lifecycle, similar to Claude Agent SDK hooks.

pub mod action_narration_guard;
pub mod approval;
pub mod environment_onboarding;
pub mod execpolicy;
pub mod inspection;
pub mod lifecycle;
pub mod loop_detection;
pub mod matcher;
pub mod path_restriction;
pub mod permission;
pub mod permission_contracts;
pub(crate) mod policy_shadow;
pub mod pre_completion_checklist;
pub mod reasoning_scheduler;
pub mod retry_advisor;
pub mod steering_followup;
pub mod validation;

pub use action_narration_guard::{ActionNarrationGuardConfig, ActionNarrationGuardMiddleware};
pub use approval::{AmendmentPersister, ApprovalOutcome, ApprovalResolver};
pub use environment_onboarding::{EnvironmentOnboardingConfig, EnvironmentOnboardingMiddleware};
pub use execpolicy::{BashDecision, BashRule, ExecPolicy, ExecPolicyHook, PatternPos};
pub use inspection::{
    InspectionAction, InspectionHookAdapter, InspectionResult, PathSecurityInspector,
    PermissionModeInspector, ToolInspectionManager, ToolInspector, ToolInspectorRequest,
    default_inspection_manager,
};
pub use lifecycle::{
    CompletionContext, CompletionHookResult, HookContext, HookResult, LifecycleHook, PreLlmContext,
    SessionStartContext, SessionStartHookResult, ToolCallSummary, TurnContext,
};
pub use loop_detection::{LoopDetectionConfig, LoopDetectionMiddleware};
pub use matcher::HookMatcher;
pub use path_restriction::{
    AllowedPaths, PathRestrictionConfig, PathRestrictionHook, extract_paths_from_text,
};
pub use permission::{PermissionDecision, PermissionMode};
pub use permission_contracts::{
    approval_outcome_from_dto, approval_outcome_from_prompt_choice, approval_outcome_to_dto,
    bash_decision_to_permission_decision, parse_permission_mode_str, permission_decision_from_dto,
    permission_decision_from_json_value, permission_decision_to_dto, permission_mode_config_to_dto,
    permission_mode_config_to_runtime, permission_mode_dto_to_config,
};
pub use pre_completion_checklist::{
    PreCompletionChecklistConfig, PreCompletionChecklistMiddleware,
};
pub use reasoning_scheduler::{
    ReasoningEffortSlot, ReasoningPreset, ReasoningScheduleConfig, ReasoningScheduler,
};
pub use retry_advisor::RetryAdvisorHook;
pub use steering_followup::{
    CONTINUE_TASK_PROMPT, STREAM_TIMEOUT_CONTINUATION_PREFIX, SteeringFollowUpProvider,
    is_steering_only_query,
};
pub use validation::ValidationHook;

use std::collections::HashMap;
use std::sync::Arc;

/// Registry for managing lifecycle hooks with matchers
#[derive(Clone)]
pub struct HookRegistry {
    hooks: HashMap<String, Vec<Arc<dyn LifecycleHook>>>,
    /// Hooks that participate in `on_pre_completion` (session/turn-level)
    completion_hooks: Vec<Arc<dyn LifecycleHook>>,
}

impl HookRegistry {
    /// Create a new empty hook registry
    #[must_use]
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            completion_hooks: Vec::new(),
        }
    }

    /// Register a hook with a matcher
    pub fn register(&mut self, matcher: HookMatcher, hook: Arc<dyn LifecycleHook>) {
        match matcher {
            HookMatcher::Tool { name } => {
                self.hooks.entry(name.into_owned()).or_default().push(hook);
            }
            HookMatcher::All => {
                // Store "all" hooks separately or apply to all tools
                self.hooks.entry("*".to_string()).or_default().push(hook);
            }
        }
    }

    /// Register a hook for `on_pre_completion` (called when LLM signals intent to stop).
    pub fn register_completion_hook(&mut self, hook: Arc<dyn LifecycleHook>) {
        self.completion_hooks.push(hook);
    }

    /// Get hooks matching a tool name
    pub fn get_matching(&self, tool_name: &str) -> Vec<Arc<dyn LifecycleHook>> {
        let mut result = Vec::new();

        // Add tool-specific hooks
        if let Some(hooks) = self.hooks.get(tool_name) {
            result.extend(hooks.iter().map(Arc::clone));
        }

        // Add "all" hooks
        if let Some(hooks) = self.hooks.get("*") {
            result.extend(hooks.iter().map(Arc::clone));
        }

        result
    }

    /// Get all completion hooks (for `on_pre_completion`)
    #[must_use]
    pub fn completion_hooks(&self) -> &[Arc<dyn LifecycleHook>] {
        &self.completion_hooks
    }

    /// Check if any hooks are registered
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty() && self.completion_hooks.is_empty()
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}
