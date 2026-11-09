//! Privilege intersection for delegated work.
//!
//! A parent agent may narrow what a subagent can do, never widen it. Every delegation path
//! passes the requested tool list through [`intersect_tools`], so a `transfer_to` call that
//! asks for a tool the parent does not hold cannot escalate privilege.

use crate::acl::TaskInvocationPolicy;
use std::collections::BTreeSet;

/// Ceiling on what one registered agent may do when it is dispatched or invoked.
///
/// `max_tools` of `None` means the agent keeps whatever allowlist it was built with. A
/// `Some(list)` is a hard ceiling: a delegation can narrow it further but never widen it.
#[derive(Debug, Clone, Default)]
pub struct AgentPrivileges {
    /// Maximum tool set this agent may ever use.
    pub max_tools: Option<Vec<String>>,
    /// Which other agents this agent may invoke.
    pub task_policy: TaskInvocationPolicy,
    /// Default persona applied when a delegation does not override it.
    pub persona: Option<String>,
}

impl AgentPrivileges {
    /// Privileges with no tool ceiling and an allow-all task policy.
    pub fn unrestricted() -> Self {
        Self::default()
    }

    /// Set the maximum tool set.
    #[must_use]
    pub fn with_max_tools(mut self, tools: Vec<String>) -> Self {
        self.max_tools = Some(tools);
        self
    }

    /// Set the task invocation policy.
    #[must_use]
    pub fn with_task_policy(mut self, policy: TaskInvocationPolicy) -> Self {
        self.task_policy = policy;
        self
    }

    /// Set the default persona.
    #[must_use]
    pub fn with_persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }
}

/// Combine a parent and a target ceiling into the ceiling that applies to one delegation.
///
/// Both sides constrain the result: when each names a set, only tools in both survive.
pub fn tool_ceiling(parent: Option<&[String]>, target: Option<&[String]>) -> Option<Vec<String>> {
    match (parent, target) {
        (None, None) => None,
        (None, Some(list)) | (Some(list), None) => Some(intersect_tools(None, list)),
        (Some(parent), Some(target)) => Some(intersect_tools(Some(parent), target)),
    }
}

/// Intersect a requested tool list with the parent's maximum privileges.
///
/// `parent_allowed` of `None` means the parent has no explicit allowlist, so the request
/// passes through unchanged (deduplicated and ordered). Otherwise only tools present in both
/// lists survive. Order is deterministic to keep prompts and logs stable.
pub fn intersect_tools(parent_allowed: Option<&[String]>, requested: &[String]) -> Vec<String> {
    let requested_set: BTreeSet<&str> = requested.iter().map(String::as_str).collect();

    let Some(parent) = parent_allowed else {
        return requested_set.into_iter().map(str::to_string).collect();
    };

    let parent_set: BTreeSet<&str> = parent.iter().map(String::as_str).collect();
    requested_set
        .intersection(&parent_set)
        .map(|name| (*name).to_string())
        .collect()
}

/// Tools the request asked for that the parent does not hold.
///
/// Used for operator-facing warnings so a silently narrowed delegation is still visible.
pub fn rejected_tools(parent_allowed: Option<&[String]>, requested: &[String]) -> Vec<String> {
    let Some(parent) = parent_allowed else {
        return Vec::new();
    };
    let parent_set: BTreeSet<&str> = parent.iter().map(String::as_str).collect();
    let mut rejected: BTreeSet<&str> = BTreeSet::new();
    for name in requested {
        if !parent_set.contains(name.as_str()) {
            rejected.insert(name.as_str());
        }
    }
    rejected.into_iter().map(str::to_string).collect()
}

/// Resolve the effective tool list for a delegated task.
///
/// Returns `None` when the subagent should keep its own current allowlist: that happens only
/// when neither the request nor the parent constrains the set. When the parent has a maximum
/// set, that set is the ceiling even if the request named no tools.
pub fn effective_tools(
    parent_allowed: Option<&[String]>,
    requested: Option<&[String]>,
) -> Option<Vec<String>> {
    match (parent_allowed, requested) {
        (None, None) => None,
        (None, Some(requested)) => Some(intersect_tools(None, requested)),
        (Some(parent), None) => Some(intersect_tools(Some(parent), parent)),
        (Some(parent), Some(requested)) => Some(intersect_tools(Some(parent), requested)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    #[test]
    fn unrestricted_parent_passes_request_through() {
        let requested = names(&["fs", "bash"]);
        assert_eq!(intersect_tools(None, &requested), names(&["bash", "fs"]));
    }

    #[test]
    fn intersection_drops_tools_the_parent_lacks() {
        let parent = names(&["fs", "semantic_search"]);
        let requested = names(&["fs", "bash"]);
        assert_eq!(intersect_tools(Some(&parent), &requested), names(&["fs"]));
    }

    #[test]
    fn intersection_never_widens_privilege() {
        let parent = names(&["fs"]);
        let requested = names(&["bash", "code_execution"]);
        assert!(intersect_tools(Some(&parent), &requested).is_empty());
    }

    #[test]
    fn duplicates_collapse_deterministically() {
        let requested = names(&["fs", "fs", "bash"]);
        assert_eq!(intersect_tools(None, &requested), names(&["bash", "fs"]));
    }

    #[test]
    fn rejected_tools_report_escalation_attempts() {
        let parent = names(&["fs"]);
        let requested = names(&["fs", "bash", "bash"]);
        assert_eq!(rejected_tools(Some(&parent), &requested), names(&["bash"]));
        assert!(rejected_tools(None, &requested).is_empty());
    }

    #[test]
    fn effective_tools_keeps_current_set_when_nothing_constrains() {
        assert!(effective_tools(None, None).is_none());
    }

    #[test]
    fn effective_tools_applies_parent_ceiling_without_a_request() {
        let parent = names(&["fs", "semantic_search"]);
        assert_eq!(
            effective_tools(Some(&parent), None),
            Some(names(&["fs", "semantic_search"]))
        );
    }

    #[test]
    fn effective_tools_intersects_when_both_are_set() {
        let parent = names(&["fs", "semantic_search"]);
        let requested = names(&["semantic_search", "bash"]);
        assert_eq!(
            effective_tools(Some(&parent), Some(&requested)),
            Some(names(&["semantic_search"]))
        );
    }

    #[test]
    fn tool_ceiling_combines_parent_and_target_limits() {
        let parent = names(&["fs", "bash"]);
        let target = names(&["fs", "semantic_search"]);
        assert_eq!(
            tool_ceiling(Some(&parent), Some(&target)),
            Some(names(&["fs"]))
        );
        assert_eq!(tool_ceiling(None, Some(&target)), Some(target.clone()));
        assert_eq!(
            tool_ceiling(Some(&parent), None),
            Some(names(&["bash", "fs"]))
        );
        assert_eq!(tool_ceiling(None, None), None);
    }

    #[test]
    fn privileges_builders_set_fields() {
        let privileges = AgentPrivileges::unrestricted()
            .with_max_tools(names(&["fs"]))
            .with_persona("You review code.");

        assert_eq!(
            privileges.max_tools.as_deref(),
            Some(names(&["fs"]).as_slice())
        );
        assert_eq!(privileges.persona.as_deref(), Some("You review code."));
        assert!(privileges.task_policy.is_empty());
    }
}
