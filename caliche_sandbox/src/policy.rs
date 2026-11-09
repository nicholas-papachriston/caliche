//! Tool policy evaluation.

use crate::config::ToolPolicy;

/// Normalize ergonomic tool aliases to canonical sandbox names.
///
/// Mirrors Caliche registry aliases so `shell`/`web`/`memory` honor the
/// same allow/deny lists as `bash`/`http`/`workspace`.
#[must_use]
pub fn normalize_tool_name(tool_name: &str) -> &str {
    let trimmed = tool_name.trim();
    if trimmed.eq_ignore_ascii_case("shell") {
        "bash"
    } else if trimmed.eq_ignore_ascii_case("web") {
        "http"
    } else if trimmed.eq_ignore_ascii_case("memory") {
        "workspace"
    } else {
        trimmed
    }
}

fn policy_names_match(policy_name: &str, tool_name: &str) -> bool {
    normalize_tool_name(policy_name) == normalize_tool_name(tool_name)
}

/// Check if a tool is allowed by the policy (allow list and deny list).
#[must_use]
pub fn evaluate_tool_policy(policy: &ToolPolicy, tool_name: &str) -> bool {
    if policy
        .deny
        .iter()
        .any(|denied| policy_names_match(denied, tool_name))
    {
        return false;
    }
    if policy.allow.is_empty() {
        return true;
    }
    policy
        .allow
        .iter()
        .any(|allowed| policy_names_match(allowed, tool_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_allows_shell_alias_for_bash() {
        let policy = ToolPolicy::default();
        assert!(evaluate_tool_policy(&policy, "bash"));
        assert!(evaluate_tool_policy(&policy, "shell"));
        assert!(evaluate_tool_policy(&policy, " Shell "));
        assert!(!evaluate_tool_policy(&policy, "http"));
        assert!(!evaluate_tool_policy(&policy, "web"));
    }

    #[test]
    fn deny_list_honors_aliases() {
        let policy = ToolPolicy {
            allow: vec![],
            deny: vec!["shell".into()],
        };
        assert!(!evaluate_tool_policy(&policy, "bash"));
        assert!(!evaluate_tool_policy(&policy, "shell"));
    }
}
