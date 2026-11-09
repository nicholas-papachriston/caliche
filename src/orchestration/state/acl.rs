//! Namespaces and write ACLs for [`SharedState`](super::SharedState).
//!
//! Shared state is a single JSON document that every agent in a run can read. Without a
//! policy any agent can also overwrite any branch, which lets one compromised subagent steer
//! the whole run. A [`StateWritePolicy`] limits which JSON pointer paths an `agent_id` may
//! write, and [`agent_namespace`] gives each agent a private branch by convention.

use crate::orchestration::acl::{GlobRule, PolicyEffect, glob_match};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// Root under which per-agent private branches live.
pub const AGENT_NAMESPACE_ROOT: &str = "/agents";

/// Outcome of a write authorization check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateWriteDecision {
    /// The write is permitted.
    Allow,
    /// The write is refused.
    Deny {
        /// Pattern that produced the denial, or `"<no matching rule>"` when the policy is
        /// closed by default and nothing matched.
        pattern: String,
    },
}

impl StateWriteDecision {
    /// Whether the decision permits the write.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Convert a denial into an error.
    pub fn into_result(self, agent_id: &str, path: &str) -> anyhow::Result<()> {
        match self {
            Self::Allow => Ok(()),
            Self::Deny { pattern } => anyhow::bail!(
                "Agent '{agent_id}' may not write shared state path '{path}': denied by rule '{pattern}'"
            ),
        }
    }
}

/// Per-agent write rules over shared state JSON pointer paths.
///
/// An empty policy allows every write, which keeps existing callers working. Once any rule
/// exists for an agent, that agent's rule set alone decides its writes, and a path with no
/// matching rule is refused.
#[derive(Debug, Clone, Default)]
pub struct StateWritePolicy {
    per_agent: BTreeMap<String, Vec<GlobRule>>,
    default_rules: Vec<GlobRule>,
    /// When true, an agent with no rules at all may still write anywhere.
    open_by_default: bool,
}

impl StateWritePolicy {
    /// Policy that permits every write.
    pub fn allow_all() -> Self {
        Self {
            per_agent: BTreeMap::new(),
            default_rules: Vec::new(),
            open_by_default: true,
        }
    }

    /// Policy that refuses every write unless a rule permits it.
    pub fn deny_by_default() -> Self {
        Self {
            per_agent: BTreeMap::new(),
            default_rules: Vec::new(),
            open_by_default: false,
        }
    }

    /// Allow `agent_id` to write paths matching `pattern`.
    #[must_use]
    pub fn allow(mut self, agent_id: impl Into<String>, pattern: impl Into<String>) -> Self {
        self.per_agent
            .entry(agent_id.into())
            .or_default()
            .push(GlobRule::new(pattern, PolicyEffect::Allow));
        self
    }

    /// Refuse `agent_id` writes to paths matching `pattern`.
    #[must_use]
    pub fn deny(mut self, agent_id: impl Into<String>, pattern: impl Into<String>) -> Self {
        self.per_agent
            .entry(agent_id.into())
            .or_default()
            .push(GlobRule::new(pattern, PolicyEffect::Deny));
        self
    }

    /// Add a rule that applies to agents without their own rules.
    #[must_use]
    pub fn default_rule(mut self, pattern: impl Into<String>, effect: PolicyEffect) -> Self {
        self.default_rules.push(GlobRule::new(pattern, effect));
        self
    }

    /// Give `agent_id` exclusive write access to its own namespace branch.
    #[must_use]
    pub fn with_private_namespace(self, agent_id: impl Into<String>) -> Self {
        let agent_id = agent_id.into();
        let pattern = format!("{}/*", agent_namespace(&agent_id));
        self.allow(agent_id, pattern)
    }

    /// Whether the policy carries no rules.
    pub fn is_empty(&self) -> bool {
        self.per_agent.is_empty() && self.default_rules.is_empty()
    }

    /// Decide whether `agent_id` may write `path`.
    pub fn evaluate(&self, agent_id: &str, path: &str) -> StateWriteDecision {
        let normalized = normalize_path(path);
        let rules = self
            .per_agent
            .get(agent_id)
            .filter(|rules| !rules.is_empty())
            .map(Vec::as_slice)
            .unwrap_or(self.default_rules.as_slice());

        if rules.is_empty() {
            return if self.open_by_default {
                StateWriteDecision::Allow
            } else {
                StateWriteDecision::Deny {
                    pattern: "<no matching rule>".to_string(),
                }
            };
        }

        let winner = rules
            .iter()
            .filter(|rule| path_matches(&rule.pattern, &normalized))
            .max_by(|left, right| {
                left.specificity().cmp(&right.specificity()).then_with(|| {
                    match (left.effect, right.effect) {
                        (PolicyEffect::Deny, PolicyEffect::Allow) => Ordering::Greater,
                        (PolicyEffect::Allow, PolicyEffect::Deny) => Ordering::Less,
                        _ => Ordering::Equal,
                    }
                })
            });

        match winner {
            Some(rule) => match rule.effect {
                PolicyEffect::Allow => StateWriteDecision::Allow,
                PolicyEffect::Deny => StateWriteDecision::Deny {
                    pattern: rule.pattern.clone(),
                },
            },
            None => StateWriteDecision::Deny {
                pattern: "<no matching rule>".to_string(),
            },
        }
    }

    /// Evaluate and convert a denial into an error.
    pub fn authorize(&self, agent_id: &str, path: &str) -> anyhow::Result<()> {
        self.evaluate(agent_id, path).into_result(agent_id, path)
    }
}

/// Private branch reserved for one agent, for example `/agents/reviewer`.
pub fn agent_namespace(agent_id: &str) -> String {
    format!("{AGENT_NAMESPACE_ROOT}/{}", sanitize_agent_id(agent_id))
}

/// Keep namespace keys free of glob metacharacters and path separators so a
/// hostile agent id cannot match another agent's branch.
pub fn sanitize_agent_id(agent_id: &str) -> String {
    let cleaned: String = agent_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned
    }
}

/// Join an agent namespace with a relative path inside it.
pub fn namespaced_path(agent_id: &str, path: &str) -> String {
    let suffix = path.trim_start_matches('/');
    if suffix.is_empty() {
        return agent_namespace(agent_id);
    }
    format!("{}/{}", agent_namespace(agent_id), suffix)
}

/// Normalize a JSON pointer path to a single leading slash and no trailing slash.
fn normalize_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "/" {
        return "/".to_string();
    }
    let core = trimmed.trim_matches('/');
    format!("/{core}")
}

/// Match a path pattern, where a trailing `/*` also covers the branch root itself.
///
/// Without this, `"/agents/reviewer/*"` would allow `/agents/reviewer/notes` but refuse
/// `/agents/reviewer`, which is a surprising split for a namespace grant.
fn path_matches(pattern: &str, path: &str) -> bool {
    if glob_match(pattern, path) {
        return true;
    }
    pattern
        .strip_suffix("/*")
        .is_some_and(|prefix| glob_match(prefix, path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_open_policy_allows_writes() {
        let policy = StateWritePolicy::allow_all();
        assert!(policy.is_empty());
        assert!(policy.evaluate("reviewer", "/anything").is_allowed());
    }

    #[test]
    fn deny_by_default_refuses_unmatched_paths() {
        let policy = StateWritePolicy::deny_by_default();
        assert_eq!(
            policy.evaluate("reviewer", "/plan"),
            StateWriteDecision::Deny {
                pattern: "<no matching rule>".to_string()
            }
        );
    }

    #[test]
    fn private_namespace_grants_own_branch_only() {
        let policy = StateWritePolicy::deny_by_default().with_private_namespace("reviewer");

        assert!(policy.evaluate("reviewer", "/agents/reviewer").is_allowed());
        assert!(
            policy
                .evaluate("reviewer", "/agents/reviewer/notes")
                .is_allowed()
        );
        assert!(!policy.evaluate("reviewer", "/agents/planner").is_allowed());
        assert!(!policy.evaluate("reviewer", "/plan").is_allowed());
    }

    #[test]
    fn hostile_agent_id_cannot_glob_other_namespaces() {
        let policy = StateWritePolicy::deny_by_default().with_private_namespace("*");
        assert!(policy.evaluate("*", "/agents/_").is_allowed());
        assert!(policy.evaluate("*", "/agents/_/notes").is_allowed());
        assert!(!policy.evaluate("*", "/agents/planner").is_allowed());
        assert!(!policy.evaluate("*", "/agents/planner/notes").is_allowed());
    }

    #[test]
    fn literal_deny_outranks_longer_wildcard_allow() {
        let policy = StateWritePolicy::deny_by_default()
            .allow("planner", "/plan*")
            .deny("planner", "/plan");
        assert!(!policy.evaluate("planner", "/plan").is_allowed());
        assert!(policy.evaluate("planner", "/plan/steps").is_allowed());
    }

    #[test]
    fn sanitize_agent_id_strips_separators_and_globs() {
        assert_eq!(sanitize_agent_id("reviewer"), "reviewer");
        assert_eq!(sanitize_agent_id("a/../b"), "a____b");
        assert_eq!(sanitize_agent_id("*"), "_");
        assert_eq!(sanitize_agent_id(""), "unknown");
        assert_eq!(agent_namespace("a/b"), "/agents/a_b");
    }

    #[test]
    fn explicit_deny_beats_broader_allow() {
        let policy = StateWritePolicy::deny_by_default()
            .allow("planner", "/plan*")
            .deny("planner", "/plan/secrets*");

        assert!(policy.evaluate("planner", "/plan/steps").is_allowed());
        assert!(!policy.evaluate("planner", "/plan/secrets/key").is_allowed());
    }

    #[test]
    fn default_rules_apply_to_agents_without_rules() {
        let policy = StateWritePolicy::deny_by_default()
            .default_rule("/scratch*", PolicyEffect::Allow)
            .allow("planner", "/plan*");

        assert!(policy.evaluate("unknown", "/scratch/tmp").is_allowed());
        assert!(!policy.evaluate("unknown", "/plan").is_allowed());
        // Planner has its own rules, so the default set no longer applies to it.
        assert!(!policy.evaluate("planner", "/scratch/tmp").is_allowed());
    }

    #[test]
    fn paths_normalize_before_matching() {
        let policy = StateWritePolicy::deny_by_default().allow("a", "/plan*");
        assert!(policy.evaluate("a", "plan/steps").is_allowed());
        assert!(policy.evaluate("a", "/plan/steps/").is_allowed());
    }

    #[test]
    fn namespaced_path_joins_branch_and_suffix() {
        assert_eq!(agent_namespace("reviewer"), "/agents/reviewer");
        assert_eq!(
            namespaced_path("reviewer", "notes"),
            "/agents/reviewer/notes"
        );
        assert_eq!(
            namespaced_path("reviewer", "/notes"),
            "/agents/reviewer/notes"
        );
        assert_eq!(namespaced_path("reviewer", ""), "/agents/reviewer");
    }

    #[test]
    fn authorize_reports_the_denying_rule() {
        let policy = StateWritePolicy::deny_by_default().deny("planner", "/plan*");
        let err = policy
            .authorize("planner", "/plan")
            .expect_err("deny rule must refuse");
        let message = err.to_string();
        assert!(message.contains("planner"));
        assert!(message.contains("/plan"));
    }
}
