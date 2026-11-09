//! Access control for subagent (task) invocation.
//!
//! An agent declares which other agents it may invoke with a small glob rule map, for
//! example `{"*": "deny", "reviewer": "allow"}`. The most specific rule wins, so a broad
//! `*` deny can be relaxed for named targets without ordering games.
//!
//! Defaults stay permissive for backward compatibility: an empty policy allows every
//! target. Callers that want a closed posture start from [`TaskInvocationPolicy::deny_all`]
//! or configure `orchestration.task_acl` in the agent config.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Effect of a single glob rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyEffect {
    /// The invocation is permitted.
    Allow,
    /// The invocation is refused.
    Deny,
}

impl PolicyEffect {
    /// Parse an effect from a configuration string (`allow` / `deny`).
    ///
    /// `ask` is accepted and mapped to [`Self::Deny`]: there is no interactive prompt in a
    /// non-interactive dispatch path, so the safe reading of "ask" is "not automatically
    /// permitted".
    pub fn parse(value: &str) -> anyhow::Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "allow" | "true" | "yes" => Ok(Self::Allow),
            "deny" | "false" | "no" | "ask" => Ok(Self::Deny),
            other => anyhow::bail!("Unknown task ACL effect '{other}' (expected allow or deny)"),
        }
    }
}

/// A single glob rule inside a policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobRule {
    /// Glob pattern supporting `*` (any run of characters) and `?` (one character).
    pub pattern: String,
    /// What happens when [`Self::pattern`] matches.
    pub effect: PolicyEffect,
}

impl GlobRule {
    /// Build a rule from a pattern and effect.
    pub fn new(pattern: impl Into<String>, effect: PolicyEffect) -> Self {
        Self {
            pattern: pattern.into(),
            effect,
        }
    }

    /// Whether this rule's pattern matches `candidate`.
    pub fn matches(&self, candidate: &str) -> bool {
        glob_match(&self.pattern, candidate)
    }

    /// Specificity score used to pick a winner when several rules match.
    ///
    /// Literal (wildcard-free) patterns outrank wildcard patterns; among wildcard patterns
    /// the one with more literal characters wins, which makes `reviewer` beat `rev*` and
    /// `rev*` beat `*`.
    pub fn specificity(&self) -> (u8, usize) {
        let has_wildcard = self.pattern.contains('*') || self.pattern.contains('?');
        let literal_len = self
            .pattern
            .chars()
            .filter(|c| *c != '*' && *c != '?')
            .count();
        (u8::from(!has_wildcard), literal_len)
    }
}

fn cmp_matching_rules(left: &GlobRule, right: &GlobRule) -> std::cmp::Ordering {
    left.specificity()
        .cmp(&right.specificity())
        .then_with(|| match (left.effect, right.effect) {
            (PolicyEffect::Deny, PolicyEffect::Allow) => std::cmp::Ordering::Greater,
            (PolicyEffect::Allow, PolicyEffect::Deny) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Equal,
        })
}

/// Outcome of an authorization check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationDecision {
    /// The caller may invoke the target.
    Allow,
    /// The caller may not invoke the target.
    Deny {
        /// Pattern that produced the denial, for operator-facing messages.
        pattern: String,
    },
}

impl InvocationDecision {
    /// Whether the decision permits the invocation.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Convert a denial into an error; allow returns `Ok(())`.
    pub fn into_result(self, caller: &str, target: &str) -> anyhow::Result<()> {
        match self {
            Self::Allow => Ok(()),
            Self::Deny { pattern } => anyhow::bail!(
                "Agent '{caller}' may not invoke '{target}': denied by task ACL rule '{pattern}'"
            ),
        }
    }
}

/// Glob rules that decide which targets an agent may invoke.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskInvocationPolicy {
    #[serde(default)]
    rules: Vec<GlobRule>,
}

impl TaskInvocationPolicy {
    /// Policy that permits every target (the default when nothing is configured).
    pub fn allow_all() -> Self {
        Self::default()
    }

    /// Policy that refuses every target.
    pub fn deny_all() -> Self {
        Self {
            rules: vec![GlobRule::new("*", PolicyEffect::Deny)],
        }
    }

    /// Build a policy from explicit rules.
    pub fn from_rules(rules: impl IntoIterator<Item = GlobRule>) -> Self {
        Self {
            rules: rules.into_iter().collect(),
        }
    }

    /// Build a policy from a configuration map such as `{"*": "deny", "reviewer": "allow"}`.
    pub fn from_config_map(map: &BTreeMap<String, String>) -> anyhow::Result<Self> {
        let mut rules = Vec::with_capacity(map.len());
        for (pattern, effect) in map {
            rules.push(GlobRule::new(pattern.clone(), PolicyEffect::parse(effect)?));
        }
        Ok(Self { rules })
    }

    /// Whether the policy has no rules and therefore allows every target.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Rules in declaration order.
    pub fn rules(&self) -> &[GlobRule] {
        &self.rules
    }

    /// Add one rule and return the policy, for builder-style construction.
    #[must_use]
    pub fn with_rule(mut self, pattern: impl Into<String>, effect: PolicyEffect) -> Self {
        self.rules.push(GlobRule::new(pattern, effect));
        self
    }

    /// Decide whether `target` may be invoked.
    ///
    /// With no matching rule the target is allowed, which keeps existing deployments working
    /// until they opt into a policy.
    pub fn evaluate(&self, target: &str) -> InvocationDecision {
        self.evaluate_aliases(&[target])
    }

    /// Decide using every known alias of a target (raw name, agent id, registered name).
    ///
    /// A name-based deny must still apply when the caller invokes the same agent by UUID.
    /// Equal-specificity ties prefer [`PolicyEffect::Deny`].
    pub fn evaluate_aliases(&self, aliases: &[&str]) -> InvocationDecision {
        let winner = self
            .rules
            .iter()
            .filter(|rule| {
                aliases
                    .iter()
                    .copied()
                    .any(|alias| !alias.is_empty() && rule.matches(alias))
            })
            .max_by(|left, right| cmp_matching_rules(left, right));

        match winner {
            Some(rule) => match rule.effect {
                PolicyEffect::Allow => InvocationDecision::Allow,
                PolicyEffect::Deny => InvocationDecision::Deny {
                    pattern: rule.pattern.clone(),
                },
            },
            None => InvocationDecision::Allow,
        }
    }

    /// Evaluate and convert a denial into an error.
    pub fn authorize(&self, caller: &str, target: &str) -> anyhow::Result<()> {
        self.evaluate(target).into_result(caller, target)
    }

    /// Evaluate aliases and convert a denial into an error naming `target`.
    pub fn authorize_aliases(
        &self,
        caller: &str,
        target: &str,
        aliases: &[&str],
    ) -> anyhow::Result<()> {
        self.evaluate_aliases(aliases).into_result(caller, target)
    }
}

/// Match `text` against a glob `pattern` supporting `*` and `?`.
///
/// Byte-oriented and allocation-free; patterns and agent names are ASCII in practice, and a
/// multi-byte character simply behaves as several `?`-matchable bytes.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern = pattern.as_bytes();
    let text = text.as_bytes();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut match_after_star = 0usize;

    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == b'?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == b'*' {
            star = Some(pi);
            pi += 1;
            match_after_star = ti;
        } else if let Some(star_idx) = star {
            pi = star_idx + 1;
            match_after_star += 1;
            ti = match_after_star;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == b'*' {
        pi += 1;
    }
    pi == pattern.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_policy_allows_every_target() {
        let policy = TaskInvocationPolicy::allow_all();
        assert!(policy.is_empty());
        assert!(policy.evaluate("reviewer").is_allowed());
        assert!(policy.evaluate("anything-else").is_allowed());
    }

    #[test]
    fn deny_all_refuses_every_target() {
        let policy = TaskInvocationPolicy::deny_all();
        assert_eq!(
            policy.evaluate("reviewer"),
            InvocationDecision::Deny {
                pattern: "*".to_string()
            }
        );
    }

    #[test]
    fn exact_rule_overrides_wildcard_deny() {
        let mut map = BTreeMap::new();
        map.insert("*".to_string(), "deny".to_string());
        map.insert("reviewer".to_string(), "allow".to_string());
        let policy = TaskInvocationPolicy::from_config_map(&map).expect("valid config map");

        assert!(policy.evaluate("reviewer").is_allowed());
        assert!(!policy.evaluate("shell-runner").is_allowed());
    }

    #[test]
    fn longer_wildcard_beats_shorter_wildcard() {
        let policy = TaskInvocationPolicy::from_rules([
            GlobRule::new("*", PolicyEffect::Allow),
            GlobRule::new("danger-*", PolicyEffect::Deny),
        ]);

        assert!(policy.evaluate("reviewer").is_allowed());
        assert_eq!(
            policy.evaluate("danger-rm"),
            InvocationDecision::Deny {
                pattern: "danger-*".to_string()
            }
        );
    }

    #[test]
    fn question_mark_matches_single_character() {
        assert!(glob_match("agent-?", "agent-1"));
        assert!(!glob_match("agent-?", "agent-12"));
    }

    #[test]
    fn glob_star_matches_across_separators() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("rev*er", "reviewer"));
        assert!(!glob_match("rev*er", "reviewing"));
    }

    #[test]
    fn authorize_reports_the_denying_pattern() {
        let policy = TaskInvocationPolicy::deny_all();
        let err = policy
            .authorize("planner", "shell-runner")
            .expect_err("deny-all must refuse");
        let message = err.to_string();
        assert!(message.contains("planner"));
        assert!(message.contains("shell-runner"));
        assert!(message.contains('*'));
    }

    #[test]
    fn ask_effect_is_treated_as_deny() {
        assert_eq!(
            PolicyEffect::parse("ask").expect("ask parses"),
            PolicyEffect::Deny
        );
        assert!(PolicyEffect::parse("maybe").is_err());
    }

    #[test]
    fn aliases_apply_name_deny_to_agent_id() {
        let mut map = BTreeMap::new();
        map.insert("deployer".to_string(), "deny".to_string());
        let policy = TaskInvocationPolicy::from_config_map(&map).expect("valid");
        let id = "01900000-0000-7000-8000-000000000001";
        assert!(policy.evaluate(id).is_allowed());
        assert!(!policy.evaluate_aliases(&[id, "deployer"]).is_allowed());
    }

    #[test]
    fn aliases_allow_allowlisted_name_when_invoked_by_id() {
        let mut map = BTreeMap::new();
        map.insert("*".to_string(), "deny".to_string());
        map.insert("reviewer".to_string(), "allow".to_string());
        let policy = TaskInvocationPolicy::from_config_map(&map).expect("valid");
        let id = "01900000-0000-7000-8000-000000000002";
        assert!(!policy.evaluate(id).is_allowed());
        assert!(policy.evaluate_aliases(&[id, "reviewer"]).is_allowed());
    }

    #[test]
    fn equal_specificity_prefers_deny() {
        let policy = TaskInvocationPolicy::from_rules([
            GlobRule::new("a*", PolicyEffect::Allow),
            GlobRule::new("*a", PolicyEffect::Deny),
        ]);
        assert_eq!(
            policy.evaluate("aa"),
            InvocationDecision::Deny {
                pattern: "*a".to_string()
            }
        );
    }
}
