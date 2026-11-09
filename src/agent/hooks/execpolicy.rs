//! Fine-grained bash command policy engine.
//!
//! Evaluates tokenized bash commands against declarative rules (allow / prompt / forbidden)
//! and integrates via `LifecycleHook` for the `bash` tool.

use crate::agent::hooks::lifecycle::{HookContext, HookResult, LifecycleHook};
use anyhow::{Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

/// Decision for a matched rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BashDecision {
    /// Allow execution without prompting.
    Allow,
    /// Require user approval (phase 1: treated as deny with message).
    Prompt,
    /// Forbid execution.
    Forbidden,
}

/// One position in a pattern: either a single token (or "*" for any), or alternation of tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PatternPos {
    /// Single required token, or "*" to match any token.
    One(String),
    /// Any one of these tokens.
    Many(Vec<String>),
}

impl PatternPos {
    fn matches(&self, token: &str) -> bool {
        match self {
            Self::One(s) => s == "*" || s == token,
            Self::Many(v) => v.iter().any(|s| s == token),
        }
    }
}

/// A single bash policy rule: pattern + decision + optional justification.
#[derive(Debug, Clone)]
pub struct BashRule {
    /// Command prefix pattern (tokenized); each element is one position (string or list of alternatives).
    pub pattern: Vec<PatternPos>,
    /// Decision when this rule matches.
    pub decision: BashDecision,
    /// Optional justification (shown in deny message).
    pub justification: Option<String>,
    /// Example commands that must match this rule (validated at config load time).
    pub match_examples: Option<Vec<String>>,
    /// Example commands that must NOT match this rule (validated at config load time).
    pub not_match_examples: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PatternField {
    Single(String),
    List(Vec<PatternPos>),
}

impl<'de> Deserialize<'de> for BashRule {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct BashRuleRaw {
            pattern: PatternField,
            decision: BashDecision,
            #[serde(default)]
            justification: Option<String>,
            #[serde(default, rename = "match")]
            match_examples: Option<Vec<String>>,
            #[serde(default, rename = "not_match")]
            not_match_examples: Option<Vec<String>>,
        }
        let raw = BashRuleRaw::deserialize(deserializer)?;
        let pattern = match raw.pattern {
            PatternField::Single(s) => vec![PatternPos::One(s)],
            PatternField::List(l) => l,
        };
        Ok(Self {
            pattern,
            decision: raw.decision,
            justification: raw.justification,
            match_examples: raw.match_examples,
            not_match_examples: raw.not_match_examples,
        })
    }
}

impl Serialize for BashRule {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("BashRule", 5)?;
        st.serialize_field("pattern", &self.pattern)?;
        st.serialize_field("decision", &self.decision)?;
        st.serialize_field("justification", &self.justification)?;
        st.serialize_field("match", &self.match_examples)?;
        st.serialize_field("not_match", &self.not_match_examples)?;
        st.end()
    }
}

/// Tokenize a bash command for policy matching.
///
/// A command the shell lexer cannot parse becomes one opaque token, so a rule can still
/// match it literally instead of the policy silently seeing no tokens at all.
fn tokenize(command: &str) -> Vec<String> {
    shell_words::split(command).unwrap_or_else(|_| vec![command.to_string()])
}

impl BashRule {
    /// Returns true if the given token slice matches this rule's pattern (prefix match).
    fn matches_tokens(&self, tokens: &[String]) -> bool {
        self.pattern.len() <= tokens.len()
            && self
                .pattern
                .iter()
                .zip(tokens)
                .all(|(pos, token)| pos.matches(token))
    }

    /// Validate that `match/not_match` examples conform to this rule. Call at config load time.
    pub fn validate_examples(&self) -> Result<()> {
        if let Some(ref examples) = self.match_examples {
            for (i, cmd) in examples.iter().enumerate() {
                let tokens = tokenize(cmd);
                if !self.matches_tokens(&tokens) {
                    bail!(
                        "permissions.bash rule match example {} ('{}') does not match this rule's pattern",
                        i + 1,
                        cmd
                    );
                }
            }
        }
        if let Some(ref examples) = self.not_match_examples {
            for (i, cmd) in examples.iter().enumerate() {
                let tokens = tokenize(cmd);
                if self.matches_tokens(&tokens) {
                    bail!(
                        "permissions.bash rule not_match example {} ('{}') unexpectedly matches this rule's pattern",
                        i + 1,
                        cmd
                    );
                }
            }
        }
        Ok(())
    }
}

/// Policy engine: evaluates a command string against an ordered list of rules.
///
/// Last matching rule wins; default is Allow when no rule matches. To approximate
/// "forbidden > prompt > allow" precedence, order rules with forbidden first, then
/// prompt, then allow.
#[derive(Debug, Clone)]
pub struct ExecPolicy {
    rules: Vec<BashRule>,
}

/// Prefixes that must not be suggested as allow-rule amendments (arbitrary exec, destructive).
/// Single-token prefixes prevent overly broad rules (e.g. "python" alone would allow "python -c").
const BANNED_PREFIX_SUGGESTIONS: &[&[&str]] = &[
    &["python"],
    &["python3"],
    &["bash"],
    &["sh"],
    &["rm"],
    &["python", "-c"],
    &["python3", "-c"],
    &["bash", "-c"],
    &["bash", "-lc"],
    &["sh", "-c"],
    &["rm", "-rf"],
    &["rm", "-r"],
];

fn compute_proposed_amendment(tokens: &[String]) -> Option<Vec<String>> {
    if tokens.is_empty() {
        return None;
    }
    for banned in BANNED_PREFIX_SUGGESTIONS {
        if tokens.len() >= banned.len()
            && tokens[..banned.len()]
                .iter()
                .zip(banned.iter())
                .all(|(a, b)| a == *b)
        {
            return None;
        }
    }
    Some(tokens.to_vec())
}

impl ExecPolicy {
    /// Build an execution policy from rules.
    #[must_use]
    pub const fn new(rules: Vec<BashRule>) -> Self {
        Self { rules }
    }

    /// Evaluate a bash command string. Tokenizes with shell-words, then applies rules in order.
    /// Returns (decision, justification) where justification is from the last matching rule.
    #[must_use]
    pub fn evaluate(&self, command: &str) -> (BashDecision, Option<String>) {
        self.evaluate_tokens(&tokenize(command))
    }

    /// Evaluate an already tokenized command, so callers that also need the tokens do not
    /// tokenize twice. Returns the decision and justification of the last matching rule.
    #[must_use]
    pub fn evaluate_tokens(&self, tokens: &[String]) -> (BashDecision, Option<String>) {
        let mut decision = BashDecision::Allow;
        let mut justification = None;

        for rule in &self.rules {
            if rule.matches_tokens(tokens) {
                decision = rule.decision;
                justification = rule.justification.clone();
            }
        }

        (decision, justification)
    }
}

/// Lifecycle hook that enforces `ExecPolicy` for the `bash` tool only.
#[derive(Debug)]
pub struct ExecPolicyHook {
    policy: Arc<ExecPolicy>,
}

impl ExecPolicyHook {
    /// Create a hook that uses the given policy.
    #[must_use]
    pub const fn new(policy: Arc<ExecPolicy>) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl LifecycleHook for ExecPolicyHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        if ctx.tool_name.as_ref() != "bash" {
            return HookResult::allow();
        }

        let Some(command) = ctx.tool_input.get("command").and_then(|v| v.as_str()) else {
            return HookResult::allow();
        };

        let tokens = tokenize(command);
        let (decision, justification) = self.policy.evaluate_tokens(&tokens);

        let proposed_amendment = match decision {
            BashDecision::Prompt => compute_proposed_amendment(&tokens),
            BashDecision::Allow | BashDecision::Forbidden => None,
        };
        let permission =
            crate::agent::hooks::permission_contracts::bash_decision_to_permission_decision(
                decision,
                justification,
                proposed_amendment,
            );
        match permission {
            crate::agent::hooks::permission::PermissionDecision::Allow => HookResult::allow(),
            crate::agent::hooks::permission::PermissionDecision::Deny { reason } => {
                HookResult::deny(reason.to_string())
            }
            crate::agent::hooks::permission::PermissionDecision::RequireApproval {
                reason,
                proposed_amendment,
            } => HookResult::require_approval(reason.to_string(), proposed_amendment),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(
        pattern: Vec<PatternPos>,
        decision: BashDecision,
        justification: Option<&str>,
    ) -> BashRule {
        BashRule {
            pattern,
            decision,
            justification: justification.map(String::from),
            match_examples: None,
            not_match_examples: None,
        }
    }

    #[test]
    fn tokenizes_and_matches_single_prefix_rule() {
        let policy = ExecPolicy::new(vec![rule(
            vec![
                PatternPos::One("git".to_string()),
                PatternPos::One("status".to_string()),
            ],
            BashDecision::Allow,
            Some("safe read-only"),
        )]);

        let (d, j) = policy.evaluate("git status");
        assert_eq!(d, BashDecision::Allow);
        assert_eq!(j.as_deref(), Some("safe read-only"));

        let (d, _) = policy.evaluate("git diff");
        assert_eq!(d, BashDecision::Allow); // no match -> default allow
    }

    #[test]
    fn matches_alternation_positions() {
        let policy = ExecPolicy::new(vec![rule(
            vec![
                PatternPos::One("git".to_string()),
                PatternPos::Many(vec![
                    "status".to_string(),
                    "diff".to_string(),
                    "log".to_string(),
                ]),
            ],
            BashDecision::Allow,
            None,
        )]);

        assert_eq!(policy.evaluate("git status").0, BashDecision::Allow);
        assert_eq!(policy.evaluate("git diff").0, BashDecision::Allow);
        assert_eq!(policy.evaluate("git log").0, BashDecision::Allow);
        assert_eq!(policy.evaluate("git push").0, BashDecision::Allow); // no match
    }

    #[test]
    fn last_matching_bash_rule_wins() {
        let policy = ExecPolicy::new(vec![
            rule(
                vec![PatternPos::One("*".to_string())],
                BashDecision::Prompt,
                Some("catch-all prompt"),
            ),
            rule(
                vec![
                    PatternPos::One("rm".to_string()),
                    PatternPos::One("-rf".to_string()),
                ],
                BashDecision::Forbidden,
                Some("Use trash-put or git clean instead"),
            ),
        ]);

        let (d, j) = policy.evaluate("rm -rf /tmp/foo");
        assert_eq!(d, BashDecision::Forbidden);
        assert!(j.as_deref().unwrap().contains("trash-put"));

        let (d, j) = policy.evaluate("echo hello");
        assert_eq!(d, BashDecision::Prompt);
        assert_eq!(j.as_deref(), Some("catch-all prompt"));
    }

    #[test]
    fn allows_by_default_when_no_rule_matches() {
        let policy = ExecPolicy::new(vec![rule(
            vec![PatternPos::One("rm".to_string())],
            BashDecision::Forbidden,
            None,
        )]);

        let (d, _) = policy.evaluate("cargo build");
        assert_eq!(d, BashDecision::Allow);
    }

    #[test]
    fn validates_matching_and_nonmatching_examples() {
        let r = BashRule {
            pattern: vec![
                PatternPos::One("git".to_string()),
                PatternPos::Many(vec!["status".to_string(), "diff".to_string()]),
            ],
            decision: BashDecision::Allow,
            justification: None,
            match_examples: Some(vec!["git status".to_string(), "git diff".to_string()]),
            not_match_examples: Some(vec!["git push".to_string()]),
        };
        assert!(r.validate_examples().is_ok());
    }

    #[test]
    fn rejects_match_examples_that_do_not_match() {
        let r = BashRule {
            pattern: vec![
                PatternPos::One("git".to_string()),
                PatternPos::One("status".to_string()),
            ],
            decision: BashDecision::Allow,
            justification: None,
            match_examples: Some(vec!["git push".to_string()]),
            not_match_examples: None,
        };
        assert!(r.validate_examples().is_err());
    }

    #[test]
    fn rejects_not_match_examples_that_match() {
        let r = BashRule {
            pattern: vec![
                PatternPos::One("git".to_string()),
                PatternPos::One("status".to_string()),
            ],
            decision: BashDecision::Allow,
            justification: None,
            match_examples: None,
            not_match_examples: Some(vec!["git status".to_string()]),
        };
        assert!(r.validate_examples().is_err());
    }
}
