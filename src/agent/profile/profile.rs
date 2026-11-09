//! Agent profile data model: YAML frontmatter + body (system prompt).

use crate::agent::hooks::BashRule;
use crate::orchestration::TaskInvocationPolicy;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Permission settings for a profile (edit allow/deny, bash rules).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilePermission {
    /// Permission mode: `default`, `accept-edits`, `accept-all`, `deny-all`, or `interactive`.
    #[serde(default)]
    pub edit: Option<String>,
    /// Bash command policy rules (list form). Map form ("git diff*": allow) can be added later.
    #[serde(default)]
    pub bash: Option<Vec<BashRule>>,
}

/// Agent profile: frontmatter fields + body (system prompt).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Human-readable description.
    #[serde(default)]
    pub description: Option<String>,
    /// Mode: "primary" (selectable as main agent) or "subagent" (invocable via @mention).
    #[serde(default)]
    pub mode: Option<String>,
    /// Model ID (e.g. "anthropic/claude-3-5-haiku-20241022" or "gpt-4o").
    #[serde(default)]
    pub model: Option<String>,
    /// Temperature for generation (0.0–1.0).
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Max steps/iterations per turn.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Tools: list of tool names (allowlist) or map of tool_name -> bool (enable/disable).
    #[serde(default)]
    pub tools: Option<serde_norway::Value>,
    /// Permission (edit, bash rules).
    #[serde(default)]
    pub permission: Option<ProfilePermission>,
    /// Subagents this profile may invoke, as glob pattern -> `allow` or `deny`.
    ///
    /// Example frontmatter: `task_permissions: {"*": "deny", "reviewer": "allow"}`.
    /// When absent, every target that the caller can reach is permitted.
    #[serde(default)]
    pub task_permissions: Option<BTreeMap<String, String>>,
    /// Display color (e.g. "#ff6b6b").
    #[serde(default)]
    pub color: Option<String>,
    /// System prompt body (Markdown after the frontmatter). Set by loader, not from YAML.
    #[serde(skip)]
    pub system_prompt_body: Option<String>,
    /// Unknown frontmatter keys for forward compatibility.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_norway::Value>,
}

impl AgentProfile {
    /// Whether this profile is a primary agent (selectable as main).
    pub fn is_primary(&self) -> bool {
        self.mode
            .as_deref()
            .map(|m| m.eq_ignore_ascii_case("primary"))
            .unwrap_or(false)
    }

    /// Whether this profile is a subagent (invocable via @mention).
    pub fn is_subagent(&self) -> bool {
        self.mode
            .as_deref()
            .map(|m| m.eq_ignore_ascii_case("subagent"))
            .unwrap_or(false)
    }

    /// Resolve enabled tool names: from list (allowlist) or from map (keys where value is true).
    pub fn enabled_tools(&self) -> Option<Vec<String>> {
        let v = self.tools.as_ref()?;
        if let Some(arr) = v.as_sequence() {
            let names: Vec<String> = arr
                .iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect();
            return Some(names);
        }
        if let Some(map) = v.as_mapping() {
            let names: Vec<String> = map
                .iter()
                .filter(|(_, val)| val.as_bool().unwrap_or(false))
                .filter_map(|(k, _)| k.as_str().map(String::from))
                .collect();
            return Some(names);
        }
        None
    }

    /// Task invocation policy declared by this profile.
    ///
    /// Returns an allow-all policy when the profile declares no `task_permissions`, so an
    /// existing profile keeps its current reach.
    pub fn task_policy(&self) -> anyhow::Result<TaskInvocationPolicy> {
        match self.task_permissions.as_ref() {
            Some(map) => TaskInvocationPolicy::from_config_map(map),
            None => Ok(TaskInvocationPolicy::allow_all()),
        }
    }

    /// System prompt: body if present, else None.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt_body
            .as_deref()
            .filter(|s| !s.trim().is_empty())
    }
}
