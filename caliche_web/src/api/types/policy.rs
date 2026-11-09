//! Permissions page payloads for `/api/policy`.

use serde::{Deserialize, Serialize};

/// Deployment document written by PUT `/api/policy`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyDocument {
    /// File version. The server accepts 1.
    pub version: u64,
    /// Preset name.
    pub preset: String,
    /// Optional default-effect override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    /// Production profile.
    #[serde(default, skip_serializing_if = "is_false")]
    pub production: bool,
    /// Sandbox override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxDocument>,
    /// Tool rules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolRuleDocument>,
}

/// `[sandbox]` table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SandboxDocument {
    /// Requirement token.
    pub require: String,
}

/// One tool rule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolRuleDocument {
    /// Tool family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    /// Exact tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Effect token.
    pub effect: String,
}

/// One effective-rule row.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct EffectiveRule {
    /// Rule id.
    pub id: String,
    /// Layer name.
    pub layer: String,
    /// Effect name.
    pub effect: String,
    /// Provenance.
    pub source: String,
    /// Matcher label.
    pub matcher: String,
}

/// GET `/api/policy/effective` body.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct EffectivePolicy {
    /// False when the file has not been written.
    pub file_present: bool,
    /// Path the server loaded.
    pub source_path: String,
    /// Document a save writes back.
    pub document: PolicyDocument,
    /// Compiled preset.
    pub preset: String,
    /// Compiled default effect.
    pub default_effect: String,
    /// Compiled sandbox requirement.
    pub sandbox: String,
    /// Host label for the panel.
    pub host: String,
    /// Capability panel text.
    pub capability_panel: String,
    /// Effective rules.
    pub rules: Vec<EffectiveRule>,
    /// Session grants.
    pub grants: Vec<EffectiveRule>,
    /// Pending approval ids.
    pub pending: Vec<String>,
    /// Preset the live tool gate uses.
    pub live_gate: String,
}

const fn is_false(value: &bool) -> bool {
    !*value
}

#[cfg(test)]
mod tests {
    use super::EffectivePolicy;

    #[test]
    fn effective_policy_json_keeps_the_preset() {
        let raw = r#"{
            "file_present": false,
            "source_path": "policy.toml",
            "document": { "version": 1, "preset": "locked" },
            "preset": "locked",
            "default_effect": "deny",
            "sandbox": "required",
            "host": "macos_seatbelt",
            "capability_panel": "network: sandboxed\n",
            "rules": [],
            "grants": [],
            "pending": [],
            "live_gate": "trusted"
        }"#;
        let policy: EffectivePolicy = serde_json::from_str(raw).expect("policy json");
        assert_eq!(policy.preset, "locked");
        assert_eq!(policy.live_gate, "trusted");
        assert!(!policy.file_present);
    }
}
