//! Wire DTOs for permission modes, hook decisions, and approval outcomes.

use serde::{Deserialize, Serialize};

/// Client/server permission mode (mirrors harness `PermissionModeConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionModeDto {
    #[default]
    Default,
    AcceptEdits,
    AcceptAll,
    DenyAll,
    Interactive,
}

/// Hook/middleware permission decision on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum PermissionDecisionDto {
    Allow,
    Deny {
        reason: String,
    },
    RequireApproval {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proposed_amendment: Option<Vec<String>>,
    },
}

/// User resolution of a require-approval prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ApprovalOutcomeDto {
    Allow,
    AllowAndPersist { rule_prefix: Vec<String> },
    Deny,
}

/// Compact CLI/UI prompt choice before mapping to [`ApprovalOutcomeDto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPromptChoiceDto {
    Yes,
    No,
    Always,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_decision_dto_roundtrips_require_approval() {
        let dto = PermissionDecisionDto::RequireApproval {
            reason: "dangerous".into(),
            proposed_amendment: Some(vec!["git".into(), "status".into()]),
        };
        let value = serde_json::to_value(&dto).expect("serialize");
        assert_eq!(value["decision"], "require_approval");
        assert_eq!(value["reason"], "dangerous");
        let decoded: PermissionDecisionDto = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded, dto);
    }

    #[test]
    fn approval_outcome_dto_roundtrips_persist() {
        let dto = ApprovalOutcomeDto::AllowAndPersist {
            rule_prefix: vec!["rm".into()],
        };
        let value = serde_json::to_value(&dto).expect("serialize");
        assert_eq!(value["outcome"], "allow_and_persist");
        let decoded: ApprovalOutcomeDto = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded, dto);
    }

    #[test]
    fn permission_mode_dto_serde_kebab_case() {
        let value = serde_json::to_value(PermissionModeDto::AcceptEdits).expect("serialize");
        assert_eq!(value, "accept-edits");
        let decoded: PermissionModeDto =
            serde_json::from_value(serde_json::json!("deny-all")).expect("deserialize");
        assert_eq!(decoded, PermissionModeDto::DenyAll);
    }
}
