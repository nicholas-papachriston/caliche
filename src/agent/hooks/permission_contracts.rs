//! Canonical permission/approval wire mapping and parse helpers.
//!
//! Hosts (CLI/API) and plugin adapters should call these instead of
//! re-implementing permission-mode spellings or decision JSON shapes.

use crate::agent::hooks::approval::ApprovalOutcome;
use crate::agent::hooks::execpolicy::BashDecision;
use crate::agent::hooks::permission::{PermissionDecision, PermissionMode};
use crate::harness::config::PermissionModeConfig;
use beef::lean::Cow;
use caliche_contracts::permission::{
    ApprovalOutcomeDto, ApprovalPromptChoiceDto, PermissionDecisionDto, PermissionModeDto,
};

/// Parse a permission-mode string into harness config.
///
/// The spellings are `default`, `accept-edits`, `accept-all`, `deny-all`,
/// and `interactive`. Empty/`None` maps to [`PermissionModeConfig::Default`].
/// Unknown tokens are errors. The token table lives in
/// [`caliche_policy::parse_mode_overlay`].
///
/// # Errors
///
/// Returns an error when `value` is not a known permission mode.
pub fn parse_permission_mode_str(value: Option<&str>) -> Result<PermissionModeConfig, String> {
    caliche_policy::parse_mode_overlay(value)
        .map(permission_mode_from_overlay)
        .map_err(|error| match error {
            caliche_policy::PolicyError::UnknownPermissionMode(token) => format!(
                "Invalid permission mode: {token}. Valid modes: default, accept-edits, accept-all, deny-all, interactive"
            ),
            other => other.to_string(),
        })
}

const fn permission_mode_from_overlay(
    overlay: caliche_policy::ModeOverlay,
) -> PermissionModeConfig {
    match overlay {
        caliche_policy::ModeOverlay::Default => PermissionModeConfig::Default,
        caliche_policy::ModeOverlay::AcceptEdits => PermissionModeConfig::AcceptEdits,
        caliche_policy::ModeOverlay::AcceptAll => PermissionModeConfig::AcceptAll,
        caliche_policy::ModeOverlay::DenyAll => PermissionModeConfig::DenyAll,
        caliche_policy::ModeOverlay::Interactive => PermissionModeConfig::Interactive,
    }
}

/// Map harness config mode ↔ contracts DTO.
#[must_use]
pub const fn permission_mode_config_to_dto(mode: PermissionModeConfig) -> PermissionModeDto {
    match mode {
        PermissionModeConfig::Default => PermissionModeDto::Default,
        PermissionModeConfig::AcceptEdits => PermissionModeDto::AcceptEdits,
        PermissionModeConfig::AcceptAll => PermissionModeDto::AcceptAll,
        PermissionModeConfig::DenyAll => PermissionModeDto::DenyAll,
        PermissionModeConfig::Interactive => PermissionModeDto::Interactive,
    }
}

/// Map contracts DTO ↔ harness config mode.
#[must_use]
pub const fn permission_mode_dto_to_config(mode: PermissionModeDto) -> PermissionModeConfig {
    match mode {
        PermissionModeDto::Default => PermissionModeConfig::Default,
        PermissionModeDto::AcceptEdits => PermissionModeConfig::AcceptEdits,
        PermissionModeDto::AcceptAll => PermissionModeConfig::AcceptAll,
        PermissionModeDto::DenyAll => PermissionModeConfig::DenyAll,
        PermissionModeDto::Interactive => PermissionModeConfig::Interactive,
    }
}

/// Map harness config mode ↔ runtime [`PermissionMode`].
#[must_use]
pub const fn permission_mode_config_to_runtime(mode: PermissionModeConfig) -> PermissionMode {
    match mode {
        PermissionModeConfig::Default => PermissionMode::Default,
        PermissionModeConfig::AcceptEdits => PermissionMode::AcceptEdits,
        PermissionModeConfig::AcceptAll => PermissionMode::AcceptAll,
        PermissionModeConfig::DenyAll => PermissionMode::DenyAll,
        PermissionModeConfig::Interactive => PermissionMode::Interactive,
    }
}

/// Map runtime decision → contracts DTO.
#[must_use]
pub fn permission_decision_to_dto(decision: &PermissionDecision) -> PermissionDecisionDto {
    match decision {
        PermissionDecision::Allow => PermissionDecisionDto::Allow,
        PermissionDecision::Deny { reason } => PermissionDecisionDto::Deny {
            reason: reason.to_string(),
        },
        PermissionDecision::RequireApproval {
            reason,
            proposed_amendment,
        } => PermissionDecisionDto::RequireApproval {
            reason: reason.to_string(),
            proposed_amendment: proposed_amendment.clone(),
        },
    }
}

/// Map contracts DTO → runtime decision.
#[must_use]
pub fn permission_decision_from_dto(dto: PermissionDecisionDto) -> PermissionDecision {
    match dto {
        PermissionDecisionDto::Allow => PermissionDecision::Allow,
        PermissionDecisionDto::Deny { reason } => PermissionDecision::Deny {
            reason: Cow::owned(reason),
        },
        PermissionDecisionDto::RequireApproval {
            reason,
            proposed_amendment,
        } => PermissionDecision::RequireApproval {
            reason: Cow::owned(reason),
            proposed_amendment,
        },
    }
}

/// Parse a JSON value into a runtime permission decision.
///
/// Supports:
/// - `"allow"` string
/// - tagged objects `{ "decision": "allow"|"deny"|"require_approval", ... }`
/// - nested `{ "permission_decision": { ... } }`
pub fn permission_decision_from_json_value(
    value: &serde_json::Value,
) -> Result<PermissionDecision, String> {
    if let Some(s) = value.as_str() {
        return match s.trim().to_ascii_lowercase().as_str() {
            "allow" => Ok(PermissionDecision::Allow),
            "deny" => Ok(PermissionDecision::deny("Hook denied")),
            "require_approval" => Ok(PermissionDecision::require_approval(
                "Approval required",
                None,
            )),
            other => Err(format!("Unknown permission decision: {other}")),
        };
    }

    let root = if let Some(nested) = value.get("permission_decision") {
        nested
    } else {
        value
    };

    if let Ok(dto) = serde_json::from_value::<PermissionDecisionDto>(root.clone()) {
        return Ok(permission_decision_from_dto(dto));
    }

    // Untagged fallback: { "decision": "...", "reason": "...", "proposed_amendment": [...] }
    let decision = root
        .get("decision")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "missing decision field".to_string())?;
    let reason = root
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("Hook denied");
    let proposed_amendment = root.get("proposed_amendment").and_then(|v| {
        v.as_array().map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
    });
    match decision.trim().to_ascii_lowercase().as_str() {
        "allow" => Ok(PermissionDecision::Allow),
        "deny" => Ok(PermissionDecision::deny(reason.to_string())),
        "require_approval" => Ok(PermissionDecision::require_approval(
            reason.to_string(),
            proposed_amendment,
        )),
        other => Err(format!("Unknown permission decision: {other}")),
    }
}

/// Map exec-policy bash decision into a runtime permission decision.
#[must_use]
pub fn bash_decision_to_permission_decision(
    decision: BashDecision,
    justification: Option<String>,
    proposed_amendment: Option<Vec<String>>,
) -> PermissionDecision {
    match decision {
        BashDecision::Allow => PermissionDecision::Allow,
        BashDecision::Forbidden => {
            let reason = justification.unwrap_or_else(|| "Forbidden by policy".to_string());
            PermissionDecision::deny(reason)
        }
        BashDecision::Prompt => {
            let reason = justification.map_or_else(
                || "Requires user approval".to_string(),
                |j| format!("Requires user approval: {j}"),
            );
            PermissionDecision::require_approval(reason, proposed_amendment)
        }
    }
}

/// Map approval outcome ↔ contracts DTO.
#[must_use]
pub fn approval_outcome_to_dto(outcome: &ApprovalOutcome) -> ApprovalOutcomeDto {
    match outcome {
        ApprovalOutcome::Allow => ApprovalOutcomeDto::Allow,
        ApprovalOutcome::AllowAndPersist { rule_prefix } => ApprovalOutcomeDto::AllowAndPersist {
            rule_prefix: rule_prefix.clone(),
        },
        ApprovalOutcome::Deny => ApprovalOutcomeDto::Deny,
    }
}

/// Map contracts DTO → approval outcome.
#[must_use]
pub fn approval_outcome_from_dto(dto: ApprovalOutcomeDto) -> ApprovalOutcome {
    match dto {
        ApprovalOutcomeDto::Allow => ApprovalOutcome::Allow,
        ApprovalOutcomeDto::AllowAndPersist { rule_prefix } => {
            ApprovalOutcome::AllowAndPersist { rule_prefix }
        }
        ApprovalOutcomeDto::Deny => ApprovalOutcome::Deny,
    }
}

/// Map a compact y/n/a prompt choice into an approval outcome.
#[must_use]
pub fn approval_outcome_from_prompt_choice(
    choice: ApprovalPromptChoiceDto,
    proposed_amendment: Option<&[String]>,
) -> ApprovalOutcome {
    match choice {
        ApprovalPromptChoiceDto::Yes => ApprovalOutcome::Allow,
        ApprovalPromptChoiceDto::No => ApprovalOutcome::Deny,
        ApprovalPromptChoiceDto::Always => ApprovalOutcome::AllowAndPersist {
            rule_prefix: proposed_amendment
                .map(<[std::string::String]>::to_vec)
                .unwrap_or_default(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_permission_mode_spellings() {
        assert_eq!(
            parse_permission_mode_str(Some("accept-edits")).expect("parse"),
            PermissionModeConfig::AcceptEdits
        );
        assert_eq!(
            parse_permission_mode_str(Some("accept-all")).expect("parse"),
            PermissionModeConfig::AcceptAll
        );
        assert_eq!(
            parse_permission_mode_str(Some("deny-all")).expect("parse"),
            PermissionModeConfig::DenyAll
        );
        assert_eq!(
            parse_permission_mode_str(Some("interactive")).expect("parse"),
            PermissionModeConfig::Interactive
        );
        assert_eq!(
            parse_permission_mode_str(Some("default")).expect("parse"),
            PermissionModeConfig::Default
        );
        assert!(parse_permission_mode_str(Some("accept_all")).is_err());
        assert!(parse_permission_mode_str(Some("denyall")).is_err());
        assert!(parse_permission_mode_str(Some("prompt")).is_err());
        let unknown = parse_permission_mode_str(Some("nope")).expect_err("unknown mode");
        assert!(unknown.contains("nope"), "{unknown}");
        assert!(unknown.contains("default"), "{unknown}");
        assert!(unknown.contains("accept-edits"), "{unknown}");
        assert!(unknown.contains("accept-all"), "{unknown}");
        assert!(unknown.contains("deny-all"), "{unknown}");
        assert!(unknown.contains("interactive"), "{unknown}");
        assert!(parse_permission_mode_str(Some("yolo")).is_err());
        assert_eq!(
            parse_permission_mode_str(None).expect("missing"),
            PermissionModeConfig::Default
        );
        assert_eq!(
            parse_permission_mode_str(Some("")).expect("empty"),
            PermissionModeConfig::Default
        );
        assert_eq!(
            parse_permission_mode_str(Some("  ")).expect("blank"),
            PermissionModeConfig::Default
        );
    }

    #[test]
    fn permission_decision_json_supports_nested_and_require_approval() {
        let nested = serde_json::json!({
            "permission_decision": {
                "decision": "require_approval",
                "reason": "ask",
                "proposed_amendment": ["git", "push"]
            }
        });
        let decision = permission_decision_from_json_value(&nested).expect("parse");
        assert!(decision.is_approval_required());
        assert_eq!(
            decision.proposed_amendment(),
            Some([String::from("git"), String::from("push")].as_slice())
        );

        let allow =
            permission_decision_from_json_value(&serde_json::json!("allow")).expect("allow");
        assert!(allow.is_allowed());
    }

    #[test]
    fn bash_prompt_maps_to_require_approval() {
        let decision = bash_decision_to_permission_decision(
            BashDecision::Prompt,
            Some("network".into()),
            Some(vec!["curl".into()]),
        );
        assert!(decision.is_approval_required());
        assert!(
            decision
                .approval_reason()
                .is_some_and(|r| r.contains("network"))
        );
        assert_eq!(
            decision.proposed_amendment(),
            Some([String::from("curl")].as_slice())
        );
    }

    #[test]
    fn approval_prompt_choice_always_persists_prefix() {
        let outcome = approval_outcome_from_prompt_choice(
            ApprovalPromptChoiceDto::Always,
            Some(&["rm".into(), "-rf".into()]),
        );
        match outcome {
            ApprovalOutcome::AllowAndPersist { rule_prefix } => {
                assert_eq!(rule_prefix, vec!["rm", "-rf"]);
            }
            other => panic!("expected AllowAndPersist, got {other:?}"),
        }
    }
}
