//! Permission system for controlling tool execution.

use beef::lean::Cow;
use serde::{Deserialize, Serialize};

use crate::utils::beef_helpers::{deserialize_cow_str, serialize_cow_str};

/// Permission decision for tool execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Allow the tool to execute
    Allow,
    /// Deny the tool execution with a reason
    Deny {
        /// Reason for denial
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        reason: Cow<'static, str>,
    },
    /// Require user approval before executing; blocks until resolved
    RequireApproval {
        /// Reason shown to the user
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        reason: Cow<'static, str>,
        /// Optional command prefix to append as allow rule if user chooses "always"
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proposed_amendment: Option<Vec<String>>,
    },
}

impl PermissionDecision {
    /// Create an allow decision
    #[must_use]
    pub const fn allow() -> Self {
        Self::Allow
    }

    /// Create a deny decision with a reason
    pub fn deny(reason: impl Into<Cow<'static, str>>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Create a require-approval decision
    pub fn require_approval(
        reason: impl Into<Cow<'static, str>>,
        proposed_amendment: Option<Vec<String>>,
    ) -> Self {
        Self::RequireApproval {
            reason: reason.into(),
            proposed_amendment,
        }
    }

    /// Check if the decision is to allow
    #[must_use]
    pub const fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Check if the decision is to deny
    #[must_use]
    pub const fn is_denied(&self) -> bool {
        matches!(self, Self::Deny { .. })
    }

    /// Check if approval is required (blocks until resolved)
    #[must_use]
    pub const fn is_approval_required(&self) -> bool {
        matches!(self, Self::RequireApproval { .. })
    }

    /// Get the denial reason if denied
    #[must_use]
    pub fn denial_reason(&self) -> Option<&str> {
        match self {
            Self::Deny { reason } => Some(reason.as_ref()),
            Self::Allow | Self::RequireApproval { .. } => None,
        }
    }

    /// Get the approval reason if approval required
    #[must_use]
    pub fn approval_reason(&self) -> Option<&str> {
        match self {
            Self::RequireApproval { reason, .. } => Some(reason.as_ref()),
            Self::Allow | Self::Deny { .. } => None,
        }
    }

    /// Get the proposed amendment if approval required
    #[must_use]
    pub const fn proposed_amendment(&self) -> Option<&[String]> {
        match self {
            Self::RequireApproval {
                proposed_amendment: Some(prefix),
                ..
            } => Some(prefix.as_slice()),
            Self::RequireApproval {
                proposed_amendment: None,
                ..
            }
            | Self::Allow
            | Self::Deny { .. } => None,
        }
    }
}

/// Permission mode for automatic permission handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionMode {
    /// Default mode - use hooks for permission decisions
    #[default]
    Default,
    /// Auto-accept file edit operations
    AcceptEdits,
    /// Auto-accept all tool calls
    AcceptAll,
    /// Deny all tool calls
    DenyAll,
    /// Interactive mode - prompt for dangerous operations
    Interactive,
}
