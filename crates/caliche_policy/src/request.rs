//! Input to the sealed decider.

use crate::action::Action;
use crate::posture::{HostCapability, ModeOverlay};
use crate::principal::{ApprovalWaiter, Principal};
use serde_json::Value;

/// Session fields that affect a decision. The overlay never lifts a deny.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionScope {
    /// Session id.
    pub session_id: String,
    /// Profile id, when one is selected.
    pub profile_id: Option<String>,
    /// Mode overlay for this call.
    pub overlay: ModeOverlay,
    /// Production profile (`CALICHE_ENV=production`).
    pub production: bool,
    /// Host isolation the caller claims. Not probed by `decide`.
    pub host: HostCapability,
    /// Included in [`crate::ActionBinding`] so a reload invalidates approvals.
    pub policy_version: u64,
    /// Working directory included in the binding.
    pub cwd: String,
}

impl SessionScope {
    /// Non-production session with the default overlay.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            profile_id: None,
            overlay: ModeOverlay::Default,
            production: false,
            host: HostCapability::Unspecified,
            policy_version: 1,
            cwd: ".".to_owned(),
        }
    }
}

/// One authorization request.
///
/// `raw_args` is hashed into the binding and is not logged by shadow mode.
pub struct Request<'a> {
    /// Who is asking.
    pub principal: &'a Principal,
    /// Whether a human can answer.
    pub waiter: ApprovalWaiter,
    /// Session scope.
    pub session: &'a SessionScope,
    /// Action.
    pub action: Action,
    /// Exact tool arguments. Used only for the binding hash.
    pub raw_args: &'a Value,
}

impl<'a> Request<'a> {
    /// Assemble a request.
    #[must_use]
    pub const fn new(
        principal: &'a Principal,
        waiter: ApprovalWaiter,
        session: &'a SessionScope,
        action: Action,
        raw_args: &'a Value,
    ) -> Self {
        Self {
            principal,
            waiter,
            session,
            action,
            raw_args,
        }
    }
}
