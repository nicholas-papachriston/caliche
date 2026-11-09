//! Typed reasons attached to a [`crate::Decision`].

use serde::{Deserialize, Serialize};

/// Stable reason code. The human message is separate and must not carry secrets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// No deployment default and no matching rule.
    MissingPolicy,
    /// Preset or mode string was rejected at load time (also used in messages).
    UnknownPolicy,
    /// `Ask` with no human waiter.
    NoApprover,
    /// A facet checker failed.
    CheckerError,
    /// Approval answered by a principal that is not human.
    NonHumanApprover,
    /// Deployment layer produced deny.
    DeploymentDeny,
    /// Floor refused a vendor auth file or foreign keychain item.
    VendorAuth,
    /// Floor refused a vendor-subscription credential.
    ByoCredential,
    /// Floor refused a metadata or private egress target.
    Ssrf,
    /// Production unattended use hit `danger` or a noop sandbox.
    ProductionUnsafe,
    /// Deployment default supplied the effect.
    DeploymentDefault,
    /// A layer rule supplied the effect.
    MatchedRule,
    /// Session grant lowered a non-hard ask.
    SessionGrant,
    /// `AcceptAll` lowered a non-hard ask. Deny is never lowered.
    AcceptAll,
    /// `AcceptEdits` lowered a non-hard ask for an edit.
    AcceptEdits,
    /// `DenyAll` denied a non-read action.
    DenyAll,
    /// `Interactive` raised an allow to ask.
    Interactive,
    /// Shell plan was opaque, so the effect cannot be allow.
    OpaqueShell,
    /// A filesystem path is outside the workspace.
    OutsideWorkspace,
    /// A filesystem path form is refused.
    PathRejected,
    /// No workspace root, so a filesystem path cannot be placed.
    NoWorkspace,
    /// A write targets a `.git` entry inside the workspace.
    GitMetadata,
    /// The POSIX denylist matched a shell segment.
    PosixDenylist,
    /// The PowerShell denylist matched a command.
    PowerShellDenylist,
    /// The cmd denylist matched a command.
    CmdDenylist,
    /// The host cannot enforce a sandbox obligation.
    UnmetSandbox,
}

impl ReasonCode {
    /// Stable snake-case name for logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MissingPolicy => "missing_policy",
            Self::UnknownPolicy => "unknown_policy",
            Self::NoApprover => "no_approver",
            Self::CheckerError => "checker_error",
            Self::NonHumanApprover => "non_human_approver",
            Self::DeploymentDeny => "deployment_deny",
            Self::VendorAuth => "vendor_auth",
            Self::ByoCredential => "byo_credential",
            Self::Ssrf => "ssrf",
            Self::ProductionUnsafe => "production_unsafe",
            Self::DeploymentDefault => "deployment_default",
            Self::MatchedRule => "matched_rule",
            Self::SessionGrant => "session_grant",
            Self::AcceptAll => "accept_all",
            Self::AcceptEdits => "accept_edits",
            Self::DenyAll => "deny_all",
            Self::Interactive => "interactive",
            Self::OpaqueShell => "opaque_shell",
            Self::OutsideWorkspace => "outside_workspace",
            Self::PathRejected => "path_rejected",
            Self::NoWorkspace => "no_workspace",
            Self::GitMetadata => "git_metadata",
            Self::PosixDenylist => "posix_denylist",
            Self::PowerShellDenylist => "powershell_denylist",
            Self::CmdDenylist => "cmd_denylist",
            Self::UnmetSandbox => "unmet_sandbox",
        }
    }
}

/// Why [`crate::Decide::decide`] chose an effect.
///
/// The message is a static or owned string. It is not deserialized from
/// untrusted input, so it can stay a `'static` cow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reason {
    code: ReasonCode,
    message: beef::lean::Cow<'static, str>,
}

impl Reason {
    /// Build a reason from a code and a human message.
    #[must_use]
    pub fn new(code: ReasonCode, message: impl Into<beef::lean::Cow<'static, str>>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Machine-readable code.
    #[must_use]
    pub const fn code(&self) -> ReasonCode {
        self.code
    }

    /// Human message. Do not put secrets or raw credentials here.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_ref()
    }

    pub(crate) fn missing_policy() -> Self {
        Self::new(
            ReasonCode::MissingPolicy,
            "missing or unknown policy denies the action",
        )
    }

    pub(crate) fn no_approver() -> Self {
        Self::new(ReasonCode::NoApprover, "no human waiter; ask becomes deny")
    }

    pub(crate) fn checker_error() -> Self {
        Self::new(ReasonCode::CheckerError, "checker error denies the action")
    }

    pub(crate) fn non_human_approver() -> Self {
        Self::new(
            ReasonCode::NonHumanApprover,
            "only a human principal may answer an approval",
        )
    }

    pub(crate) fn vendor_auth() -> Self {
        Self::new(
            ReasonCode::VendorAuth,
            "floor denies reading vendor auth files",
        )
    }

    pub(crate) fn byo_credential() -> Self {
        Self::new(
            ReasonCode::ByoCredential,
            "floor denies vendor subscription credentials; bring your own key",
        )
    }

    pub(crate) fn ssrf() -> Self {
        Self::new(
            ReasonCode::Ssrf,
            "floor denies metadata, link-local, loopback, or private egress",
        )
    }

    pub(crate) fn production_unsafe() -> Self {
        Self::new(
            ReasonCode::ProductionUnsafe,
            "production refuses the danger preset and a noop sandbox",
        )
    }

    pub(crate) fn deployment_default() -> Self {
        Self::new(ReasonCode::DeploymentDefault, "deployment default")
    }

    pub(crate) fn matched_rule() -> Self {
        Self::new(ReasonCode::MatchedRule, "matched policy rule")
    }

    pub(crate) fn session_grant() -> Self {
        Self::new(ReasonCode::SessionGrant, "session grant")
    }

    pub(crate) fn accept_all() -> Self {
        Self::new(ReasonCode::AcceptAll, "accept-all lowered a non-hard ask")
    }

    pub(crate) fn opaque_shell() -> Self {
        Self::new(
            ReasonCode::OpaqueShell,
            "unreadable shell command is not allowed",
        )
    }

    pub(crate) fn outside_workspace() -> Self {
        Self::new(
            ReasonCode::OutsideWorkspace,
            "path is outside the workspace",
        )
    }

    pub(crate) fn path_rejected() -> Self {
        Self::new(ReasonCode::PathRejected, "filesystem path is rejected")
    }

    pub(crate) fn git_metadata() -> Self {
        Self::new(
            ReasonCode::GitMetadata,
            "writes to the repository's .git directory are denied",
        )
    }

    pub(crate) fn no_workspace() -> Self {
        Self::new(
            ReasonCode::NoWorkspace,
            "no workspace root; filesystem path is denied",
        )
    }

    pub(crate) fn posix_denylist() -> Self {
        Self::new(
            ReasonCode::PosixDenylist,
            "posix denylist refuses the command",
        )
    }

    pub(crate) fn powershell_denylist() -> Self {
        Self::new(
            ReasonCode::PowerShellDenylist,
            "powershell denylist refuses the command",
        )
    }

    pub(crate) fn cmd_denylist() -> Self {
        Self::new(ReasonCode::CmdDenylist, "cmd denylist refuses the command")
    }

    pub(crate) fn unmet_sandbox(network: bool) -> Self {
        let message = if network {
            "network not sandboxed"
        } else {
            "filesystem not sandboxed"
        };
        Self::new(ReasonCode::UnmetSandbox, message)
    }
}
