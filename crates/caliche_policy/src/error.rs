//! Load and parse errors for the policy crate.
//!
//! Unknown presets, permission modes, and globs fail here. They do not become
//! a silent [`crate::Effect::Allow`].

use thiserror::Error;

/// Failure while parsing or compiling a policy snapshot.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// Permission-mode string is not a known overlay.
    #[error("unknown permission mode '{0}'")]
    UnknownPermissionMode(String),
    /// Preset name is not one of the named presets.
    #[error("unknown policy preset '{0}'")]
    UnknownPreset(String),
    /// Effect string is not `allow`, `ask`, or `deny`.
    #[error("unknown effect '{0}'")]
    UnknownEffect(String),
    /// A profile rule tried to grant `allow`.
    #[error("profile allow rule '{0}' would raise rights")]
    ProfileAllowRaisesRights(String),
    /// A project rule tried to grant `allow` without a deployment opt-in.
    #[error("project allow rule '{0}' would raise rights; deployment did not opt in")]
    ProjectAllowRaisesRights(String),
    /// The floor layer cannot grant new allows.
    #[error("floor rule '{0}' cannot allow")]
    FloorCannotAllow(String),
    /// A [`crate::PolicySource`] tried to supply the compiled-in floor.
    #[error("policy source cannot supply the floor layer")]
    SourceCannotSupplyFloor,
    /// Rules returned by a source did not all use the source's declared layer.
    #[error("policy source layer does not match its rules")]
    SourceLayerMismatch,
    /// Glob used a construct this crate does not accept.
    #[error("bad glob '{0}'")]
    BadGlob(String),
    /// `danger` was selected while the deployment is a production profile.
    #[error("preset danger is refused in production")]
    DangerRefusedInProduction,
    /// `noop` sandbox was selected while the deployment is a production profile.
    #[error("noop sandbox is refused in production")]
    NoopRefusedInProduction,
    /// Pending id was empty or missing the `apr_` prefix.
    #[error("invalid pending id '{0}'")]
    InvalidPendingId(String),
    /// `policy.toml` `version` is not 1.
    #[error("unsupported policy.toml version {0}")]
    UnknownPolicyVersion(u64),
    /// `policy.toml` failed to parse or named an unknown field.
    #[error("invalid policy.toml: {0}")]
    InvalidPolicy(String),
    /// Sandbox requirement token is not a known value.
    #[error("unknown sandbox requirement '{0}'")]
    UnknownSandbox(String),
    /// Tool class token is not a known family.
    #[error("unknown tool class '{0}'")]
    UnknownToolClass(String),
    /// A tool rule named both `class` and `name`, or named neither.
    #[error("tool rule must set class or name, not both")]
    ToolRuleTarget,
    /// Host token is not a known capability.
    #[error("unknown host capability '{0}'")]
    UnknownHost(String),
    /// The deployment file could not be written.
    #[error("failed to write policy.toml: {0}")]
    PolicyWrite(String),
}
