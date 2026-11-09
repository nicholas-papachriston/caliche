//! Rules and the matchers they carry.
//!
//! A rule names an effect. It is not a second `decide`. Only
//! [`crate::CompiledPolicy`] implements [`crate::Decide`].

use crate::action::ToolClass;
use crate::effect::Effect;
use crate::posture::{Layer, ShellDialect};
use serde::{Deserialize, Serialize};

/// Where a rule was loaded from. Shown later by `policy show`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleSource {
    /// Compiled into the binary.
    Builtin {
        /// Stable id such as `floor.vendor_auth`.
        id: &'static str,
    },
    /// File line.
    File {
        /// Path.
        path: String,
        /// 1-based line.
        line: u32,
    },
    /// Environment variable shim.
    Env {
        /// Variable name.
        name: &'static str,
    },
}

/// Pointer stored on a [`crate::Decision`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleRef {
    /// Layer that contributed the winning effect.
    pub layer: Layer,
    /// Provenance (`builtin:floor.vendor_auth`, `file:line`, `env:VAR`).
    pub source: String,
}

impl RuleRef {
    pub(crate) fn builtin(layer: Layer, id: &'static str) -> Self {
        Self {
            layer,
            source: format!("builtin:{id}"),
        }
    }
}

/// What a rule looks at. Matchers produce a yes/no. They do not produce effects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Matcher {
    /// Tool family.
    ToolClass(ToolClass),
    /// Exact tool name.
    ToolName(String),
    /// `ApprovalAnswer` from a non-human principal.
    NonHumanApproval,
    /// Vendor auth path or a keychain item Caliche does not own.
    VendorAuth,
    /// `CredentialClass::VendorSubscription`.
    VendorSubscription,
    /// Metadata, link-local, loopback, or private host.
    Ssrf,
    /// Facet failure.
    CheckerError,
    /// Shell content for one dialect. Sets that dialect's facet bit.
    ShellContent {
        /// Dialect the content rule applies to.
        dialect: ShellDialect,
    },
}

impl Matcher {
    pub(crate) const fn is_shell_content(&self) -> bool {
        matches!(self, Self::ShellContent { .. })
    }

    pub(crate) const fn specificity(&self) -> u8 {
        match self {
            Self::ToolClass(_) => 1,
            Self::ToolName(_) => 2,
            Self::ShellContent { .. } => 3,
            Self::NonHumanApproval
            | Self::VendorAuth
            | Self::VendorSubscription
            | Self::Ssrf
            | Self::CheckerError => 4,
        }
    }
}

/// One compiled rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// Stable id within the snapshot.
    pub id: String,
    /// Layer that owns the rule.
    pub layer: Layer,
    /// Effect when the matcher hits.
    pub effect: Effect,
    /// When true, session overlays cannot lower an ask produced by this rule.
    pub hard: bool,
    /// Provenance.
    pub source: RuleSource,
    /// Matcher.
    pub matcher: Matcher,
}

impl Rule {
    /// Deny a tool class at a layer.
    #[must_use]
    pub fn deny_tool_class(layer: Layer, class: ToolClass, id: impl Into<String>) -> Self {
        Self::effect_tool_class(layer, class, Effect::Deny, id)
    }

    /// Ask for a tool class at a layer.
    #[must_use]
    pub fn ask_tool_class(layer: Layer, class: ToolClass, id: impl Into<String>) -> Self {
        Self::effect_tool_class(layer, class, Effect::Ask, id)
    }

    /// Allow a tool class at a layer.
    ///
    /// Profile allows are a compile error. Project allows are a compile error
    /// unless the deployment opted in. Neither can lift a deployment deny.
    #[must_use]
    pub fn allow_tool_class(layer: Layer, class: ToolClass, id: impl Into<String>) -> Self {
        Self::effect_tool_class(layer, class, Effect::Allow, id)
    }

    /// Shell content rule. Compiling one sets the dialect facet bit.
    #[must_use]
    pub fn shell_content(
        layer: Layer,
        dialect: ShellDialect,
        effect: Effect,
        id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            layer,
            effect,
            hard: effect == Effect::Deny,
            source: RuleSource::Builtin { id: "caller" },
            matcher: Matcher::ShellContent { dialect },
        }
    }

    /// Project allow loaded from a file path. Used with `trust_project_allows`.
    #[must_use]
    pub fn project_allow_file(class: ToolClass, path: impl Into<String>, line: u32) -> Self {
        let path = path.into();
        Self {
            id: format!("project-allow:{path}:{line}"),
            layer: Layer::Project,
            effect: Effect::Allow,
            hard: false,
            source: RuleSource::File { path, line },
            matcher: Matcher::ToolClass(class),
        }
    }

    fn effect_tool_class(
        layer: Layer,
        class: ToolClass,
        effect: Effect,
        id: impl Into<String>,
    ) -> Self {
        let id = id.into();
        Self {
            id,
            layer,
            effect,
            hard: effect == Effect::Deny,
            source: RuleSource::Builtin { id: "caller" },
            matcher: Matcher::ToolClass(class),
        }
    }

    pub(crate) fn floor(id: &'static str, effect: Effect, hard: bool, matcher: Matcher) -> Self {
        Self {
            id: id.to_owned(),
            layer: Layer::Floor,
            effect,
            hard,
            source: RuleSource::Builtin { id },
            matcher,
        }
    }

    pub(crate) fn rule_ref(&self) -> RuleRef {
        let source = match &self.source {
            RuleSource::Builtin { id } => format!("builtin:{id}"),
            RuleSource::File { path, line } => format!("{path}:{line}"),
            RuleSource::Env { name } => format!("env:{name}"),
        };
        RuleRef {
            layer: self.layer,
            source,
        }
    }
}

/// Rules a [`crate::PolicySource`] contributes at one layer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuleSet {
    /// Rules. Every rule's layer must match the source's declared layer.
    pub rules: Vec<Rule>,
}

impl RuleSet {
    /// Empty set.
    #[must_use]
    pub const fn new() -> Self {
        Self { rules: Vec::new() }
    }
}
