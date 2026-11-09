//! Shell analyzer trait.
//!
//! Dialect denylists live in [`crate::posix`], [`crate::pwsh`], and [`crate::cmd`].
//! This module is the gate and a stub:
//!
//! - When the dialect's content facet is **off**, [`CompiledPolicy::analyze_shell`](crate::CompiledPolicy::analyze_shell)
//!   returns `None` without calling [`ShellAnalyzer::analyze`]. No parse, no
//!   allocation of a plan.
//! - When the facet is **on**, both stages are required. The stub does not
//!   parse or resolve; it returns an opaque plan so the command cannot be allowed.

use crate::action::ShellPlan;
use crate::error::PolicyError;
use crate::posture::ShellDialect;

/// Dialect analyzer. Implementations produce facts, never effects.
pub trait ShellAnalyzer {
    /// Dialect this analyzer understands.
    fn dialect(&self) -> ShellDialect;

    /// Parse and resolve `command`.
    ///
    /// # Errors
    ///
    /// Implementations return an error for fatal analyzer failures. Callers
    /// treat that as a checker error (deny), not as allow.
    fn analyze(&self, command: &str) -> Result<ShellPlan, PolicyError>;
}

/// Stub analyzer. Does not parse or resolve.
///
/// When the facet is on, [`ShellAnalyzer::analyze`] still runs so tests can
/// count calls. The plan is opaque. When the facet is off, the policy does
/// not call this method.
#[derive(Clone, Debug)]
pub struct StubShellAnalyzer {
    dialect: ShellDialect,
}

impl StubShellAnalyzer {
    /// Stub for `dialect`.
    #[must_use]
    pub const fn new(dialect: ShellDialect) -> Self {
        Self { dialect }
    }
}

impl ShellAnalyzer for StubShellAnalyzer {
    fn dialect(&self) -> ShellDialect {
        self.dialect
    }

    fn analyze(&self, _command: &str) -> Result<ShellPlan, PolicyError> {
        Ok(ShellPlan {
            opaque: true,
            segment_count: 0,
            denied: false,
        })
    }
}
