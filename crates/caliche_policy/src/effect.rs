//! Allow / Ask / Deny lattice. Deny wins.

use serde::{Deserialize, Serialize};

/// Decision effect. The lattice order is `Allow < Ask < Deny`.
///
/// Joining two effects takes the maximum, so a single deny beats any allow.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// The action may proceed.
    Allow = 0,
    /// A human waiter must answer before the action proceeds.
    Ask = 1,
    /// The action is refused.
    Deny = 2,
}

impl Effect {
    /// Lattice join. Deny beats ask, and ask beats allow.
    #[must_use]
    pub const fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Deny, _) | (_, Self::Deny) => Self::Deny,
            (Self::Ask, _) | (_, Self::Ask) => Self::Ask,
            (Self::Allow, Self::Allow) => Self::Allow,
        }
    }

    /// Stable wire name (`allow`, `ask`, `deny`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Ask => "ask",
            Self::Deny => "deny",
        }
    }
}

impl core::fmt::Display for Effect {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}
