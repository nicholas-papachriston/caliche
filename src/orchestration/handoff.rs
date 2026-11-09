//! Orchestration-facing extensions for handoff/turn contracts.
//!
//! Ownership of the core contracts lives in `agent::kernel::turn`.

pub use crate::agent::kernel::turn::{Handoff, TurnResult};

impl Handoff {
    /// Convert this handoff into delegation options for flow execution.
    pub fn to_delegation_options(&self) -> super::coordinator::DelegationOptions {
        super::coordinator::DelegationOptions {
            persona: self.persona.clone(),
            tools: self.tools.clone(),
            instructions: self.instructions.clone(),
        }
    }
}
