//! Orchestration primitives for Caliche and external agent harnesses.
//!
//! This crate has no dependency on the Caliche engine. Hosts can use it as a
//! pure coordination layer: task ACLs, privilege intersection, output quarantine,
//! and dispatch to any [`HarnessRuntime`] (local agents, Pi, Hermes, or a custom
//! subprocess).
//!
//! The root `caliche` crate re-exports these types from
//! `caliche::orchestration` and supplies a local-agent runtime adapter.

pub mod acl;
pub mod command;
pub mod external;
pub mod harness;
pub mod privilege;
pub mod quarantine;

pub use acl::{GlobRule, InvocationDecision, PolicyEffect, TaskInvocationPolicy, glob_match};
pub use command::{
    CommandPolicy, HARNESS_COMMAND_ALLOWLIST_ENV, command_allowed_by_allowlist, is_forbidden_shell,
};
pub use external::{ExternalCliRuntime, ExternalCliSpec, PromptPlacement};
pub use harness::{HarnessRequest, HarnessResponse, HarnessRuntime, HarnessUsage, StaticRuntime};
pub use privilege::{
    AgentPrivileges, effective_tools, intersect_tools, rejected_tools, tool_ceiling,
};
pub use quarantine::{
    UNTRUSTED_BEGIN, UNTRUSTED_END, is_exactly_quarantined, is_quarantined, quarantine_once,
    quarantine_untrusted,
};
