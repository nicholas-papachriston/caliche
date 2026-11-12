//! Typed policy core for Caliche.
//!
//! The crate ships types, a compiled-in floor, fail-closed parsing, and a
//! shadow compare. Set `CALICHE_POLICY_SHADOW=1` to log differences.
//! The tool path calls [`evaluate_tool_gate`] before hooks. That gate uses
//! [`Preset::Trusted`] plus the session overlay, so a floor deny always blocks
//! and [`ModeOverlay::Interactive`] asks for a side effect.
//! Hooks still run after an allow and after a settled ask.
//! [`resolve_fs_path`] resolves filesystem and patch paths. A path outside
//! the workspace asks. A rejected path form denies.
//!
//! MCP connect and MCP tool calls also refuse [`Effect::Deny`] from this
//! snapshot. [`Effect::Ask`] does not block those calls.
//!
//! # Limits
//!
//! - POSIX, `PowerShell`, and cmd denylists run when that dialect's facet is
//!   on (`standard`, `locked`, `review`). [`Preset::Trusted`] leaves the
//!   facets off, so the analyzers are not called.
//! - [`approval::ApprovalBroker`] records one pending id, takes one human
//!   answer, and rechecks the action binding. There is no approval UI and no
//!   HTTP approval route.
//! - There is no Windows kernel sandbox.
//! - Unattended Windows requires Docker or WSL2 ([`HostCapability::WindowsDockerOrWsl2`]).
//!   [`HostCapability::WindowsJobOnly`] does not confine files or network.
//!   [`policy_show`] prints that gap. `decide` asks or denies shell and code
//!   execution when the session names a host that cannot meet the obligation.
//!   An unnamed host does not change the effect.
//! - Workstation preset default is [`Preset::Standard`]. New deployments use
//!   [`Preset::Locked`]. [`Preset::Trusted`] leaves the shell analyzer off and
//!   keeps the workspace-write sandbox on.
//! - Bring-your-own provider keys only. The floor denies vendor auth files and
//!   [`CredentialClass::VendorSubscription`].
//! - [`parse_policy_toml`] loads version-1 `policy.toml`. Unknown keys fail
//!   the load. [`open_policy_file`] and [`render_effective`] feed `policy show`
//!   and the Permissions page. The live tool gate stays on [`Preset::Trusted`].
//!
//! # Examples
//!
//! ```
//! use caliche_policy::{Effect, PolicyBuilder, Preset};
//!
//! let policy = PolicyBuilder::preset(Preset::new_deploy_default())
//!     .build()
//!     .expect("locked preset compiles");
//! assert_eq!(policy.deployment_default(), Some(Effect::Deny));
//! assert_eq!(Preset::workstation_default(), Preset::Standard);
//! ```

#![deny(missing_docs)]

mod action;
mod analyzer;
mod approval;
mod capability;
mod cmd;
mod credential;
mod document;
mod effect;
mod error;
mod fs;
mod git_discovery;
mod ids;
mod matching;
mod parse;
mod paths;
mod policy;
mod posix;
mod posture;
mod principal;
mod pwsh;
mod reason;
mod request;
mod rule;
mod shadow;
mod source;
mod tool_name;

pub use action::{
    Action, FsOp, HttpMethod, PathPlace, ResolvedPath, ShellPlan, ToolClass, ToolFacts,
};
pub use analyzer::{ShellAnalyzer, StubShellAnalyzer};
pub use approval::{
    APPROVAL_TTL, ActionCard, ApprovalAnswer, ApprovalBroker, ApprovalRequest, BrokerError,
    OpenedApproval, Settlement,
};
pub use capability::{
    ObligationGap, host_enforces_filesystem, host_enforces_network, host_enforces_process,
    obligation_gap, policy_show,
};
pub use cmd::CmdShellAnalyzer;
pub use credential::CredentialClass;
pub use document::{
    EffectivePolicy, EffectiveRule, PolicyDocument, SandboxDocument, ToolRuleDocument,
    default_policy_path, host_for_process, host_for_this_process, open_policy_file, parse_host,
    parse_policy_toml, render_effective, view_policy, write_policy_file,
};
pub use effect::Effect;
pub use error::PolicyError;
pub use fs::{
    ADD_FILE_MARKER, COMMONDIR_FILE, CONFIG_FILE, DELETE_FILE_MARKER, FSMONITOR_KEY, FsFault,
    FsRoots, GIT_DIR_NAME, GITFILE_PREFIX, GITLINK_MODE, HEAD_FILE, INDEX_FILE, MODULES_DIR,
    MOVE_TO_MARKER, OBJECTS_DIR, PATCH_PATH_MARKERS, REFS_DIR, SAFE_BARE_REPOSITORY_KEY,
    SHA256_OBJECT_FORMAT, SPLIT_INDEX_EXTENSION, UPDATE_FILE_MARKER, annotate_fs_action,
    is_git_dir_name, patch_header_paths, raw_paths_from_args, resolve_fs_path,
};
pub use git_discovery::{
    ClassifyFault, Discovery, GIT_CONFIG_GLOBAL_VAR, GIT_CONFIG_NOSYSTEM_VAR, GIT_DIR_LINKS,
    ISOLATED_GIT_CONFIG, LOOSE_HEAD_CONTENTS, WorkspaceGit, classify, clear_git_config, common_dir,
    common_dir_path, discover, gitfile_path, is_git_dir, plant_linked_worktree, read_gitfile,
    write_loose_git_dir,
};
pub use ids::{ActionBinding, ActionId, ActionKind, BindingRecheck, PendingId};
pub use parse::{parse_effect, parse_mode_overlay, parse_preset};
pub use paths::{SsrfClass, classify_ssrf_host, is_vendor_auth_path};
pub use policy::{CompiledPolicy, Decide, Decision, PolicyBuilder};
pub use posix::PosixShellAnalyzer;
pub use posture::{
    ActiveFacets, FsConfine, HostCapability, Layer, ModeOverlay, NetConfine, Obligations, Preset,
    ProcConfine, SandboxBackend, SandboxRequirement, ShellDialect,
};
pub use principal::{ApprovalWaiter, BotOrigin, HumanChannel, Principal, Subject};
pub use pwsh::PowerShellAnalyzer;
pub use reason::{Reason, ReasonCode};
pub use request::{Request, SessionScope};
pub use rule::{Matcher, Rule, RuleRef, RuleSet, RuleSource};
pub use shadow::{
    SHADOW_ENV, ShadowComparison, compare_shadow, evaluate_action, evaluate_shadow,
    evaluate_tool_gate, observe_shadow, shadow_enabled_value,
};
pub use source::{PolicySource, ingest_source, source_rules};
pub use tool_name::{TOOL_ALIASES, tool_name_as_run};
