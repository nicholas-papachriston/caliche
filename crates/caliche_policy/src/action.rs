//! What is being authorized.

use crate::credential::CredentialClass;
use crate::ids::{ActionId, ActionKind, PendingId};
use crate::posture::ShellDialect;
use serde::{Deserialize, Serialize};

/// Tool family. Some families still carry opaque facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolClass {
    /// Filesystem reads and writes.
    Fs,
    /// Shell (`bash` today; `shell` later).
    Shell,
    /// HTTP tool.
    Http,
    /// `web_fetch` / `web_search`.
    WebFetch,
    /// MCP tool call (`server__tool`).
    Mcp,
    /// `execute_python` and other code execution. Content is not analyzed.
    CodeExec,
    /// Patch apply.
    Patch,
    /// Git subcommands.
    Git,
    /// SQL execution.
    Sql,
    /// Workspace helper.
    Workspace,
    /// Plugin or unknown tool.
    Other,
}

/// Filesystem operation carried in facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FsOp {
    /// Read a file.
    Read,
    /// Write or edit a file.
    Write,
    /// Delete a file.
    Delete,
    /// List or search.
    List,
    /// Unclassified.
    Other,
}

/// HTTP method. GET is not exempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HttpMethod {
    /// GET.
    Get,
    /// HEAD.
    Head,
    /// POST.
    Post,
    /// PUT.
    Put,
    /// PATCH.
    Patch,
    /// DELETE.
    Delete,
    /// Anything else.
    Other,
}

/// Where a resolved path sits relative to the workspace.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathPlace {
    /// The caller did not supply a workspace, so containment was not checked.
    #[default]
    Unchecked,
    /// The path is the workspace or a file inside it.
    Inside,
    /// The path is inside the workspace, in or at a `.git` entry (any case,
    /// at any depth). Host-side `git` runs commands named in repository
    /// config, so the gate denies a write here. A read is treated as inside.
    GitMetadata,
    /// The path is outside the workspace. The gate asks before use. Accept-all
    /// cannot lower that ask for a write.
    Outside,
    /// The resolver ran with no workspace root. The gate denies.
    NoWorkspace,
    /// The path form is refused (stream, device, or short name).
    Rejected,
}

/// Canonical path a resolver produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPath {
    /// Canonical path text. Separators are `/`.
    pub canonical: String,
    /// Containment. Missing on older snapshots means [`PathPlace::Unchecked`].
    #[serde(default)]
    pub place: PathPlace,
}

impl ResolvedPath {
    /// Wrap a path string that has not been checked against a workspace.
    #[must_use]
    pub fn new(canonical: impl Into<String>) -> Self {
        Self {
            canonical: canonical.into(),
            place: PathPlace::Unchecked,
        }
    }

    /// A path form the resolver refused.
    #[must_use]
    pub fn rejected(canonical: impl Into<String>) -> Self {
        Self {
            canonical: canonical.into(),
            place: PathPlace::Rejected,
        }
    }
}

/// Shell plan. The POSIX analyzer fills this. Other dialects stay opaque.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellPlan {
    /// Parser could not read the command. Never yields allow.
    pub opaque: bool,
    /// Number of simple commands. The stub analyzer reports zero.
    pub segment_count: u32,
    /// The POSIX denylist matched a segment. A floor deny rule consumes this.
    #[serde(default)]
    pub denied: bool,
}

/// Facts a facet produced. Facets do not return effects.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolFacts {
    /// Filesystem facts.
    Fs {
        /// Operation.
        op: FsOp,
        /// Paths the caller resolved.
        paths: Vec<ResolvedPath>,
    },
    /// Shell facts. `plan: None` means the dialect facet is off (no parse).
    Shell {
        /// Dialect the caller selected.
        dialect: ShellDialect,
        /// `None` when the analyzer did not run.
        plan: Option<ShellPlan>,
    },
    /// HTTP or web fetch facts.
    Http {
        /// Method.
        method: HttpMethod,
        /// Hostname only, no userinfo.
        host: String,
    },
    /// MCP tool facts.
    Mcp {
        /// Server id.
        server: String,
        /// Tool name.
        tool: String,
    },
    /// Code execution. The body is not inspected.
    CodeExec {
        /// Language label.
        lang: String,
    },
    /// Patch facts.
    Patch {
        /// Paths named by the patch.
        paths: Vec<ResolvedPath>,
    },
    /// Keychain item. Items Caliche does not own are vendor secrets.
    Keychain {
        /// True only for stores this process created.
        owned_by_caliche: bool,
    },
    /// No structured facts.
    Opaque,
    /// The facet failed (parse, canonicalize, DNS, analyzer).
    CheckerError {
        /// Short reason. Do not include secrets.
        detail: String,
    },
}

/// Action presented to `decide`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    /// Tool call.
    Tool {
        /// Family.
        class: ToolClass,
        /// Tool name (`bash`, `fs`, …).
        name: String,
        /// Facts from the class backend. Plugins supply facts, not effects.
        facts: ToolFacts,
    },
    /// Outbound connection.
    Egress {
        /// Hostname.
        host: String,
        /// Port.
        port: u16,
    },
    /// MCP connect.
    McpConnect {
        /// Server id.
        server: String,
    },
    /// Model selection.
    ModelUse {
        /// Provider id.
        provider: String,
        /// Model id.
        model: String,
        /// Credential class.
        cred: CredentialClass,
    },
    /// Spawn an external harness.
    HarnessSpawn {
        /// Harness id.
        harness: String,
        /// Canonical executable path. Empty means the path was not checked.
        exe: String,
    },
    /// Invoke another agent.
    TaskInvoke {
        /// Target alias.
        target: String,
    },
    /// Answer a pending approval.
    ApprovalAnswer {
        /// Pending id.
        pending: PendingId,
    },
    /// Config write.
    ConfigWrite {
        /// Target path.
        target: String,
    },
    /// The facet could not build facts.
    Unresolved {
        /// Short reason. Do not include secrets.
        detail: String,
    },
}

impl Action {
    /// Tool action.
    #[must_use]
    pub fn tool(class: ToolClass, name: impl Into<String>, facts: ToolFacts) -> Self {
        Self::Tool {
            class,
            name: name.into(),
            facts,
        }
    }

    /// Identity used in logs and bindings. The name is not the argument hash.
    #[must_use]
    pub fn id(&self) -> ActionId {
        match self {
            Self::Tool { name, .. } => ActionId::new(ActionKind::Tool, name.clone()),
            Self::Egress { host, .. } => ActionId::new(ActionKind::Egress, host.clone()),
            Self::McpConnect { server } => ActionId::new(ActionKind::McpConnect, server.clone()),
            Self::ModelUse {
                provider, model, ..
            } => ActionId::new(ActionKind::ModelUse, format!("{provider}/{model}")),
            Self::HarnessSpawn { harness, .. } => {
                ActionId::new(ActionKind::HarnessSpawn, harness.clone())
            }
            Self::TaskInvoke { target } => ActionId::new(ActionKind::TaskInvoke, target.clone()),
            Self::ApprovalAnswer { pending } => {
                ActionId::new(ActionKind::ApprovalAnswer, pending.as_str())
            }
            Self::ConfigWrite { target } => ActionId::new(ActionKind::ConfigWrite, target.clone()),
            Self::Unresolved { .. } => ActionId::new(ActionKind::Unresolved, "unresolved"),
        }
    }

    /// Tool name when this is a tool call.
    #[must_use]
    pub const fn tool_name(&self) -> Option<&str> {
        match self {
            Self::Tool { name, .. } => Some(name.as_str()),
            _ => None,
        }
    }

    /// Tool class when this is a tool call.
    #[must_use]
    pub const fn tool_class(&self) -> Option<ToolClass> {
        match self {
            Self::Tool { class, .. } => Some(*class),
            _ => None,
        }
    }
}
