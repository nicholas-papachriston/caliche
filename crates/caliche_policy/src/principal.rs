//! Who is asking, and whether a human can answer.

use serde::{Deserialize, Serialize};

/// Stable subject label (user id, plugin id, or other subject). Not a secret.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Subject(String);

impl Subject {
    /// Wrap a non-secret subject id.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Borrow the id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Channel a human uses to answer an approval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HumanChannel {
    /// CLI tty prompt.
    CliTty,
    /// Web UI session.
    UiSession,
    /// API caller authenticated as a human subject.
    ApiJwt,
    /// ACP client.
    AcpClient,
}

/// Where an unattended principal came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BotOrigin {
    /// Inbound webhook.
    Webhook,
    /// Cron or scheduler.
    Cron,
    /// Gateway adapter.
    Gateway,
    /// HTTP session create with no approval waiter.
    UnattendedPost,
}

/// Principal on whose behalf an action runs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Principal {
    /// A person at a CLI, UI, API, or ACP channel.
    Human {
        /// Subject id.
        id: Subject,
        /// Channel that can answer approvals.
        channel: HumanChannel,
    },
    /// Unattended automation.
    Bot {
        /// Subject id.
        id: Subject,
        /// How this principal was triggered.
        origin: BotOrigin,
    },
    /// External harness Caliche is driving.
    Peer {
        /// Harness id.
        harness: String,
    },
    /// Plugin or extension.
    Plugin {
        /// Plugin id.
        id: String,
    },
    /// Child agent.
    Subagent {
        /// Parent session id.
        parent: String,
        /// Nesting depth.
        depth: u8,
    },
}

impl Principal {
    /// Human principal.
    #[must_use]
    pub fn human(id: impl Into<String>, channel: HumanChannel) -> Self {
        Self::Human {
            id: Subject::new(id),
            channel,
        }
    }

    /// Unattended principal.
    #[must_use]
    pub fn bot(id: impl Into<String>, origin: BotOrigin) -> Self {
        Self::Bot {
            id: Subject::new(id),
            origin,
        }
    }

    /// Whether this principal may answer an approval.
    ///
    /// Only [`Principal::Human`] returns true. Model output, tools, MCP, plugins,
    /// peers, and subagents cannot.
    #[must_use]
    pub const fn is_human(&self) -> bool {
        matches!(self, Self::Human { .. })
    }
}

/// Whether this session can block on a human answer.
///
/// Computed per session. An unattended caller cannot set it to pretend a
/// person is present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ApprovalWaiter {
    /// A human channel is attached.
    Present(HumanChannel),
    /// Cron, webhook, or any session with nobody to ask.
    Absent,
}
