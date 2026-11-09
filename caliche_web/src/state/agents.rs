//! Sub-agent state management.
//!
//! Tracks agents spawned from the main chat interface, including local Caliche
//! agents and external agents (Claude Code, Codex, etc.).

use chrono::{DateTime, Utc};

/// The runtime source / executor for a sub-agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSource {
    /// Caliche backend session (same API the parent tab is connected to).
    Local,
    /// A remote Caliche endpoint (e.g. a Fargate fleet node).
    #[allow(
        dead_code,
        reason = "nothing builds it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    Remote,
}

impl std::fmt::Display for AgentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentSource::Local => write!(f, "local"),
            AgentSource::Remote => write!(f, "remote"),
        }
    }
}

/// Lifecycle status of a sub-agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStatus {
    /// Queued but not yet started
    Pending,
    /// Actively streaming output
    Running,
    /// Completed successfully
    Completed,
    /// Failed with an error (captured in the last output chunk)
    Failed,
    /// Manually cancelled
    Cancelled,
}

impl AgentStatus {
    pub fn label(&self) -> &str {
        match self {
            AgentStatus::Pending => "pending",
            AgentStatus::Running => "running",
            AgentStatus::Completed => "completed",
            AgentStatus::Failed => "failed",
            AgentStatus::Cancelled => "cancelled",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, AgentStatus::Pending | AgentStatus::Running)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentStatus::Cancelled | AgentStatus::Completed | AgentStatus::Failed
        )
    }
}

/// A single output chunk from the agent, with a timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentOutputChunk {
    pub content: String,
    /// Whether this is a tool-call/system event rather than LLM text
    pub is_event: bool,
}

/// A single command injected into a running sub-agent from the fleet UI.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectedCommand {
    pub text: String,
}

/// A sub-agent managed from the main chat UI.
#[derive(Debug, Clone, PartialEq)]
pub struct SubAgent {
    /// Unique identifier (UUID-style string generated locally)
    pub id: String,
    /// Short display name
    pub name: String,
    /// The task / prompt the agent was given
    pub task: String,
    /// Runtime source
    pub source: AgentSource,
    /// Current lifecycle status
    pub status: AgentStatus,
    /// Streamed output chunks
    pub output: Vec<AgentOutputChunk>,
    /// Commands injected after creation (e.g. from fleet broadcast)
    pub injected_commands: Vec<InjectedCommand>,
    /// Cumulative token usage
    pub token_usage: u32,
    /// When the agent last produced output or changed status
    pub updated_at: DateTime<Utc>,
    /// Optional backend session ID when the agent has been registered with the
    /// server. Populated once `ApiClient::spawn_coding_run` succeeds; used by
    /// cancel/open-output flows to address the right server session.
    pub backend_session_id: Option<String>,
    /// Provider override requested when spawning, if any.
    pub provider: Option<String>,
    /// Model override requested when spawning, if any.
    pub model: Option<String>,
}

impl SubAgent {
    /// Create a new pending agent.
    pub fn new(name: impl Into<String>, task: impl Into<String>, source: AgentSource) -> Self {
        let now = Utc::now();
        Self {
            id: crate::state::agents::new_id(),
            name: name.into(),
            task: task.into(),
            source,
            status: AgentStatus::Pending,
            output: Vec::new(),
            injected_commands: Vec::new(),
            token_usage: 0,
            updated_at: now,
            backend_session_id: None,
            provider: None,
            model: None,
        }
    }

    /// Attach a backend session ID once the server acknowledges the spawn.
    pub fn attach_backend_session(&mut self, session_id: impl Into<String>) {
        self.backend_session_id = Some(session_id.into());
        self.updated_at = Utc::now();
    }

    /// Record a provider/model selection made when spawning.
    pub fn set_provider_and_model(&mut self, provider: Option<String>, model: Option<String>) {
        self.provider = provider;
        self.model = model;
    }

    /// Append a text output chunk from the backend stream / response.
    pub fn push_output(&mut self, text: impl Into<String>) {
        self.output.push(AgentOutputChunk {
            content: text.into(),
            is_event: false,
        });
        self.updated_at = Utc::now();
    }

    /// Append a tool-call / system event to the output log.
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn push_event(&mut self, text: impl Into<String>) {
        self.output.push(AgentOutputChunk {
            content: text.into(),
            is_event: true,
        });
        self.updated_at = Utc::now();
    }

    /// Inject a command or context update into this task from the fleet UI.
    /// Has no effect on terminal tasks (Done/Failed/Cancelled).
    pub fn inject_command(&mut self, text: impl Into<String>) {
        if self.status.is_terminal() {
            return;
        }
        self.injected_commands
            .push(InjectedCommand { text: text.into() });
        self.updated_at = Utc::now();
    }

    /// Full concatenated text output (text chunks only).
    pub fn full_output(&self) -> String {
        self.output
            .iter()
            .filter(|c| !c.is_event)
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Human-readable "time since" string.
    pub fn time_ago(&self) -> String {
        let secs = (Utc::now() - self.updated_at).num_seconds();
        if secs < 60 {
            format!("{}s ago", secs)
        } else if secs < 3600 {
            format!("{}m ago", secs / 60)
        } else {
            format!("{}h ago", secs / 3600)
        }
    }
}

/// Generate a stable unique ID for sub-agents.
fn new_id() -> String {
    format!("agent-{}", uuid::Uuid::new_v4())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_active_terminal_classification() {
        assert!(AgentStatus::Pending.is_active());
        assert!(AgentStatus::Running.is_active());
        assert!(!AgentStatus::Completed.is_active());
        assert!(!AgentStatus::Failed.is_active());
        assert!(!AgentStatus::Cancelled.is_active());

        assert!(!AgentStatus::Pending.is_terminal());
        assert!(!AgentStatus::Running.is_terminal());
        assert!(AgentStatus::Completed.is_terminal());
        assert!(AgentStatus::Failed.is_terminal());
        assert!(AgentStatus::Cancelled.is_terminal());
    }

    #[test]
    fn spawn_then_attach_then_complete_transitions() {
        let mut agent = SubAgent::new("agent-1", "do the thing", AgentSource::Local);
        assert_eq!(agent.status, AgentStatus::Pending);
        assert!(agent.backend_session_id.is_none());
        assert!(agent.status.is_active());

        agent.attach_backend_session("session-xyz");
        agent.status = AgentStatus::Running;
        assert_eq!(agent.backend_session_id.as_deref(), Some("session-xyz"));
        assert!(agent.status.is_active());

        agent.push_output("partial");
        agent.push_output(" result");
        agent.status = AgentStatus::Completed;
        assert!(agent.status.is_terminal());
        assert_eq!(agent.full_output(), "partial result");
    }

    #[test]
    fn inject_command_noop_for_terminal_agents() {
        let mut agent = SubAgent::new("agent-2", "task", AgentSource::Local);
        agent.status = AgentStatus::Cancelled;
        agent.inject_command("please continue");
        assert!(agent.injected_commands.is_empty());
    }

    #[test]
    fn set_provider_and_model_records_overrides() {
        let mut agent = SubAgent::new("agent-3", "task", AgentSource::Local);
        agent.set_provider_and_model(Some("openai".into()), Some("gpt-4o-mini".into()));
        assert_eq!(agent.provider.as_deref(), Some("openai"));
        assert_eq!(agent.model.as_deref(), Some("gpt-4o-mini"));
    }
}
