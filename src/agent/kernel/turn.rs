use crate::messages::TokenUsage;
use serde::{Deserialize, Serialize};

/// Represents a handoff from one agent to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handoff {
    /// The target agent identifier to transfer control to.
    pub target_agent: String,
    /// Human-readable reason for the handoff.
    pub reason: String,
    /// Override the target agent's system prompt (persona) for this handoff only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    /// Restrict the target agent to these tool names for this handoff.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Detailed instructions for the target agent (prepended to the task message).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

impl Handoff {
    /// Create a new handoff.
    pub fn new(target_agent: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            target_agent: target_agent.into(),
            reason: reason.into(),
            persona: None,
            tools: None,
            instructions: None,
        }
    }

    /// Set persona (system prompt override) for the target subagent.
    pub fn with_persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }

    /// Set allowed tools for the target subagent for this handoff.
    #[must_use]
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set detailed instructions for the target subagent.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// Result of an agent turn execution.
#[derive(Debug, Clone)]
pub enum TurnResult {
    /// Standard reply with response text and token usage.
    Reply(String, TokenUsage),
    /// Handoff to another agent with handoff details and token usage.
    Handoff(Handoff, TokenUsage),
    /// Explicit termination of the flow.
    Stop(TokenUsage),
}

impl TurnResult {
    /// Get token usage from this result.
    #[must_use]
    pub const fn token_usage(&self) -> &TokenUsage {
        match self {
            Self::Reply(_, usage) => usage,
            Self::Handoff(_, usage) => usage,
            Self::Stop(usage) => usage,
        }
    }

    /// Extract response text if this is a reply.
    #[must_use]
    pub fn as_reply(&self) -> Option<&str> {
        match self {
            Self::Reply(text, _) => Some(text),
            _ => None,
        }
    }

    /// Extract the handoff if this is a handoff.
    #[must_use]
    pub const fn as_handoff(&self) -> Option<&Handoff> {
        match self {
            Self::Handoff(handoff, _) => Some(handoff),
            _ => None,
        }
    }

    /// Check if this result is a handoff.
    #[must_use]
    pub const fn is_handoff(&self) -> bool {
        matches!(self, Self::Handoff(_, _))
    }

    /// Check if this result is an explicit stop.
    #[must_use]
    pub const fn is_stop(&self) -> bool {
        matches!(self, Self::Stop(_))
    }
}
