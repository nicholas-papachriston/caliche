//! Conversation export functionality.
//!
//! Exports conversations to various formats (markdown, JSON, etc.)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A conversation message for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMessage {
    pub role: String,
    pub content: String,
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Provider chain-of-thought. Omitted unless the caller asked to include it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// A conversation export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationExport {
    pub session_id: String,
    pub provider: String,
    pub model: String,
    pub created_at: DateTime<Utc>,
    pub messages: Vec<ExportMessage>,
    pub total_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_cost: Option<f64>,
    /// Plan document (if created during Plan mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
}

impl ConversationExport {
    /// Create a new export
    pub fn new(session_id: &str, provider: &str, model: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            created_at: Utc::now(),
            messages: Vec::new(),
            total_tokens: 0,
            estimated_cost: None,
            plan: None,
        }
    }

    /// Add a message to the export. Thinking is omitted.
    pub fn add_message(&mut self, role: &str, content: &str, tokens: Option<u64>) {
        self.add_message_with_reasoning(role, content, None, tokens);
    }

    /// Add a message, optionally keeping provider chain-of-thought.
    ///
    /// Empty reasoning is dropped. Callers that build a public or default
    /// export pass `None` so the file stays text-only.
    pub fn add_message_with_reasoning(
        &mut self,
        role: &str,
        content: &str,
        reasoning: Option<&str>,
        tokens: Option<u64>,
    ) {
        let reasoning_content = reasoning
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_string);
        self.messages.push(ExportMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Some(Utc::now()),
            tokens,
            tool_name: None,
            reasoning_content,
        });
        if let Some(t) = tokens {
            self.total_tokens += t;
        }
    }

    /// Add a tool call message
    pub fn add_tool_call(&mut self, tool_name: &str, result: &str) {
        self.messages.push(ExportMessage {
            role: "tool".to_string(),
            content: result.to_string(),
            timestamp: Some(Utc::now()),
            tokens: None,
            tool_name: Some(tool_name.to_string()),
            reasoning_content: None,
        });
    }

    /// Set estimated cost
    pub fn set_cost(&mut self, cost: f64) {
        self.estimated_cost = Some(cost);
    }

    /// Set the plan document (markdown content)
    pub fn set_plan(&mut self, plan: String) {
        self.plan = Some(plan);
    }

    /// Export to markdown format
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Header
        md.push_str("# Conversation Export\n\n");
        md.push_str(&format!("**Session:** {}\n", self.session_id));
        md.push_str(&format!("**Provider:** {}\n", self.provider));
        md.push_str(&format!("**Model:** {}\n", self.model));
        md.push_str(&format!(
            "**Date:** {}\n",
            self.created_at.format("%Y-%m-%d %H:%M:%S UTC")
        ));
        md.push_str(&format!("**Total Tokens:** {}\n", self.total_tokens));
        if let Some(cost) = self.estimated_cost {
            md.push_str(&format!("**Estimated Cost:** ${:.4}\n", cost));
        }
        md.push_str("\n---\n\n");

        // Plan document (if available)
        if let Some(ref plan) = self.plan {
            md.push_str("## Plan\n\n");
            md.push_str(plan);
            md.push_str("\n---\n\n");
        }

        // Messages
        for msg in &self.messages {
            let role_display = match msg.role.as_str() {
                "user" => "**User**",
                "assistant" => "**Assistant**",
                "system" => "*System*",
                "tool" => {
                    if let Some(ref name) = msg.tool_name {
                        &format!("*Tool: {}*", name)
                    } else {
                        "*Tool*"
                    }
                }
                _ => &msg.role,
            };

            md.push_str(&format!("### {}\n\n", role_display));

            // Format content - preserve code blocks
            md.push_str(&msg.content);
            md.push_str("\n\n");
            if let Some(reasoning) = msg
                .reasoning_content
                .as_deref()
                .filter(|text| !text.is_empty())
            {
                md.push_str("**Thinking**\n\n");
                md.push_str(reasoning);
                md.push_str("\n\n");
            }

            if let Some(tokens) = msg.tokens {
                md.push_str(&format!("*({} tokens)*\n\n", tokens));
            }
        }

        md
    }

    /// Export to JSON (pretty-printed).
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export to JSON Lines (one message per line)
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut lines = Vec::new();

        // Metadata line
        let meta = serde_json::json!({
            "type": "metadata",
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
        });
        lines.push(serde_json::to_string(&meta)?);

        // Message lines
        for msg in &self.messages {
            let line = serde_json::json!({
                "type": "message",
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "tokens": msg.tokens,
                "tool_name": msg.tool_name,
                "reasoning_content": msg.reasoning_content,
            });
            lines.push(serde_json::to_string(&line)?);
        }

        Ok(lines.join("\n"))
    }

    /// Save to a file
    pub async fn save(&self, path: &Path, format: ExportFormat) -> anyhow::Result<()> {
        let content = match format {
            ExportFormat::Markdown => self.to_markdown(),
            ExportFormat::Json => self.to_json()?,
            ExportFormat::Jsonl => self.to_jsonl()?,
        };

        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Load from a JSON file
    pub async fn load(path: &Path) -> anyhow::Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        serde_json::from_str(&content).map_err(Into::into)
    }
}

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Markdown,
    Json,
    Jsonl,
}

impl ExportFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Self {
        match path.extension().and_then(|s| s.to_str()) {
            Some("md") | Some("markdown") => ExportFormat::Markdown,
            Some("json") => ExportFormat::Json,
            Some("jsonl") => ExportFormat::Jsonl,
            _ => ExportFormat::Markdown, // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_markdown() {
        let mut export = ConversationExport::new("test-123", "anthropic", "claude-3-5-sonnet");
        export.add_message("user", "Hello, how are you?", Some(10));
        export.add_message("assistant", "I'm doing well, thank you!", Some(15));

        let md = export.to_markdown();
        assert!(md.contains("# Conversation Export"));
        assert!(md.contains("claude-3-5-sonnet"));
        assert!(md.contains("Hello, how are you?"));
    }

    #[test]
    fn test_export_json() {
        let mut export = ConversationExport::new("test-123", "anthropic", "claude-3-5-sonnet");
        export.add_message("user", "Hello", Some(5));

        let Ok(json) = export.to_json() else {
            panic!("json export should serialize");
        };
        assert!(json.contains("\"session_id\""));
        assert!(json.contains("test-123"));
        assert!(!json.contains("reasoning_content"));
    }

    #[test]
    fn export_include_thinking_keeps_reasoning() {
        let mut export = ConversationExport::new("test-123", "anthropic", "claude");
        export.add_message_with_reasoning("assistant", "the answer", Some("hidden cot"), None);
        export.add_message_with_reasoning("assistant", "plain", Some("  "), None);

        let md = export.to_markdown();
        assert!(md.contains("the answer"));
        assert!(md.contains("**Thinking**"));
        assert!(md.contains("hidden cot"));
        assert_eq!(
            md.matches("**Thinking**").count(),
            1,
            "blank reasoning is omitted"
        );

        let json = export.to_json().expect("json");
        assert!(json.contains("hidden cot"));
        let parsed: ConversationExport = serde_json::from_str(&json).expect("parse");
        assert_eq!(
            parsed.messages[0].reasoning_content.as_deref(),
            Some("hidden cot")
        );
        assert!(parsed.messages[1].reasoning_content.is_none());
    }
}
