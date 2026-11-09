//! Interaction modes for agent sessions (CLI and API).
//!
//! This module defines the different modes an agent can operate in:
//! - **Agent** - Full capabilities including file writes (default)
//! - **Plan** - Interview user, gather requirements, build plan document (read-only)
//! - **Ask** - Answer questions, explain code (read-only)

use crate::agent::tools::metadata::is_likely_side_effecting_tool;
use crate::agent::tools::names::{
    TOOL_APPLY_PATCH, TOOL_BASH, TOOL_EXECUTE_PYTHON, TOOL_FILESYSTEM, TOOL_HTTP,
    TOOL_SQL_EXECUTION,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Interaction mode for agent sessions.
///
/// Controls what tools are available and how the agent behaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InteractionMode {
    /// Full agent capabilities including file writes (default)
    #[default]
    Agent,
    /// Planning mode - interview user, build plan document
    Plan,
    /// Ask mode - answer questions, explain code
    Ask,
}

impl InteractionMode {
    /// Returns true if this mode restricts write operations.
    #[must_use]
    pub const fn is_readonly(&self) -> bool {
        matches!(self, Self::Plan | Self::Ask)
    }

    /// Get the display name for this mode.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Agent => "Agent",
            Self::Plan => "Plan",
            Self::Ask => "Ask",
        }
    }

    /// Get a description of this mode.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Agent => "Full capabilities - can read and write files",
            Self::Plan => "Planning mode - interview and document requirements (read-only)",
            Self::Ask => "Ask mode - answer questions and explain code (read-only)",
        }
    }
}

impl fmt::Display for InteractionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for InteractionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "agent" => Ok(Self::Agent),
            "plan" => Ok(Self::Plan),
            "ask" => Ok(Self::Ask),
            _ => Err(format!(
                "Unknown mode: '{s}'. Valid modes: agent, plan, ask"
            )),
        }
    }
}

/// Tools that have write/side-effect capabilities.
///
/// These tools are blocked in Plan and Ask modes.
/// - fs: filesystem operations (write/edit/delete are side-effecting)
/// - `apply_patch`: freeform multi-file mutations
/// - bash: runs shell commands that can mutate state
/// - `execute_python`: runs Python code that can mutate state
/// - `sql_execution`: executes SQL (INSERT/UPDATE/DELETE)
/// - http: supports POST/PUT/DELETE/PATCH methods that mutate remote state
pub(crate) const WRITE_TOOLS: &[&str] = &[
    TOOL_FILESYSTEM,
    TOOL_APPLY_PATCH,
    TOOL_BASH,
    TOOL_EXECUTE_PYTHON,
    TOOL_SQL_EXECUTION,
    TOOL_HTTP,
];

/// Tool name prefixes that indicate write/side-effect capabilities.
/// Used for MCP and custom tools that may have write behavior (e.g. "create_*", "write_*").
pub const WRITE_TOOL_PREFIXES: &[&str] = &["write_", "create_", "edit_", "delete_", "update_"];

/// Returns true if the tool has write/side-effect capabilities and should be blocked in Plan and Ask modes.
#[must_use]
pub fn is_write_tool(name: &str) -> bool {
    if WRITE_TOOLS.contains(&name) {
        return true;
    }
    if is_likely_side_effecting_tool(name) {
        return true;
    }
    WRITE_TOOL_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// Returns the built-in tool names considered side-effecting in readonly modes.
#[must_use]
pub const fn readonly_blocked_tools() -> &'static [&'static str] {
    WRITE_TOOLS
}

/// System prompt for Plan mode.
pub(crate) const PLAN_MODE_PROMPT: &str = r"You are in PLANNING MODE. Your role is to:
1. Interview the user to understand their requirements
2. Ask clarifying questions one at a time
3. Summarize what you understand after each answer
4. Build a structured plan document

Do NOT write code or make changes. Focus on understanding and planning.
When you have enough information, present the plan for user approval.

Use ONLY these read-only tools to understand the codebase: fs (op=read/list/grep), workspace, semantic_search (if available). Use write_plan only once to save the final plan document (e.g. PLAN.md) when ready; do not use write_plan for source code or multiple files. Do NOT use fs with op=write/edit/delete, bash, execute_python, git with operation=commit, sql_execution, or http.
";

/// System prompt for Ask mode.
pub(crate) const ASK_MODE_PROMPT: &str = r"You are in ASK MODE. Your role is to:
1. Answer questions about code and concepts
2. Explain how things work
3. Provide information and guidance

Do NOT make changes to files. Use read-only tools to gather context when needed.
";

/// Returns the system prompt prefix for a readonly interaction mode, if any.
#[must_use]
pub const fn readonly_mode_prompt(mode: InteractionMode) -> Option<&'static str> {
    match mode {
        InteractionMode::Plan => Some(PLAN_MODE_PROMPT),
        InteractionMode::Ask => Some(ASK_MODE_PROMPT),
        InteractionMode::Agent => None,
    }
}

/// Prompt for extracting plan data from conversation turns.
#[must_use]
pub const fn plan_extraction_prompt() -> &'static str {
    PLAN_EXTRACTION_PROMPT
}

/// Prompt for extracting plan data from conversation turns.
pub(crate) const PLAN_EXTRACTION_PROMPT: &str = r#"Based on the conversation exchange, extract any planning information into this JSON structure. Only include fields that have new information from this exchange:
{
  "title": "optional title if a project name was discussed",
  "new_requirements": ["any new requirements mentioned in this exchange"],
  "clarification": {"question": "the question asked", "answer": "the answer given"} or null,
  "new_tasks": ["any specific tasks identified"],
  "notes": ["any important decisions or constraints mentioned"]
}
Return ONLY valid JSON, no explanation. If no planning information was exchanged, return: {"new_requirements": [], "new_tasks": [], "notes": []}
"#;

/// A task in a plan document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanTask {
    /// Task description
    pub description: String,
    /// Whether the task is completed
    pub completed: bool,
    /// Optional sub-tasks
    pub subtasks: Vec<String>,
}

impl PlanTask {
    /// Create a new task.
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            completed: false,
            subtasks: Vec::new(),
        }
    }

    /// Add a subtask.
    pub fn with_subtask(mut self, subtask: impl Into<String>) -> Self {
        self.subtasks.push(subtask.into());
        self
    }
}

/// A clarification question and answer pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clarification {
    /// The question asked
    pub question: String,
    /// The user's answer
    pub answer: String,
}

/// Plan document that evolves during Plan mode.
///
/// This document captures requirements, clarifications, and tasks
/// as the agent interviews the user.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanDocument {
    /// Optional title for the plan
    pub title: Option<String>,
    /// High-level requirements gathered from the user
    pub requirements: Vec<String>,
    /// Questions asked and answers received
    pub clarifications: Vec<Clarification>,
    /// Tasks to be completed
    pub tasks: Vec<PlanTask>,
    /// Additional notes
    pub notes: Vec<String>,
}

impl PlanDocument {
    /// Create a new empty plan document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the plan title.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = Some(title.into());
    }

    /// Add a requirement.
    pub fn add_requirement(&mut self, requirement: impl Into<String>) {
        self.requirements.push(requirement.into());
    }

    /// Add a clarification (question and answer).
    pub fn add_clarification(&mut self, question: impl Into<String>, answer: impl Into<String>) {
        self.clarifications.push(Clarification {
            question: question.into(),
            answer: answer.into(),
        });
    }

    /// Add a task.
    pub fn add_task(&mut self, task: PlanTask) {
        self.tasks.push(task);
    }

    /// Add a note.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Check if the plan is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.requirements.is_empty()
            && self.clarifications.is_empty()
            && self.tasks.is_empty()
            && self.notes.is_empty()
    }

    /// Render the plan as markdown.
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Title
        if let Some(ref title) = self.title {
            md.push_str(&format!("# {title}\n\n"));
        } else {
            md.push_str("# Plan\n\n");
        }

        // Requirements
        if !self.requirements.is_empty() {
            md.push_str("## Requirements\n\n");
            for req in &self.requirements {
                md.push_str(&format!("- {req}\n"));
            }
            md.push('\n');
        }

        // Clarifications
        if !self.clarifications.is_empty() {
            md.push_str("## Clarifications\n\n");
            for clarification in &self.clarifications {
                md.push_str(&format!(
                    "- **Q:** {}\n  **A:** {}\n",
                    clarification.question, clarification.answer
                ));
            }
            md.push('\n');
        }

        // Tasks
        if !self.tasks.is_empty() {
            md.push_str("## Tasks\n\n");
            for (i, task) in self.tasks.iter().enumerate() {
                let checkbox = if task.completed { "[x]" } else { "[ ]" };
                md.push_str(&format!("{}. {} {}\n", i + 1, checkbox, task.description));
                for subtask in &task.subtasks {
                    md.push_str(&format!("   - {subtask}\n"));
                }
            }
            md.push('\n');
        }

        // Notes
        if !self.notes.is_empty() {
            md.push_str("## Notes\n\n");
            for note in &self.notes {
                md.push_str(&format!("- {note}\n"));
            }
            md.push('\n');
        }

        md
    }

    /// Parse LLM-extracted JSON and merge into the plan document.
    ///
    /// The JSON should have the structure from `PLAN_EXTRACTION_PROMPT`.
    pub fn merge_extraction(&mut self, json_str: &str) -> Result<bool, serde_json::Error> {
        #[derive(Deserialize)]
        struct PlanExtraction {
            #[serde(default)]
            title: Option<String>,
            #[serde(default)]
            new_requirements: Vec<String>,
            #[serde(default)]
            clarification: Option<Clarification>,
            #[serde(default)]
            new_tasks: Vec<String>,
            #[serde(default)]
            notes: Vec<String>,
        }

        // Try to parse - handle potential markdown code fences
        let clean_json = caliche_contracts::text::extract_json(json_str).unwrap_or(json_str);

        let extraction: PlanExtraction = serde_json::from_str(clean_json)?;

        let mut had_changes = false;

        // Only set title if we don't have one and extraction provides one
        if self.title.is_none() && extraction.title.is_some() {
            self.title = extraction.title;
            had_changes = true;
        }

        // Add new requirements (deduplicate)
        for req in extraction.new_requirements {
            if !req.is_empty() && !self.requirements.contains(&req) {
                self.requirements.push(req);
                had_changes = true;
            }
        }

        // Add clarification if present
        if let Some(c) = extraction.clarification
            && !c.question.is_empty()
            && !c.answer.is_empty()
        {
            self.clarifications.push(c);
            had_changes = true;
        }

        // Add new tasks (deduplicate by description)
        for task_desc in extraction.new_tasks {
            if !task_desc.is_empty() && !self.tasks.iter().any(|t| t.description == task_desc) {
                self.tasks.push(PlanTask::new(task_desc));
                had_changes = true;
            }
        }

        // Add notes (deduplicate)
        for note in extraction.notes {
            if !note.is_empty() && !self.notes.contains(&note) {
                self.notes.push(note);
                had_changes = true;
            }
        }

        Ok(had_changes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merged(reply: &str) -> (Result<bool, String>, Vec<String>) {
        let mut plan = PlanDocument::new();
        let changed = plan.merge_extraction(reply).map_err(|e| e.to_string());
        (changed, plan.requirements)
    }

    #[test]
    fn extraction_with_an_uppercase_fence_tag_merges() {
        assert_eq!(
            merged("```JSON\n{\"new_requirements\": [\"offline\"]}\n```"),
            (Ok(true), vec!["offline".to_string()])
        );
    }

    #[test]
    fn extraction_after_text_merges() {
        assert_eq!(
            merged("Here is the update:\n```json\n{\"new_requirements\": [\"offline\"]}\n```"),
            (Ok(true), vec!["offline".to_string()])
        );
    }

    #[test]
    fn interaction_mode_defaults_to_agent() {
        let mode: InteractionMode = Default::default();
        assert_eq!(mode, InteractionMode::Agent);
    }

    #[test]
    fn interaction_mode_readonly_only_for_plan_and_ask() {
        assert!(!InteractionMode::Agent.is_readonly());
        assert!(InteractionMode::Plan.is_readonly());
        assert!(InteractionMode::Ask.is_readonly());
    }

    #[test]
    fn interaction_mode_parses_case_insensitive_names() {
        assert_eq!(
            "agent".parse::<InteractionMode>().unwrap(),
            InteractionMode::Agent
        );
        assert_eq!(
            "plan".parse::<InteractionMode>().unwrap(),
            InteractionMode::Plan
        );
        assert_eq!(
            "ask".parse::<InteractionMode>().unwrap(),
            InteractionMode::Ask
        );
        assert_eq!(
            "AGENT".parse::<InteractionMode>().unwrap(),
            InteractionMode::Agent
        );
        assert!("invalid".parse::<InteractionMode>().is_err());
    }

    #[test]
    fn interaction_mode_serde_uses_lowercase_wire_names() {
        let mode = InteractionMode::Plan;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"plan\"");

        let parsed: InteractionMode = serde_json::from_str("\"ask\"").unwrap();
        assert_eq!(parsed, InteractionMode::Ask);
    }

    #[test]
    fn is_write_tool_blocks_builtins_and_write_prefixes() {
        assert!(is_write_tool("fs"));
        assert!(is_write_tool("bash"));
        assert!(is_write_tool("http"));
        assert!(is_write_tool("write_plan"));
        assert!(is_write_tool("create_ticket"));
        assert!(!is_write_tool("semantic_search"));
        assert!(!is_write_tool("workspace"));
        assert!(!is_write_tool("web_search"));
        assert!(!is_write_tool("web_fetch"));
    }

    #[test]
    fn readonly_mode_prompt_present_only_for_plan_and_ask() {
        assert!(readonly_mode_prompt(InteractionMode::Agent).is_none());
        assert!(
            readonly_mode_prompt(InteractionMode::Plan)
                .is_some_and(|p| p.contains("PLANNING MODE"))
        );
        assert!(readonly_mode_prompt(InteractionMode::Ask).is_some_and(|p| p.contains("ASK MODE")));
    }

    #[test]
    fn plan_document_markdown_includes_sections() {
        let mut plan = PlanDocument::new();
        plan.set_title("Test Plan");
        plan.add_requirement("Implement feature X");
        plan.add_clarification("What framework?", "Axum");
        plan.add_task(PlanTask::new("Create module").with_subtask("Add tests"));
        plan.add_note("Consider performance");

        let md = plan.to_markdown();
        assert!(md.contains("# Test Plan"));
        assert!(md.contains("## Requirements"));
        assert!(md.contains("- Implement feature X"));
        assert!(md.contains("## Clarifications"));
        assert!(md.contains("**Q:** What framework?"));
        assert!(md.contains("## Tasks"));
        assert!(md.contains("[ ] Create module"));
        assert!(md.contains("## Notes"));
    }
}
