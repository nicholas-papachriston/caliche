//! Task planning and decomposition support.
//!
//! This module provides functionality for breaking down complex tasks
//! into smaller, manageable steps that the agent can execute.

use crate::harness::events::{AgentEvent, EventStream};
use serde::{Deserialize, Serialize};

/// A single task step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStep {
    /// Unique step ID
    pub id: String,
    /// Step description
    pub description: String,
    /// Step status
    pub status: StepStatus,
    /// Dependencies (IDs of steps that must complete first)
    #[serde(default)]
    pub dependencies: Vec<String>,
    /// Output from this step (if completed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Tools likely needed for this step
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suggested_tools: Vec<String>,
    /// Estimated complexity (1-5)
    #[serde(default)]
    pub complexity: u8,
}

/// Status of a task step
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum StepStatus {
    /// Not started yet
    #[default]
    Pending,
    /// Currently in progress
    InProgress,
    /// Successfully completed
    Completed,
    /// Failed with error
    Failed,
    /// Skipped (dependency failed or user cancelled)
    Skipped,
}

/// A task plan with multiple steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPlan {
    /// Plan ID
    pub id: String,
    /// Original task description
    pub task: String,
    /// Steps in the plan
    pub steps: Vec<TaskStep>,
    /// Current step index being executed
    pub current_step: usize,
    /// Overall plan status
    pub status: PlanStatus,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Completed timestamp
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Status of a task plan
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum PlanStatus {
    /// Plan is being created
    #[default]
    Planning,
    /// Ready to execute
    Ready,
    /// Currently executing
    Executing,
    /// Successfully completed
    Completed,
    /// Failed during execution
    Failed,
    /// Cancelled by user
    Cancelled,
}

impl TaskPlan {
    /// Create a new task plan
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            task: task.into(),
            steps: Vec::new(),
            current_step: 0,
            status: PlanStatus::Planning,
            created_at: chrono::Utc::now(),
            completed_at: None,
        }
    }

    /// Add a step to the plan
    pub fn add_step(&mut self, description: impl Into<String>) -> &mut TaskStep {
        let step = TaskStep {
            id: format!("step_{}", self.steps.len() + 1),
            description: description.into(),
            status: StepStatus::Pending,
            dependencies: Vec::new(),
            output: None,
            suggested_tools: Vec::new(),
            complexity: 1,
        };
        let step_idx = self.steps.len();
        self.steps.push(step);
        // Safe: we just pushed, so the index exists
        &mut self.steps[step_idx]
    }

    /// Mark plan as ready to execute
    pub fn finalize(&mut self) {
        self.status = PlanStatus::Ready;
    }

    /// Get the next step to execute
    pub fn next_step(&self) -> Option<&TaskStep> {
        self.steps.iter().find(|s| s.status == StepStatus::Pending)
    }

    /// Get a mutable reference to the current step
    pub fn current_step_mut(&mut self) -> Option<&mut TaskStep> {
        let idx = self.current_step;
        self.steps.get_mut(idx)
    }

    /// Mark the current step as complete and advance
    pub fn complete_current_step(&mut self, output: impl Into<String>) {
        if let Some(step) = self.steps.get_mut(self.current_step) {
            step.status = StepStatus::Completed;
            step.output = Some(output.into());
        }

        // Advance to next pending step
        self.current_step = self
            .steps
            .iter()
            .position(|s| s.status == StepStatus::Pending)
            .unwrap_or(self.steps.len());

        // Check if all steps complete
        if self.steps.iter().all(|s| s.status == StepStatus::Completed) {
            self.status = PlanStatus::Completed;
            self.completed_at = Some(chrono::Utc::now());
        }
    }

    /// Mark the current step as failed
    pub fn fail_current_step(&mut self, error: impl Into<String>) {
        if let Some(step) = self.steps.get_mut(self.current_step) {
            step.status = StepStatus::Failed;
            step.output = Some(error.into());
        }
        self.status = PlanStatus::Failed;
    }

    /// Get progress as percentage (0-100)
    pub fn progress(&self) -> u8 {
        if self.steps.is_empty() {
            return 100;
        }
        let completed = self
            .steps
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count();
        ((completed as f64 / self.steps.len() as f64) * 100.0) as u8
    }

    /// Get step descriptions as a list
    pub fn step_descriptions(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.description.clone()).collect()
    }

    /// Check if all dependencies for a step are satisfied
    pub fn dependencies_satisfied(&self, step_id: &str) -> bool {
        if let Some(step) = self.steps.iter().find(|s| s.id == step_id) {
            step.dependencies.iter().all(|dep_id| {
                self.steps
                    .iter()
                    .find(|s| &s.id == dep_id)
                    .map(|s| s.status == StepStatus::Completed)
                    .unwrap_or(true) // If dependency not found, assume satisfied
            })
        } else {
            true
        }
    }
}

/// Task planner that creates plans from task descriptions
#[derive(Debug)]
pub struct TaskPlanner {
    event_stream: Option<EventStream>,
}

impl TaskPlanner {
    /// Create a new task planner
    pub fn new() -> Self {
        Self { event_stream: None }
    }

    /// Create with event stream for emitting planning events
    pub fn with_events(event_stream: EventStream) -> Self {
        Self {
            event_stream: Some(event_stream),
        }
    }

    /// Parse a plan from LLM output
    ///
    /// Expects output in format:
    /// 1. First step description
    /// 2. Second step description
    ///    ...
    pub fn parse_plan(&self, task: &str, llm_output: &str) -> TaskPlan {
        let mut plan = TaskPlan::new(task);

        // Parse numbered steps
        for line in llm_output.lines() {
            let trimmed = line.trim();

            // Match patterns like "1.", "1)", "- ", "* "
            if let Some(desc) = Self::extract_step_description(trimmed)
                && !desc.is_empty()
            {
                let step = plan.add_step(desc);

                // Detect suggested tools from description
                step.suggested_tools = Self::detect_tools(trimmed);

                // Estimate complexity
                step.complexity = Self::estimate_complexity(trimmed);
            }
        }

        // If no steps found, treat entire output as single step
        if plan.steps.is_empty() && !llm_output.trim().is_empty() {
            plan.add_step(llm_output.trim());
        }

        plan.finalize();

        // Emit planning event
        if let Some(ref stream) = self.event_stream {
            stream.emit(AgentEvent::Planning {
                steps: plan.step_descriptions(),
            });
        }

        plan
    }

    /// Extract step description from a line
    fn extract_step_description(line: &str) -> Option<String> {
        let trimmed = line.trim();

        // Match "1. description" or "1) description"
        if let Some(idx) = trimmed.find(['.', ')']) {
            let prefix = &trimmed[..idx];
            if prefix.chars().all(|c| c.is_ascii_digit()) {
                return Some(trimmed[idx + 1..].trim().to_string());
            }
        }

        // Match "- description" or "* description"
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            return Some(trimmed[2..].trim().to_string());
        }

        None
    }

    /// Detect tools that might be needed based on description.
    ///
    /// Returns owned `String`s because `TaskStep.suggested_tools` is `Vec<String>`,
    /// but builds from `&'static str` slices to avoid repeated heap work.
    fn detect_tools(description: &str) -> Vec<String> {
        let lower = description.to_lowercase();
        let mut tools: Vec<&str> = Vec::new();

        if lower.contains("file") || lower.contains("read") || lower.contains("write") {
            tools.push("fs");
        }
        if lower.contains("edit") || lower.contains("modify") || lower.contains("update") {
            tools.push("fs");
        }
        if lower.contains("run") || lower.contains("execute") || lower.contains("command") {
            tools.push("bash");
        }
        if lower.contains("search") || lower.contains("find") || lower.contains("grep") {
            tools.push("fs");
        }
        if lower.contains("list") || lower.contains("directory") {
            tools.push("fs");
        }
        if lower.contains("git") {
            tools.push("git");
        }
        if lower.contains("test") {
            tools.push("bash");
        }

        tools.into_iter().map(String::from).collect()
    }

    /// Estimate complexity from description
    pub fn estimate_complexity(description: &str) -> u8 {
        let lower = description.to_lowercase();

        let mut complexity = 1;

        // Complex keywords
        if lower.contains("implement") || lower.contains("create") || lower.contains("build") {
            complexity += 2;
        }
        if lower.contains("refactor") || lower.contains("restructure") {
            complexity += 2;
        }
        if lower.contains("fix") || lower.contains("debug") {
            complexity += 1;
        }
        if lower.contains("test") {
            complexity += 1;
        }
        if lower.contains("multiple") || lower.contains("all") {
            complexity += 1;
        }

        complexity.min(5)
    }

    /// Generate a planning prompt for the LLM
    pub fn generate_planning_prompt(task: &str) -> String {
        format!(
            r#"Break down the following task into a series of concrete steps that can be executed sequentially.

Task: {}

Instructions:
- List each step on a separate line, numbered (1., 2., etc.)
- Each step should be actionable and specific
- Consider dependencies between steps
- Keep steps atomic (one main action per step)
- Include any setup or verification steps needed

Steps:"#,
            task
        )
    }
}

impl Default for TaskPlanner {
    fn default() -> Self {
        Self::new()
    }
}
