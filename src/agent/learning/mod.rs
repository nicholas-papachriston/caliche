//! Autonomous skill learning loop.
//!
//! Feature-gated behind `learning-loop`. Periodically evaluates the agent's
//! recent tool-call trajectory and either creates a new skill plugin or patches
//! an existing one when the experience is worth capturing.
//!
//! Inspired by the Hermes Agent closed learning loop:
//! observe -> distill -> reuse -> refine.
//!
//! Distillation selects the *weakest* valid hypothesis (Bennett 2023) and
//! requires parent-holdout generalisation before create/patch.

pub mod checkpoint;
pub mod harness_rank;
pub mod memory_nudge;
pub mod skill_manage;
pub mod weakness;

pub use harness_rank::{HarnessImprovementSuggestion, RankedHarnessSuggestion, rank_by_weakness};
pub use weakness::{DistillReport, weakness_score};

use crate::agent::tools::tool::ToolCall;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the learning loop checkpoint system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Number of tool calls between self-evaluation checkpoints. Default: 15.
    pub interval_tool_calls: usize,
    /// Minimum tool calls in a coherent sequence to justify skill creation. Default: 5.
    pub min_tool_calls_for_skill: usize,
    /// Path for the persistent MEMORY.md file. When `None`, memory nudges are skipped.
    pub memory_file_path: Option<PathBuf>,
    /// Automatically create skills from experience. Default: `true`.
    pub auto_create_skills: bool,
    /// Automatically patch existing skills when a better path is found. Default: `true`.
    pub auto_patch_skills: bool,
    /// Maximum number of skills the loop will auto-create per session. Default: 5.
    pub max_skills_per_session: usize,
    /// Prefer weakest valid skill hypotheses over most-specific ones. Default: `true`.
    #[serde(default = "default_true")]
    pub prefer_weak_skills: bool,
    /// Require parent-holdout generalisation before create/patch. Default: `true`.
    #[serde(default = "default_true")]
    pub require_holdout_pass: bool,
    /// Minimum holdout extent in `[0.0, 1.0]` to accept a skill. Default: `1.0`.
    #[serde(default = "default_min_holdout_extent")]
    pub min_holdout_extent: f64,
}

const fn default_true() -> bool {
    true
}

const fn default_min_holdout_extent() -> f64 {
    1.0
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            interval_tool_calls: 15,
            min_tool_calls_for_skill: 5,
            memory_file_path: None,
            auto_create_skills: true,
            auto_patch_skills: true,
            max_skills_per_session: 5,
            prefer_weak_skills: true,
            require_holdout_pass: true,
            min_holdout_extent: 1.0,
        }
    }
}

/// Outcome of an episode (coherent tool-call sequence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeOutcome {
    Success,
    PartialSuccess,
    Failure,
    Abandoned,
}

/// Record of a single tool call within an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub arguments_summary: String,
    pub success: bool,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

impl ToolCallRecord {
    pub fn from_tool_call(tc: &ToolCall, success: bool, duration_ms: u64) -> Self {
        let arguments_summary = serde_json::to_string(&tc.arguments)
            .map(|s| {
                if s.len() > 200 {
                    format!("{}...", caliche_contracts::text::truncate_utf8(&s, 200))
                } else {
                    s
                }
            })
            .unwrap_or_default();
        Self {
            tool_name: tc.name.as_ref().to_string(),
            arguments_summary,
            success,
            duration_ms,
            timestamp: Utc::now(),
        }
    }
}

/// Record of an error encountered during an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub tool_name: String,
    pub error_message: String,
    pub recovered: bool,
    pub timestamp: DateTime<Utc>,
}

/// Record of a user correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionRecord {
    pub original_action: String,
    pub correction: String,
    pub timestamp: DateTime<Utc>,
}

/// A coherent episode of tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub tool_calls: Vec<ToolCallRecord>,
    pub errors: Vec<ErrorRecord>,
    pub user_corrections: Vec<CorrectionRecord>,
    pub outcome: EpisodeOutcome,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
}

impl Episode {
    fn new() -> Self {
        Self {
            tool_calls: Vec::new(),
            errors: Vec::new(),
            user_corrections: Vec::new(),
            outcome: EpisodeOutcome::Abandoned,
            started_at: Utc::now(),
            ended_at: None,
        }
    }
}

/// Lightweight summary for progressive skill disclosure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub usage_count: u32,
    pub last_used: DateTime<Utc>,
}

/// Tracks episodes within a session.
#[derive(Debug)]
pub struct EpisodeTracker {
    current_episode: Episode,
    completed_episodes: VecDeque<Episode>,
    tool_calls_since_checkpoint: usize,
    skills_created_this_session: usize,
    max_completed_episodes: usize,
}

impl EpisodeTracker {
    pub fn new() -> Self {
        Self {
            current_episode: Episode::new(),
            completed_episodes: VecDeque::new(),
            tool_calls_since_checkpoint: 0,
            skills_created_this_session: 0,
            max_completed_episodes: 20,
        }
    }

    /// Record a tool call in the current episode. On failure, the caller is
    /// expected to populate `error_message` via [`Self::record_tool_error`]
    /// when the error message is different from the arguments summary.
    pub fn record_tool_call(&mut self, record: ToolCallRecord) {
        if !record.success {
            self.current_episode.errors.push(ErrorRecord {
                tool_name: record.tool_name.clone(),
                error_message: record.arguments_summary.clone(),
                recovered: false,
                timestamp: record.timestamp,
            });
        }
        self.current_episode.tool_calls.push(record);
        self.tool_calls_since_checkpoint += 1;
    }

    /// Overwrite the last recorded error's message (useful when the caller
    /// has a better-formed error message than the raw arguments summary).
    pub fn amend_last_error_message(&mut self, error_message: String) {
        if let Some(err) = self.current_episode.errors.last_mut() {
            err.error_message = error_message;
        }
    }

    /// Record a user correction in the current episode.
    pub fn record_user_correction(&mut self, original: String, correction: String) {
        self.current_episode
            .user_corrections
            .push(CorrectionRecord {
                original_action: original,
                correction,
                timestamp: Utc::now(),
            });
    }

    /// Mark the most recent error as recovered.
    pub fn mark_last_error_recovered(&mut self) {
        if let Some(err) = self.current_episode.errors.last_mut() {
            err.recovered = true;
        }
    }

    /// Check if a checkpoint should fire based on tool call count.
    pub fn should_checkpoint(&self, interval: usize) -> bool {
        self.tool_calls_since_checkpoint >= interval
    }

    /// Finalize the current episode and rotate it.
    pub fn finalize_episode(&mut self, outcome: EpisodeOutcome) {
        self.current_episode.outcome = outcome;
        self.current_episode.ended_at = Some(Utc::now());
        let episode = std::mem::replace(&mut self.current_episode, Episode::new());
        self.completed_episodes.push_back(episode);
        if self.completed_episodes.len() > self.max_completed_episodes {
            self.completed_episodes.pop_front();
        }
        self.tool_calls_since_checkpoint = 0;
    }

    /// Get the current episode for analysis.
    pub fn current_episode(&self) -> &Episode {
        &self.current_episode
    }

    /// Number of skills created this session.
    pub fn skills_created(&self) -> usize {
        self.skills_created_this_session
    }

    /// Increment the skills-created counter.
    pub fn increment_skills_created(&mut self) {
        self.skills_created_this_session += 1;
    }
}

impl Default for EpisodeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// The main learning loop component. Held by `Agent` (or `AgentService`)
/// and queried after tool execution to decide whether to create/patch skills.
#[derive(Debug)]
pub struct LearningLoop {
    pub config: CheckpointConfig,
    pub episode_tracker: Arc<RwLock<EpisodeTracker>>,
    /// Optional persistent plugin store. When `Some`, `CheckpointAction`s
    /// are forwarded here by the chat loop to durably create/patch skills.
    pub plugin_store: Option<Arc<crate::plugins::PluginStore>>,
}

impl LearningLoop {
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            episode_tracker: Arc::new(RwLock::new(EpisodeTracker::new())),
            plugin_store: None,
        }
    }

    /// Attach a persistent plugin store so that skill create/patch/delete
    /// actions survive process restarts.
    pub fn with_plugin_store(mut self, store: Arc<crate::plugins::PluginStore>) -> Self {
        self.plugin_store = Some(store);
        self
    }

    /// Record a tool call and check whether a checkpoint is due.
    /// Returns `true` when the caller should run `evaluate_checkpoint`.
    pub async fn record_and_check(&self, record: ToolCallRecord) -> bool {
        let mut tracker = self.episode_tracker.write().await;
        tracker.record_tool_call(record);
        tracker.should_checkpoint(self.config.interval_tool_calls)
    }

    /// Mark the latest failed tool call as recovered by a later successful action.
    pub async fn mark_last_error_recovered(&self) {
        let mut tracker = self.episode_tracker.write().await;
        tracker.mark_last_error_recovered();
    }
}
