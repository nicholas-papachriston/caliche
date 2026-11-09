//! Self-evaluation checkpoint: analyse recent trajectory, decide whether an
//! experience is worth capturing as a skill or memory nudge.
//!
//! Skill distillation maximises *weakness* among hypotheses valid on the child
//! episode, then accepts only candidates that generalise to a parent holdout
//! split (Bennett 2023).

use super::weakness::{
    DistillReport, DistillReportParts, SkillHypothesis, build_candidates, child_episode,
    holdout_split, select_hypothesis,
};
use super::{CheckpointConfig, Episode, EpisodeOutcome, LearningLoop, SkillSummary};
use crate::plugins::schema::{PluginDefinition, PluginSpec, PluginType, SkillPluginSpec};
use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

/// Result of evaluating a checkpoint.
#[derive(Debug)]
pub enum CheckpointAction {
    /// No action -- the episode is routine.
    Skip,
    /// Create a new skill from the episode trajectory.
    CreateSkill {
        definition: Box<PluginDefinition>,
        report: DistillReport,
    },
    /// Patch an existing skill (identified by id) with updated steps.
    PatchSkill {
        skill_id: String,
        patch: SkillPatch,
        report: DistillReport,
    },
    /// Persist durable facts to MEMORY.md.
    MemoryNudge(Vec<String>),
}

/// A patch to apply to an existing skill plugin.
#[derive(Debug, Clone)]
pub struct SkillPatch {
    pub new_description: Option<String>,
    pub new_system_prompt_template: Option<String>,
    pub additional_required_tools: Vec<String>,
}

/// Criteria for deciding whether an episode is worth capturing.
#[derive(Debug)]
struct EpisodeAnalysis {
    tool_call_count: usize,
    unique_tools: usize,
    had_error_recovery: bool,
    had_user_correction: bool,
    is_multi_step_workflow: bool,
}

impl EpisodeAnalysis {
    fn from_episode(episode: &Episode) -> Self {
        let tool_call_count = episode.tool_calls.len();
        let unique_tools = {
            let mut names: Vec<&str> = episode
                .tool_calls
                .iter()
                .map(|tc| tc.tool_name.as_str())
                .collect();
            names.sort_unstable();
            names.dedup();
            names.len()
        };
        let had_error_recovery = episode.errors.iter().any(|e| e.recovered);
        let had_user_correction = !episode.user_corrections.is_empty();
        let is_multi_step_workflow = unique_tools >= 2 && tool_call_count >= 3;

        Self {
            tool_call_count,
            unique_tools,
            had_error_recovery,
            had_user_correction,
            is_multi_step_workflow,
        }
    }

    /// Whether this episode meets the criteria for skill creation.
    fn worth_capturing(&self, config: &CheckpointConfig) -> bool {
        if self.tool_call_count < config.min_tool_calls_for_skill {
            return false;
        }
        self.had_error_recovery
            || self.had_user_correction
            || self.is_multi_step_workflow
            || self.unique_tools >= 3
            || self.tool_call_count >= config.min_tool_calls_for_skill * 2
    }
}

impl LearningLoop {
    /// Run the self-evaluation checkpoint on the current episode.
    ///
    /// Returns one or more actions the caller should take (create skill,
    /// patch existing, nudge memory, or skip).
    pub async fn evaluate_checkpoint(
        &self,
        existing_skills: &[SkillSummary],
    ) -> Result<Vec<CheckpointAction>> {
        let tracker = self.episode_tracker.read().await;
        let episode = tracker.current_episode();
        let analysis = EpisodeAnalysis::from_episode(episode);

        let mut actions = Vec::new();

        if !analysis.worth_capturing(&self.config) {
            tracing::debug!(
                tool_calls = analysis.tool_call_count,
                "Checkpoint: episode not worth capturing"
            );
            return Ok(vec![CheckpointAction::Skip]);
        }

        if tracker.skills_created() >= self.config.max_skills_per_session {
            tracing::info!(
                "Checkpoint: skill creation limit reached ({}/{})",
                tracker.skills_created(),
                self.config.max_skills_per_session
            );
            return Ok(vec![CheckpointAction::Skip]);
        }

        let workflow_signature = build_workflow_signature(episode);
        let distill = distill_skill(episode, &analysis, &self.config);

        match distill {
            DistillOutcome::Accepted { hypothesis, report } => {
                log_distill_report(&report);
                if let Some(matching) = find_matching_skill(existing_skills, &workflow_signature) {
                    if self.config.auto_patch_skills && analysis.had_error_recovery {
                        let patch = build_skill_patch(&hypothesis);
                        actions.push(CheckpointAction::PatchSkill {
                            skill_id: matching.id.clone(),
                            patch,
                            report,
                        });
                    }
                } else if self.config.auto_create_skills {
                    let definition = build_skill_definition(&hypothesis);
                    actions.push(CheckpointAction::CreateSkill {
                        definition: Box::new(definition),
                        report,
                    });
                }
            }
            DistillOutcome::Rejected { reports } => {
                for report in &reports {
                    log_distill_report(report);
                }
                tracing::info!(
                    candidates = reports.len(),
                    "Checkpoint: no skill hypothesis passed weakness+holdout gate"
                );
            }
        }

        let facts = extract_durable_facts(episode);
        if !facts.is_empty() {
            actions.push(CheckpointAction::MemoryNudge(facts));
        }

        if actions.is_empty() {
            actions.push(CheckpointAction::Skip);
        }

        Ok(actions)
    }

    /// Finalize the current episode after a checkpoint evaluation.
    pub async fn finalize_current_episode(&self, outcome: EpisodeOutcome) {
        let mut tracker = self.episode_tracker.write().await;
        tracker.finalize_episode(outcome);
    }

    /// Increment the created-skills counter.
    pub async fn mark_skill_created(&self) {
        let mut tracker = self.episode_tracker.write().await;
        tracker.increment_skills_created();
    }
}

enum DistillOutcome {
    Accepted {
        hypothesis: SkillHypothesis,
        report: DistillReport,
    },
    Rejected {
        reports: Vec<DistillReport>,
    },
}

fn distill_skill(
    episode: &Episode,
    analysis: &EpisodeAnalysis,
    config: &CheckpointConfig,
) -> DistillOutcome {
    let (train_len, holdout_tools) = holdout_split(episode);
    let child = child_episode(episode, train_len);
    let candidates = build_candidates(episode, analysis.tool_call_count);

    let mut rejected: Vec<DistillReport> = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        let valid = super::weakness::is_valid_on_child(candidate, &child);
        let extent = super::weakness::holdout_extent(candidate, &holdout_tools);
        let holdout_ok = extent + f64::EPSILON >= config.min_holdout_extent.clamp(0.0, 1.0);
        rejected.push(DistillReport::from_parts(DistillReportParts {
            candidate: candidate.label,
            specificity: candidate.specificity,
            quantified_steps: candidate.quantified_steps,
            tool_coverage: candidate.tool_coverage(),
            valid_on_child: valid,
            holdout_extent: extent,
            holdout_passed: holdout_ok || holdout_tools.is_empty(),
            accepted: false,
        }));
    }

    match select_hypothesis(
        &candidates,
        &child,
        &holdout_tools,
        config.prefer_weak_skills,
        config.min_holdout_extent,
        config.require_holdout_pass,
    ) {
        Some((idx, mut report)) => {
            report.accepted = true;
            DistillOutcome::Accepted {
                hypothesis: candidates[idx].clone(),
                report,
            }
        }
        None => DistillOutcome::Rejected { reports: rejected },
    }
}

fn log_distill_report(report: &DistillReport) {
    tracing::info!(
        candidate = report.candidate,
        specificity = report.specificity,
        quantified_steps = report.quantified_steps,
        tool_coverage = report.tool_coverage,
        weakness = report.weakness,
        valid_on_child = report.valid_on_child,
        holdout_extent = report.holdout_extent,
        holdout_passed = report.holdout_passed,
        accepted = report.accepted,
        "learning distill weakness report"
    );
}

/// Build a signature string from the tool-call sequence for matching against
/// existing skills.
fn build_workflow_signature(episode: &Episode) -> String {
    episode
        .tool_calls
        .iter()
        .map(|tc| tc.tool_name.as_str())
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// Find an existing skill whose name or description overlaps with the
/// workflow signature.
fn find_matching_skill<'a>(
    skills: &'a [SkillSummary],
    signature: &str,
) -> Option<&'a SkillSummary> {
    let sig_tools: Vec<&str> = signature.split(" -> ").collect();
    skills.iter().find(|s| {
        let desc_lower = s.description.to_ascii_lowercase();
        sig_tools
            .iter()
            .any(|tool| desc_lower.contains(&tool.to_ascii_lowercase()))
    })
}

/// Construct a `PluginDefinition` (type = Skill) from a selected hypothesis.
fn build_skill_definition(hypothesis: &SkillHypothesis) -> PluginDefinition {
    let now = Utc::now();
    PluginDefinition {
        id: format!("auto-skill-{}", uuid::Uuid::now_v7()),
        name: hypothesis.name.clone(),
        version: "1.0.0".to_string(),
        description: hypothesis.description.clone(),
        author: Some("learning-loop".to_string()),
        tags: vec![
            "auto-generated".to_string(),
            format!("distill:{}", hypothesis.label),
            "weakness-selected".to_string(),
        ],
        plugin_type: PluginType::Skill,
        spec: PluginSpec::Skill(SkillPluginSpec {
            system_prompt_template: hypothesis.system_prompt.clone(),
            parameters: HashMap::new(),
            required_tools: hypothesis.required_tools.clone(),
            restrict_tools: false,
            max_iterations: None,
        }),
        created_at: now,
        updated_at: now,
    }
}

/// Build a patch for an existing skill from the selected weak hypothesis.
fn build_skill_patch(hypothesis: &SkillHypothesis) -> SkillPatch {
    SkillPatch {
        new_description: Some(hypothesis.description.clone()),
        new_system_prompt_template: Some(hypothesis.system_prompt.clone()),
        additional_required_tools: hypothesis.required_tools.clone(),
    }
}

/// Extract durable facts from an episode worth persisting to MEMORY.md.
///
/// Facts are abstracted (paths/IDs stripped) so memory stays weak/general.
fn extract_durable_facts(episode: &Episode) -> Vec<String> {
    let mut facts = Vec::new();

    for correction in &episode.user_corrections {
        facts.push(format!(
            "User corrected: \"{}\" -> \"{}\"",
            super::weakness::abstract_text(&correction.original_action),
            super::weakness::abstract_text(&correction.correction)
        ));
    }

    for error in &episode.errors {
        if error.recovered {
            facts.push(format!(
                "Tool `{}` can fail with: {}. Recovery is possible.",
                error.tool_name,
                super::weakness::abstract_text(&error.error_message)
            ));
        }
    }

    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::learning::{
        CheckpointConfig, Episode, EpisodeOutcome, ErrorRecord, ToolCallRecord,
    };
    use chrono::Utc;

    fn record(tool: &str, args: &str, success: bool) -> ToolCallRecord {
        ToolCallRecord {
            tool_name: tool.to_owned(),
            arguments_summary: args.to_owned(),
            success,
            duration_ms: 1,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn distill_prefers_weak_hypothesis() {
        let episode = Episode {
            tool_calls: vec![
                record("read_file", r#"{"path":"/tmp/a/b/c.rs"}"#, true),
                record("edit_file", r#"{"path":"/tmp/a/b/c.rs"}"#, false),
                record("edit_file", r#"{"path":"/tmp/a/b/c.rs"}"#, true),
                record("bash", r#"{"command":"cargo test suite42"}"#, true),
            ],
            errors: vec![ErrorRecord {
                tool_name: "edit_file".into(),
                error_message: "failed /tmp/a/b/c.rs".into(),
                recovered: true,
                timestamp: Utc::now(),
            }],
            user_corrections: Vec::new(),
            outcome: EpisodeOutcome::Success,
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
        };
        let analysis = EpisodeAnalysis::from_episode(&episode);
        let config = CheckpointConfig::default();
        match distill_skill(&episode, &analysis, &config) {
            DistillOutcome::Accepted { hypothesis, report } => {
                assert!(report.accepted);
                assert_ne!(hypothesis.label, "specific");
                assert!(report.weakness > 0.0);
            }
            DistillOutcome::Rejected { reports } => {
                panic!("expected accepted distill, got rejected: {reports:?}");
            }
        }
    }

    #[test]
    fn holdout_gate_can_reject_all() {
        let episode = Episode {
            tool_calls: vec![
                record("read_file", "{}", true),
                record("unique_parent_only_tool", "{}", true),
            ],
            errors: Vec::new(),
            user_corrections: Vec::new(),
            outcome: EpisodeOutcome::Success,
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
        };
        // Build candidates only from the train prefix tools by constructing a
        // mismatched required_tools set via config requiring holdout pass at 1.0.
        // With train_len=1, holdout is unique_parent_only_tool; candidates built
        // from the full episode include both tools so they should still pass.
        // Force rejection by requiring holdout while prefer_weak uses candidates
        // that omit the holdout tool:
        let analysis = EpisodeAnalysis {
            tool_call_count: 2,
            unique_tools: 2,
            had_error_recovery: false,
            had_user_correction: false,
            is_multi_step_workflow: false,
        };
        let mut config = CheckpointConfig::default();
        config.require_holdout_pass = true;
        config.min_holdout_extent = 1.0;
        // Manually select among candidates that only list read_file.
        let child = child_episode(&episode, 1);
        let candidates = vec![SkillHypothesis {
            label: "specific",
            name: "narrow".into(),
            description: "only read".into(),
            system_prompt: "## Procedure\n1. Use `read_file` -- x\n".into(),
            required_tools: vec!["read_file".into()],
            specificity: 0,
            quantified_steps: 1,
        }];
        let selected = select_hypothesis(
            &candidates,
            &child,
            &["unique_parent_only_tool".into()],
            true,
            1.0,
            true,
        );
        assert!(selected.is_none());
        // Full distill from the episode should still accept because candidates
        // include both tools.
        match distill_skill(&episode, &analysis, &config) {
            DistillOutcome::Accepted { report, .. } => assert!(report.holdout_passed),
            DistillOutcome::Rejected { .. } => {
                panic!("full-episode candidates should cover holdout")
            }
        }
    }
}
