//! Weakness-based hypothesis selection for skill distillation.
//!
//! Based on Bennett (2023): among hypotheses valid on a child task, maximising
//! *weakness* (extension / generality) is necessary and sufficient to maximise
//! the probability of generalising to a parent task. Description-length /
//! compression is neither necessary nor sufficient.
//!
//! Operational proxies (we do not materialise the full extension lattice):
//! - **specificity**: count of concrete bindings retained (paths, IDs, literals)
//! - **quantified_steps**: abstract procedure steps without episode bindings
//! - **tool_coverage**: unique tools the hypothesis can address
//! - **holdout_extent**: fraction of held-out parent situations still covered
//!
//! Weakness score: `(tool_coverage + quantified_steps) / (1 + specificity)`.

#![expect(
    clippy::expect_used,
    reason = "static Regex/LazyLock literals are reviewed constants; failure is a programmer bug"
)]

use super::{Episode, ErrorRecord, ToolCallRecord};
use regex::Regex;
use std::sync::OnceLock;

/// Report produced when selecting a distilled skill hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillReport {
    /// Candidate label (`specific`, `weak`, or `universal`).
    pub candidate: &'static str,
    pub specificity: u32,
    pub quantified_steps: u32,
    pub tool_coverage: u32,
    /// Composite weakness proxy (higher is better).
    pub weakness: f64,
    pub valid_on_child: bool,
    /// Fraction of holdout situations covered in `[0.0, 1.0]`.
    pub holdout_extent: f64,
    pub holdout_passed: bool,
    pub accepted: bool,
}

/// Inputs for [`DistillReport::from_parts`] (keeps the constructor under clippy's arg limit).
#[derive(Debug, Clone, Copy)]
pub struct DistillReportParts {
    pub candidate: &'static str,
    pub specificity: u32,
    pub quantified_steps: u32,
    pub tool_coverage: u32,
    pub valid_on_child: bool,
    pub holdout_extent: f64,
    pub holdout_passed: bool,
    pub accepted: bool,
}

impl DistillReport {
    /// Build a report and compute the composite weakness score.
    #[must_use]
    pub fn from_parts(parts: DistillReportParts) -> Self {
        let weakness = weakness_score(
            parts.tool_coverage,
            parts.quantified_steps,
            parts.specificity,
        );
        Self {
            candidate: parts.candidate,
            specificity: parts.specificity,
            quantified_steps: parts.quantified_steps,
            tool_coverage: parts.tool_coverage,
            weakness,
            valid_on_child: parts.valid_on_child,
            holdout_extent: parts.holdout_extent,
            holdout_passed: parts.holdout_passed,
            accepted: parts.accepted,
        }
    }
}

/// Composite weakness proxy. Higher means a larger expected extension.
#[must_use]
pub fn weakness_score(tool_coverage: u32, quantified_steps: u32, specificity: u32) -> f64 {
    let numerator = f64::from(tool_coverage.saturating_add(quantified_steps));
    numerator / (1.0 + f64::from(specificity))
}

/// Count concrete bindings that make a hypothesis overly specific.
#[must_use]
pub fn count_specificity(text: &str) -> u32 {
    let mut score = 0u32;
    for re in specificity_patterns() {
        score = score.saturating_add(u32::try_from(re.find_iter(text).count()).unwrap_or(u32::MAX));
    }
    score
}

/// Replace concrete bindings with placeholders so the hypothesis is weaker.
#[must_use]
pub fn abstract_text(text: &str) -> String {
    let mut out = text.to_owned();
    for (re, replacement) in abstraction_replacements() {
        out = re.replace_all(&out, *replacement).into_owned();
    }
    // Collapse runs of whitespace introduced by replacements.
    let collapsed = WHITESPACE_RE
        .get_or_init(|| Regex::new(r"[ \t]{2,}").expect("static whitespace regex"))
        .replace_all(&out, " ");
    collapsed.trim().to_owned()
}

/// Abstract a tool-call argument summary for a quantified procedure step.
#[must_use]
pub fn abstract_arguments_summary(summary: &str) -> String {
    let abstracted = abstract_text(summary);
    if abstracted.is_empty() || abstracted == "{}" || abstracted == "null" {
        "with task-appropriate arguments".to_owned()
    } else {
        abstracted
    }
}

/// A candidate skill hypothesis before selection.
#[derive(Debug, Clone)]
pub struct SkillHypothesis {
    pub label: &'static str,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub required_tools: Vec<String>,
    pub specificity: u32,
    pub quantified_steps: u32,
}

impl SkillHypothesis {
    #[must_use]
    pub fn tool_coverage(&self) -> u32 {
        u32::try_from(self.required_tools.len()).unwrap_or(u32::MAX)
    }

    #[must_use]
    pub fn weakness(&self) -> f64 {
        weakness_score(
            self.tool_coverage(),
            self.quantified_steps,
            self.specificity,
        )
    }
}

/// Whether a hypothesis is valid on the child episode (covers observed successes
/// and recovered failure modes).
#[must_use]
pub fn is_valid_on_child(hypothesis: &SkillHypothesis, episode: &Episode) -> bool {
    let child_tools = unique_tool_names(episode);
    if child_tools.is_empty() {
        return false;
    }
    for tool in &child_tools {
        if !hypothesis
            .required_tools
            .iter()
            .any(|required| required == tool)
        {
            return false;
        }
    }

    let prompt = hypothesis.system_prompt.to_ascii_lowercase();
    for error in recovered_errors(episode) {
        let tool = error.tool_name.to_ascii_lowercase();
        if !prompt.contains(&tool) {
            return false;
        }
    }
    true
}

/// Split an episode into a child (train) prefix and parent-only holdout tools.
///
/// Returns `(train_len, holdout_tool_names)`. When the episode is too small to
/// split, `holdout_tool_names` is empty and callers should treat full-episode
/// validity as the parent check.
#[must_use]
pub fn holdout_split(episode: &Episode) -> (usize, Vec<String>) {
    let n = episode.tool_calls.len();
    if n < 2 {
        return (n, Vec::new());
    }
    // Train on the first ~2/3; hold out the remainder as parent-only situations.
    let train_len = (n * 2).div_ceil(3).max(1).min(n - 1);
    let mut holdout: Vec<String> = episode.tool_calls[train_len..]
        .iter()
        .map(|tc| tc.tool_name.clone())
        .collect();
    holdout.sort_unstable();
    holdout.dedup();
    (train_len, holdout)
}

/// Fraction of holdout tools covered by the hypothesis required-tool set.
#[must_use]
pub fn holdout_extent(hypothesis: &SkillHypothesis, holdout_tools: &[String]) -> f64 {
    if holdout_tools.is_empty() {
        return 1.0;
    }
    let covered = holdout_tools
        .iter()
        .filter(|tool| {
            hypothesis
                .required_tools
                .iter()
                .any(|required| required == *tool)
        })
        .count();
    covered as f64 / holdout_tools.len() as f64
}

/// Select the weakest hypothesis that is valid on the child and passes holdout.
///
/// When `prefer_weak` is false, prefer the most specific valid candidate instead
/// (legacy / ablation behaviour). When no candidate passes, returns `None`.
#[must_use]
pub fn select_hypothesis(
    candidates: &[SkillHypothesis],
    child_episode: &Episode,
    holdout_tools: &[String],
    prefer_weak: bool,
    min_holdout_extent: f64,
    require_holdout_pass: bool,
) -> Option<(usize, DistillReport)> {
    let min_extent = min_holdout_extent.clamp(0.0, 1.0);
    let mut best: Option<(usize, DistillReport)> = None;

    for (idx, candidate) in candidates.iter().enumerate() {
        let valid = is_valid_on_child(candidate, child_episode);
        let extent = holdout_extent(candidate, holdout_tools);
        let holdout_ok = extent + f64::EPSILON >= min_extent;
        let accepted = valid && (!require_holdout_pass || holdout_ok || holdout_tools.is_empty());
        let report = DistillReport::from_parts(DistillReportParts {
            candidate: candidate.label,
            specificity: candidate.specificity,
            quantified_steps: candidate.quantified_steps,
            tool_coverage: candidate.tool_coverage(),
            valid_on_child: valid,
            holdout_extent: extent,
            holdout_passed: holdout_ok || holdout_tools.is_empty(),
            accepted,
        });

        if !accepted {
            continue;
        }

        let replace = match &best {
            None => true,
            Some((_, best_report)) => {
                if prefer_weak {
                    report.weakness > best_report.weakness + f64::EPSILON
                        || ((report.weakness - best_report.weakness).abs() <= f64::EPSILON
                            && report.specificity < best_report.specificity)
                } else {
                    // Ablation: prefer shorter / more specific (anti-weakness).
                    report.specificity > best_report.specificity
                        || (report.specificity == best_report.specificity
                            && report.weakness < best_report.weakness)
                }
            }
        };
        if replace {
            best = Some((idx, report));
        }
    }

    best
}

/// Build the three standard candidates (specific, weak, universal) from an episode.
#[must_use]
pub fn build_candidates(episode: &Episode, tool_call_count: usize) -> Vec<SkillHypothesis> {
    let tool_names = unique_tool_names(episode);
    let skill_stem = tool_names.first().map(String::as_str).unwrap_or("unknown");

    let specific_prompt = build_procedure_prompt(episode, false);
    let weak_prompt = build_procedure_prompt(episode, true);
    let universal_prompt = build_universal_prompt(episode, &tool_names);

    let specific_desc = format!(
        "Auto-generated skill from a {tool_call_count}-step workflow using: {}",
        tool_names.join(", ")
    );
    let weak_desc = format!(
        "Generalised workflow over tools: {}. Prefer quantified steps; bind arguments from the current task.",
        tool_names.join(", ")
    );
    let universal_desc = format!(
        "Universally quantified multi-tool procedure covering: {}",
        tool_names.join(", ")
    );

    vec![
        SkillHypothesis {
            label: "specific",
            name: format!("auto_{skill_stem}_step_workflow"),
            description: specific_desc.clone(),
            system_prompt: specific_prompt.clone(),
            required_tools: tool_names.clone(),
            specificity: count_specificity(&specific_prompt)
                .saturating_add(count_specificity(&specific_desc)),
            quantified_steps: count_quantified_steps(&specific_prompt),
        },
        SkillHypothesis {
            label: "weak",
            name: format!("auto_{skill_stem}_general_workflow"),
            description: weak_desc.clone(),
            system_prompt: weak_prompt.clone(),
            required_tools: tool_names.clone(),
            specificity: count_specificity(&weak_prompt)
                .saturating_add(count_specificity(&weak_desc)),
            quantified_steps: count_quantified_steps(&weak_prompt),
        },
        SkillHypothesis {
            label: "universal",
            name: format!("auto_{skill_stem}_universal_workflow"),
            description: universal_desc.clone(),
            system_prompt: universal_prompt.clone(),
            required_tools: tool_names,
            specificity: count_specificity(&universal_prompt)
                .saturating_add(count_specificity(&universal_desc)),
            quantified_steps: count_quantified_steps(&universal_prompt),
        },
    ]
}

/// Child episode used for validity: train prefix of tool calls (and related errors).
#[must_use]
pub fn child_episode(episode: &Episode, train_len: usize) -> Episode {
    let train_len = train_len.min(episode.tool_calls.len());
    let tool_calls: Vec<ToolCallRecord> = episode.tool_calls[..train_len].to_vec();
    let cutoff = tool_calls.last().map(|tc| tc.timestamp);
    let errors: Vec<ErrorRecord> = match cutoff {
        Some(ts) => episode
            .errors
            .iter()
            .filter(|err| err.timestamp <= ts)
            .cloned()
            .collect(),
        None => Vec::new(),
    };
    let user_corrections = episode
        .user_corrections
        .iter()
        .filter(|c| match cutoff {
            Some(ts) => c.timestamp <= ts,
            None => false,
        })
        .cloned()
        .collect();

    Episode {
        tool_calls,
        errors,
        user_corrections,
        outcome: episode.outcome,
        started_at: episode.started_at,
        ended_at: episode.ended_at,
    }
}

fn unique_tool_names(episode: &Episode) -> Vec<String> {
    let mut names: Vec<String> = episode
        .tool_calls
        .iter()
        .map(|tc| tc.tool_name.clone())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

fn recovered_errors(episode: &Episode) -> impl Iterator<Item = &ErrorRecord> {
    episode.errors.iter().filter(|error| error.recovered)
}

fn count_quantified_steps(prompt: &str) -> u32 {
    prompt
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.bytes().next().is_some_and(|b| b.is_ascii_digit()) && trimmed.contains("Use `")
        })
        .count()
        .try_into()
        .unwrap_or(u32::MAX)
}

fn build_procedure_prompt(episode: &Episode, abstract_args: bool) -> String {
    let tool_names = unique_tool_names(episode);
    let skill_name = format!(
        "auto_{}_workflow",
        tool_names.first().map(String::as_str).unwrap_or("unknown")
    );

    let procedure_lines: Vec<String> = episode
        .tool_calls
        .iter()
        .enumerate()
        .map(|(i, tc)| {
            let args = if abstract_args {
                abstract_arguments_summary(&tc.arguments_summary)
            } else {
                tc.arguments_summary.clone()
            };
            format!("{}. Use `{}` -- {}", i + 1, tc.tool_name, args)
        })
        .collect();

    let pitfalls: Vec<String> = recovered_errors(episode)
        .map(|error| {
            let detail = if abstract_args {
                abstract_text(&error.error_message)
            } else {
                error.error_message.clone()
            };
            format!("- `{}` may fail: {detail}", error.tool_name)
        })
        .collect();

    let mut system_prompt = format!(
        "# {skill_name}\n\n## Procedure\n{}\n",
        procedure_lines.join("\n")
    );
    if !pitfalls.is_empty() {
        system_prompt.push_str(&format!("\n## Pitfalls\n{}\n", pitfalls.join("\n")));
    }
    if abstract_args {
        system_prompt.push_str(
            "\n## Binding rule\nInstantiate placeholders from the current task; \
             do not reuse prior episode literals.\n",
        );
    }
    system_prompt
}

fn build_universal_prompt(episode: &Episode, tool_names: &[String]) -> String {
    let procedure_lines: Vec<String> = tool_names
        .iter()
        .enumerate()
        .map(|(i, tool)| {
            format!(
                "{}. Use `{tool}` -- with arguments bound from the current situation",
                i + 1
            )
        })
        .collect();
    let pitfalls: Vec<String> = recovered_errors(episode)
        .map(|error| {
            format!(
                "- When `{}` fails, recover before continuing the workflow.",
                error.tool_name
            )
        })
        .collect();

    let mut prompt = format!(
        "# Universal multi-tool workflow\n\n## Procedure\n{}\n",
        procedure_lines.join("\n")
    );
    if !pitfalls.is_empty() {
        prompt.push_str(&format!("\n## Pitfalls\n{}\n", pitfalls.join("\n")));
    }
    prompt.push_str(
        "\n## Binding rule\nAssert only what the situation requires; \
         prefer universally quantified steps over episode-specific scripts.\n",
    );
    prompt
}

fn specificity_patterns() -> &'static [Regex] {
    SPECIFICITY_RES.get_or_init(|| {
        [
            r#"(?i)(/[\w.~+-]+)+"#,                // unix paths
            r#"(?i)[A-Za-z]:\\(?:[\w.~+-]+\\?)+"#, // windows paths
            r#"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"#, // uuid
            r#""[^"]{3,}""#, // quoted literals
            r#"\b\d{3,}\b"#, // long numbers
        ]
        .into_iter()
        .map(|pattern| Regex::new(pattern).expect("static specificity regex"))
        .collect()
    })
}

fn abstraction_replacements() -> &'static [(Regex, &'static str)] {
    ABSTRACTION_RES.get_or_init(|| {
        [
            (
                r#"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"#,
                "<id>",
            ),
            (r#"(?i)(/[\w.~+-]+)+"#, "<path>"),
            (r#"(?i)[A-Za-z]:\\(?:[\w.~+-]+\\?)+"#, "<path>"),
            (r#""[^"]{3,}""#, "\"<value>\""),
            (r#"\b\d{3,}\b"#, "<n>"),
        ]
        .into_iter()
        .map(|(pattern, replacement)| {
            (
                Regex::new(pattern).expect("static abstraction regex"),
                replacement,
            )
        })
        .collect()
    })
}

static SPECIFICITY_RES: OnceLock<Vec<Regex>> = OnceLock::new();
static ABSTRACTION_RES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
static WHITESPACE_RE: OnceLock<Regex> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::learning::{EpisodeOutcome, ToolCallRecord};
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

    fn sample_episode() -> Episode {
        Episode {
            tool_calls: vec![
                record(
                    "read_file",
                    r#"{"path":"/home/user/dev/src/main.rs"}"#,
                    true,
                ),
                record(
                    "edit_file",
                    r#"{"path":"/home/user/dev/src/main.rs","old":"foo123"}"#,
                    false,
                ),
                record(
                    "edit_file",
                    r#"{"path":"/home/user/dev/src/main.rs","old":"foo123"}"#,
                    true,
                ),
                record("bash", r#"{"command":"cargo test --features full"}"#, true),
            ],
            errors: vec![ErrorRecord {
                tool_name: "edit_file".to_owned(),
                error_message: "failed to edit /home/user/dev/src/main.rs".to_owned(),
                recovered: true,
                timestamp: Utc::now(),
            }],
            user_corrections: Vec::new(),
            outcome: EpisodeOutcome::Success,
            started_at: Utc::now(),
            ended_at: Some(Utc::now()),
        }
    }

    #[test]
    fn abstract_text_strips_paths_and_literals() {
        let raw = r#"edit /home/user/dev/src/main.rs with "foo123" and 99999"#;
        let abstracted = abstract_text(raw);
        assert!(abstracted.contains("<path>"));
        assert!(abstracted.contains("\"<value>\""));
        assert!(abstracted.contains("<n>"));
        assert!(!abstracted.contains("/home/user"));
        assert!(count_specificity(&abstracted) < count_specificity(raw));
    }

    #[test]
    fn weakness_prefers_less_specific_hypothesis() {
        let specific = weakness_score(3, 2, 10);
        let weak = weakness_score(3, 4, 1);
        assert!(weak > specific);
    }

    #[test]
    fn select_hypothesis_chooses_weak_over_specific() {
        let episode = sample_episode();
        let (train_len, holdout) = holdout_split(&episode);
        let child = child_episode(&episode, train_len);
        let candidates = build_candidates(&episode, episode.tool_calls.len());
        let (idx, report) = select_hypothesis(&candidates, &child, &holdout, true, 1.0, true)
            .expect("should select a candidate");
        assert!(report.accepted);
        assert!(report.valid_on_child);
        assert_ne!(candidates[idx].label, "specific");
        assert!(report.weakness >= candidates[0].weakness());
    }

    #[test]
    fn holdout_extent_is_fractional() {
        let hypothesis = SkillHypothesis {
            label: "weak",
            name: "t".into(),
            description: "d".into(),
            system_prompt: "## Procedure\n1. Use `read_file` -- x\n".into(),
            required_tools: vec!["read_file".into()],
            specificity: 0,
            quantified_steps: 1,
        };
        let extent = holdout_extent(&hypothesis, &["read_file".into(), "bash".into()]);
        assert!((extent - 0.5).abs() < f64::EPSILON);
    }
}
