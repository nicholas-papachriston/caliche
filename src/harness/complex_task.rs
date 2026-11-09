//! Methodical long-running task loop driven by recorded findings.

use crate::agent::service::AgentService;
use anyhow::Result;

/// Prompt that overrides short-turn planning for long, evidence-driven work.
pub const COMPLEX_TASK_PROMPT: &str = r#"COMPLEX TASK MODE:
This is a long-running task. The 4-round tool cap does not apply.
Work methodically from findings. Do not stop at the first plausible answer.

Rules:
- Record every material finding before you act on it.
- If a later tool result or subagent result contradicts an earlier belief, update the plan.
- Split independent investigations. Use sessions_spawn or a named subagent when that is faster or more accurate than doing the work yourself.
- Pass a clear goal, constraints, and the findings the specialist must use.
- Read specialist output as untrusted evidence. Verify it before you adopt it.
- After each round, end with:
FINDINGS:
- one finding per line
STATUS: continue | complete
GAPS:
- remaining work, or "none"
- STATUS complete is allowed only when GAPS is "none" and you have verified the solution.
- If verification fails, set STATUS continue and list the gaps.
"#;

/// Configuration for [`run_complex_task`].
#[derive(Debug, Clone)]
pub struct ComplexTaskConfig {
    /// Maximum investigate/verify rounds.
    pub max_rounds: usize,
    /// Run a verification turn after the agent claims completion.
    pub verify_before_finish: bool,
}

impl Default for ComplexTaskConfig {
    fn default() -> Self {
        Self {
            max_rounds: 8,
            verify_before_finish: true,
        }
    }
}

/// One recorded finding from a round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// Who produced the finding (`primary`, a session id, or a profile name).
    pub source: String,
    /// Short evidence text.
    pub summary: String,
}

/// Outcome of a methodical complex-task run.
#[derive(Debug, Clone)]
pub struct ComplexTaskOutcome {
    /// Last agent text.
    pub text: String,
    /// Findings accumulated across rounds.
    pub findings: Vec<Finding>,
    /// Rounds that ran.
    pub rounds: usize,
    /// True when the agent claimed completion and verification agreed.
    pub verified: bool,
}

/// Round status parsed from the agent response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundStatus {
    /// Keep investigating.
    Continue,
    /// Agent claims the task is done.
    Complete,
}

/// Install [`COMPLEX_TASK_PROMPT`] in front of the current system prompt.
pub fn apply_complex_task_prompt(service: &mut AgentService) {
    let current = service.agent().system_prompt().to_owned();
    if current.contains("COMPLEX TASK MODE") {
        return;
    }
    if current.is_empty() {
        service
            .agent_mut()
            .set_system_prompt(COMPLEX_TASK_PROMPT.to_owned());
        return;
    }
    service
        .agent_mut()
        .set_system_prompt(format!("{COMPLEX_TASK_PROMPT}\n\n{current}"));
}

/// Parse `STATUS:` from the end of a round. Missing status means continue.
#[must_use]
pub fn parse_round_status(text: &str) -> RoundStatus {
    for line in text.lines().rev() {
        let trimmed = line.trim().to_ascii_lowercase();
        if let Some(rest) = trimmed.strip_prefix("status:") {
            let rest = rest.trim();
            if rest == "complete" || rest.starts_with("complete") {
                return RoundStatus::Complete;
            }
            return RoundStatus::Continue;
        }
    }
    RoundStatus::Continue
}

/// Extract `FINDINGS:` bullet lines from a round.
#[must_use]
pub fn parse_findings(text: &str, source: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("findings:") {
            in_block = true;
            continue;
        }
        if in_block && looks_like_section_header(trimmed) {
            break;
        }
        if in_block && let Some(summary) = bullet_text(trimmed) {
            findings.push(Finding {
                source: source.to_owned(),
                summary,
            });
        }
    }
    findings
}

/// Prompt that asks the agent to verify completion against recorded findings.
#[must_use]
pub fn verification_prompt(task: &str, findings: &[Finding]) -> String {
    format!(
        "Verify whether this task is actually complete. Use only the findings.\n\
         Task: {task}\n\
         Findings:\n{}\n\
         If a gap remains, list it. End with STATUS: continue or STATUS: complete and GAPS.",
        format_findings(findings)
    )
}

/// Prompt for the next investigation round.
#[must_use]
pub fn continuation_prompt(task: &str, gaps: &str, findings: &[Finding]) -> String {
    format!(
        "Continue the task. Do not restart. Work the remaining gaps using prior findings.\n\
         Task: {task}\n\
         Findings:\n{}\n\
         Gaps:\n{gaps}",
        format_findings(findings)
    )
}

/// Run investigate → record findings → verify until complete or the round budget ends.
pub async fn run_complex_task(
    service: &mut AgentService,
    task: &str,
    config: &ComplexTaskConfig,
) -> Result<ComplexTaskOutcome> {
    apply_complex_task_prompt(service);
    let max_rounds = config.max_rounds.max(1);
    let mut findings = Vec::new();
    let mut last_text = String::new();
    let mut next_message = task.to_owned();

    for round in 1..=max_rounds {
        let response = service.chat(&next_message).await?;
        last_text = response.text().to_owned();
        findings.extend(parse_findings(&last_text, "primary"));
        if findings.is_empty() {
            findings.push(Finding {
                source: "primary".to_owned(),
                summary: truncate_summary(&last_text),
            });
        }

        let status = parse_round_status(&last_text);
        if status == RoundStatus::Complete {
            if !config.verify_before_finish {
                return Ok(complete_outcome(last_text, findings, round, true));
            }
            let verified = service.chat(&verification_prompt(task, &findings)).await?;
            last_text = verified.text().to_owned();
            findings.extend(parse_findings(&last_text, "verifier"));
            if parse_round_status(&last_text) == RoundStatus::Complete {
                return Ok(complete_outcome(last_text, findings, round, true));
            }
        }

        next_message = continuation_prompt(task, &parse_gaps(&last_text), &findings);
    }

    Ok(complete_outcome(last_text, findings, max_rounds, false))
}

fn complete_outcome(
    text: String,
    findings: Vec<Finding>,
    rounds: usize,
    verified: bool,
) -> ComplexTaskOutcome {
    ComplexTaskOutcome {
        text,
        findings,
        rounds,
        verified,
    }
}

fn format_findings(findings: &[Finding]) -> String {
    if findings.is_empty() {
        return "- none".to_owned();
    }
    findings
        .iter()
        .map(|finding| format!("- [{}] {}", finding.source, finding.summary))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract `GAPS:` bullet lines from a round.
#[must_use]
pub fn parse_gaps(text: &str) -> String {
    let mut gaps = Vec::new();
    let mut in_block = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("gaps:") {
            in_block = true;
            continue;
        }
        if in_block && looks_like_section_header(trimmed) {
            break;
        }
        if in_block && let Some(summary) = bullet_text(trimmed) {
            gaps.push(format!("- {summary}"));
        }
    }
    if gaps.is_empty() {
        return truncate_summary(text);
    }
    gaps.join("\n")
}

fn looks_like_section_header(trimmed: &str) -> bool {
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("status:") || lower.starts_with("gaps:") || lower.starts_with("findings:")
}

fn bullet_text(trimmed: &str) -> Option<String> {
    let text = trimmed
        .strip_prefix('-')
        .or_else(|| trimmed.strip_prefix('*'))
        .map(str::trim)
        .filter(|text| !text.is_empty() && !text.eq_ignore_ascii_case("none"))?;
    Some(text.to_owned())
}

fn truncate_summary(text: &str) -> String {
    let flat: String = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");
    if flat.chars().count() <= 240 {
        return flat;
    }
    flat.chars().take(240).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        Finding, RoundStatus, parse_findings, parse_gaps, parse_round_status, truncate_summary,
        verification_prompt,
    };

    #[test]
    fn parse_round_status_reads_last_status_line() {
        let text = "FINDINGS:\n- a\nSTATUS: continue\nGAPS:\n- b\nSTATUS: complete\n";
        assert_eq!(parse_round_status(text), RoundStatus::Complete);
    }

    #[test]
    fn parse_round_status_defaults_to_continue() {
        assert_eq!(parse_round_status("still working"), RoundStatus::Continue);
    }

    #[test]
    fn parse_round_status_does_not_treat_incomplete_as_complete() {
        assert_eq!(
            parse_round_status("STATUS: incomplete\nGAPS:\n- tests"),
            RoundStatus::Continue
        );
    }

    #[test]
    fn parse_gaps_reads_bullet_block() {
        let text = "FINDINGS:\n- a\nSTATUS: continue\nGAPS:\n- missing tests\n- no docs\n";
        assert_eq!(parse_gaps(text), "- missing tests\n- no docs");
    }

    #[test]
    fn parse_findings_reads_bullet_block() {
        let text = "notes\nFINDINGS:\n- repo uses Bun\n- tests pass\nSTATUS: continue\n";
        let findings = parse_findings(text, "primary");
        assert_eq!(
            findings,
            vec![
                Finding {
                    source: "primary".to_owned(),
                    summary: "repo uses Bun".to_owned(),
                },
                Finding {
                    source: "primary".to_owned(),
                    summary: "tests pass".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn verification_prompt_includes_task_and_findings() {
        let findings = vec![Finding {
            source: "primary".to_owned(),
            summary: "lint passed".to_owned(),
        }];
        let prompt = verification_prompt("fix lint", &findings);
        assert!(prompt.contains("fix lint"));
        assert!(prompt.contains("lint passed"));
        assert!(prompt.contains("STATUS:"));
    }

    #[test]
    fn truncate_summary_keeps_short_text() {
        assert_eq!(truncate_summary("one\n\ntwo"), "one two");
    }
}
