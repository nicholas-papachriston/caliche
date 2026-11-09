//! Project context (AGENTS.md and similar files) sized to the model's
//! context window.
//!
//! The service keeps the file and the section it put in the system prompt.
//! When the model or the window changes, the section is sized again and
//! swapped in place, so a smaller window cuts the file and a larger one
//! restores it.

use super::AgentService;
use super::lifecycle::apply_context_window_limit;
use caliche_contracts::text::truncate_utf8;
use std::path::PathBuf;
use std::time::Duration;

/// Default share, in percent, of the model's context window that project
/// context may take.
pub const DEFAULT_PROJECT_CONTEXT_SHARE: u8 = 25;

/// Characters per token when sizing project context. Code and Markdown run
/// about 3 to 4; the low end keeps the cut inside the budget.
const PROJECT_CONTEXT_CHARS_PER_TOKEN: usize = 3;

/// How long to wait for a local server to report the model's window.
const REPORTED_WINDOW_TIMEOUT: Duration = Duration::from_secs(3);

/// A project instruction file and the share of the window it may take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectContext {
    /// Path to the file.
    pub path: PathBuf,
    /// File name, for display.
    pub source_filename: String,
    /// The whole file.
    pub content: String,
    /// Percent of the context window the section may take (1 to 100).
    pub share_percent: u8,
}

impl ProjectContext {
    /// The system-prompt section for this file.
    ///
    /// With a known context window, the section takes at most
    /// `share_percent` of it. A larger file is cut at a line end, and a last
    /// line names the file and says where it was cut, so the agent can read
    /// the rest. Without a window, the whole file is used. The section has
    /// no trailing whitespace, so a prompt builder that trims it keeps it
    /// as it is, and a later resize finds it in the prompt.
    #[must_use]
    pub fn prompt_section(&self, context_window: Option<usize>) -> String {
        let header = format!("# Project Context (from {})\n\n", self.source_filename);
        let whole = format!("{header}{}", self.content.trim_end());
        let Some(window) = context_window else {
            return whole;
        };
        let budget = window.saturating_mul(usize::from(self.share_percent.clamp(1, 100))) / 100
            * PROJECT_CONTEXT_CHARS_PER_TOKEN;
        if whole.len() <= budget {
            return whole;
        }
        let total = self.content.lines().count();
        let cut_note = |kept: usize| {
            format!(
                "\n\n[{} is cut to fit this model's {window}-token context: these are its first {kept} of {total} lines. Read {} for the rest.]",
                self.source_filename,
                self.path.display()
            )
        };
        let room = budget.saturating_sub(header.len() + cut_note(total).len());
        let head = truncate_utf8(&self.content, room);
        let head = head.rfind('\n').map_or("", |end| &head[..end]);
        format!("{header}{head}{}", cut_note(head.lines().count()))
    }
}

/// The project context and the section now in the system prompt.
#[derive(Debug, Clone)]
pub(super) struct BudgetedProjectContext {
    context: ProjectContext,
    section: String,
}

impl AgentService {
    /// The window that sizes project context: an explicit window (flag,
    /// config, `CALICHE_CONTEXT_WINDOW` or `/context-window`) first, else the
    /// window the provider's server reported for the active model, else
    /// `None`.
    #[must_use]
    pub fn project_context_window(&self) -> Option<usize> {
        self.context_window_override()
            .or(self.reported_context_window)
    }

    /// Ask the provider's server for the active model's window and, without
    /// an explicit window, use it for compaction and the context meter. A
    /// server that does not answer leaves the current window.
    pub(super) async fn apply_reported_context_window(&mut self) {
        self.reported_context_window = None;
        if self.context_window_override().is_some() {
            return;
        }
        let Some(probe) = self.agent.llm_provider().model_load_probe() else {
            return;
        };
        let Ok(client) = reqwest::Client::builder()
            .timeout(REPORTED_WINDOW_TIMEOUT)
            .build()
        else {
            return;
        };
        let Some(reported) = probe.context_window(&client).await else {
            return;
        };
        match apply_context_window_limit(
            &mut self.agent,
            self.provider,
            &self.model,
            Some(reported),
        ) {
            Ok(()) => self.reported_context_window = Some(reported),
            Err(err) => tracing::warn!(reported, "ignoring the reported context window: {err:#}"),
        }
    }

    /// Size `context` to this model's window and remember it, so a later
    /// model or window change sizes it again. Returns the section; the
    /// caller places it in the system prompt.
    pub fn budget_project_context(&mut self, context: ProjectContext) -> String {
        let section = context.prompt_section(self.project_context_window());
        self.project_context = Some(BudgetedProjectContext {
            context,
            section: section.clone(),
        });
        section
    }

    /// The project-context section now in the system prompt, if any.
    #[must_use]
    pub fn project_context_section(&self) -> Option<&str> {
        self.project_context
            .as_ref()
            .map(|budgeted| budgeted.section.as_str())
    }

    /// Carry `other`'s project context over, for a copy of its prompt.
    pub fn inherit_project_context(&mut self, other: &Self) {
        self.project_context.clone_from(&other.project_context);
    }

    /// Size the project context for the active model again. An unknown
    /// window keeps the current section.
    pub(super) fn rebudget_project_context(&mut self) {
        self.rebudget_project_context_for(self.project_context_window());
    }

    /// Size the project context for `window` and swap the new section into
    /// the system prompt in place of the old one.
    pub(super) fn rebudget_project_context_for(&mut self, window: Option<usize>) {
        let Some(window) = window else {
            return;
        };
        let Some(budgeted) = self.project_context.as_mut() else {
            return;
        };
        let section = budgeted.context.prompt_section(Some(window));
        if section == budgeted.section {
            return;
        }
        let stale = std::mem::replace(&mut budgeted.section, section);
        self.replace_project_section(&stale);
    }

    /// Put the current section where `stale` is in the system prompt.
    pub(super) fn replace_project_section(&mut self, stale: &str) {
        let Some(current) = self.project_context_section().map(str::to_owned) else {
            return;
        };
        if stale == current {
            return;
        }
        let prompt = self.agent.system_prompt();
        if prompt.contains(stale) {
            let swapped = prompt.replacen(stale, &current, 1);
            self.agent.set_system_prompt(swapped);
        }
        if let Some(base) = self.base_system_prompt.as_mut()
            && base.contains(stale)
        {
            *base = base.replacen(stale, &current, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An AGENTS.md of 2000 lines (52,000 characters, about 17K tokens).
    fn large_agents_md(share_percent: u8) -> ProjectContext {
        ProjectContext {
            path: PathBuf::from("/repo/AGENTS.md"),
            source_filename: "AGENTS.md".to_string(),
            content: (0..2000)
                .map(|i| format!("rule {i:04}: keep this line\n"))
                .collect(),
            share_percent,
        }
    }

    /// A 32K window gives project context 25%: 8192 tokens, 24,576
    /// characters. The file is cut at a line end, with a note.
    #[test]
    fn a_32k_window_gets_the_head_of_a_large_agents_md_and_a_cut_note() {
        let section = large_agents_md(25).prompt_section(Some(32_768));

        assert!(
            section
                .starts_with("# Project Context (from AGENTS.md)\n\nrule 0000: keep this line\n"),
            "{section}"
        );
        assert!(
            section.ends_with(
                "rule 0937: keep this line\n\n[AGENTS.md is cut to fit this model's 32768-token context: these are its first 938 of 2000 lines. Read /repo/AGENTS.md for the rest.]"
            ),
            "{section}"
        );
        assert!(!section.contains("rule 0938"));
        assert_eq!(section.len(), 24_558);
        assert!(section.len() <= 32_768 / 4 * 3);
    }

    /// A 200K window has room for the whole file: no cut, no note.
    #[test]
    fn a_200k_window_gets_the_whole_agents_md() {
        let context = large_agents_md(25);

        let section = context.prompt_section(Some(200_000));

        assert_eq!(
            section,
            format!(
                "# Project Context (from AGENTS.md)\n\n{}",
                context.content.trim_end()
            )
        );
    }

    /// No window: the whole file.
    #[test]
    fn an_unknown_window_keeps_the_whole_agents_md() {
        let context = large_agents_md(25);

        let section = context.prompt_section(None);

        assert_eq!(
            section,
            format!(
                "# Project Context (from AGENTS.md)\n\n{}",
                context.content.trim_end()
            )
        );
    }

    /// The share is configurable: 50% of 32K keeps more lines.
    #[test]
    fn a_larger_share_keeps_more_of_the_file() {
        let section = large_agents_md(50).prompt_section(Some(32_768));

        assert!(
            section.ends_with(
                "these are its first 1883 of 2000 lines. Read /repo/AGENTS.md for the rest.]"
            ),
            "{section}"
        );
        assert!(section.len() <= 32_768 / 2 * 3);
    }
}
