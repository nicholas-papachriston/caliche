//! Agent Skills System
//!
//! Skills are reusable capability packages that can be applied to agents.
//! A skill bundles together:
//! - System prompt additions (expertise, guidelines)
//! - Required tools
//! - Lifecycle hooks
//! - Configuration settings
//!
//! # Example
//!
//! ```rust,no_run
//! use caliche::agent::skills::{Skill, SkillBuilder};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a code review skill
//! let code_review_skill = SkillBuilder::new("code_review")
//!     .description("Expert code reviewer with security focus")
//!     .system_prompt_addition(r#"
//!         You are an expert code reviewer. Focus on:
//!         - Security vulnerabilities
//!         - Performance issues
//!         - Code style and maintainability
//!     "#)
//!     .require_tool("fs")
//!     .build();
//!
//! # use std::sync::Arc;
//! # let llm = Arc::new(caliche::llm::OpenAIProvider::try_new("key".to_string(), "model".to_string())?);
//! # let tools = Arc::new(caliche::ToolRegistry::new());
//! # let mut agent = caliche::agent::core::Agent::new(llm, tools);
//! // Apply skill to agent
//! agent.apply_skill(&code_review_skill);
//! # Ok(())
//! # }
//! ```

mod catalog;
mod remote;

pub use remote::{
    InstallReport, RemoteSkillSource, SkillFreshness, SkillLock, SkillLockEntry, SkillUpdateAction,
    SkillUpdatePreview, apply_updates, content_hash, install_checkout, install_source,
    parse_remote_source, preview_updates, skill_freshness, skill_source_allowlist, source_allowed,
};

use crate::agent::hooks::{HookMatcher, LifecycleHook};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::OnceLock;

/// Compact skill index entry for progressive disclosure (pi-style).
///
/// The model sees name/description/location and loads the full skill body via `fs`
/// when needed, instead of dumping every skill into the system prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillIndexEntry {
    pub name: String,
    pub description: String,
    /// Absolute or workspace-relative path to the skill file (e.g. SKILL.md).
    pub location: String,
}

impl SkillIndexEntry {
    #[must_use]
    pub fn from_skill(skill: &Skill) -> Option<Self> {
        skill.source_location.as_ref().map(|location| Self {
            name: skill.name.clone(),
            description: skill.description.clone(),
            location: location.clone(),
        })
    }
}

/// Format skill index entries for the system prompt (progressive disclosure).
#[must_use]
pub fn format_skills_for_prompt(skills: &[SkillIndexEntry]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "\n\n<available_skills>\n\
         Skills are optional. When a skill is relevant, read its location with fs op=read \
         before following its instructions. Do not invent skill contents.\n",
    );
    for skill in skills {
        let _ = write!(
            out,
            "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    <location>{}</location>\n  </skill>\n",
            xml_escape(&skill.name),
            xml_escape(&skill.description),
            xml_escape(&skill.location),
        );
    }
    out.push_str("</available_skills>");
    out
}

fn xml_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A skill is a reusable capability package for agents.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique identifier for the skill
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Text to append to the agent's system prompt
    pub system_prompt_addition: Option<String>,
    /// Tools required by this skill
    pub required_tools: HashSet<String>,
    /// Whole words or phrases that select this skill. The name also matches.
    pub triggers: Vec<String>,
    /// Hooks to register when skill is applied
    hooks: Vec<(HookMatcher, Arc<dyn LifecycleHook>)>,
    /// Maximum iterations override (if skill needs more/fewer)
    pub max_iterations: Option<usize>,
    /// Whether this skill should restrict tools to only required ones
    pub restrict_tools: bool,
    /// On-disk location for progressive disclosure (discovered skills).
    pub source_location: Option<String>,
}

impl Skill {
    /// Create a new skill builder
    pub fn builder(name: impl Into<String>) -> SkillBuilder {
        SkillBuilder::new(name)
    }

    /// Get the skill's hooks
    #[must_use]
    pub fn hooks(&self) -> &[(HookMatcher, Arc<dyn LifecycleHook>)] {
        &self.hooks
    }

    /// Check if a tool is required by this skill
    #[must_use]
    pub fn requires_tool(&self, tool_name: &str) -> bool {
        self.required_tools.contains(tool_name)
    }
}

/// Builder for creating skills
#[derive(Debug, Default)]
pub struct SkillBuilder {
    name: String,
    description: String,
    system_prompt_addition: Option<String>,
    required_tools: HashSet<String>,
    triggers: Vec<String>,
    hooks: Vec<(HookMatcher, Arc<dyn LifecycleHook>)>,
    max_iterations: Option<usize>,
    restrict_tools: bool,
    source_location: Option<String>,
}

impl SkillBuilder {
    /// Create a new skill builder with a name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            system_prompt_addition: None,
            required_tools: HashSet::new(),
            triggers: Vec::new(),
            hooks: Vec::new(),
            max_iterations: None,
            restrict_tools: false,
            source_location: None,
        }
    }

    /// Set the skill description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add text to append to the system prompt
    pub fn system_prompt_addition(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt_addition = Some(prompt.into());
        self
    }

    /// Add a required tool for this skill
    pub fn require_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.required_tools.insert(tool_name.into());
        self
    }

    /// Phrases that select this skill. Each entry is a whole word or a phrase.
    pub fn triggers<I, S>(mut self, triggers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.triggers.extend(triggers.into_iter().map(Into::into));
        self
    }

    /// Add multiple required tools
    pub fn require_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for tool in tools {
            self.required_tools.insert(tool.into());
        }
        self
    }

    /// Add a lifecycle hook
    pub fn add_hook(mut self, matcher: HookMatcher, hook: Arc<dyn LifecycleHook>) -> Self {
        self.hooks.push((matcher, hook));
        self
    }

    /// Set maximum iterations for this skill
    #[must_use]
    pub const fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Restrict the agent to only use required tools when skill is active
    #[must_use]
    pub const fn restrict_to_required_tools(mut self) -> Self {
        self.restrict_tools = true;
        self
    }

    /// Set on-disk skill location for progressive disclosure.
    pub fn source_location(mut self, location: impl Into<String>) -> Self {
        self.source_location = Some(location.into());
        self
    }

    /// Build the skill
    #[must_use]
    pub fn build(self) -> Skill {
        Skill {
            name: self.name,
            description: self.description,
            system_prompt_addition: self.system_prompt_addition,
            required_tools: self.required_tools,
            triggers: self.triggers,
            hooks: self.hooks,
            max_iterations: self.max_iterations,
            restrict_tools: self.restrict_tools,
            source_location: self.source_location,
        }
    }
}

// ============================================================================
// Pre-built Skills
// ============================================================================

/// Detect skills to apply based on message intent (keyword/heuristic).
/// Returns one or more skills; multiple can apply (e.g. debugging + `code_analysis`).
#[must_use]
pub fn detect_skills(query: &str) -> Vec<Skill> {
    detect_skills_with_catalog(query, discovered_skills())
}

/// Detect skills using built-ins plus an explicit discovered catalog.
#[must_use]
pub fn detect_skills_with_catalog(query: &str, discovered: &[Skill]) -> Vec<Skill> {
    detect_skills_with_catalogs(query, &[discovered])
}

/// Built-in skills. Add a row in `catalog.rs` to register another one.
#[must_use]
pub fn builtin_skills() -> Vec<Skill> {
    catalog::builtin_skills()
}

/// Name and description of each built-in skill, in catalog order.
#[must_use]
pub fn builtin_summaries() -> Vec<(&'static str, &'static str)> {
    catalog::BUILTIN_SKILLS
        .iter()
        .map(|skill| (skill.name, skill.description))
        .collect()
}

/// Detect skills across several catalogs. Only matching skills are cloned.
#[must_use]
pub fn detect_skills_with_catalogs(query: &str, catalogs: &[&[Skill]]) -> Vec<Skill> {
    let normalized = query.trim().to_lowercase();
    let words = catalog::query_words(&normalized);
    let mut out = catalog::matching_builtins(&words);
    for group in catalogs {
        append_matching_discovered(&words, group, &mut out);
    }
    out
}

fn append_matching_discovered(words: &[&str], discovered: &[Skill], out: &mut Vec<Skill>) {
    for skill in discovered {
        if skill.name.trim().is_empty() {
            continue;
        }
        if !catalog::skill_matches(words, &skill.name, &skill.triggers) {
            continue;
        }
        if out.iter().any(|existing| existing.name == skill.name) {
            continue;
        }
        out.push(skill.clone());
    }
}

static DISCOVERED_SKILLS: OnceLock<Vec<Skill>> = OnceLock::new();

/// Resolve auto-discovered skills once per process.
///
/// The slice is process-wide. Callers clone only the skills they apply.
pub fn discovered_skills() -> &'static [Skill] {
    DISCOVERED_SKILLS
        .get_or_init(|| {
            let options = crate::discovery::auto::DiscoveryOptions {
                // Rules are discovered separately; skill index is for progressive
                // disclosure of on-disk SKILL.md entries only.
                rules: false,
                skills: true,
                commands: false,
                subagents: false,
                mcp: false,
            };
            let snapshot = crate::discovery::auto::discover_all(&options);
            let mut out = Vec::new();
            for s in snapshot.skills {
                let mut builder = SkillBuilder::new(s.name)
                    .description(s.description)
                    .source_location(s.source_path.display().to_string());
                // Keep the body available for explicit `/skill` apply, but discovery
                // defaults to progressive disclosure via the skill index.
                if !s.prompt.trim().is_empty() {
                    builder = builder.system_prompt_addition(s.prompt);
                }
                for tool in s.required_tools {
                    builder = builder.require_tool(tool);
                }
                if !s.triggers.is_empty() {
                    builder = builder.triggers(s.triggers);
                }
                out.push(builder.build());
            }
            out
        })
        .as_slice()
}

/// Pre-built skills. Each function reads one row from [`catalog`].
pub mod prebuilt {
    use super::{Skill, catalog};

    /// Create a code analysis skill
    #[must_use]
    pub fn code_analysis() -> Skill {
        catalog::require("code_analysis")
    }

    /// Create a code review skill
    #[must_use]
    pub fn code_review() -> Skill {
        catalog::require("code_review")
    }

    /// Create a debugging skill
    #[must_use]
    pub fn debugging() -> Skill {
        catalog::require("debugging")
    }

    /// Create a documentation skill
    #[must_use]
    pub fn documentation() -> Skill {
        catalog::require("documentation")
    }

    /// Create a refactoring skill
    #[must_use]
    pub fn refactoring() -> Skill {
        catalog::require("refactoring")
    }

    /// Create a research skill
    #[must_use]
    pub fn research() -> Skill {
        catalog::require("research")
    }

    /// Create a testing skill
    #[must_use]
    pub fn testing() -> Skill {
        catalog::require("testing")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_skill_index_for_progressive_disclosure() {
        let entries = vec![SkillIndexEntry {
            name: "review".to_string(),
            description: "Code review help".to_string(),
            location: "/repo/.agents/skills/review/SKILL.md".to_string(),
        }];
        let prompt = format_skills_for_prompt(&entries);
        assert!(prompt.contains("<available_skills>"));
        assert!(prompt.contains("<name>review</name>"));
        assert!(prompt.contains("SKILL.md"));
        assert!(prompt.contains("fs op=read"));
    }

    #[test]
    fn builtin_catalog_names_are_unique_and_load() {
        let skills = builtin_skills();
        assert_eq!(skills.len(), 7);
        let mut names = std::collections::HashSet::new();
        for skill in &skills {
            assert!(names.insert(skill.name.clone()), "duplicate {}", skill.name);
            assert!(
                skill
                    .system_prompt_addition
                    .as_ref()
                    .is_some_and(|body| !body.trim().is_empty())
            );
            assert!(!skill.triggers.is_empty());
        }
        assert_eq!(prebuilt::debugging().max_iterations, Some(100));
    }

    #[test]
    fn disk_skill_matches_its_triggers_not_its_description() {
        let skill = SkillBuilder::new("bench")
            .description("aspects of performance testing")
            .triggers(["benchmark"])
            .system_prompt_addition("measure the real path")
            .build();
        let aspects = detect_skills_with_catalog("performance aspects", &[skill.clone()]);
        assert!(aspects.iter().all(|item| item.name != "bench"));
        let hit = detect_skills_with_catalog("run a benchmark", &[skill]);
        assert!(hit.iter().any(|item| item.name == "bench"));
    }

    #[test]
    fn testing_skill_ignores_meta_talk_and_the_word_aspects() {
        let meta = detect_skills_with_catalog("im testing it via you", &[]);
        assert!(meta.iter().all(|skill| skill.name != "testing"));
        let aspects = detect_skills_with_catalog("performance aspects and efficiency", &[]);
        assert!(aspects.iter().all(|skill| skill.name != "testing"));
        let write = detect_skills_with_catalog("add a unit test for the parser", &[]);
        assert!(write.iter().any(|skill| skill.name == "testing"));
        let review = detect_skills_with_catalog("please review the pull request", &[]);
        assert!(review.iter().any(|skill| skill.name == "code_review"));
        let latest = detect_skills_with_catalog("what is the latest release", &[]);
        assert!(latest.iter().all(|skill| skill.name != "research"));
    }

    #[test]
    fn index_entry_requires_source_location() {
        let skill = SkillBuilder::new("local")
            .description("no path")
            .system_prompt_addition("body")
            .build();
        assert!(SkillIndexEntry::from_skill(&skill).is_none());

        let skill = SkillBuilder::new("disk")
            .description("on disk")
            .source_location("/tmp/SKILL.md")
            .build();
        let entry = SkillIndexEntry::from_skill(&skill).expect("indexed");
        assert_eq!(entry.location, "/tmp/SKILL.md");
    }
}
