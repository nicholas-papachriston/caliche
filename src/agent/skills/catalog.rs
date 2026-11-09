//! Built-in skills.
//!
//! Add a skill by appending one [`BuiltinSkill`] row. Detection uses the name
//! (underscores as spaces) and `triggers`. Each trigger is a whole word or a
//! phrase. A substring such as `spec` inside `aspects` does not match.

use super::{Skill, SkillBuilder};

pub(super) struct BuiltinSkill {
    pub name: &'static str,
    pub description: &'static str,
    pub triggers: &'static [&'static str],
    pub prompt: &'static str,
    pub tools: &'static [&'static str],
    pub max_iterations: Option<usize>,
}

impl BuiltinSkill {
    fn to_skill(&self) -> Skill {
        let mut builder = SkillBuilder::new(self.name)
            .description(self.description)
            .system_prompt_addition(self.prompt)
            .require_tools(self.tools.iter().copied())
            .triggers(self.triggers.iter().copied());
        if let Some(max) = self.max_iterations {
            builder = builder.max_iterations(max);
        }
        builder.build()
    }

    fn matches(&self, words: &[&str]) -> bool {
        self.triggers
            .iter()
            .any(|trigger| trigger_matches(words, trigger))
    }
}

pub(super) static BUILTIN_SKILLS: &[BuiltinSkill] = &[
    BuiltinSkill {
        name: "code_analysis",
        description: "Analyzes code for patterns, issues, and improvements",
        triggers: &["analyze", "analyzes", "analysis", "understand the code"],
        prompt: r"
You are an expert code analyst. When analyzing code:
- Identify potential bugs and edge cases
- Look for performance bottlenecks
- Check for security vulnerabilities
- Suggest refactoring opportunities
- Consider maintainability and readability
- Use fs op=grep for targeted searches rather than reading entire files
",
        tools: &["fs"],
        max_iterations: None,
    },
    BuiltinSkill {
        name: "code_review",
        description: "Reviews code changes with focus on quality and security",
        triggers: &["review", "reviewing", "code review", "pull request"],
        prompt: r"
You are a senior code reviewer. When reviewing code:
- Focus on correctness and logic errors
- Check for security vulnerabilities (injection, auth issues)
- Verify error handling is appropriate
- Ensure tests cover edge cases
- Look for code style consistency
- Provide constructive, actionable feedback
",
        tools: &["fs", "git"],
        max_iterations: None,
    },
    BuiltinSkill {
        name: "debugging",
        description: "Systematic debugging and issue investigation",
        triggers: &["debug", "debugging", "bug", "stack trace"],
        prompt: r"
You are an expert debugger. When debugging:
- Gather evidence systematically before forming hypotheses
- Check logs, error messages, and stack traces
- Reproduce the issue if possible
- Narrow down the root cause through elimination
- Verify fixes don't introduce regressions
- Document findings clearly
",
        tools: &["fs", "bash", "git"],
        max_iterations: Some(100),
    },
    BuiltinSkill {
        name: "documentation",
        description: "Creates and improves documentation",
        triggers: &["documentation", "readme", "docs"],
        prompt: r"
You are a technical writer. When creating documentation:
- Write clear, concise explanations
- Include practical examples
- Structure content logically
- Consider the target audience
- Keep documentation up to date with code
- Use consistent formatting and style
",
        tools: &["fs"],
        max_iterations: None,
    },
    BuiltinSkill {
        name: "refactoring",
        description: "Safe code refactoring with minimal risk",
        triggers: &["refactor", "refactoring"],
        prompt: r"
You are a refactoring expert. When refactoring:
- Make small, incremental changes
- Preserve existing behavior
- Run tests after each change
- Use semantic-preserving transformations
- Improve readability without over-engineering
- Document breaking changes clearly
",
        tools: &["fs", "bash"],
        max_iterations: None,
    },
    BuiltinSkill {
        name: "research",
        description: "Research and information gathering",
        triggers: &["research", "look up", "find out"],
        prompt: r#"
You are a research assistant. When researching:
- NEVER fabricate statistics, benchmarks, or quantitative comparisons. If you do not have a cited number, do not invent one.
- Every factual claim must trace to a tool result or search citation. When citing search results, include the source reference (e.g. [1], [2]).
- Clearly separate "verified from sources" from "general knowledge / inference".
- Identify gaps: say "I could not find data on X" rather than making something up.
- Present balanced perspectives.
- Summarize findings concisely.

When using the research tool: pass a topic that decomposes into 3-6 focused sub-questions (e.g. "deep dive on caliche architecture: core agent framework, orchestration, dripstone, sub-crates"). The runner scouts for tightly scoped sub-questions, runs research sub-agents in parallel (each with narrow responsibility for speed and fidelity), then compiles findings. Prefer topics that benefit from parallel sub-agents over single broad queries.
When the research tool returns (codebase exploration): the sub-agent returns grounded answers. When you synthesize results into a document or summary, only use type, trait, module, or API names that appeared in the tool output. Do not infer or invent names (e.g. do not assume "Memory" or "ConversationMemory"); if a name was not in the research results, omit it or say "see source" and the file path.
When exploring the codebase with fs: use workspace(action=codebase_context) or semantic_search first, then read only the files you need—at most a few parallel fs(op=read) per round. Do not re-read the same path in the next round unless you need a different line range or the user asks a new question. Cap exploratory reads per turn; synthesize early instead of re-fetching the same large files.
"#,
        tools: &["web_search", "web_fetch", "http", "perplexity_search"],
        max_iterations: None,
    },
    BuiltinSkill {
        name: "testing",
        description: "Test creation and quality assurance",
        triggers: &["test", "tests", "unit test", "coverage"],
        prompt: r"
You are a QA engineer. When the user asks for tests:
- Write the tests they asked for
- Cover edge cases and error conditions
- Test both happy path and failure scenarios
- Use appropriate testing patterns (unit, integration)
- Ensure tests are maintainable and readable
- Do not invent benchmark numbers or performance reports
",
        tools: &["fs", "bash"],
        max_iterations: None,
    },
];

pub(super) fn require(name: &str) -> Skill {
    match BUILTIN_SKILLS.iter().find(|skill| skill.name == name) {
        Some(skill) => skill.to_skill(),
        None => SkillBuilder::new(name).build(),
    }
}

pub(super) fn builtin_skills() -> Vec<Skill> {
    BUILTIN_SKILLS.iter().map(BuiltinSkill::to_skill).collect()
}

pub(super) fn matching_builtins(words: &[&str]) -> Vec<Skill> {
    BUILTIN_SKILLS
        .iter()
        .filter(|skill| skill.matches(words))
        .map(BuiltinSkill::to_skill)
        .collect()
}

pub(super) fn query_words(query: &str) -> Vec<&str> {
    query
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect()
}

pub(super) fn trigger_matches(words: &[&str], trigger: &str) -> bool {
    let trigger = trigger.trim().to_lowercase();
    let parts: Vec<&str> = trigger
        .split_whitespace()
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() || parts.len() > words.len() {
        return false;
    }
    words
        .windows(parts.len())
        .any(|window| window == parts.as_slice())
}

pub(super) fn skill_matches(words: &[&str], name: &str, triggers: &[String]) -> bool {
    if !triggers.is_empty() {
        return triggers
            .iter()
            .any(|trigger| trigger_matches(words, trigger));
    }
    trigger_matches(words, &name.to_lowercase().replace('_', " "))
}
