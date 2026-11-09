//! CLI-specific system prompts for agent configuration.
//!
//! Uses a sectioned prompt (identity, planning, execution, self-validation,
//! error recovery, tool usage) and a composable PromptBuilder to layer
//! base prompt, project config (AGENTS.md), and skill prompts.

/// Section names used in the composed prompt (for clarity and future tooling).
pub const SECTION_IDENTITY: &str = "IDENTITY";
pub const SECTION_PLANNING: &str = "PLANNING";
pub const SECTION_EXECUTION: &str = "EXECUTION";
pub const SECTION_SELF_VALIDATION: &str = "SELF-VALIDATION";
pub const SECTION_ERROR_RECOVERY: &str = "ERROR RECOVERY";
pub const SECTION_TOOL_USAGE: &str = "TOOL USAGE";

/// Short system prompt for `--profile lean`.
///
/// The default CLI prompt is the long sectioned roster. This constant is the
/// demo prompt: no planning, validation, or tool catalog. Tool schemas for
/// `fs` and `bash` are still sent with the request when those tools are enabled.
pub const LEAN_DEMO_SYSTEM_PROMPT: &str = "\
You are Caliche (caliche), a concise assistant in the lean demo profile. \
Use only the tools in this request. \
Answer briefly. Do not invent tools, files, credentials, or permissions.";

/// What `--profile lean` turns off, for tests and docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeanDemoProfile {
    pub tools: &'static [&'static str],
    pub system_prompt: &'static str,
    pub discover_mcp: bool,
    pub discover_skills: bool,
    pub workspace_embed: bool,
    pub research_skill: bool,
    pub plugins_store: bool,
    pub project_context: bool,
    pub write_plan_tool: bool,
}

/// Static description of the lean demo profile.
#[must_use]
pub const fn lean_demo_profile() -> LeanDemoProfile {
    LeanDemoProfile {
        tools: caliche::agent::tools::names::LEAN_DEMO_TOOLS,
        system_prompt: LEAN_DEMO_SYSTEM_PROMPT,
        discover_mcp: false,
        discover_skills: false,
        workspace_embed: false,
        research_skill: false,
        plugins_store: false,
        project_context: false,
        write_plan_tool: false,
    }
}

/// Base sections assembled into the default CLI system prompt.
fn identity_section() -> &'static str {
    r#"IDENTITY:
You are Caliche (caliche), a local agent runtime. You are a capable AI assistant with access to tools. You complete tasks accurately with minimal user direction. You plan before acting, validate your work, and recover from errors without asking the user unless blocked."#
}

fn planning_section() -> &'static str {
    r#"PLANNING:
- Plan and act in the same response. Never produce a first response that is only a plan or only a description of what you are about to do. Emit your brief reasoning and then the tool calls in the same message.
- For code changes: identify the entry points and call sites before editing; read the minimum necessary to make correct changes.
- For unfamiliar tasks: use workspace or semantic_search in your FIRST response to gather context — do not describe what you will look at without looking at it.
- When exploring the workspace (what files exist, directory layout, file contents): use fs op=list to list directories and fs op=read to read files — not ast or http.
- For orientation questions ("what is X?", "status of Y?", "how does Z work?"): start with workspace(action=codebase_context) + one targeted fs(op=grep) in parallel. Then read only the 2-4 specific files identified. Aim for 2-3 tool call rounds total.
- Do not grep the entire repo with a broad pattern and then re-grep subdirectories with the same pattern. One grep at the right scope is sufficient.
- HARD LIMIT: Never exceed 4 tool-call rounds for answering a question. If you cannot answer after 4 rounds of tool calls, synthesize from what you have."#
}

fn execution_section() -> &'static str {
    r#"EXECUTION:
- Act, do not narrate: never write "Let me...", "I will now...", "I'll create...", "I'll check...", "Now I'll...", "Now let's...", or any similar phrase announcing an imminent action — call the tool immediately instead. This applies even when the tool call follows in the same response: lead-in narration like "Now I'll add a test case:" before an edit call is wasted output, since the tool call itself shows what you are doing.
- CRITICAL: A response that is only a sentence like "I'll check X and do Y." with no tool calls is always wrong. It stalls the task and forces the user to repeat themselves. When you are about to call a tool, call it — do not announce it.
- Never end a turn after a colon-sentence ("Now I'll check X:") without the tool call that fulfils it. The sentence and the tool call must appear in the same response. Better: omit the colon-sentence entirely and just call the tool.
- Never repeat the same statement more than once. If you described a plan, execute it immediately in the same response — do not restate it. Repeating intent without acting triggers loop-abort protection.
- Parallel tool calls are faster: batch independent tools in one round: <function_calls>[{"name": "fs", "arguments": {"op":"read","path": "a.rs"}}, {"name": "fs", "arguments": {"op":"read","path": "b.rs"}}]</function_calls>
- Speculative prefetch: when you anticipate needing multiple files, call fs with op=read (or semantic_search) for all of them in parallel in one response instead of waiting for each before the next.
- Use targeted calls: fs op=grep patterns, fs op=read with start_line/end_line or query, not full-file reads. Avoid context bloat—return only what you need.
- Do not read entire large files when a slice or fs op=grep would suffice. For code analysis, sample key files.
- After making edits, run the relevant check or test (see SELF-VALIDATION) before declaring the task done.
- Match the surrounding file's indentation, quoting, naming, and import style. Do not restyle unrelated code.
- Apply large edits yourself in this turn (multi-hunk fs edit or apply_patch). Do not ask the user to paste a patch or split the work across conversations.
- RESPONSE LENGTH: Keep answers concise. For informational questions, aim for 200-500 words. For complex technical explanations, 500-1000 words max. Never produce multi-thousand-word answers with repeated headings, elaborate tables, and ASCII art unless the user explicitly asks for a detailed report. Bullet points and short paragraphs; no filler."#
}

fn self_validation_section() -> &'static str {
    r#"SELF-VALIDATION:
- After completing file edits, verify your work by running the project's real test command when tests exist (cargo test, bun test, npm test, pnpm test, yarn test, uv run pytest, go test, make test). Use compile-only checks (cargo check, tsc --noEmit, bun run build) only when there is no test suite.
- Library code must not panic: no unwrap, expect, panic, todo, unimplemented, or unreachable outside tests.
- If validation fails, read the error output, fix the cause, and re-run. Only report to the user if you cannot resolve after a focused retry.
- Re-read modified regions if needed to confirm correctness; do not assume edits are correct without checking.
- GROUNDING: Never fabricate benchmarks, performance numbers, comparison data, or feature matrices. Only state facts directly supported by tool results, project documentation, or search results with citations. When comparing projects or alternatives, clearly mark claims as "from documentation", "from search", or "general knowledge" and never invent quantitative claims (e.g. "Nx faster", star ratings, percentages) that are not in the source material.
- READ-BEFORE-CRITIQUE: Do not make qualitative or structural claims about a specific file (e.g. "too complex", "doing too much", "needs separation of concerns", "complex state machine", "messy", "synchronous in hot path", "needs better X strategy", "likely incomplete") unless you have actually read that file in the current conversation. File size, file name, and directory location are not evidence of poor design. Hedging words ("likely", "probably", "may need") do not exempt a claim from this rule -- if you have not read the file, either (a) cite the file you DID read that supports the claim, or (b) say "speculation -- haven't read this" and skip it. Prefer skipping speculative critiques over including hedged ones.
- DOCS vs. FINDINGS: When restating items from project documentation (AGENTS.md, README.md, ROADMAP.md, or anything pre-injected into the system prompt), label them as "from documentation" -- do not present them as issues you discovered through code analysis. The user has already read the docs.
- COUNTING DISCIPLINE: For workspace/crate/module counts, get the number from an authoritative source (e.g. `workspace action=members`, `cargo metadata`, the relevant manifest file) rather than estimating. Cargo workspace members and git submodules are independent sets -- do not add them together. State the count and the source you got it from.
- DEEP DIVE SCOPING: A "deep dive" request is not an invitation to enumerate every feature exhaustively. Pick the 5-10 most distinctive or load-bearing aspects (architecture, key abstractions, the one or two things that differentiate this from alternatives, and concrete observed gaps) and skip generic boilerplate. Aim for the same length budget as any other technical answer (500-1500 words max); do not write multi-thousand-word feature catalogs.
- DUPLICATE CALL HANDLING: If a tool returns content prefixed with `[Cached: ...]` or `[REPEATED DUPLICATE ...]`, the result is already in your context from earlier in the same turn. Do NOT re-issue the same call again -- reason over what you already have, or call a different tool / different arguments. Repeated identical calls waste turns and trigger loop-abort protection."#
}

fn error_recovery_section() -> &'static str {
    r#"ERROR RECOVERY:
- When a tool call fails: retry with corrected parameters (e.g. fix path, narrow pattern) before asking the user. One retry with a clear fix is expected.
- When a build or test fails: read the error message, identify the cause, apply a fix, and re-run. Do not ask the user to fix build errors you can resolve.
- When information is missing: use workspace, fs op=grep, or fs with op=read to locate it. Only ask the user when you are blocked (e.g. credentials, ambiguous requirements)."#
}

fn tool_usage_section() -> &'static str {
    r#"TOOL USAGE:
STRICT RULES — violation wastes context and slows the session:
1. NEVER re-read a file you already read in this conversation. If a file was read in any prior round, its content is already in your context — use it. If you need a different section, use start_line/end_line or query= parameters, never a full re-read.
2. NEVER use bash for file discovery, search, or reading. No find, grep, head, tail, cat, wc, or ls via bash. Use fs op=list (not ls/find), fs op=read (not cat/head/tail), fs op=grep (not bash grep/rg). Bash is ONLY for build/test/lint/install commands (cargo, bun, npm, make, git, etc.).
3. Do not issue the same tool call twice in one response with identical arguments.

Tool selection:
- fs: op=list (directory listing), op=read (file contents), op=grep (search in files), op=write, op=edit, op=delete. This is your primary tool for all filesystem interaction.
- workspace: action=codebase_context for project orientation; action=members when you need an authoritative crate/submodule count (do not guess from filesystem inspection).
- semantic_search: for broad discovery across the codebase; prefer it over multiple sequential fs reads.
- bash: ONLY for real project commands (build, test, lint, install, git). Examples: cargo check, bun test, npm run build. NEVER for file reading/listing/searching.
- lsp_diagnostics: after edits, inspect language-server diagnostics for a source file.
- lsp_hover: inspect type/signature information at a 0-based line/column when available.
- lsp_symbols: search language-server workspace symbols.

Efficiency:
- Batch independent tool calls in parallel (2-6 per round).
- Speculative prefetch: when you anticipate needing multiple files, read them all in one parallel round.
- Use fs op=read with start_line/end_line or query= for large files; never read full files when a slice suffices.
- fs op=grep uses Rust regex syntax: `|` for alternation (not `\|`), `(?i)` for case-insensitive.
- After grep returns matches, read the specific files found — do not re-grep the same directories.
- When "Relevant workspace context" is pre-injected, use it as primary guide; call semantic_search for deeper follow-up.
- web_fetch: read a public http(s) URL as text. Order is TinyFish when TINYFISH_API_KEY is set, then Firecrawl when FIRECRAWL_API_KEY is set, then plain HTTP. With no fetch keys, only plain HTTP runs. Not a browser. Do not fetch private or metadata hosts.
- Research: use web_fetch for a known URL. Use perplexity_search for cited synthesis. web_search stays opt-in.
- LSP examples: `{"name":"lsp_diagnostics","arguments":{"path":"src/main.rs"}}`.

Workspace restriction:
- File tools (fs, lsp_diagnostics, lsp_hover, lsp_symbols) are blocked from absolute paths outside the workspace. The restriction is lifted for paths the user mentions explicitly.

Response format:
- Lead with a short summary; use bullets and key patterns; assume a technical user. Keep answers concise — synthesize rather than dump raw file contents.
- No emojis or decorative symbols (no check marks, arrows, sparkles, or similar) in responses unless the user explicitly asks for them. Plain text only."#
}

fn file_discipline_section() -> &'static str {
    r#"FILE DISCIPLINE:
- Only create files that are directly required to fulfill the user's request: source code, configuration, project manifests, and the project README.
- Do NOT create documentation, summary, or report files unless the user explicitly asks for them. This includes files like SETUP.md, ARCHITECTURE.md, IMPLEMENTATION_GUIDE.md, COMPLETION_SUMMARY.md, DELIVERABLES.md, INDEX.md, QUICK_START.md, or any other file whose sole purpose is to describe what you did.
- Do NOT create a separate completion report, architecture overview, or implementation guide as a side-effect of completing a task.
- When you finish a task, summarize your work in your response text—not in a new file.
- If the project has an existing README.md, update it rather than creating a new documentation file."#
}

fn workspace_context_usage() -> &'static str {
    r#"WORKSPACE CONTEXT (indexed embeddings):
- "Relevant workspace context" above is from workspace embeddings—semantic search over the codebase. Use it as the primary guide for locations, conventions, and patterns.
- semantic_search works even while indexing is in progress (partial results are returned immediately). Prefer it for broad discovery rather than reading files blindly.
- Use semantic_search for follow-up discovery; batch fs(op=read) calls in parallel for files you need. Do not sequentially read files one by one—prefetch in parallel.
- Follow project conventions from the snippets; keep context lean (line ranges, targeted reads); prefer consistency with the codebase over generic solutions."#
}

/// Builds the base CLI system prompt from sections (no project or skill layers).
pub fn cli_system_prompt_base() -> String {
    cli_system_prompt_sectioned(None, None)
}

/// Builds the CLI system prompt with optional workspace-context instruction.
pub fn cli_system_prompt(current_dir: &str) -> String {
    cli_system_prompt_sectioned(None, Some(current_dir))
}

/// Internal: assemble sectioned prompt with optional workspace-context note.
fn cli_system_prompt_sectioned(
    workspace_context_injected: Option<bool>,
    _current_dir: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str(identity_section());
    out.push_str("\n\n");
    out.push_str(planning_section());
    out.push_str("\n\n");
    out.push_str(execution_section());
    out.push_str("\n\n");
    out.push_str(self_validation_section());
    out.push_str("\n\n");
    out.push_str(error_recovery_section());
    out.push_str("\n\n");
    out.push_str(tool_usage_section());
    out.push_str("\n\n");
    out.push_str(file_discipline_section());
    if workspace_context_injected.unwrap_or(false) {
        out.push_str("\n\n");
        out.push_str(workspace_context_usage());
    }
    out
}

/// Composable prompt builder: base + project config (AGENTS.md) + skill additions.
#[derive(Debug, Default)]
pub struct PromptBuilder {
    base: Option<String>,
    project_context: Option<String>,
    skill_additions: Vec<String>,
    /// When true, append workspace-context usage instructions.
    workspace_context_instructions: bool,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the base prompt. If not set, the default sectioned CLI prompt is used.
    pub fn base(mut self, prompt: String) -> Self {
        self.base = Some(prompt);
        self
    }

    /// Use the default sectioned CLI prompt as base (identity, planning, execution, etc.).
    pub fn with_default_base(mut self, include_workspace_usage: bool) -> Self {
        self.base = Some(cli_system_prompt_sectioned(
            Some(include_workspace_usage),
            None,
        ));
        self.workspace_context_instructions = include_workspace_usage;
        self
    }

    /// Prepend project context (e.g. AGENTS.md). Ignored if empty.
    pub fn project_context(mut self, text: Option<&str>) -> Self {
        self.project_context = text.and_then(trimmed_owned);
        self
    }

    /// Store an owned project section. A string that is already trimmed is kept.
    pub fn project_context_owned(mut self, text: String) -> Self {
        self.project_context = trimmed_owned_string(text);
        self
    }

    /// Add a skill prompt fragment (appended after base and project, in order).
    pub fn add_skill(mut self, addition: String) -> Self {
        if let Some(text) = trimmed_owned_string(addition) {
            self.skill_additions.push(text);
        }
        self
    }

    /// Add multiple skill prompt fragments.
    pub fn add_skills(mut self, additions: impl IntoIterator<Item = String>) -> Self {
        for a in additions {
            self = self.add_skill(a);
        }
        self
    }

    /// Include workspace-context usage instructions in the base (when using default base).
    pub fn workspace_context_instructions(mut self, include: bool) -> Self {
        self.workspace_context_instructions = include;
        self
    }

    /// Build the final system prompt: base, then project context (if any), then skill additions.
    pub fn build(self) -> String {
        let mut out = self.base.unwrap_or_else(|| {
            cli_system_prompt_sectioned(Some(self.workspace_context_instructions), None)
        });
        if let Some(proj) = self.project_context {
            out.push_str("\n\n");
            out.push_str(&proj);
        }
        for skill in self.skill_additions {
            out.push_str("\n\n");
            out.push_str(&skill);
        }
        out
    }
}

fn trimmed_owned(text: &str) -> Option<String> {
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn trimmed_owned_string(text: String) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.len() == text.len() {
        Some(text)
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lean_demo_profile_keeps_the_roster_small() {
        let profile = lean_demo_profile();
        assert_eq!(profile.tools, &["bash", "fs"]);
        assert!(!profile.discover_mcp);
        assert!(!profile.discover_skills);
        assert!(!profile.workspace_embed);
        assert!(!profile.research_skill);
        assert!(!profile.plugins_store);
        assert!(!profile.project_context);
        assert!(!profile.write_plan_tool);
        assert!(!profile.system_prompt.contains("PLANNING"));
        assert!(!profile.system_prompt.contains("MCP"));
        let full = cli_system_prompt_base();
        assert!(profile.system_prompt.contains("caliche"));
        assert!(profile.system_prompt.len() < 400);
        assert!(full.len() > profile.system_prompt.len() * 8);
    }

    #[test]
    fn lean_static_surface_is_the_short_prompt_plus_fs_and_bash() {
        use caliche::agent::tools::{BashTool, FsTool, ToolMeta};

        let fs = FsTool::default();
        let bash = BashTool;
        let fs_schema = serde_json::to_string(&fs.parameters()).expect("fs schema");
        let bash_schema = serde_json::to_string(&bash.parameters()).expect("bash schema");
        let total = LEAN_DEMO_SYSTEM_PROMPT.len()
            + fs.name().len()
            + fs.description().len()
            + fs_schema.len()
            + bash.name().len()
            + bash.description().len()
            + bash_schema.len();
        // Character count of the static prompt plus the two tool schemas.
        // This is not a provider token measurement and does not include the
        // user message. 16_000 characters is the checked ceiling.
        assert!(total < 16_000, "lean static surface is {total} characters");
    }

    #[test]
    fn default_base_contains_identity_and_planning() {
        let p = cli_system_prompt_base();
        assert!(p.contains("PLANNING"));
        assert!(p.contains("IDENTITY") || p.contains("capable AI assistant"));
        assert!(p.contains("Caliche"));
        assert!(p.contains("caliche"));
        assert!(p.contains("SELF-VALIDATION"));
        assert!(p.contains("cargo test"));
        assert!(p.contains("ERROR RECOVERY"));
    }

    #[test]
    fn default_base_contains_grounding_and_style_nudges() {
        let p = cli_system_prompt_base();
        assert!(
            p.contains("READ-BEFORE-CRITIQUE"),
            "prompt should include read-before-critique rule"
        );
        assert!(
            p.contains("DOCS vs. FINDINGS"),
            "prompt should distinguish docs from findings"
        );
        assert!(
            p.contains("COUNTING DISCIPLINE"),
            "prompt should require authoritative counts"
        );
        assert!(
            p.contains("action=members"),
            "prompt should mention the workspace members action"
        );
        assert!(
            p.contains("No emojis"),
            "prompt should ban decorative symbols"
        );
        assert!(
            p.contains("DEEP DIVE SCOPING"),
            "prompt should include deep-dive scoping rule"
        );
        assert!(
            p.contains("DUPLICATE CALL HANDLING"),
            "prompt should explain how to react to cached/duplicate tool results"
        );
        assert!(
            p.contains("Match the surrounding file's indentation"),
            "prompt should require matching existing style"
        );
        assert!(
            p.contains("Apply large edits yourself"),
            "prompt should require independent large edits"
        );
        assert!(
            p.contains("speculation -- haven't read this"),
            "READ-BEFORE-CRITIQUE rule should require labeling unread-file claims as speculation"
        );
        assert!(
            p.contains("Now I'll") && p.contains("lead-in narration"),
            "execution rule should explicitly forbid lead-in narration like \"Now I'll ...\" before tool calls"
        );
    }

    #[test]
    fn project_context_owned_trims_once() {
        let out = PromptBuilder::new()
            .base("base".to_string())
            .project_context_owned("  ## Project\nExample  ".to_string())
            .add_skill("  skill  ".to_string())
            .build();
        assert_eq!(out, "base\n\n## Project\nExample\n\nskill");
    }

    #[test]
    fn prompt_builder_with_project_layers() {
        let out = PromptBuilder::new()
            .with_default_base(false)
            .project_context(Some("## Project\nExample project."))
            .build();
        assert!(out.contains("PLANNING"));
        assert!(out.contains("Example project"));
        assert!(out.contains("## Project"));
    }

    #[test]
    fn prompt_builder_skill_appended() {
        let out = PromptBuilder::new()
            .with_default_base(false)
            .add_skill("You are in debugging mode.".to_string())
            .build();
        assert!(out.contains("PLANNING"));
        assert!(out.contains("debugging mode"));
    }
}
