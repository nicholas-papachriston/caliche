//! Project initialization command handler.
//!
//! Handles creating AGENTS.md and other project configuration files.

use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use crate::traits::factory::ServiceConfig;
use anyhow::Result;
use async_trait::async_trait;
use caliche_contracts::text::truncate_utf8;
use std::path::{Path, PathBuf};

impl InitArgs {
    pub fn from_flags(no_ai: bool, output: &Option<String>, force: bool) -> Self {
        Self {
            no_ai,
            output: output.clone(),
            force,
        }
    }
}

/// Arguments for the init command.
#[derive(Debug, Clone, Default)]
pub struct InitArgs {
    /// Skip AI analysis and use static template generation instead.
    pub no_ai: bool,
    /// Output file path. Defaults to `AGENTS.md` in the current directory.
    pub output: Option<String>,
    /// Overwrite existing file without prompting (skips all confirmation prompts).
    pub force: bool,
}

/// Handler for the `init` command - initialize project configuration.
pub struct InitHandler;

#[async_trait]
impl CommandHandler for InitHandler {
    type Args = InitArgs;

    async fn execute(
        &self,
        args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        if args.no_ai {
            run_init_command(&args)
                .await
                .map_err(|e| CliError::Command(e.to_string()))?;
        } else {
            run_init_command_ai(&args, ctx)
                .await
                .map_err(|e| CliError::Command(e.to_string()))?;
        }

        Ok(CommandResult::Success(Some(
            "Project initialized successfully".to_string(),
        )))
    }

    fn name(&self) -> &'static str {
        "init"
    }

    fn description(&self) -> &'static str {
        "Initialize agent configuration for a project (AI-powered by default)"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Analyzes the codebase with AI and creates a detailed AGENTS.md with real project \
             context: structure, build commands, conventions, important files, and notes.\n\n\
             If AGENTS.md already exists, you can choose to update it (AI reads the existing \
             content and enriches it) or overwrite it entirely.\n\n\
             Use --no-ai to skip AI analysis and generate a static template instead.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "caliche init",
            "caliche init --force",
            "caliche init --output docs/agent-context.md",
            "caliche init --no-ai",
        ]
    }

    fn requires_credentials(&self) -> bool {
        false
    }

    fn supports_offline(&self) -> bool {
        false
    }
}

/// Resolve the output path from args, defaulting to AGENTS.md in cwd.
fn resolve_output_path(cwd: &Path, output: &Option<String>) -> PathBuf {
    match output {
        Some(p) => {
            let p = PathBuf::from(p);
            if p.is_absolute() { p } else { cwd.join(p) }
        }
        None => cwd.join("AGENTS.md"),
    }
}

/// Run the init command with static template generation.
pub async fn run_init_command(args: &InitArgs) -> Result<()> {
    use std::io::IsTerminal;

    let cwd = std::env::current_dir()?;
    let output_path = resolve_output_path(&cwd, &args.output);
    let unattended =
        args.force || !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal());

    if output_path.exists() && !args.force {
        if unattended {
            anyhow::bail!(
                "{} already exists. Re-run with --force to overwrite.",
                output_path.display()
            );
        }
        use dialoguer::{Confirm, theme::ColorfulTheme};
        let overwrite = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt(format!(
                "{} already exists. Overwrite?",
                output_path.display()
            ))
            .default(false)
            .interact()?;

        if !overwrite {
            println!("Initialization cancelled.");
            return Ok(());
        }
    }

    // Detect project type
    let project_type = detect_project_type(&cwd);
    println!("Detected project type: {}", project_type);

    // Get project name
    let default_name = cwd
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("my-project")
        .to_string();

    let project_name = if unattended {
        default_name
    } else {
        use dialoguer::{Input, theme::ColorfulTheme};
        Input::with_theme(&ColorfulTheme::default())
            .with_prompt("Project name")
            .default(default_name)
            .interact_text()?
    };

    // Generate content and write primary output file
    let content = generate_project_context(&project_name, &project_type, &cwd);

    tokio::fs::write(&output_path, &content).await?;
    println!("\nCreated: {}", output_path.display());

    let create_agentignore = if unattended {
        false
    } else {
        use dialoguer::{Confirm, theme::ColorfulTheme};
        Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("Create .agentignore file?")
            .default(true)
            .interact()?
    };

    if create_agentignore {
        let agentignore_path = cwd.join(".agentignore");
        let agentignore_content = generate_agentignore(&project_type);
        tokio::fs::write(&agentignore_path, &agentignore_content).await?;
        println!("Created: {}", agentignore_path.display());
    }

    println!(
        "\nProject initialized! Edit {} to customize agent behavior.",
        output_path.display()
    );

    Ok(())
}

/// Answers provided by the user during the interactive questionnaire.
struct UserAnswers {
    /// Whether to update an existing file rather than overwrite it.
    update_mode: bool,
    /// Existing file content when in update mode.
    existing_content: Option<String>,
    /// User-supplied project description (may be empty to let AI infer).
    description: String,
    /// Special conventions or rules the user wants to enforce.
    conventions: String,
    /// Key files or areas the user wants the agent to know about.
    key_files: String,
    /// Additional notes to include.
    notes: String,
}

/// Run an interactive questionnaire before AI analysis.
///
/// Returns `None` if the user cancelled.
fn run_questionnaire(output_path: &std::path::Path, force: bool) -> Result<Option<UserAnswers>> {
    use dialoguer::{Input, Select, theme::ColorfulTheme};

    let theme = ColorfulTheme::default();
    let existing_content: Option<String> = if output_path.exists() {
        std::fs::read_to_string(output_path)
            .ok()
            .filter(|s| !s.trim().is_empty())
    } else {
        None
    };

    let update_mode = if existing_content.is_some() && !force {
        println!("\n{} already exists.", output_path.display());
        let choice = Select::with_theme(&theme)
            .with_prompt("What would you like to do?")
            .items(&[
                "Update — AI reads the existing file and enriches it with fresh analysis",
                "Overwrite — AI generates a completely new file from scratch",
                "Cancel",
            ])
            .default(0)
            .interact()?;

        match choice {
            0 => true,
            1 => false,
            _ => return Ok(None),
        }
    } else {
        false
    };

    println!("\nAnswer a few questions to guide the AI analysis.");
    println!("Press Enter to skip any question and let the AI infer from the code.\n");

    let description: String = Input::with_theme(&theme)
        .with_prompt("Project description (what does this project do?)")
        .allow_empty(true)
        .interact_text()?;

    let conventions: String = Input::with_theme(&theme)
        .with_prompt(
            "Special conventions or rules (e.g. 'use Bun not npm', 'prefer type-safe APIs')",
        )
        .allow_empty(true)
        .interact_text()?;

    let key_files: String = Input::with_theme(&theme)
        .with_prompt(
            "Key files or areas the agent must know about (comma-separated, or leave blank)",
        )
        .allow_empty(true)
        .interact_text()?;

    let notes: String = Input::with_theme(&theme)
        .with_prompt("Anything else to include (env vars, deployment notes, gotchas)?")
        .allow_empty(true)
        .interact_text()?;

    Ok(Some(UserAnswers {
        update_mode,
        existing_content,
        description,
        conventions,
        key_files,
        notes,
    }))
}

/// System prompt used when the AI subagent analyzes the codebase.
const AI_INIT_SYSTEM_PROMPT: &str = r#"You are a codebase analyst producing an AGENTS.md file — a project context document
that gives an AI coding agent everything it needs to work effectively in this repository.

Your job:
1. Use your read tools to thoroughly explore the project: list the directory tree, read README files,
   read the primary package manifest (Cargo.toml, package.json, pyproject.toml, go.mod, etc.),
   read Makefiles or CI configs, and sample 3-5 source files to understand code style and conventions.
2. For workspaces/monorepos: read the root manifest and each member's manifest (e.g. Cargo: root and member Cargo.toml; npm/yarn: root and package package.json).
3. Check for linter/formatter configs: .clippy.toml, .eslintrc, ruff.toml, .editorconfig, .prettierrc.
4. Check for AGENTS.md or .cursor/rules/ if they exist — incorporate those conventions.
5. Synthesize everything into a detailed, accurate AGENTS.md.

Output ONLY valid Markdown — no preamble, no explanation, no code fences around the whole document.
Use this exact structure (all sections required, never leave placeholder comments):

# <project name>

## Project Overview
<2-4 sentences: what the project does, who it's for, what makes it notable>

## Repository Structure
<bullet list of key directories and files with a one-line description of each>

## Build Commands
<the actual commands to build, test, lint, and format — inferred from real config files>

## Code Conventions
<concrete rules inferred from linter configs, existing code style, and any rules files>

## Testing
<how to run tests, where test files live, any test conventions>

## Important Files
<bullet list: key entry points, config files, and files an agent should read before making changes>

## Environment and Configuration
<required env vars, optional env vars with defaults, .env file conventions>

## Notes
<workspace structure if a monorepo, feature flags, known gotchas, deployment notes>
"#;

/// Build the user-facing prompt for the AI analysis, incorporating questionnaire answers.
fn build_user_prompt(cwd: &std::path::Path, answers: &UserAnswers) -> String {
    let mut prompt = format!(
        "Analyze the codebase in `{}`. Thoroughly explore the project before writing anything:\n\
         1. List the top-level directory contents\n\
         2. Read the primary package manifest (Cargo.toml, package.json, etc.)\n\
         3. Read the README if it exists\n\
         4. Read 3-5 source files to understand code style\n\
         5. Check for linter/formatter configs\n\n",
        cwd.display()
    );

    if answers.update_mode {
        if let Some(existing) = &answers.existing_content {
            prompt.push_str(
                "The project already has an AGENTS.md. Update it with fresh, accurate information \
                 from your analysis. Keep any sections that are already accurate, improve sections \
                 that are vague or missing, and add any new sections that would help an agent.\n\n\
                 Existing AGENTS.md content:\n```\n",
            );
            prompt.push_str(existing);
            prompt.push_str("\n```\n\n");
        }
    }

    if !answers.description.is_empty() {
        prompt.push_str(&format!(
            "The user describes the project as: \"{}\"\n\n",
            answers.description
        ));
    }

    if !answers.conventions.is_empty() {
        prompt.push_str(&format!(
            "The user specifies these conventions/rules that must appear in the output: \"{}\"\n\n",
            answers.conventions
        ));
    }

    if !answers.key_files.is_empty() {
        prompt.push_str(&format!(
            "The user says these files/areas are important: \"{}\"\n\n",
            answers.key_files
        ));
    }

    if !answers.notes.is_empty() {
        prompt.push_str(&format!(
            "Additional notes from the user to include: \"{}\"\n\n",
            answers.notes
        ));
    }

    prompt.push_str(
        "Output only the raw Markdown content of AGENTS.md with no surrounding commentary.",
    );

    prompt
}

/// Run the init command with AI-powered codebase analysis (the default mode).
///
/// Runs an interactive questionnaire to gather user context, then spawns a
/// constrained read-only subagent that thoroughly explores the codebase and
/// produces a detailed, populated AGENTS.md.
pub async fn run_init_command_ai(args: &InitArgs, ctx: &mut CommandContext<'_>) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let output_path = resolve_output_path(&cwd, &args.output);

    // Interactive questionnaire — returns None if user cancelled
    let answers = match run_questionnaire(&output_path, args.force)? {
        Some(a) => a,
        None => {
            println!("Initialization cancelled.");
            return Ok(());
        }
    };

    println!("\nAnalyzing codebase... (this may take 15-30 seconds)");

    let config = ServiceConfig::new()
        .tools(vec![
            "fs".to_string(),
            "list_files".to_string(),
            "glob".to_string(),
            "fs".to_string(),
        ])
        .system_prompt(AI_INIT_SYSTEM_PROMPT)
        .working_dir(cwd.clone());

    let mut service = ctx
        .service_factory
        .create_service(config)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create AI service: {}", e))?;

    let user_prompt = build_user_prompt(&cwd, &answers);

    let response = service
        .chat(&user_prompt)
        .await
        .map_err(|e| anyhow::anyhow!("AI analysis failed: {}", e))?;

    let content = extract_markdown_content(&response.text());

    tokio::fs::write(&output_path, &content).await?;

    let action = if answers.update_mode {
        "Updated"
    } else {
        "Created"
    };
    println!("{}: {} (AI-generated)", action, output_path.display());
    println!("Review the file and make any adjustments before committing.");

    Ok(())
}

/// Run AI-powered init from the REPL, reusing the existing session's provider.
///
/// Unlike `run_init_command_ai`, this creates its own constrained `AgentService`
/// directly (no `CommandContext` required), making it callable from the slash-command
/// executor where only `&mut AgentService` is available.
pub async fn run_init_ai_for_repl(output: Option<String>, force: bool) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let output_path = resolve_output_path(&cwd, &output);

    let answers = match run_questionnaire(&output_path, force)? {
        Some(a) => a,
        None => {
            println!("Initialization cancelled.");
            return Ok(());
        }
    };

    println!("\nAnalyzing codebase... (this may take 15-30 seconds)");

    // Spawn a constrained sub-service using env-var credentials (same as the session).
    let mut sub_service = caliche::AgentService::from_config(
        caliche::AgentServiceConfig::new().with_tool_names(vec![
            "fs".to_string(),
            "list_files".to_string(),
            "glob".to_string(),
            "fs".to_string(),
        ]),
    )
    .await
    .map_err(|e| anyhow::anyhow!("Failed to create analysis service: {}", e))?;

    sub_service
        .agent_mut()
        .set_system_prompt(AI_INIT_SYSTEM_PROMPT.to_string());

    let user_prompt = build_user_prompt(&cwd, &answers);

    let response = sub_service
        .chat(&user_prompt)
        .await
        .map_err(|e| anyhow::anyhow!("AI analysis failed: {}", e))?;

    let content = extract_markdown_content(&response.text());

    tokio::fs::write(&output_path, &content).await?;

    let action = if answers.update_mode {
        "Updated"
    } else {
        "Created"
    };
    println!("{}: {} (AI-generated)", action, output_path.display());
    println!("Review the file and make any adjustments before committing.");

    Ok(())
}

/// Strip any leading/trailing prose that wraps the Markdown content.
///
/// If the model surrounds the output with an explanation like "Here is the AGENTS.md:",
/// this removes that text. If the content starts with `#` (a Markdown heading) directly,
/// it is returned unchanged.
fn extract_markdown_content(raw: &str) -> String {
    let trimmed = raw.trim();

    // Find the first line that starts a Markdown heading — that's the real content start.
    if let Some(pos) = trimmed.find("\n# ").or_else(|| {
        if trimmed.starts_with("# ") {
            Some(0)
        } else {
            None
        }
    }) {
        trimmed[pos..].trim_start().to_string()
    } else {
        trimmed.to_string()
    }
}

/// Detect project type (distinguishes typescript vs javascript).
fn detect_project_type(path: &Path) -> String {
    if path.join("Cargo.toml").exists() {
        "rust".to_string()
    } else if path.join("package.json").exists() {
        if path.join("tsconfig.json").exists() {
            "typescript".to_string()
        } else {
            "javascript".to_string()
        }
    } else if path.join("pyproject.toml").exists() || path.join("setup.py").exists() {
        "python".to_string()
    } else if path.join("go.mod").exists() {
        "go".to_string()
    } else if path.join("pom.xml").exists() || path.join("build.gradle").exists() {
        "java".to_string()
    } else if path.join("Gemfile").exists() {
        "ruby".to_string()
    } else if path.join("composer.json").exists() {
        "php".to_string()
    } else {
        "generic".to_string()
    }
}

/// Generate project context content for AGENTS.md.
fn generate_project_context(project_name: &str, project_type: &str, cwd: &Path) -> String {
    let structure = analyze_codebase_structure(cwd);
    let conventions = analyze_conventions(cwd, project_type);
    let build_commands = detect_build_commands(cwd, project_type);

    let type_specific = match project_type {
        "rust" => {
            r#"
## Build Commands
- `cargo build` - Build the project
- `cargo build --release` - Build for release
- `cargo test` - Run tests
- `cargo clippy` - Run linter
- `cargo fmt` - Format code

## Code Style
- Follow Rust idioms and best practices
- Handle errors properly - avoid unwrap()
- Prefer references over cloning
- Use `anyhow::Result` for error handling
"#
        }
        "typescript" | "javascript" | "node" => {
            r#"
## Build Commands
- `npm install` - Install dependencies
- `npm run build` - Build the project
- `npm test` - Run tests
- `npm run lint` - Run linter

## Code Style
- Use TypeScript when possible
- Prefer async/await over callbacks
- Use ESLint and Prettier
"#
        }
        "python" => {
            r#"
## Build Commands
- `uv sync` or `pip install -e .` - Install in development mode
- `pytest` - Run tests
- `ruff check .` - Run linter
- `black .` - Format code

## Code Style
- Use type hints
- Follow PEP 8 style guide
- Use uv for package management
"#
        }
        "go" => {
            r#"
## Build Commands
- `go build` - Build the project
- `go test ./...` - Run tests
- `go fmt ./...` - Format code
- `golangci-lint run` - Run linter

## Code Style
- Follow effective Go guidelines
- Handle errors explicitly
- Use standard library when possible
"#
        }
        _ => {
            r#"
## Build Commands
- Add your build commands here

## Code Style
- Add your code style guidelines here
"#
        }
    };

    format!(
        r#"# {project_name}

## Project Overview
<!-- Describe your project here -->

## Codebase Structure
{structure}

## Build Commands
{build_commands}

## Code Conventions
{conventions}

{type_specific}
## Testing
- Run tests before committing
- Ensure all tests pass

## Important Files
<!-- List key files the agent should know about -->

## Notes
<!-- Any additional notes for the agent -->
"#,
        project_name = project_name,
        structure = structure,
        build_commands = build_commands,
        conventions = conventions,
        type_specific = type_specific
    )
}

/// Generate .agentignore content based on project type.
fn generate_agentignore(project_type: &str) -> String {
    let mut lines = vec![
        "# Agent ignore file",
        "# Files and directories listed here will be ignored by the agent",
        "",
        "# Common ignores",
        ".git/",
        ".env",
        ".env.*",
        "*.log",
        "",
    ];

    match project_type {
        "rust" => {
            lines.extend(&["# Rust", "target/", "Cargo.lock"]);
        }
        "typescript" | "javascript" => {
            lines.extend(&[
                "# Node.js",
                "node_modules/",
                "dist/",
                "build/",
                ".next/",
                "coverage/",
            ]);
        }
        "python" => {
            lines.extend(&[
                "# Python",
                "__pycache__/",
                "*.pyc",
                ".venv/",
                "venv/",
                ".pytest_cache/",
                "*.egg-info/",
            ]);
        }
        "go" => {
            lines.extend(&["# Go", "vendor/"]);
        }
        _ => {}
    }

    lines.join("\n")
}

/// Project configuration loaded from AGENTS.md or similar files.
#[derive(Debug, Clone)]
pub struct ProjectConfig {
    /// Path to the config file.
    pub path: PathBuf,
    /// Filename of the loaded config (for display).
    pub source_filename: String,
    /// Content of the config file.
    pub content: String,
}

const MAX_PROJECT_CONTEXT_BYTES: usize = 12_000;

impl ProjectConfig {
    /// This file as project context that takes at most `share_percent` of
    /// the model's context window.
    #[must_use]
    pub fn into_context(self, share_percent: u8) -> caliche::ProjectContext {
        caliche::ProjectContext {
            path: self.path,
            source_filename: self.source_filename,
            content: self.content,
            share_percent,
        }
    }
}

/// Load project configuration from AGENTS.md or similar project files.
/// Searches from CWD up to (and including) the git root.
pub async fn load_project_config() -> Option<ProjectConfig> {
    let discovery =
        caliche::discovery::auto::discover_all(&caliche::discovery::auto::DiscoveryOptions {
            rules: true,
            skills: false,
            commands: false,
            subagents: false,
            mcp: false,
        });
    let best_rule = discovery.rules.into_iter().max_by_key(|r| r.scope.rank());
    if let Some(rule) = best_rule {
        let filename = rule
            .source_path
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("rule.md")
            .to_string();
        return Some(ProjectConfig {
            path: rule.source_path,
            source_filename: filename,
            content: rule.content,
        });
    }

    use crate::utils::find_git_root;

    let cwd = std::env::current_dir().ok()?;
    let git_root = find_git_root(&cwd);

    let search_dirs: Vec<PathBuf> = if let Some(root) = &git_root {
        let mut dirs = Vec::new();
        let mut current = cwd.clone();
        loop {
            dirs.push(current.clone());
            if current == *root {
                break;
            }
            current = current.parent()?.to_path_buf();
        }
        dirs
    } else {
        vec![cwd]
    };

    let search_filenames = [
        "AGENTS.md",
        "AGENT.md",
        ".caliche.md",
        ".caliche/config.md",
        "agent.md",
    ];

    for dir in search_dirs {
        for filename in search_filenames {
            let path = dir.join(filename);
            if path.exists() {
                match tokio::fs::read_to_string(&path).await {
                    Ok(content) if !content.trim().is_empty() => {
                        let content = if content.len() > MAX_PROJECT_CONTEXT_BYTES {
                            format!(
                                "{}\n\n[project context truncated to {} bytes]",
                                truncate_utf8(&content, MAX_PROJECT_CONTEXT_BYTES),
                                MAX_PROJECT_CONTEXT_BYTES
                            )
                        } else {
                            content
                        };
                        return Some(ProjectConfig {
                            path,
                            source_filename: filename.to_string(),
                            content,
                        });
                    }
                    Err(e) => {
                        tracing::debug!("Failed to read project config {}: {}", path.display(), e);
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// Analyze codebase structure
fn analyze_codebase_structure(cwd: &Path) -> String {
    let mut structure = String::new();

    // Find main entry points
    let entry_points = vec![
        "main.rs",
        "lib.rs",
        "index.ts",
        "index.js",
        "main.py",
        "main.go",
        "src/main.rs",
        "src/lib.rs",
        "src/index.ts",
        "src/index.js",
    ];

    let mut found_entries = Vec::new();
    for entry in entry_points {
        let path = cwd.join(entry);
        if path.exists() {
            found_entries.push(entry);
        }
    }

    if !found_entries.is_empty() {
        structure.push_str("### Entry Points\n");
        for entry in found_entries {
            structure.push_str(&format!("- `{}`\n", entry));
        }
        structure.push('\n');
    }

    // Detect common directories
    let common_dirs = vec![
        ("src", "Source code"),
        ("tests", "Test files"),
        ("docs", "Documentation"),
        ("examples", "Example code"),
        ("scripts", "Build/utility scripts"),
    ];

    let mut found_dirs = Vec::new();
    for (dir, desc) in common_dirs {
        let path = cwd.join(dir);
        if path.is_dir() {
            found_dirs.push((dir, desc));
        }
    }

    if !found_dirs.is_empty() {
        structure.push_str("### Key Directories\n");
        for (dir, desc) in found_dirs {
            structure.push_str(&format!("- `{}/` - {}\n", dir, desc));
        }
    }

    if structure.is_empty() {
        structure.push_str("<!-- Codebase structure will be analyzed here -->\n");
    }

    structure
}

/// Analyze code conventions
fn analyze_conventions(cwd: &Path, project_type: &str) -> String {
    let mut conventions = String::new();

    // Check for common config files
    let config_files = match project_type {
        "rust" => vec![("Cargo.toml", "Rust package configuration")],
        "node" | "typescript" | "javascript" => vec![
            ("package.json", "Node.js package configuration"),
            ("tsconfig.json", "TypeScript configuration"),
        ],
        "python" => vec![
            ("pyproject.toml", "Python project configuration"),
            ("setup.py", "Python setup script"),
        ],
        "go" => vec![("go.mod", "Go module definition")],
        _ => vec![],
    };

    for (file, desc) in config_files {
        if cwd.join(file).exists() {
            conventions.push_str(&format!("- {}: {}\n", file, desc));
        }
    }

    // Check for linter configs
    let linter_configs = vec![
        (".clippy.toml", "Rust Clippy configuration"),
        (".eslintrc", "ESLint configuration"),
        (".prettierrc", "Prettier configuration"),
        ("ruff.toml", "Ruff (Python) configuration"),
    ];

    for (file, desc) in linter_configs {
        if cwd.join(file).exists() {
            conventions.push_str(&format!("- {}: {}\n", file, desc));
        }
    }

    if conventions.is_empty() {
        conventions.push_str("<!-- Code conventions will be detected here -->\n");
    }

    conventions
}

/// Detect build commands from project files
fn detect_build_commands(cwd: &Path, project_type: &str) -> String {
    let mut commands = String::new();

    match project_type {
        "rust" => {
            if cwd.join("Cargo.toml").exists() {
                commands.push_str("- `cargo build` - Build the project\n");
                commands.push_str("- `cargo build --release` - Build for release\n");
                commands.push_str("- `cargo test` - Run tests\n");
                commands.push_str("- `cargo clippy` - Run linter\n");
                commands.push_str("- `cargo fmt` - Format code\n");
            }
        }
        "node" | "typescript" | "javascript" => {
            if cwd.join("package.json").exists() {
                // Try to read package.json scripts
                if let Ok(content) = std::fs::read_to_string(cwd.join("package.json")) {
                    if content.contains("\"scripts\"") {
                        commands.push_str("- `npm install` - Install dependencies\n");
                        commands.push_str("- `npm run build` - Build (if script exists)\n");
                        commands.push_str("- `npm test` - Run tests\n");
                        commands.push_str("- `npm run lint` - Run linter (if script exists)\n");
                    }
                }
            }
        }
        "python" => {
            if cwd.join("pyproject.toml").exists() || cwd.join("setup.py").exists() {
                commands.push_str("- `uv sync` or `pip install -e .` - Install dependencies\n");
                commands.push_str("- `pytest` - Run tests\n");
                commands.push_str("- `ruff check .` - Run linter\n");
            }
        }
        "go" => {
            if cwd.join("go.mod").exists() {
                commands.push_str("- `go build` - Build the project\n");
                commands.push_str("- `go test ./...` - Run tests\n");
                commands.push_str("- `go fmt ./...` - Format code\n");
            }
        }
        _ => {}
    }

    if commands.is_empty() {
        commands.push_str("<!-- Build commands will be detected here -->\n");
    }

    commands
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_handler_metadata() {
        let handler = InitHandler;
        assert_eq!(handler.name(), "init");
        assert!(!handler.description().is_empty());
        assert!(!handler.requires_credentials());
        assert!(!handler.supports_offline());
    }

    #[test]
    fn test_detect_project_type() {
        let path = PathBuf::from(".");
        let _project_type = detect_project_type(path.as_path());
    }

    #[test]
    fn test_generate_project_context() {
        let content = generate_project_context("test-project", "rust", Path::new("."));
        assert!(content.contains("test-project"));
        assert!(content.contains("cargo build"));
    }

    #[test]
    fn test_generate_agentignore() {
        let content = generate_agentignore("rust");
        assert!(content.contains("target/"));

        let content = generate_agentignore("python");
        assert!(content.contains("__pycache__/"));
    }

    #[test]
    fn test_truncate_utf8_respects_char_boundaries() {
        let text = "abc—def";
        let truncated = truncate_utf8(text, 4);
        assert_eq!(truncated, "abc");
    }
}
