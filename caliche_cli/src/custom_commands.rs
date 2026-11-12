//! Custom user-defined commands loaded from Markdown files.
//!
//! Users can create custom commands by placing Markdown files in
//! `~/.caliche/commands/`. Each file becomes a slash command.
//!
//! Example: `~/.caliche/commands/test.md` becomes `/test`
//!
//! ```markdown
//! ---
//! description: Run tests with coverage
//! model: anthropic/claude-3-5-sonnet
//! ---
//! Run the test suite with coverage and suggest fixes for failures.
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Metadata for a custom command (parsed from YAML front matter)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommandMetadata {
    /// Description shown in help
    #[serde(default)]
    pub description: String,
    /// Optional model override
    #[serde(default)]
    pub model: Option<String>,
    /// Optional provider override
    #[serde(default)]
    pub provider: Option<String>,
    /// Optional agent/mode to use
    #[serde(default)]
    pub agent: Option<String>,
    /// Whether this command requires confirmation
    #[serde(default)]
    pub confirm: bool,
    /// Arguments the command accepts
    #[serde(default)]
    pub args: Vec<String>,
}

/// A custom command definition
#[derive(Debug, Clone)]
pub struct CustomCommand {
    /// Command name (without leading /)
    pub name: String,
    /// Metadata from front matter
    pub metadata: CommandMetadata,
    /// The prompt template (after front matter)
    pub prompt: String,
    /// Source file path
    pub source: PathBuf,
}

impl CustomCommand {
    /// Parse a custom command from file contents
    pub fn parse(name: &str, content: &str, source: PathBuf) -> Result<Self> {
        let (metadata, prompt) = parse_front_matter(content)?;

        Ok(Self {
            name: name.to_string(),
            metadata,
            prompt,
            source,
        })
    }

    /// Render the prompt with `args` and resolve its `@` references.
    ///
    /// A command that is not from the user's own command directory comes
    /// from the workspace, so its references may name only files inside the
    /// workspace: a project command cannot pull in a file from elsewhere on
    /// the host.
    pub async fn resolve_prompt(
        &self,
        args: &[&str],
        resolver: crate::context_refs::ContextResolver,
    ) -> crate::context_refs::ResolvedContext {
        let resolver = if self.name.starts_with("user:") {
            resolver
        } else {
            resolver.workspace_only()
        };
        resolver.resolve_prompt(&self.render_prompt(args)).await
    }

    /// Render the prompt with arguments
    pub fn render_prompt(&self, args: &[&str]) -> String {
        let mut prompt = self.prompt.clone();

        // Replace numbered placeholders like {0}, {1}, etc.
        for (i, arg) in args.iter().enumerate() {
            prompt = prompt.replace(&format!("{{{}}}", i), arg);
        }

        // Replace named placeholders from metadata.args
        for (i, arg_name) in self.metadata.args.iter().enumerate() {
            if let Some(value) = args.get(i) {
                prompt = prompt.replace(&format!("{{{}}}", arg_name), value);
            }
        }

        // Replace $PLACEHOLDER style placeholders (uppercase, alphanumeric + underscore)
        for (i, arg) in args.iter().enumerate() {
            if let Some(arg_name) = self.metadata.args.get(i) {
                // Replace $ARG_NAME style
                let placeholder = format!("${}", arg_name.to_uppercase());
                prompt = prompt.replace(&placeholder, arg);
            }
        }

        // Replace {args} with all arguments joined
        prompt = prompt.replace("{args}", &args.join(" "));

        prompt
    }
}

/// Parse YAML front matter from Markdown content
fn parse_front_matter(content: &str) -> Result<(CommandMetadata, String)> {
    let content = content.trim();

    // Check for front matter delimiter
    if !content.starts_with("---") {
        return Ok((CommandMetadata::default(), content.to_string()));
    }

    // Find the closing delimiter
    let rest = &content[3..];
    if let Some(end_pos) = rest.find("\n---") {
        let yaml_str = &rest[..end_pos].trim();
        let body = rest[end_pos + 4..].trim();

        let metadata: CommandMetadata = serde_norway::from_str(yaml_str).unwrap_or_default();

        Ok((metadata, body.to_string()))
    } else {
        // No closing delimiter, treat whole thing as content
        Ok((CommandMetadata::default(), content.to_string()))
    }
}

/// Custom command registry
#[derive(Debug, Default)]
pub struct CommandRegistry {
    /// Loaded commands indexed by name
    commands: HashMap<String, CustomCommand>,
}

impl CommandRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the user commands directory path
    pub fn user_commands_dir() -> Result<PathBuf> {
        // Try XDG config first, then fall back to ~/.caliche
        if let Some(config_dir) = dirs::config_dir() {
            Ok(config_dir.join("caliche").join("commands"))
        } else {
            let home = dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Could not determine home directory"))?;
            Ok(home.join(".caliche").join("commands"))
        }
    }

    /// Get the project commands directory path
    pub fn project_commands_dir() -> Option<PathBuf> {
        std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(".caliche").join("commands"))
    }

    /// Load all custom commands from disk (user and project directories)
    pub async fn load() -> Result<Self> {
        let mut registry = Self::new();

        // Load from user directory
        let user_dir = Self::user_commands_dir()?;
        if user_dir.exists() {
            Self::load_from_directory(&mut registry, &user_dir, "user").await?;
        }

        // Load from project directory (overrides user commands with same name)
        if let Some(cwd) = std::env::current_dir()
            .ok()
            .and_then(|cwd| cwd.canonicalize().ok())
            .map(|cwd| caliche_sandbox::external_tool_path(&cwd))
        {
            let commands = tokio::task::spawn_blocking(move || load_project_commands(&cwd))
                .await
                .map_err(|e| anyhow::anyhow!("project command load failed: {e}"))?;
            for cmd in commands {
                registry.commands.insert(cmd.name.clone(), cmd);
            }
        }

        Self::load_from_discovery(&mut registry);

        Ok(registry)
    }

    fn load_from_discovery(registry: &mut Self) {
        let options = caliche::discovery::auto::DiscoveryOptions {
            rules: false,
            skills: false,
            commands: true,
            subagents: false,
            mcp: false,
        };
        let snapshot = caliche::discovery::auto::discover_all(&options);
        for cmd in snapshot.commands {
            let source_name = if cmd.scope == caliche::discovery::auto::SourceScope::UserHome {
                "user"
            } else {
                "project"
            };
            let key = format!("{}:{}", source_name, cmd.name);
            registry.commands.insert(
                key.clone(),
                CustomCommand {
                    name: key,
                    metadata: CommandMetadata {
                        description: cmd.description,
                        ..CommandMetadata::default()
                    },
                    prompt: cmd.prompt,
                    source: cmd.source_path,
                },
            );
        }
    }

    /// Load commands from a directory under the user's home or config dir.
    ///
    /// This follows links: users symlink their own dotfiles there, and the
    /// sandbox cannot write to it. Project commands go through
    /// [`load_project_commands`] instead.
    async fn load_from_directory(registry: &mut Self, dir: &PathBuf, prefix: &str) -> Result<()> {
        let mut entries = tokio::fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only process .md files
            if path.extension().map(|e| e == "md").unwrap_or(false)
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                // Skip files starting with underscore (private commands)
                if stem.starts_with('_') {
                    continue;
                }

                match tokio::fs::read_to_string(&path).await {
                    Ok(content) => match CustomCommand::parse(stem, &content, path.clone()) {
                        Ok(mut cmd) => {
                            // Prefix command name with source
                            let prefixed_name = format!("{}:{}", prefix, stem);
                            cmd.name = prefixed_name.clone();
                            registry.commands.insert(prefixed_name, cmd);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to parse custom command {}: {}",
                                path.display(),
                                e
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!("Failed to read custom command {}: {}", path.display(), e);
                    }
                }
            }
        }
        Ok(())
    }

    /// Get a command by name (supports both prefixed and unprefixed names)
    pub fn get(&self, name: &str) -> Option<&CustomCommand> {
        // Strip leading / if present
        let name = name.strip_prefix('/').unwrap_or(name);

        // Try exact match first
        if let Some(cmd) = self.commands.get(name) {
            return Some(cmd);
        }

        // Fall back to the bare command stem: commands are keyed as
        // `{user|project}:{stem}` but the REPL looks up `/stem` directly.
        for (key, cmd) in &self.commands {
            if let Some((_, unprefixed)) = key.split_once(':') {
                if unprefixed == name {
                    return Some(cmd);
                }
            }
        }

        None
    }

    /// Check if a command exists
    pub fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// List all available custom commands
    pub fn list(&self) -> Vec<(&str, &str)> {
        let mut commands: Vec<_> = self
            .commands
            .iter()
            .map(|(name, cmd)| {
                let desc = if cmd.metadata.description.is_empty() {
                    "Custom command"
                } else {
                    cmd.metadata.description.as_str()
                };
                (name.as_str(), desc)
            })
            .collect();

        commands.sort_by(|a, b| a.0.cmp(b.0));
        commands
    }

    /// Get the number of loaded commands
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// Load `.caliche/commands/*.md` below the canonical working directory `cwd`
/// without following any link.
///
/// A cloned repository or a sandboxed command can plant a link there that
/// would pull an outside file into a prompt, so a linked directory or file
/// is skipped with one warning naming the path and the reason. A missing
/// directory is skipped quietly.
fn load_project_commands(cwd: &Path) -> Vec<CustomCommand> {
    let dir = cwd.join(".caliche").join("commands");
    let entries = match caliche_sandbox::list_dir_beneath(cwd, &dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            tracing::warn!(path = %dir.display(), "skipping project commands: {e}");
            return Vec::new();
        }
    };
    let mut commands = Vec::new();
    for entry in entries {
        let path = dir.join(&entry.name);
        let Some(stem) = path
            .extension()
            .filter(|e| *e == "md")
            .and_then(|_| path.file_stem())
            .and_then(|s| s.to_str())
        else {
            continue;
        };
        // Skip files starting with underscore (private commands)
        if stem.starts_with('_') {
            continue;
        }
        if entry.kind != caliche_sandbox::EntryKind::File {
            tracing::warn!(path = %path.display(), "skipping project command: it is not a regular file");
            continue;
        }
        let Some(content) = caliche::discovery::auto::read_workspace_text(cwd, &path) else {
            continue;
        };
        match CustomCommand::parse(stem, &content, path.clone()) {
            Ok(mut cmd) => {
                cmd.name = format!("project:{stem}");
                commands.push(cmd);
            }
            Err(e) => {
                tracing::warn!("Failed to parse custom command {}: {}", path.display(), e);
            }
        }
    }
    commands
}

/// Create an example custom command file
pub async fn create_example_command() -> Result<PathBuf> {
    let commands_dir = CommandRegistry::user_commands_dir()?;
    tokio::fs::create_dir_all(&commands_dir).await?;

    let example_path = commands_dir.join("example.md");

    let content = r#"---
description: Example custom command - explain a file
args: [file]
---
Please read and explain the following file: {file}

Provide:
1. A brief summary of what the file does
2. Key functions/classes and their purposes
3. Any notable patterns or techniques used
"#;

    tokio::fs::write(&example_path, content).await?;

    Ok(example_path)
}

/// Print help for custom commands
pub fn print_custom_commands_help(registry: &CommandRegistry) {
    if registry.is_empty() {
        println!("\nNo custom commands found.");
        println!("Create commands in: ~/.caliche/commands/");
        println!("\nExample: ~/.caliche/commands/test.md");
        println!("  ---");
        println!("  description: Run tests");
        println!("  ---");
        println!("  Run the test suite and report failures.");
        return;
    }

    println!("\nCustom Commands:");
    for (name, desc) in registry.list() {
        println!("  \x1b[32m/{:<15}\x1b[0m {}", name, desc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_front_matter() {
        let content = r#"---
description: Test command
model: gpt-4
---
Do something useful."#;

        let Ok((meta, body)) = parse_front_matter(content) else {
            panic!("front matter should parse");
        };
        assert_eq!(meta.description, "Test command");
        assert_eq!(meta.model, Some("gpt-4".to_string()));
        assert_eq!(body, "Do something useful.");
    }

    #[test]
    fn test_parse_no_front_matter() {
        let content = "Just a plain prompt.";
        let Ok((meta, body)) = parse_front_matter(content) else {
            panic!("plain body should parse");
        };
        assert!(meta.description.is_empty());
        assert_eq!(body, "Just a plain prompt.");
    }

    #[test]
    fn test_render_prompt_with_args() {
        let cmd = CustomCommand {
            name: "test".to_string(),
            metadata: CommandMetadata {
                args: vec!["file".to_string()],
                ..Default::default()
            },
            prompt: "Analyze {file} and {0}".to_string(),
            source: PathBuf::new(),
        };

        let rendered = cmd.render_prompt(&["test.rs"]);
        assert_eq!(rendered, "Analyze test.rs and test.rs");
    }
}

/// Project commands are read without following links.
#[cfg(all(test, unix))]
mod link_tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;

    const OUTSIDE: &str = "OUTSIDE-SECRET-TEXT";

    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn canonical_dir() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().canonicalize().expect("canonical temp dir");
        (dir, path)
    }

    fn names_and_prompts(commands: &[CustomCommand]) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = commands
            .iter()
            .map(|c| (c.name.clone(), c.prompt.clone()))
            .collect();
        out.sort();
        out
    }

    #[test]
    fn a_symlinked_project_command_file_is_skipped() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        fs::write(outside.join("secret.md"), OUTSIDE).expect("outside file");
        let dir = ws.join(".caliche/commands");
        fs::create_dir_all(&dir).expect("commands dir");
        fs::write(dir.join("ok.md"), "run the tests").expect("ok.md");
        symlink(outside.join("secret.md"), dir.join("evil.md")).expect("symlink");

        let commands = load_project_commands(&ws);

        assert_eq!(
            names_and_prompts(&commands),
            vec![("project:ok".to_string(), "run the tests".to_string())]
        );
    }

    /// A project command's `@` reference to a file outside the workspace is
    /// refused, while one inside it is attached. The same text in a user
    /// command reads the outside file, since the user wrote it.
    #[tokio::test]
    async fn a_project_command_names_no_file_outside_the_workspace() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        fs::write(outside.join("secret.txt"), OUTSIDE).expect("outside file");
        fs::write(ws.join("inside.txt"), "INSIDE-TEXT").expect("inside file");
        let dir = ws.join(".caliche/commands");
        fs::create_dir_all(&dir).expect("commands dir");
        fs::write(
            dir.join("leak.md"),
            format!(
                "summarize @{} and @inside.txt",
                outside.join("secret.txt").display()
            ),
        )
        .expect("leak.md");
        let resolver = || crate::context_refs::ContextResolver::new(ws.clone(), None, Vec::new());

        let mut commands = load_project_commands(&ws);
        assert_eq!(commands.len(), 1);
        let project = commands.remove(0);
        let resolved = project.resolve_prompt(&[], resolver()).await;
        let mut user = project.clone();
        user.name = "user:leak".to_string();
        let from_user = user.resolve_prompt(&[], resolver()).await;

        assert!(!resolved.message.contains(OUTSIDE), "{}", resolved.message);
        assert!(
            resolved
                .message
                .contains("is outside the workspace, and text from the workspace can name only files inside it"),
            "{}",
            resolved.message
        );
        assert!(
            resolved
                .message
                .ends_with("--- File: inside.txt ---\nINSIDE-TEXT"),
            "{}",
            resolved.message
        );
        assert!(
            !resolved.allowed_paths.contains(&outside.join("secret.txt")),
            "{:?}",
            resolved.allowed_paths
        );
        assert!(from_user.message.contains(OUTSIDE), "{}", from_user.message);
    }

    #[test]
    fn a_symlinked_project_commands_dir_is_skipped() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        fs::write(outside.join("secret.md"), OUTSIDE).expect("outside file");
        fs::create_dir_all(ws.join(".caliche")).expect(".caliche");
        symlink(&outside, ws.join(".caliche/commands")).expect("symlink");

        let commands = load_project_commands(&ws);

        assert_eq!(names_and_prompts(&commands), Vec::new());
    }
}
