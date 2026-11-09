//! Git-related command handlers.
//!
//! Handles PR generation, commit message generation, and code review commands.

use crate::errors::{CliError, CliResult};
use crate::handlers::host_git::HostGit;
use crate::handlers::stream_output::{PlainStreamSinkOptions, process_plain_stream};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use crate::traits::factory::ServiceConfig;
use anyhow::{Context, Result};
use async_trait::async_trait;
use caliche::AgentService;
use std::io::Write;

/// Arguments for the PR command.
#[derive(Debug, Clone, Default)]
pub struct PrArgs;

/// Arguments for the commit command.
#[derive(Debug, Clone, Default)]
pub struct CommitArgs;

/// Arguments for the review command.
#[derive(Debug, Clone, Default)]
pub struct ReviewArgs;

/// Handler for the `/pr` command - generate PR descriptions.
pub struct PrHandler;

#[async_trait]
impl CommandHandler for PrHandler {
    type Args = PrArgs;

    async fn execute(
        &self,
        _args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        // Create a service for this operation
        let mut service = ctx
            .service_factory
            .create_service(ServiceConfig::default())
            .await?;

        let mut stdout = std::io::stdout();
        run_pr_command(&mut service, &mut stdout)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "pr"
    }

    fn description(&self) -> &'static str {
        "Generate a pull request description from git changes"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Analyzes git diff and commit history to generate a PR description.\n\
             Includes title, summary, and test plan sections.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &["caliche pr", "/pr (in REPL)"]
    }
}

/// Handler for the `/commit` command - generate commit messages.
pub struct CommitHandler;

#[async_trait]
impl CommandHandler for CommitHandler {
    type Args = CommitArgs;

    async fn execute(
        &self,
        _args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let mut service = ctx
            .service_factory
            .create_service(ServiceConfig::default())
            .await?;

        let mut stdout = std::io::stdout();
        run_commit_command(&mut service, &mut stdout, true)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "commit"
    }

    fn description(&self) -> &'static str {
        "Generate an intelligent commit message from staged changes"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Analyzes staged git changes and generates a conventional commit message.\n\
             Follows the format: type(scope): description",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &["caliche commit", "/commit (in REPL)"]
    }
}

/// Handler for the `/review` command - review code changes.
pub struct ReviewHandler;

#[async_trait]
impl CommandHandler for ReviewHandler {
    type Args = ReviewArgs;

    async fn execute(
        &self,
        _args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let mut service = ctx
            .service_factory
            .create_service(ServiceConfig::default())
            .await?;

        let mut stdout = std::io::stdout();
        run_review_command(&mut service, &mut stdout)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "review"
    }

    fn description(&self) -> &'static str {
        "Review staged or unstaged code changes"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Performs an AI-powered code review on git changes.\n\
             Checks for bugs, security issues, and code quality.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &["caliche review", "/review (in REPL)"]
    }
}

/// Run the PR command implementation. Output is written to `out`.
/// For CLI, pass `&mut std::io::stdout()`. For REPL capture, pass `&mut Vec::new()` then `String::from_utf8(vec)`.
pub async fn run_pr_command<W: Write + Send>(
    service: &mut AgentService,
    out: &mut W,
) -> Result<()> {
    let git = HostGit::open(&std::env::current_dir()?).await?;
    // Get git diff and status
    let diff_output = git
        .command()
        .args(["diff", "HEAD~1..HEAD"])
        .output()
        .await?;

    let diff = String::from_utf8_lossy(&diff_output.stdout);

    if diff.trim().is_empty() {
        // Try diff against main/master
        let diff_main = git
            .command()
            .args(["diff", "main...HEAD"])
            .output()
            .await
            .ok()
            .and_then(|o| if o.status.success() { Some(o) } else { None });

        let diff_master = git
            .command()
            .args(["diff", "master...HEAD"])
            .output()
            .await
            .ok()
            .and_then(|o| if o.status.success() { Some(o) } else { None });

        let diff_content = diff_main
            .or(diff_master)
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();

        if diff_content.trim().is_empty() {
            writeln!(out, "No changes found to create a PR for.")?;
            return Ok(());
        }
    }

    // Get commit log
    let log_output = git
        .command()
        .args(["log", "--oneline", "-10", "main...HEAD"])
        .output()
        .await
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    // Get current branch
    let branch_output = git
        .command()
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .await?;
    let branch = String::from_utf8_lossy(&branch_output.stdout)
        .trim()
        .to_string();

    let diff_used = if diff.trim().is_empty() {
        let diff_main = git
            .command()
            .args(["diff", "main...HEAD"])
            .output()
            .await
            .ok();
        let diff_master = git
            .command()
            .args(["diff", "master...HEAD"])
            .output()
            .await
            .ok();
        diff_main
            .or(diff_master)
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default()
    } else {
        diff.to_string()
    };

    let prompt = format!(
        r#"Generate a pull request description for the following changes.

Branch: {}

Recent commits:
{}

Diff summary (first 3000 chars):
{}

Please provide:
1. A concise PR title (one line)
2. A summary section with bullet points
3. A test plan section

Format as markdown."#,
        branch,
        log_output,
        diff_used.chars().take(3000).collect::<String>()
    );

    writeln!(out, "\nGenerating PR description...\n")?;
    out.flush()?;

    let stream = service.stream_chat(&prompt);
    if let Err(e) = process_plain_stream(stream, out, PlainStreamSinkOptions::default()).await {
        writeln!(out, "\nFailed to generate PR description: {e}")?;
    }
    writeln!(out)?;
    Ok(())
}

/// Detect the git editor from environment variables and git config.
/// Priority: GIT_EDITOR > git config core.editor > EDITOR > vi
async fn detect_git_editor(git: &HostGit) -> Result<String> {
    // Check GIT_EDITOR environment variable first
    if let Ok(editor) = std::env::var("GIT_EDITOR") {
        return Ok(editor);
    }

    // Check git config core.editor
    let git_config_output = git
        .command()
        .args(["config", "--get", "core.editor"])
        .output()
        .await;

    if let Ok(output) = git_config_output {
        if output.status.success() {
            let editor = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !editor.is_empty() {
                return Ok(editor);
            }
        }
    }

    // Check EDITOR environment variable
    if let Ok(editor) = std::env::var("EDITOR") {
        return Ok(editor);
    }

    // Fallback to vi
    Ok("vi".to_string())
}

/// Open the git editor with a commit message and return the edited message.
async fn open_git_editor(git: &HostGit, initial_message: &str) -> Result<Option<String>> {
    let editor = detect_git_editor(git).await?;

    let temp = crate::utils::create_private_temp_file("caliche_commit_", ".txt")?;
    let temp_path = crate::utils::temp_file_path(&temp);

    // Write initial message to temp file
    tokio::fs::write(&temp_path, initial_message)
        .await
        .context("Failed to write commit message to temp file")?;

    println!("Opening editor: {}", editor);
    println!("Edit the commit message in the editor, save and close when done.");

    // Parse editor command (may contain arguments like "code --wait" or "vim +1")
    let editor_parts: Vec<&str> = editor.split_whitespace().collect();
    let editor_cmd = editor_parts[0];
    let mut editor_args: Vec<String> = editor_parts[1..].iter().map(|s| s.to_string()).collect();
    editor_args.push(temp_path.display().to_string());

    // Open editor (blocking operation)
    let status = tokio::process::Command::new(editor_cmd)
        .args(&editor_args)
        .status()
        .await
        .context(format!("Failed to execute editor: {}", editor))?;

    if !status.success() {
        anyhow::bail!("Editor exited with error code: {:?}", status.code());
    }

    // Read edited content (temp file removed when `temp` is dropped)
    let content = tokio::fs::read_to_string(&temp_path)
        .await
        .context("Failed to read edited commit message")?;

    let trimmed = content.trim().to_string();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed))
    }
}

/// Run the commit command implementation.
/// When `out` is provided (e.g. REPL capture), only the generated message is written and no interactive prompt runs.
/// When using stdout, the full interactive flow runs (generate, then y/N/e).
pub async fn run_commit_command<W: Write + Send>(
    service: &mut AgentService,
    out: &mut W,
    interactive: bool,
) -> Result<()> {
    let git = HostGit::open(&std::env::current_dir()?).await?;
    let diff_output = git.command().args(["diff", "--staged"]).output().await?;
    let diff = String::from_utf8_lossy(&diff_output.stdout);
    if diff.trim().is_empty() {
        return write_empty_staged_status(out, &git).await;
    }

    let prompt = commit_message_prompt(&diff);
    writeln!(out, "\nGenerating commit message...\n")?;
    out.flush()?;

    let stream = service.stream_chat(&prompt);
    let stream_result = process_plain_stream(
        stream,
        out,
        PlainStreamSinkOptions {
            show_tool_starts: false,
        },
    )
    .await;
    writeln!(out)?;

    let accumulated_message = match stream_result {
        Ok(result) => result.text,
        Err(e) => {
            writeln!(out, "\nFailed to generate commit message: {e}")?;
            return Ok(());
        }
    };

    let message = accumulated_message.trim();
    if message.is_empty() {
        writeln!(out, "No commit message generated.")?;
        return Ok(());
    }

    if !interactive {
        writeln!(out, "Suggested commit message:\n")?;
        writeln!(out, "{message}")?;
        return Ok(());
    }

    prompt_and_commit(out, &git, message).await
}

pub(super) async fn write_empty_staged_status<W: Write + Send>(
    out: &mut W,
    git: &HostGit,
) -> Result<()> {
    let status_output = git.command().args(["status", "--short"]).output().await?;
    let status = String::from_utf8_lossy(&status_output.stdout);
    if status.trim().is_empty() {
        writeln!(out, "No changes to commit.")?;
    } else {
        writeln!(
            out,
            "No staged changes. Stage your changes first with 'git add'."
        )?;
        writeln!(out, "\nUnstaged changes:")?;
        writeln!(out, "{status}")?;
    }
    Ok(())
}

fn commit_message_prompt(diff: &str) -> String {
    format!(
        r#"Generate a concise, descriptive git commit message for the following staged changes.

Diff:
{}

Rules:
1. First line: type(scope): short description (max 72 chars)
2. Types: feat, fix, docs, style, refactor, test, chore
3. Leave a blank line after the first line
4. Body: explain what and why (not how)
5. Use imperative mood ("add" not "added")

Provide ONLY the commit message, no explanations."#,
        diff.chars().take(4000).collect::<String>()
    )
}

async fn git_commit_with_message(git: &HostGit, message: &str) -> Result<std::process::Output> {
    git.command()
        .args(["commit", "-m", message])
        .output()
        .await
        .context("failed to run git commit")
}

async fn prompt_and_commit<W: Write + Send>(
    out: &mut W,
    git: &HostGit,
    message: &str,
) -> Result<()> {
    writeln!(out, "\nSuggested commit message:\n")?;
    writeln!(out, "{message}")?;
    writeln!(out)?;
    write!(out, "Use this message? [y/N/e(dit)] ")?;
    out.flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    apply_commit_choice(out, git, message, input.trim()).await
}

async fn apply_commit_choice<W: Write + Send>(
    out: &mut W,
    git: &HostGit,
    message: &str,
    choice: &str,
) -> Result<()> {
    match choice.to_lowercase().as_str() {
        "y" | "yes" => commit_success_or_fail(out, git, message, "Committed successfully!").await,
        "e" | "edit" => commit_after_edit(out, git, message).await,
        _ => {
            writeln!(out, "Commit cancelled. Copy the message above if needed.")?;
            Ok(())
        }
    }
}

async fn commit_success_or_fail<W: Write + Send>(
    out: &mut W,
    git: &HostGit,
    message: &str,
    success_text: &str,
) -> Result<()> {
    let commit_result = git_commit_with_message(git, message).await?;
    if commit_result.status.success() {
        writeln!(out, "{success_text}")?;
    } else {
        writeln!(
            out,
            "Commit failed: {}",
            String::from_utf8_lossy(&commit_result.stderr)
        )?;
    }
    Ok(())
}

async fn commit_after_edit<W: Write + Send>(
    out: &mut W,
    git: &HostGit,
    message: &str,
) -> Result<()> {
    match open_git_editor(git, message).await {
        Ok(Some(edited_message)) => {
            commit_success_or_fail(
                out,
                git,
                &edited_message,
                "Committed successfully with edited message!",
            )
            .await
        }
        Ok(None) => {
            writeln!(out, "Commit cancelled - empty message.")?;
            Ok(())
        }
        Err(e) => {
            writeln!(out, "Failed to open editor: {e}")?;
            writeln!(
                out,
                "Copy the message above and use: git commit -m \"<message>\""
            )?;
            Ok(())
        }
    }
}

/// Run the review command implementation. Output is written to `out`.
pub async fn run_review_command<W: Write + Send>(
    service: &mut AgentService,
    out: &mut W,
) -> Result<()> {
    let git = HostGit::open(&std::env::current_dir()?).await?;
    // Get staged changes
    let diff_output = git.command().args(["diff", "--staged"]).output().await?;

    let mut diff = String::from_utf8_lossy(&diff_output.stdout).to_string();

    if diff.trim().is_empty() {
        // Get unstaged changes
        let diff_unstaged = git.command().args(["diff"]).output().await?;

        let diff_u = String::from_utf8_lossy(&diff_unstaged.stdout);

        if diff_u.trim().is_empty() {
            writeln!(out, "No changes to review.")?;
            return Ok(());
        }

        writeln!(
            out,
            "No staged changes. Reviewing unstaged changes instead.\n"
        )?;
        diff = diff_u.to_string();
    }

    let prompt = format!(
        r#"Review the following code changes and provide feedback.

Diff:
{}

Please analyze:
1. **Potential Issues**: bugs, security issues, edge cases
2. **Code Quality**: readability, maintainability, best practices
3. **Suggestions**: improvements, alternatives
4. **Overall Assessment**: brief summary

Be constructive and specific. Reference line numbers when possible."#,
        diff.chars().take(6000).collect::<String>()
    );

    writeln!(out, "\nReviewing changes...\n")?;
    out.flush()?;

    let stream = service.stream_chat(&prompt);
    if let Err(e) = process_plain_stream(stream, out, PlainStreamSinkOptions::default()).await {
        writeln!(out, "\nFailed to review changes: {e}")?;
    }
    writeln!(out)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pr_handler_metadata() {
        let handler = PrHandler;
        assert_eq!(handler.name(), "pr");
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_commit_handler_metadata() {
        let handler = CommitHandler;
        assert_eq!(handler.name(), "commit");
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_review_handler_metadata() {
        let handler = ReviewHandler;
        assert_eq!(handler.name(), "review");
        assert!(!handler.description().is_empty());
    }
}
