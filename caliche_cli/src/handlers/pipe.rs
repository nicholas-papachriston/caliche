//! Pipe input command handler.
//!
//! Handles processing input from stdin (piped input).

use crate::cancellation::CancellationToken;
use crate::commands::OutputFormat;
use crate::context_refs::ContextResolver;
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use caliche::AgentService;

/// Arguments for the pipe command.
#[derive(Debug, Clone)]
pub struct PipeArgs {
    pub prompt: String,
    pub output_format: OutputFormat,
    /// `--sandbox`: the policy tools run under.
    pub sandbox: Option<String>,
    /// `--sandbox-backend`.
    pub sandbox_backend: Option<String>,
}

/// Handler for pipe input processing.
pub struct PipeHandler;

#[async_trait]
impl CommandHandler for PipeHandler {
    type Args = PipeArgs;

    async fn execute(
        &self,
        args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        run_pipe_input(&args, ctx.cancellation_token)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "pipe"
    }

    fn description(&self) -> &'static str {
        "Process piped input from stdin"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Reads input from stdin and processes it with the given prompt.\n\
             Useful for piping file contents or command output to the agent.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "cat file.txt | caliche --pipe \"Summarize this\"",
            "git diff | caliche --pipe \"Review these changes\"",
        ]
    }
}

/// Run the pipe input handler. Tools run in the same sandbox as the REPL
/// and `-p`.
///
/// When `cancellation_token` is `Some`, the chat/stream call is raced against
/// Ctrl+C so long pipes (`git log | caliche --pipe ...`) remain interruptible.
pub async fn run_pipe_input(
    args: &PipeArgs,
    cancellation_token: Option<&CancellationToken>,
) -> Result<()> {
    use std::io::{self, BufRead, IsTerminal};
    let prompt = args.prompt.as_str();
    let output_format = &args.output_format;

    if io::stdin().is_terminal() {
        anyhow::bail!(
            "--pipe requires piped input. Example: cat file.txt | caliche --pipe \"explain this\""
        );
    }

    let mut content = String::new();
    for line in io::stdin().lock().lines() {
        content.push_str(&line?);
        content.push('\n');
    }

    if content.trim().is_empty() {
        anyhow::bail!("No input received from stdin");
    }

    let tool_names: Option<Vec<String>> = None;
    let mut service = AgentService::from_config(
        caliche::AgentServiceConfig::new().with_tool_names_or_defaults(tool_names),
    )
    .await?;
    crate::sandbox_startup::configure_sandbox(
        &mut service,
        args.sandbox.as_deref(),
        args.sandbox_backend.as_deref(),
    )
    .await?;
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    service.set_artifact_store(Some(std::sync::Arc::new(
        caliche::ArtifactStore::for_workspace(&workspace_root, None),
    )));
    let resolved = ContextResolver::from_service(&service)
        .resolve_prompt(prompt)
        .await;
    let full_message = format!("{}\n\n---\nContent:\n{}", resolved.message, content);

    if let Some(token) = cancellation_token
        && token.is_cancelled()
    {
        anyhow::bail!("Operation cancelled by user before pipe execution");
    }

    match output_format {
        OutputFormat::Json => {
            print_pipe_json(
                &mut service,
                prompt,
                &full_message,
                &content,
                resolved.attachments,
            )
            .await?
        }
        OutputFormat::Jsonl => {
            print_pipe_jsonl(
                &mut service,
                prompt,
                &full_message,
                &content,
                resolved.attachments,
            )
            .await?
        }
        OutputFormat::Stream => {
            print_pipe_stream(
                &mut service,
                &full_message,
                resolved.attachments,
                cancellation_token,
            )
            .await?
        }
        OutputFormat::Text => {
            print_pipe_text(&mut service, &full_message, resolved.attachments).await?
        }
    }

    crate::shutdown::exit(0);
}

async fn print_pipe_json(
    service: &mut AgentService,
    prompt: &str,
    full_message: &str,
    content: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<()> {
    if !attachments.is_empty() {
        let response_text = collect_attachment_response(service, full_message, attachments).await?;
        let output = serde_json::json!({
            "prompt": prompt,
            "input_length": content.len(),
            "response": response_text,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        crate::shutdown::exit(0);
    }
    let response = service.chat(full_message).await?;
    let output = serde_json::json!({
        "prompt": prompt,
        "input_length": content.len(),
        "response": response.text(),
        "usage": {
            "prompt_tokens": response.usage().prompt_tokens,
            "completion_tokens": response.usage().completion_tokens,
            "total_tokens": response.usage().total_tokens,
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

async fn print_pipe_jsonl(
    service: &mut AgentService,
    prompt: &str,
    full_message: &str,
    content: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<()> {
    if !attachments.is_empty() {
        let response_text = collect_attachment_response(service, full_message, attachments).await?;
        let event = serde_json::json!({
            "type": "response",
            "prompt": prompt,
            "input_length": content.len(),
            "response": response_text,
        });
        println!("{}", serde_json::to_string(&event)?);
        crate::shutdown::exit(0);
    }
    let response = service.chat(full_message).await?;
    let event = serde_json::json!({
        "type": "response",
        "prompt": prompt,
        "input_length": content.len(),
        "response": response.text(),
        "usage": {
            "prompt_tokens": response.usage().prompt_tokens,
            "completion_tokens": response.usage().completion_tokens,
            "total_tokens": response.usage().total_tokens,
        }
    });
    println!("{}", serde_json::to_string(&event)?);
    Ok(())
}

async fn print_pipe_stream(
    service: &mut AgentService,
    full_message: &str,
    attachments: Vec<caliche::messages::Attachment>,
    cancellation_token: Option<&CancellationToken>,
) -> Result<()> {
    use caliche::messages::StreamEvent;
    use futures_util::StreamExt;
    use std::io::Write;

    let mut stream = if attachments.is_empty() {
        service.stream_chat(full_message)
    } else {
        service.stream_chat_with_attachments_and_cancel_flag(
            full_message,
            attachments,
            cancellation_token.map(CancellationToken::shared_cancelled_flag),
        )
    };
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextChunk { content } => {
                print!("{content}");
                std::io::stdout().flush()?;
            }
            StreamEvent::Done | StreamEvent::LimitReached { .. } => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}

async fn print_pipe_text(
    service: &mut AgentService,
    full_message: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<()> {
    if !attachments.is_empty() {
        let response_text = collect_attachment_response(service, full_message, attachments).await?;
        println!("{response_text}");
        crate::shutdown::exit(0);
    }
    let response = service.chat(full_message).await?;
    println!("{}", response.text());
    Ok(())
}

async fn collect_attachment_response(
    service: &mut AgentService,
    message: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<String> {
    use caliche::messages::StreamEvent;
    use futures_util::StreamExt;

    let mut stream =
        service.stream_chat_with_attachments_and_cancel_flag(message, attachments, None);
    let mut response_text = String::new();
    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextChunk { content } => response_text.push_str(&content),
            StreamEvent::Done | StreamEvent::LimitReached { .. } => break,
            _ => {}
        }
    }
    Ok(response_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe_handler_metadata() {
        let handler = PipeHandler;
        assert_eq!(handler.name(), "pipe");
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_pipe_args() {
        let args = PipeArgs {
            prompt: "Summarize this".to_string(),
            output_format: OutputFormat::Text,
            sandbox: None,
            sandbox_backend: None,
        };
        assert_eq!(args.prompt, "Summarize this");
    }
}
