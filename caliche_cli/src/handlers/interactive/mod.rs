//! Interactive mode and REPL handlers.
//!
//! Handles interactive chat sessions and the REPL interface.

pub mod compression;
pub mod context;
mod single_message;
pub mod skills;
pub mod welcome;

use crate::commands::OutputFormat;
use crate::context_refs::ContextResolver;
use crate::errors::{CliError, CliResult};
use crate::output_mode::OutputMode;
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use caliche::AgentService;
use caliche::messages::StreamEvent;
use futures_util::StreamExt;
use std::io;
use std::io::{IsTerminal, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamRecoveryChoice {
    Retry,
    SaveAndExit,
    Quit,
}

pub(super) fn looks_like_transient_network_error(message: &str) -> bool {
    let m = message.to_lowercase();
    // Keep this conservative: only classify as retryable if it's very likely transient.
    m.contains("failed to send request")
        || m.contains("error sending request")
        || m.contains("connection")
        || m.contains("timed out")
        || m.contains("timeout")
        || m.contains("dns")
        || m.contains("tls")
        || m.contains("broken pipe")
        || m.contains("connection reset")
        || m.contains("connection refused")
}

pub(super) fn prompt_stream_recovery() -> Result<StreamRecoveryChoice> {
    // Only prompt in an interactive terminal session.
    if !(io::stdin().is_terminal() && io::stdout().is_terminal()) {
        return Ok(StreamRecoveryChoice::Quit);
    }

    eprintln!();
    eprintln!("A network/stream error occurred. What would you like to do?");
    eprintln!("  [R]etry       - retry the request");
    eprintln!("  [S]ave & exit - save session and exit (if applicable)");
    eprintln!("  [Q]uit        - exit without saving");
    eprint!("Choice [R/S/Q]: ");
    io::stderr().flush()?;

    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    let choice = line
        .trim()
        .chars()
        .next()
        .unwrap_or('r')
        .to_ascii_lowercase();

    Ok(match choice {
        's' => StreamRecoveryChoice::SaveAndExit,
        'q' => StreamRecoveryChoice::Quit,
        _ => StreamRecoveryChoice::Retry,
    })
}

fn ensure_default_artifact_store(service: &mut AgentService) {
    if service.artifact_store().is_some() {
        return;
    }
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    service.set_artifact_store(Some(std::sync::Arc::new(
        caliche::ArtifactStore::for_workspace(&workspace_root, None),
    )));
}

pub(super) async fn collect_attachment_response(
    service: &mut AgentService,
    message: &str,
    attachments: Vec<caliche::messages::Attachment>,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<String> {
    let cancel_flag =
        cancellation_token.map(crate::cancellation::CancellationToken::shared_cancelled_flag);
    let mut stream =
        service.stream_chat_with_attachments_and_cancel_flag(message, attachments, cancel_flag);
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

/// Arguments for interactive mode.
#[derive(Debug, Clone, Default)]
pub struct InteractiveArgs {
    pub session_id: Option<String>,
    pub output_mode: Option<OutputMode>,
}

/// Arguments for single message mode.
#[derive(Debug, Clone)]
pub struct SingleMessageArgs {
    pub message: String,
    pub output_format: OutputFormat,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Arguments for resuming a session.
#[derive(Debug, Clone, Default)]
pub struct ResumeArgs;

/// Handler for interactive mode.
pub struct InteractiveHandler;

#[async_trait]
impl CommandHandler for InteractiveHandler {
    type Args = InteractiveArgs;

    async fn execute(
        &self,
        args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        Box::pin(run_interactive_mode_harness(
            args.session_id,
            args.output_mode.unwrap_or(OutputMode::Text),
            ctx.cancellation_token,
        ))
        .await
        .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Interactive)
    }

    fn name(&self) -> &'static str {
        "interactive"
    }

    fn description(&self) -> &'static str {
        "Start an interactive chat session"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Starts an interactive REPL session with the agent.\n\
             Supports slash commands, context references, and session management.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "caliche",
            "caliche --session my-session",
            "caliche --resume",
        ]
    }
}

/// Handler for single message mode.
pub struct SingleMessageHandler;

#[async_trait]
impl CommandHandler for SingleMessageHandler {
    type Args = SingleMessageArgs;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        run_single_message(
            &args.message,
            &args.output_format,
            args.provider.as_deref(),
            args.model.as_deref(),
        )
        .await
        .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "message"
    }

    fn description(&self) -> &'static str {
        "Send a single message and exit"
    }

    fn help(&self) -> Option<&'static str> {
        Some("Send a single message to the agent and receive a response.")
    }

    fn examples(&self) -> &[&'static str] {
        &[
            "caliche \"Explain dependency injection\"",
            "caliche -m \"Explain async/await\" --output-format json",
        ]
    }
}

/// Handler for resuming the last session.
pub struct ResumeHandler;

#[async_trait]
impl CommandHandler for ResumeHandler {
    type Args = ResumeArgs;

    async fn execute(
        &self,
        _args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        Box::pin(run_resume_last_session(ctx.cancellation_token))
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;

        Ok(CommandResult::Interactive)
    }

    fn name(&self) -> &'static str {
        "resume"
    }

    fn description(&self) -> &'static str {
        "Resume the last interactive session"
    }

    fn help(&self) -> Option<&'static str> {
        Some("Resumes the most recent interactive session with its conversation history.")
    }

    fn examples(&self) -> &[&'static str] {
        &["caliche --resume", "caliche -r"]
    }
}

/// Run interactive mode using the unified core module (REPL).
/// When `cancellation_token` is `Some`, Ctrl+C will cancel in-flight agent operations.
///
/// Bootstraps the ROADMAP 2.5 thin-client path: discover/start an HTTP server and
/// route message turns through remote SSE while keeping slash commands local.
pub async fn run_interactive_mode_harness(
    session_id: Option<String>,
    _output_mode: OutputMode,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<()> {
    use crate::commands::Args;
    use crate::repl::run_repl;
    use crate::thin_client::bootstrap_thin_client;
    use crate::traits::factory::ServiceConfig;
    use crate::traits::factory::create_service_factory;
    use clap::Parser;

    let service_factory = create_service_factory();
    let service = service_factory
        .create_service(ServiceConfig::default())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create service: {}", e))?;

    let args = Args::parse_from(["caliche"]);
    let cli_config = crate::config::load_standard_config().await;
    let (service, thin) = Box::pin(bootstrap_thin_client(
        service,
        &args,
        cli_config.as_ref(),
        None,
    ))
    .await
    .map_err(|e| crate::thin_client::format_bootstrap_failure(&e))?;
    let thin_pair = Some((thin.client.clone(), thin.session_id.clone()));

    let result = Box::pin(run_repl(
        service,
        session_id,
        false,
        true,
        None,
        None,
        cancellation_token,
        false,
        None,
        None,
        false,
        None,
        &[],
        thin_pair,
        None,
    ))
    .await;
    thin.shutdown().await;
    result
}

/// Run a single message and exit.
pub async fn run_single_message(
    message: &str,
    output_format: &OutputFormat,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<()> {
    let tool_names: Option<Vec<String>> = None;
    let mut service = AgentService::from_config(
        caliche::AgentServiceConfig::new()
            .with_provider(
                provider.map(|p| p.parse().ok()).flatten(),
                model.map(|m| m.to_string()),
            )
            .with_tool_names_or_defaults(tool_names),
    )
    .await?;
    ensure_default_artifact_store(&mut service);
    let resolved = ContextResolver::from_service(&service)
        .resolve_prompt(message)
        .await;
    let message = resolved.message;
    match output_format {
        OutputFormat::Json => {
            single_message::print_single_message_json(&mut service, &message, resolved.attachments)
                .await
        }
        OutputFormat::Jsonl => {
            single_message::print_single_message_jsonl(&mut service, &message, resolved.attachments)
                .await
        }
        OutputFormat::Text => {
            single_message::print_single_message_text(&mut service, &message, resolved.attachments)
                .await
        }
        OutputFormat::Stream => {
            single_message::stream_single_message_with_retry(
                &mut service,
                &message,
                &resolved.attachments,
            )
            .await
        }
    }
}

/// Run a single message using an existing service (e.g. from main).
/// When `cancellation_token` is `Some`, Ctrl+C will abort the request.
pub async fn run_single_message_with_service(
    service: &mut AgentService,
    message: String,
    output_format: &OutputFormat,
    quiet: bool,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<()> {
    use uuid::Uuid;

    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "cli_chat", request_id = %request_id);
    let _guard = span.entered();

    tracing::info!("Processing CLI chat request");
    ensure_default_artifact_store(service);
    let resolved = ContextResolver::from_service(service)
        .resolve_prompt(&message)
        .await;
    let message = resolved.message;

    if !resolved.attachments.is_empty() {
        let response_text = collect_attachment_response(
            service,
            &message,
            resolved.attachments,
            cancellation_token,
        )
        .await?;
        match output_format {
            OutputFormat::Json => {
                let json_output = serde_json::json!({
                    "user": message,
                    "agent": response_text,
                });
                println!("{}", serde_json::to_string_pretty(&json_output)?);
            }
            OutputFormat::Jsonl | OutputFormat::Stream => {
                let jsonl_output = serde_json::json!({
                    "type": "response",
                    "user": message,
                    "agent": response_text,
                });
                println!("{}", serde_json::to_string(&jsonl_output)?);
            }
            OutputFormat::Text => {
                if quiet {
                    println!("{response_text}");
                } else {
                    println!("\nUser: {message}");
                    println!("Agent: {response_text}");
                }
            }
        }
        return Ok(());
    }

    let chat_result = if let Some(token) = cancellation_token {
        tokio::select! {
            biased;
            _ = token.cancelled() => {
                return Err(anyhow::anyhow!("Operation cancelled by user"));
            }
            result = service.chat(&message) => result,
        }
    } else {
        service.chat(&message).await
    };

    match chat_result {
        Ok(chat_response) => match output_format {
            OutputFormat::Json => {
                let json_output = serde_json::json!({
                    "user": message,
                    "agent": chat_response.text(),
                    "usage": {
                        "prompt_tokens": chat_response.usage().prompt_tokens,
                        "completion_tokens": chat_response.usage().completion_tokens,
                        "total_tokens": chat_response.usage().total_tokens,
                    }
                });
                println!("{}", serde_json::to_string_pretty(&json_output)?);
            }
            OutputFormat::Jsonl => {
                let jsonl_output = serde_json::json!({
                    "type": "response",
                    "user": message,
                    "agent": chat_response.text(),
                    "usage": {
                        "prompt_tokens": chat_response.usage().prompt_tokens,
                        "completion_tokens": chat_response.usage().completion_tokens,
                        "total_tokens": chat_response.usage().total_tokens,
                    }
                });
                println!("{}", serde_json::to_string(&jsonl_output)?);
            }
            OutputFormat::Stream => {
                let event = serde_json::json!({
                    "type": "response",
                    "user": message,
                    "agent": chat_response.text(),
                    "usage": {
                        "prompt_tokens": chat_response.usage().prompt_tokens,
                        "completion_tokens": chat_response.usage().completion_tokens,
                        "total_tokens": chat_response.usage().total_tokens,
                    }
                });
                println!("{}", serde_json::to_string(&event)?);
            }
            OutputFormat::Text => {
                if quiet {
                    // Quiet mode: just print the response
                    println!("{}", chat_response.text());
                } else {
                    println!("\nUser: {}", message);
                    println!("Agent: {}", chat_response.text());
                    if let Some(total) = chat_response.usage().total_tokens {
                        println!(
                            "\nToken usage - prompt: {}, completion: {}, total: {}",
                            chat_response.usage().prompt_tokens.unwrap_or(0),
                            chat_response.usage().completion_tokens.unwrap_or(0),
                            total
                        );
                    }
                }
            }
        },
        Err(e) => {
            match output_format {
                OutputFormat::Json | OutputFormat::Jsonl | OutputFormat::Stream => {
                    let error_json = serde_json::json!({
                        "type": "error",
                        "error": e.to_string(),
                        "user": message,
                    });
                    eprintln!("{}", serde_json::to_string(&error_json)?);
                }
                OutputFormat::Text => {
                    eprintln!("Error: {}", e);
                }
            }
            crate::shutdown::exit(1);
        }
    }

    Ok(())
}

/// Run a single message with session persistence.
///
/// If `existing_session` is `Some`, the conversation history is loaded into memory
/// before sending the message. Otherwise a fresh session is created.
/// Returns the session ID so callers can reference it.
pub async fn run_single_message_session(
    service: &mut AgentService,
    message: String,
    output_format: &OutputFormat,
    quiet: bool,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
    session_store: &crate::storage::SessionStore,
    existing_session: Option<crate::session::CliSession>,
) -> Result<String> {
    let request_id = uuid::Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "cli_chat_session", request_id = %request_id);
    let _guard = span.entered();

    let mut session = single_message::resolve_or_create_session(existing_session);
    single_message::seed_session_memory(service, &session.conversation_history).await?;

    ensure_default_artifact_store(service);
    let resolved = ContextResolver::from_service(service)
        .resolve_prompt(&message)
        .await;
    let message = resolved.message;
    let session_id = session.id.clone();

    if !resolved.attachments.is_empty() {
        let response_text = collect_attachment_response(
            service,
            &message,
            resolved.attachments,
            cancellation_token,
        )
        .await?;
        single_message::persist_session_turn(
            &mut session,
            service,
            &message,
            &response_text,
            None,
            session_store,
        )
        .await;
        single_message::print_session_attachment_output(
            output_format,
            &session_id,
            &message,
            &response_text,
            quiet,
        )?;
        return Ok(session_id);
    }

    match single_message::chat_with_optional_cancel(service, &message, cancellation_token).await {
        single_message::SessionChatOutcome::Cancelled => {
            Err(anyhow::anyhow!("Operation cancelled by user"))
        }
        single_message::SessionChatOutcome::Completed(Ok(chat_response)) => {
            single_message::persist_session_turn(
                &mut session,
                service,
                &message,
                chat_response.text(),
                chat_response.usage().total_tokens,
                session_store,
            )
            .await;
            single_message::print_session_chat_output(
                output_format,
                &session_id,
                &message,
                &chat_response,
                quiet,
            )?;
            Ok(session_id)
        }
        single_message::SessionChatOutcome::Completed(Err(error)) => {
            single_message::print_session_error(output_format, &session_id, &message, &error)?;
            Ok(session_id)
        }
    }
}

/// Resume the last interactive session.
pub async fn run_resume_last_session(
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<()> {
    use crate::storage::SessionStore;

    let store = SessionStore::new().await?;

    // Get the most recent session
    let sessions = store.list_detailed_async().await?;

    if sessions.is_empty() {
        println!("No previous sessions found.");
        return Ok(());
    }

    let last_session = &sessions[0];
    println!("Resuming session: {}", last_session.id);

    Box::pin(run_interactive_mode_harness(
        Some(last_session.id.clone()),
        OutputMode::Text,
        cancellation_token,
    ))
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interactive_handler_metadata() {
        let handler = InteractiveHandler;
        assert_eq!(handler.name(), "interactive");
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_single_message_handler_metadata() {
        let handler = SingleMessageHandler;
        assert_eq!(handler.name(), "message");
        assert!(!handler.description().is_empty());
    }

    #[test]
    fn test_resume_handler_metadata() {
        let handler = ResumeHandler;
        assert_eq!(handler.name(), "resume");
        assert!(!handler.description().is_empty());
    }
}
