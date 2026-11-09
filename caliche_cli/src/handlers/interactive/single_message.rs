//! Named helpers for one-shot CLI message commands.

use super::{
    StreamRecoveryChoice, collect_attachment_response, looks_like_transient_network_error,
    prompt_stream_recovery,
};
use crate::commands::OutputFormat;
use crate::handlers::stream_output::{ColoredStdioSink, process_colored_stdio_stream};
use anyhow::{Context as _, Result};
use caliche::AgentService;

pub(super) async fn print_single_message_json(
    service: &mut AgentService,
    message: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<()> {
    if !attachments.is_empty() {
        let response_text =
            collect_attachment_response(service, message, attachments, None).await?;
        let json = serde_json::json!({ "response": response_text });
        println!("{}", serde_json::to_string_pretty(&json)?);
        return Ok(());
    }
    let response = service.chat(message).await?;
    let json = serde_json::json!({
        "response": response.text(),
        "usage": {
            "prompt_tokens": response.usage().prompt_tokens,
            "completion_tokens": response.usage().completion_tokens,
            "total_tokens": response.usage().total_tokens,
        }
    });
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}

pub(super) async fn print_single_message_jsonl(
    service: &mut AgentService,
    message: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<()> {
    if !attachments.is_empty() {
        let response_text =
            collect_attachment_response(service, message, attachments, None).await?;
        let event = serde_json::json!({
            "type": "response",
            "response": response_text,
        });
        println!("{}", serde_json::to_string(&event)?);
        return Ok(());
    }
    let response = service.chat(message).await?;
    let event = serde_json::json!({
        "type": "response",
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

pub(super) async fn print_single_message_text(
    service: &mut AgentService,
    message: &str,
    attachments: Vec<caliche::messages::Attachment>,
) -> Result<()> {
    if !attachments.is_empty() {
        let response_text =
            collect_attachment_response(service, message, attachments, None).await?;
        println!("{response_text}");
        return Ok(());
    }
    let response = service.chat(message).await?;
    println!("{}", response.text());
    if let Some(total) = response.usage().total_tokens {
        eprintln!(
            "\n\x1b[2mTokens: {} (prompt: {:?}, completion: {:?})\x1b[0m",
            total,
            response.usage().prompt_tokens,
            response.usage().completion_tokens
        );
    }
    Ok(())
}

pub(super) async fn stream_single_message_with_retry(
    service: &mut AgentService,
    message: &str,
    attachments: &[caliche::messages::Attachment],
) -> Result<()> {
    let mut attempts: u32 = 0;
    loop {
        let stream = if attachments.is_empty() {
            service.stream_chat(message)
        } else {
            service.stream_chat_with_attachments_and_cancel_flag(
                message,
                attachments.to_vec(),
                None,
            )
        };
        let mut sink = ColoredStdioSink::interactive_stream();
        let stream_result = process_colored_stdio_stream(stream, &mut sink).await;
        println!();
        match stream_result {
            Ok(result) => {
                if result.text.trim().is_empty() {
                    eprintln!("\x1b[33mWarning: No text response received from the model.\x1b[0m");
                }
                return Ok(());
            }
            Err(e) => match try_stream_recovery(&e, sink.had_text_output, &mut attempts)? {
                StreamRetryDecision::Retry => {}
                StreamRetryDecision::ExitOk => return Ok(()),
            },
        }
    }
}

enum StreamRetryDecision {
    Retry,
    ExitOk,
}

fn try_stream_recovery(
    error: &anyhow::Error,
    had_text_output: bool,
    attempts: &mut u32,
) -> Result<StreamRetryDecision> {
    let err = error.to_string();
    let retryable = looks_like_transient_network_error(&err) && !had_text_output && *attempts < 2;
    if !retryable {
        anyhow::bail!("Stream error: {err}");
    }
    match prompt_stream_recovery()? {
        StreamRecoveryChoice::Retry => {
            *attempts += 1;
            eprintln!("Retrying (attempt {} of {})...", *attempts + 1, 3);
            Ok(StreamRetryDecision::Retry)
        }
        StreamRecoveryChoice::SaveAndExit => {
            eprintln!("No session to save in single-message mode. Exiting.");
            Ok(StreamRetryDecision::ExitOk)
        }
        StreamRecoveryChoice::Quit => anyhow::bail!("Stream error: {err}"),
    }
}

pub(super) fn print_session_attachment_output(
    output_format: &OutputFormat,
    session_id: &str,
    message: &str,
    response_text: &str,
    quiet: bool,
) -> Result<()> {
    match output_format {
        OutputFormat::Json => {
            let json_output = serde_json::json!({
                "session_id": session_id,
                "user": message,
                "agent": response_text,
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        OutputFormat::Jsonl | OutputFormat::Stream => {
            let jsonl_output = serde_json::json!({
                "type": "response",
                "session_id": session_id,
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
    Ok(())
}

pub(super) fn print_session_chat_output(
    output_format: &OutputFormat,
    session_id: &str,
    message: &str,
    chat_response: &caliche::messages::ChatResponse,
    quiet: bool,
) -> Result<()> {
    let response_text = chat_response.text();
    match output_format {
        OutputFormat::Json => {
            let json_output = serde_json::json!({
                "session_id": session_id,
                "user": message,
                "agent": response_text,
                "usage": {
                    "prompt_tokens": chat_response.usage().prompt_tokens,
                    "completion_tokens": chat_response.usage().completion_tokens,
                    "total_tokens": chat_response.usage().total_tokens,
                }
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        OutputFormat::Jsonl | OutputFormat::Stream => {
            let jsonl_output = serde_json::json!({
                "type": "response",
                "session_id": session_id,
                "user": message,
                "agent": response_text,
                "usage": {
                    "prompt_tokens": chat_response.usage().prompt_tokens,
                    "completion_tokens": chat_response.usage().completion_tokens,
                    "total_tokens": chat_response.usage().total_tokens,
                }
            });
            println!("{}", serde_json::to_string(&jsonl_output)?);
        }
        OutputFormat::Text => print_session_text_response(message, chat_response, quiet),
    }
    if !quiet {
        eprintln!(
            "\x1b[90mSession:\x1b[0m \x1b[1m{session_id}\x1b[0m  \x1b[90m(continue: caliche --continue -m \"next message\")\x1b[0m"
        );
    }
    Ok(())
}

fn print_session_text_response(
    message: &str,
    chat_response: &caliche::messages::ChatResponse,
    quiet: bool,
) {
    let response_text = chat_response.text();
    if quiet {
        println!("{response_text}");
        return;
    }
    println!("\nUser: {message}");
    println!("Agent: {response_text}");
    if let Some(total) = chat_response.usage().total_tokens {
        println!(
            "\nToken usage - prompt: {}, completion: {}, total: {}",
            chat_response.usage().prompt_tokens.unwrap_or(0),
            chat_response.usage().completion_tokens.unwrap_or(0),
            total
        );
    }
}

pub(super) fn print_session_error(
    output_format: &OutputFormat,
    session_id: &str,
    message: &str,
    error: &anyhow::Error,
) -> Result<()> {
    match output_format {
        OutputFormat::Json | OutputFormat::Jsonl | OutputFormat::Stream => {
            let error_json = serde_json::json!({
                "type": "error",
                "session_id": session_id,
                "error": error.to_string(),
                "user": message,
            });
            eprintln!("{}", serde_json::to_string(&error_json)?);
        }
        OutputFormat::Text => {
            eprintln!("Error: {error}");
        }
    }
    crate::shutdown::exit(1);
}

pub(super) async fn seed_session_memory(
    service: &mut AgentService,
    history: &[caliche::messages::Message],
) -> Result<()> {
    use caliche::agent::memory::MemoryStrategy as _;
    use std::sync::Arc;

    let token_counter: Arc<dyn caliche::agent::memory::TokenCounter> =
        Arc::new(caliche::agent::memory::token_counter::ApproximateTokenCounter::default());
    let memory_config =
        caliche::agent::memory::BufferMemoryConfig::default().with_max_tokens(128_000);
    let mut memory = caliche::agent::memory::BufferMemory::new(memory_config, token_counter);
    if !history.is_empty() {
        memory
            .add_messages(history.to_vec())
            .await
            .context("Failed to seed memory with conversation history")?;
    }
    service.set_memory_strategy(Some(Box::new(memory)));
    Ok(())
}

pub(super) fn resolve_or_create_session(
    existing_session: Option<crate::session::CliSession>,
) -> crate::session::CliSession {
    match existing_session {
        Some(session) => {
            tracing::info!(
                session_id = %session.id,
                history_len = session.conversation_history.len(),
                "Continuing existing session"
            );
            session
        }
        None => {
            let id = uuid::Uuid::now_v7().to_string();
            tracing::info!(session_id = %id, "Creating new one-shot session");
            crate::session::CliSession {
                id,
                name: None,
                working_directory: std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from(".")),
                conversation_history: Vec::new(),
                context_files: Vec::new(),
                permissions_granted: Vec::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                total_tokens: 0,
                provider: None,
                model: None,
                file_content_cache: std::collections::HashMap::new(),
                files_created: Vec::new(),
            }
        }
    }
}

pub(super) async fn persist_session_turn(
    session: &mut crate::session::CliSession,
    service: &AgentService,
    message: &str,
    response_text: &str,
    extra_tokens: Option<u32>,
    session_store: &crate::storage::SessionStore,
) {
    session
        .conversation_history
        .push(caliche::messages::Message::user(message));
    session
        .conversation_history
        .push(caliche::messages::Message::assistant(response_text));
    if let Some(total) = extra_tokens {
        session.total_tokens = session.total_tokens.saturating_add(u64::from(total));
    }
    session.provider = Some(service.provider().to_string());
    session.model = Some(service.model().to_string());
    session.updated_at = chrono::Utc::now();
    if let Err(error) = session_store.save(session).await {
        tracing::warn!("Failed to persist session {}: {}", session.id, error);
    }
}

pub(super) enum SessionChatOutcome {
    Cancelled,
    Completed(Result<caliche::messages::ChatResponse>),
}

pub(super) async fn chat_with_optional_cancel(
    service: &mut AgentService,
    message: &str,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> SessionChatOutcome {
    if let Some(token) = cancellation_token {
        tokio::select! {
            biased;
            _ = token.cancelled() => SessionChatOutcome::Cancelled,
            result = service.chat(message) => SessionChatOutcome::Completed(result),
        }
    } else {
        SessionChatOutcome::Completed(service.chat(message).await)
    }
}
