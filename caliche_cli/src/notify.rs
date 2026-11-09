//! Post-turn notification hooks.
//!
//! Fires an optional user-configured shell command after each agent turn completes,
//! passing turn metadata as environment variables so users can send desktop
//! notifications, update status bars, or trigger any other side effect.
//!
//! ## Environment variables injected
//!
//! | Variable | Value |
//! |---|---|
//! | `CALICHE_TURN_STATUS` | `"success"`, `"error"`, or `"interrupted"` |
//! | `CALICHE_TURN_TOKENS` | Total tokens used (empty string if unknown) |
//! | `CALICHE_SESSION_ID` | Session identifier |
//! | `CALICHE_PROMPT_PREVIEW` | First 100 chars of the user message |
//!
//! ## Examples
//!
//! The command is executed by `sh -c`; treat injected values as data and quote
//! environment variable expansions in custom commands.
//!
//! ```toml
//! # caliche.toml — macOS desktop notification
//! notify_command = "terminal-notifier -title 'Caliche' -message \"$CALICHE_TURN_STATUS\""
//!
//! # Linux desktop notification
//! notify_command = "notify-send 'Caliche' \"Turn $CALICHE_TURN_STATUS\""
//!
//! # Conditional — only notify on success
//! notify_command = "[ \"$CALICHE_TURN_STATUS\" = success ] && terminal-notifier -message 'Done'"
//! ```

use caliche::harness::sanitize_notify_preview;
use caliche::messages::TokenUsage;

/// Fire the optional notify command after a turn completes.
///
/// This is fire-and-forget: the child process is spawned but not awaited,
/// so the agent never blocks on the notification. Errors are silently ignored
/// to ensure a broken notify command can never crash or stall the agent.
pub fn fire_notify_command(
    command: &str,
    status: &str,
    prompt_preview: &str,
    session_id: &str,
    usage: Option<&TokenUsage>,
) {
    let total_tokens = usage
        .and_then(|u| u.total_tokens)
        .map(|t| t.to_string())
        .unwrap_or_default();
    let prompt_preview = sanitize_notify_preview(prompt_preview);

    let _ = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .env("CALICHE_TURN_STATUS", status)
        .env("CALICHE_TURN_TOKENS", &total_tokens)
        .env("CALICHE_SESSION_ID", session_id)
        .env("CALICHE_PROMPT_PREVIEW", prompt_preview)
        .spawn();
}
