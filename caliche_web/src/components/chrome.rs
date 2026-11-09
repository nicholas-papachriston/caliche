//! Shared operator chrome: section headers, empty hints, field rows, preset chips.

use dioxus::prelude::*;

/// Empty list or pane. Nothing was requested that failed.
pub const EMPTY_SENTENCE: &str = "nothing here yet";

/// The request reached the server and was not a rejected token.
pub const REQUEST_FAILED_SENTENCE: &str = "request failed";

/// HTTP 401. The server rejected the bearer token.
pub const TOKEN_REJECTED_SENTENCE: &str = "token rejected";

/// Next step when a pane has no rows.
pub const RETRY_ACTION: &str = "retry";

/// Next step when inspect or history has no session to open.
pub const START_CHAT_ACTION: &str = "start a chat";

/// Fleet is every worker in this process, not a remote cluster.
pub const FLEET_SENTENCE: &str = "every worker in this process";

/// Disconnected screen. The settings tab stays on the bar so this sentence is true.
pub const DISCONNECTED_DETAIL: &str = "Start the server from your project (for example caliche --server), or open the settings tab and set the API base URL.";

/// Inspect pane whose RPC is JSON-RPC method-not-found (-32601).
pub const PANE_UNAVAILABLE_SENTENCE: &str = "this pane is not on this server";

/// How a pane load should be described.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadKind {
    /// The request succeeded and returned no rows.
    Empty,
    /// The request failed for a reason other than a rejected token.
    RequestFailed,
    /// The server returned HTTP 401.
    TokenRejected,
}

/// Classify a client error string. Only an `HTTP 401` status is a rejected token.
#[must_use]
pub fn load_kind_from_error(error: &str) -> LoadKind {
    if http_status(error) == Some(401) {
        LoadKind::TokenRejected
    } else {
        LoadKind::RequestFailed
    }
}

/// Sentence for `kind`, including the server address and any server message.
#[must_use]
pub fn load_hint_body(kind: LoadKind, address: &str, server_message: Option<&str>) -> String {
    let sentence = match kind {
        LoadKind::Empty => return EMPTY_SENTENCE.to_string(),
        LoadKind::RequestFailed => REQUEST_FAILED_SENTENCE,
        LoadKind::TokenRejected => TOKEN_REJECTED_SENTENCE,
    };
    match clipped_message(server_message) {
        Some(message) => format!("{sentence}. {address}. {message}"),
        None => format!("{sentence}. {address}"),
    }
}

/// Server-supplied text from an error that includes an `HTTP <status>` body.
#[must_use]
pub fn server_message_from_error(error: &str) -> Option<String> {
    let body = http_body_tail(error)?;
    message_from_body(body)
}

/// Title stored on a live session, or `untitled` when create omitted one.
#[must_use]
pub fn session_display_title(title: Option<&str>) -> String {
    match title.map(str::trim).filter(|title| !title.is_empty()) {
        Some(title) => title.to_string(),
        None => "untitled".to_string(),
    }
}

fn clipped_message(server_message: Option<&str>) -> Option<String> {
    let message = server_message
        .map(str::trim)
        .filter(|message| !message.is_empty())?;
    const MAX_CHARS: usize = 240;
    if message.chars().count() <= MAX_CHARS {
        return Some(message.to_string());
    }
    let clipped: String = message.chars().take(MAX_CHARS).collect();
    Some(format!("{clipped}…"))
}

fn http_status(error: &str) -> Option<u16> {
    let (status, _) = split_http_body(error)?;
    status.parse().ok()
}

fn http_body_tail(error: &str) -> Option<&str> {
    let (_, body) = split_http_body(error)?;
    let body = body.trim();
    if body.is_empty() { None } else { Some(body) }
}

fn split_http_body(error: &str) -> Option<(&str, &str)> {
    let lower = error.to_ascii_lowercase();
    let index = lower.find("http ")?;
    let after = error.get(index + 5..)?.trim_start();
    let (status, rest) = after.split_once(char::is_whitespace).unwrap_or((after, ""));
    if status.is_empty() || !status.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    Some((status, rest))
}

fn message_from_body(body: &str) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(message) = json_message(trimmed) {
        return clipped_message(Some(&message));
    }
    clipped_message(Some(trimmed))
}

fn json_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if let Some(message) = value.get("message").and_then(serde_json::Value::as_str) {
        let message = message.trim();
        if !message.is_empty() {
            return Some(message.to_string());
        }
    }
    let error = value.get("error")?;
    if let Some(message) = error.as_str() {
        let message = message.trim();
        if !message.is_empty() {
            return Some(message.to_string());
        }
    }
    let message = error.get("message")?.as_str()?.trim();
    if message.is_empty() {
        None
    } else {
        Some(message.to_string())
    }
}

/// Uppercase section title row with a muted summary and trailing actions.
#[component]
pub fn SectionHeader(title: String, summary: String, children: Element) -> Element {
    rsx! {
        div { class: "section-header",
            span { class: "section-title", "{title}" }
            span { class: "section-summary", "{summary}" }
            div { class: "section-actions",
                {children}
            }
        }
    }
}

/// Empty state that names the next action.
#[component]
pub fn EmptyHint(body: String, action: String, on_action: EventHandler<()>) -> Element {
    rsx! {
        div { class: "empty-state",
            p { class: "empty-state-body", "{body}" }
            button {
                class: "empty-state-hint",
                onclick: move |_| on_action.call(()),
                "{action}"
            }
        }
    }
}

/// Label plus one-line help above a control.
#[component]
pub fn FieldRow(label: String, help: String, children: Element) -> Element {
    rsx! {
        div { class: "field-row",
            div { class: "field-row-head",
                span { class: "field-row-label", "{label}" }
                if !help.is_empty() {
                    span { class: "field-row-help", "{help}" }
                }
            }
            {children}
        }
    }
}

/// Compact toggle used for presets, modes, patterns, and capability tags.
#[component]
pub fn ChipButton(label: String, active: bool, on_click: EventHandler<()>) -> Element {
    rsx! {
        button {
            class: if active { "chip active" } else { "chip" },
            r#type: "button",
            aria_pressed: if active { "true" } else { "false" },
            onclick: move |_| on_click.call(()),
            "{label}"
        }
    }
}
