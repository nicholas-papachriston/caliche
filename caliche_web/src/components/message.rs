//! Message component with markdown rendering - monospace terminal style.

use dioxus::prelude::*;
use pulldown_cmark::{Options, Parser, html};

use super::tool_call::ToolCallDisplay;
use crate::api::types::MessageRole;
use crate::state::Message;
use caliche_contracts::chat::strip_sources_footer;

/// Props for the Message component
#[derive(Props, Clone, PartialEq)]
pub struct MessageProps {
    /// The message to display
    pub message: Message,
    /// Whether source citations should be shown
    #[props(default = true)]
    pub show_sources: bool,
}

/// Message component that renders a single message with markdown support
#[component]
pub fn MessageComponent(props: MessageProps) -> Element {
    let message = &props.message;
    let is_user = message.role == MessageRole::User;
    let is_streaming = message.streaming;
    let is_error = message.error;

    // Render markdown to HTML
    let display_content = if props.show_sources && message.sources.is_empty() {
        message.content.clone()
    } else {
        strip_sources_footer(&message.content).to_string()
    };
    let html_content = render_markdown(&display_content);

    // Format timestamp
    let time_str = message.timestamp.format("%H:%M:%S").to_string();

    // Role prefix - terminal style
    let role_prefix = if is_user { ">" } else { "$" };
    let role_label = if is_user { "user" } else { "agent" };
    let role_color = if is_user {
        "var(--info)"
    } else {
        "var(--accent)"
    };

    // Border color for left border
    let border_color = if is_error {
        "var(--error)"
    } else if is_streaming {
        "var(--warning)"
    } else if is_user {
        "var(--info)"
    } else {
        "var(--accent)"
    };

    // Streaming cursor class
    let content_class = if is_streaming {
        "prose streaming-cursor"
    } else {
        "prose"
    };

    // CSS class for background tinting
    let msg_class = if is_user {
        "message user-msg"
    } else {
        "message assistant-msg"
    };

    rsx! {
        div { class: "{msg_class}",

            // Header: role and timestamp
            div { class: "message-header",
                style: "gap: 1ch; color: var(--muted-color); margin-bottom: 0.5em;",

                span { style: "color: {role_color};", "{role_prefix}" }
                span { style: "color: {role_color}; font-weight: bold;", "{role_label}" }
                span { style: "color: var(--muted-color);", "[{time_str}]" }

                div { class: "message-actions",
                    if is_streaming {
                        span { style: "color: var(--warning);", "[streaming...]" }
                    }
                    if is_error {
                        span { style: "color: var(--error);", "[error]" }
                    }
                }
            }

            // Message content with left border
            div { class: "message-content",
                style: "border-left: 2px solid {border_color}; padding-left: 1ch; margin-left: 1ch;",

                // Markdown content
                div { class: "{content_class}",
                    style: "line-height: var(--lh); overflow-wrap: break-word;",
                    dangerous_inner_html: "{html_content}",
                }

                // Inline media (images / file attachments sent with this message)
                if !message.media.is_empty() {
                    div { class: "msg-media-strip",
                        for item in &message.media {
                            {
                                let filename = item.filename.clone();
                                let media_type = item.media_type.clone();
                                let preview = item.preview_url.clone();
                                rsx! {
                                    if let Some(url) = preview {
                                        img {
                                            key: "{filename}",
                                            class: "msg-media-img",
                                            src: "{url}",
                                            alt: "{filename}",
                                            title: "{filename}",
                                        }
                                    } else {
                                        span {
                                            key: "{filename}",
                                            class: "msg-media-file",
                                            span {
                                                style: "color: var(--accent);",
                                                {
                                                    if media_type.starts_with("audio/") { "[aud]" }
                                                    else if media_type.starts_with("video/") { "[vid]" }
                                                    else { "[doc]" }
                                                }
                                            }
                                            span { "{filename}" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Tool calls
                if let Some(ref tool_calls) = message.tool_calls {
                    if !tool_calls.is_empty() {
                        ToolCallDisplay {
                            calls: tool_calls.clone(),
                        }
                    }
                }

                if props.show_sources && !message.sources.is_empty() {
                    div {
                        class: "message-sources",
                        style: "margin-top: 0.5em; border-top: 1px dashed var(--border-color); padding-top: 0.5em; color: var(--muted-color); font-size: var(--font-size-xs);",
                        div { style: "margin-bottom: 0.25em;", "sources" }
                        for (idx, source) in message.sources.iter().enumerate() {
                            {
                                let label = format!("[{}] {}", idx + 1, source.resource);
                                rsx! {
                                    div {
                                        key: "{idx}-{source.resource}",
                                        style: "padding-left: 1ch;",
                                        "{label}"
                                    }
                                }
                            }
                        }
                    }
                }

                // Token usage - with prompt/completion breakdown
                if let Some(ref usage) = message.token_usage {
                    div { class: "token-usage",
                        style: "margin-top: 0.5em; color: var(--muted-color); font-size: var(--font-size-xs);",
                        {
                            let total = usage.total_tokens.unwrap_or(0);
                            let prompt = usage.prompt_tokens;
                            let completion = usage.completion_tokens;
                            match (prompt, completion) {
                                (Some(p), Some(c)) => format!("↑{p} ↓{c} ={total} tokens"),
                                _ => format!("{total} tokens"),
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Render markdown content to HTML with monospace-friendly styling
fn render_markdown(content: &str) -> String {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(content, options);
    let mut html_output = String::new();
    html::push_html(&mut html_output, parser);

    // Remove wrapping <p> tags for simple single-paragraph content to avoid extra spacing
    let trimmed = html_output.trim();
    if trimmed.starts_with("<p>") && trimmed.ends_with("</p>") {
        // Check if there's only one paragraph (no other <p> tags inside)
        let inner = &trimmed[3..trimmed.len() - 4];
        if !inner.contains("<p>") {
            html_output = inner.to_string();
        }
    }

    // Compact styling - minimal margins
    html_output = html_output
        // Paragraphs: minimal bottom margin only
        .replace("<p>", "<p style=\"margin: 0; margin-bottom: 0.25em;\">")
        // Remove empty paragraphs
        .replace("<p style=\"margin: 0; margin-bottom: 0.25em;\"></p>", "")
        // Headers: compact with color
        .replace("<h1>", "<h1 style=\"margin: 0.5em 0 0.25em 0; color: var(--accent); font-size: 1.1em;\">")
        .replace("<h2>", "<h2 style=\"margin: 0.5em 0 0.25em 0; color: var(--accent); font-size: 1.05em;\">")
        .replace("<h3>", "<h3 style=\"margin: 0.5em 0 0.25em 0; color: var(--accent); font-size: 1em;\">")
        .replace("<h4>", "<h4 style=\"margin: 0.5em 0 0.25em 0; color: var(--accent); font-size: 1em;\">")
        // Lists: compact
        .replace("<ul>", "<ul style=\"margin: 0.25em 0; padding-left: 2ch;\">")
        .replace("<ol>", "<ol style=\"margin: 0.25em 0; padding-left: 2ch;\">")
        .replace("<li>", "<li style=\"margin: 0;\">")
        // Code
        .replace("<code>", "<code style=\"background: var(--secondary-bg); padding: 0 0.5ch;\">")
        .replace("<pre>", "<pre style=\"background: var(--secondary-bg); padding: 0.5ch 1ch; border: 1px solid var(--border-color); overflow-x: auto; margin: 0.25em 0;\">")
        // Links
        .replace("<a ", "<a style=\"color: var(--accent); text-decoration: underline;\" ")
        // Blockquotes
        .replace("<blockquote>", "<blockquote style=\"border-left: 2px solid var(--border-color); padding-left: 1ch; margin: 0.25em 0; color: var(--muted-color);\">")
        // Tables
        .replace("<table>", "<table style=\"border-collapse: collapse; margin: 0.25em 0; display: block; overflow-x: auto;\">")
        .replace("<th>", "<th style=\"border: 1px solid var(--border-color); padding: 0.25em 1ch; text-align: left; background: var(--secondary-bg);\">")
        .replace("<td>", "<td style=\"border: 1px solid var(--border-color); padding: 0.25em 1ch;\">")
        // Strong/bold
        .replace("<strong>", "<strong style=\"color: var(--foreground);\">")
        // Horizontal rules
        .replace("<hr>", "<hr style=\"border: none; border-top: 1px dashed var(--border-color); margin: 0.5em 0;\">")
        .replace("<hr />", "<hr style=\"border: none; border-top: 1px dashed var(--border-color); margin: 0.5em 0;\">");

    html_output
}
