//! Tool call display component - monospace terminal style.

use caliche_contracts::text::truncate_utf8;
use dioxus::prelude::*;

use crate::state::ToolCall;

/// Props for the ToolCallDisplay component
#[derive(Props, Clone, PartialEq)]
pub struct ToolCallDisplayProps {
    /// Tool calls to display
    pub calls: Vec<ToolCall>,
}

/// Display a list of tool calls - collapsible section
#[component]
pub fn ToolCallDisplay(props: ToolCallDisplayProps) -> Element {
    if props.calls.is_empty() {
        return rsx! {};
    }

    let call_count = props.calls.len();
    let completed_count = props.calls.iter().filter(|c| !c.is_running()).count();
    let running_count = call_count - completed_count;

    // Auto-expand when tools are actively running so the user can see activity
    let mut expanded = use_signal(|| running_count > 0);
    let is_expanded = *expanded.read();

    let toggle_icon = if is_expanded { "[-]" } else { "[+]" };

    // Status indicator
    let status_text = if running_count > 0 {
        format!("{} running", running_count)
    } else {
        format!("{} done", completed_count)
    };

    let status_color = if running_count > 0 {
        "var(--warning)"
    } else {
        "var(--accent)"
    };

    rsx! {
        div { class: "tool-calls",
            style: "margin-top: 0.5em; border-top: 1px dashed var(--border-color); padding-top: 0.5em;",

            // Collapsible header
            button {
                style: "width: 100%; display: flex; gap: 1ch; align-items: center; background: transparent; border: none; color: var(--muted-color); text-align: left; cursor: pointer; padding: 0;",
                onclick: move |_| {
                    let current = *expanded.read();
                    expanded.set(!current);
                },

                span { "{toggle_icon}" }
                span { "tools({call_count})" }
                if running_count > 0 {
                    span { class: "tool-running-indicator", style: "color: {status_color};" }
                } else {
                    span { style: "color: {status_color};", "[{status_text}]" }
                }
            }

            // Expandable content
            if is_expanded {
                div { style: "margin-top: 0.5em; margin-left: 2ch;",
                    for call in &props.calls {
                        ToolCallItem {
                            key: "{call.id}",
                            call: call.clone(),
                        }
                    }
                }
            }
        }
    }
}

/// Props for a single tool call item
#[derive(Props, Clone, PartialEq)]
struct ToolCallItemProps {
    call: ToolCall,
}

/// Display a single tool call - terminal style
#[component]
fn ToolCallItem(props: ToolCallItemProps) -> Element {
    let call = &props.call;
    let is_running = call.is_running();
    let is_success = call.success.unwrap_or(true);
    let mut expanded = use_signal(|| false);

    let status_icon = if is_running {
        "..."
    } else if is_success {
        "ok"
    } else {
        "err"
    };

    let status_color = if is_running {
        "var(--warning)"
    } else if is_success {
        "var(--accent)"
    } else {
        "var(--error)"
    };

    let has_details = call.arguments.is_some() || call.result.is_some();
    let is_expanded = *expanded.read();
    let toggle_icon = if is_expanded { "[-]" } else { "[+]" };

    rsx! {
        div { class: "tool-call-item",
            style: "border-left: 2px solid {status_color}; padding-left: 1ch; margin: 0.25em 0;",

            // Header line
            button {
                style: "width: 100%; display: flex; gap: 1ch; align-items: center; background: transparent; border: none; color: inherit; text-align: left; cursor: pointer; padding: 0;",
                onclick: move |_| {
                    if has_details {
                        let current = *expanded.read();
                        expanded.set(!current);
                    }
                },

                if has_details {
                    span { style: "color: var(--muted-color);", "{toggle_icon}" }
                }
                span { style: "color: var(--text-color);", "{call.name}" }
                span { style: "color: var(--muted-color);", "{call.id}" }
                span { style: "color: {status_color};", "[{status_icon}]" }
            }

            // Expandable details
            if is_expanded && has_details {
                div { style: "margin-left: 2ch; margin-top: 0.25em;",
                    // Arguments
                    if let Some(ref args) = call.arguments {
                        div { style: "margin-bottom: 0.5em;",
                            div { style: "color: var(--muted-color);", "args:" }
                            pre { style: "background: var(--secondary-bg); padding: 0.5em 1ch; border: 1px solid var(--border-color); overflow-x: auto; max-height: 8em; overflow-y: auto; margin: 0;",
                                {format_json(args)}
                            }
                        }
                    }

                    // Result
                    if let Some(ref result) = call.result {
                        div {
                            div { style: "color: var(--muted-color);", "result:" }
                            pre { style: "background: var(--secondary-bg); padding: 0.5em 1ch; border: 1px solid var(--border-color); overflow-x: auto; max-height: 10em; overflow-y: auto; margin: 0; color: {status_color};",
                                {truncate_result(result, 500)}
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Format JSON for display
fn format_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

/// Truncate result for display
fn truncate_result(result: &str, max_len: usize) -> String {
    if result.len() <= max_len {
        result.to_string()
    } else {
        format!("{}...\n[truncated]", truncate_utf8(result, max_len))
    }
}
