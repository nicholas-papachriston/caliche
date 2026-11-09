//! Message list component - monospace terminal style with auto-scroll.

use dioxus::prelude::*;

use super::message::MessageComponent;
use crate::api::types::InteractionMode;
use crate::state::Message;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScrollSnapshot {
    message_count: usize,
    streaming_content_len: usize,
    trigger: u64,
}

fn next_scroll_snapshot(previous: ScrollSnapshot, messages: &[Message]) -> ScrollSnapshot {
    let message_count = messages.len();
    let streaming_content_len = messages
        .iter()
        .filter(|message| message.streaming)
        .map(|message| message.content.len())
        .sum();

    if message_count == previous.message_count
        && streaming_content_len == previous.streaming_content_len
    {
        previous
    } else {
        ScrollSnapshot {
            message_count,
            streaming_content_len,
            trigger: previous.trigger.saturating_add(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Message;

    fn streaming_message(content: &str) -> Message {
        let mut message = Message::streaming_assistant();
        message.content = content.to_string();
        message
    }

    #[test]
    fn scroll_snapshot_is_stable_for_identical_inputs() {
        let messages = vec![streaming_message("hello")];
        let first = next_scroll_snapshot(
            ScrollSnapshot {
                message_count: 0,
                streaming_content_len: 0,
                trigger: 0,
            },
            &messages,
        );
        let second = next_scroll_snapshot(first, &messages);

        assert_eq!(first.trigger, 1);
        assert_eq!(second, first);
    }

    #[test]
    fn scroll_snapshot_advances_for_new_messages_and_streaming_content_only() {
        let mut messages = vec![streaming_message("hello")];
        let mut snapshot = next_scroll_snapshot(
            ScrollSnapshot {
                message_count: 0,
                streaming_content_len: 0,
                trigger: 0,
            },
            &messages,
        );

        messages[0].content.push_str(" world");
        snapshot = next_scroll_snapshot(snapshot, &messages);
        assert_eq!(snapshot.trigger, 2);

        messages.push(Message::user("question"));
        snapshot = next_scroll_snapshot(snapshot, &messages);
        assert_eq!(snapshot.trigger, 3);

        let unchanged = next_scroll_snapshot(snapshot, &messages);
        assert_eq!(unchanged, snapshot);
    }

    #[test]
    fn finalized_content_change_does_not_trigger_stream_scroll_loop() {
        let mut message = streaming_message("hello");
        let mut snapshot = next_scroll_snapshot(
            ScrollSnapshot {
                message_count: 0,
                streaming_content_len: 0,
                trigger: 0,
            },
            &[message.clone()],
        );

        message.streaming = false;
        message.content.push_str(" finalized");
        snapshot = next_scroll_snapshot(snapshot, &[message.clone()]);
        assert_eq!(snapshot.trigger, 2);

        let unchanged = next_scroll_snapshot(snapshot, &[message]);
        assert_eq!(unchanged, snapshot);
    }
}

/// Props for the MessageList component
#[derive(Props, Clone, PartialEq)]
pub struct MessageListProps {
    /// Messages to display
    pub messages: Vec<Message>,
    /// Current interaction mode (used for empty-state hints)
    #[props(default = InteractionMode::Agent)]
    pub mode: InteractionMode,
    /// Whether source citations should be shown
    #[props(default = true)]
    pub show_sources: bool,
}

/// Message list component that displays all messages with auto-scroll
#[component]
pub fn MessageList(props: MessageListProps) -> Element {
    let mut sorted_messages = props.messages.clone();
    sorted_messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    let is_empty = sorted_messages.is_empty();

    // Simple auto-scroll: always scroll to bottom when messages change
    // Use a signal to track the "scroll version" that changes on each update
    let mut scroll_trigger = use_signal(|| 0u64);
    let mut scroll_snapshot = use_signal(|| ScrollSnapshot {
        message_count: 0,
        streaming_content_len: 0,
        trigger: 0,
    });

    let previous = *scroll_snapshot.read();
    let next = next_scroll_snapshot(previous, &sorted_messages);
    if next != previous {
        scroll_snapshot.set(next);
        scroll_trigger.set(next.trigger);
    }

    // Scroll to bottom whenever the trigger changes.
    // The Signal read must be inside the closure so Dioxus tracks it as a dependency.
    use_effect(move || {
        let trigger = *scroll_trigger.read();
        if trigger > 0 {
            spawn(async move {
                let js = r#"
                    var el = document.getElementById('message-list-container');
                    if (el) { el.scrollTop = el.scrollHeight; }
                "#;
                let _ = dioxus::document::eval(js);
            });
        }
    });

    rsx! {
        div {
            id: "message-list-container",
            class: "message-list",
            style: "flex: 1; min-height: 0; overflow-y: auto; padding: 1ch; border: 1px solid var(--border-color); -webkit-overflow-scrolling: touch;",

            if is_empty {
                {
                    let (mode_label, mode_desc, hints) = match props.mode {
                        InteractionMode::Agent => (
                            "agent mode",
                            "full capabilities — read, write, execute",
                            vec![
                                "\"read my config and suggest improvements\"",
                                "@src/ analyze the code structure",
                                "@diff what changed since last commit?",
                                "/checkpoint before-refactor",
                            ],
                        ),
                        InteractionMode::Ask => (
                            "ask mode",
                            "read-only — questions and explanations only",
                            vec![
                                "\"explain how the auth middleware works\"",
                                "@status what files are staged?",
                                "@diff summarize these changes",
                                "/agent when ready to make changes",
                            ],
                        ),
                        InteractionMode::Plan => (
                            "plan mode",
                            "planning — requirements and architecture",
                            vec![
                                "\"I need to build a REST API with auth\"",
                                "\"help me design the database schema\"",
                                "\"what are the tradeoffs between X and Y?\"",
                                "/agent when ready to implement",
                            ],
                        ),
                    };
                    rsx! {
                        div { class: "empty-state",
                            style: "display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 10em; color: var(--muted-color); padding: 2ch; gap: 1.5em;",

                            div { style: "border: 1px solid var(--accent); color: var(--accent); padding: 0.75em 2em; text-align: center; line-height: var(--lh);",
                                div { style: "font-weight: bold;", "{mode_label}" }
                                div { style: "font-size: var(--font-size-sm);", "{mode_desc}" }
                            }

                            div { style: "font-size: var(--font-size-sm); text-align: left; max-width: 50ch;",
                                div { style: "color: var(--muted-color); margin-bottom: 0.5em;", "try:" }
                                for hint in &hints {
                                    div { class: "empty-state-hint",
                                        style: "color: var(--muted-color); padding-left: 1ch; margin-bottom: 0.25em; font-size: var(--font-size-sm);",
                                        span { style: "color: var(--accent); margin-right: 0.5ch;", "›" }
                                        "{hint}"
                                    }
                                }
                            }

                            div { style: "font-size: var(--font-size-xs); color: var(--border-color); text-align: center; letter-spacing: 0.05em;",
                                "@file  ·  @diff  ·  @status  ·  @clipboard  ·  /cmd"
                            }
                        }
                    }
                }
            } else {
                div { class: "messages",
                    for message in sorted_messages {
                        MessageComponent {
                            key: "{message.message_id}",
                            message: message.clone(),
                            show_sources: props.show_sources,
                        }
                    }
                }
            }
        }
    }
}
