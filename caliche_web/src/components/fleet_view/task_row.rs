use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use super::helpers::{
    BTN_ACCENT, BTN_DANGER, BTN_MUTED, INPUT_STYLE, agent_status_color, truncate,
};
use crate::state::agents::SubAgent;

#[derive(Props, Clone, PartialEq)]
pub struct TaskRowProps {
    pub agent: SubAgent,
    pub task_expanded: Signal<Option<String>>,
    pub selected_ids: Signal<HashSet<String>>,
    pub task_inject: Signal<HashMap<String, String>>,
    pub show_output: Signal<HashSet<String>>,
    pub on_cancel: EventHandler<String>,
    pub on_open_output: EventHandler<String>,
    pub on_inject: EventHandler<String>,
}

#[component]
pub fn TaskRow(props: TaskRowProps) -> Element {
    let agent = props.agent.clone();
    let task_id = agent.id.clone();
    let mut task_expanded = props.task_expanded;
    let mut selected_ids = props.selected_ids;
    let mut task_inject = props.task_inject;
    let mut show_output = props.show_output;
    let on_cancel = props.on_cancel;
    let on_open_output = props.on_open_output;
    let on_inject = props.on_inject;

    let is_exp = task_expanded.read().as_deref() == Some(&task_id);
    let is_sel = selected_ids.read().contains(&task_id);
    let is_terminal = agent.status.is_terminal();
    let sc = agent_status_color(&agent.status);
    let sl = agent.status.label().to_string();
    let name = agent.name.clone();
    let task_short = truncate(&agent.task, 50);
    let tog_id = task_id.clone();
    let cancel_id = task_id.clone();
    let output_id = task_id.clone();
    let inject_id = task_id.clone();
    let inject_val = task_inject
        .read()
        .get(&task_id)
        .cloned()
        .unwrap_or_default();
    let inj_count = agent.injected_commands.len();

    rsx! {
        div {
            key: "{task_id}",
            style: if is_sel { "border-bottom: 1px solid var(--border-color); background: color-mix(in srgb, var(--accent) 6%, transparent);" } else { "border-bottom: 1px solid var(--border-color);" },

            div {
                style: "display: flex; align-items: center; padding: 0.4em 1ch; gap: 0.5ch; cursor: pointer;",
                onclick: move |_| {
                    let mut exp = task_expanded.write();
                    if exp.as_deref() == Some(&tog_id) {
                        *exp = None;
                    } else {
                        *exp = Some(tog_id.clone());
                    }
                },

                if !is_terminal {
                    {
                        let chk_id = task_id.clone();
                        rsx! {
                            input {
                                r#type: "checkbox",
                                style: "accent-color: var(--accent); flex-shrink: 0; cursor: pointer;",
                                checked: is_sel,
                                onclick: move |e| e.stop_propagation(),
                                onchange: move |e| {
                                    if e.checked() {
                                        selected_ids.write().insert(chk_id.clone());
                                    } else {
                                        selected_ids.write().remove(&chk_id);
                                    }
                                },
                            }
                        }
                    }
                } else {
                    span { style: "width: 1.25ch; flex-shrink: 0;" }
                }
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs); width: 0.9ch; flex-shrink: 0;",
                    if is_exp { "▼" } else { "▶" }
                }
                span { style: "font-size: var(--font-size-xs); color: {sc}; flex-shrink: 0;", "●" }
                span { style: "font-size: var(--font-size-sm); font-weight: 500; flex-shrink: 0;", "{name}" }
                span { style: "color: var(--muted-color); font-size: var(--font-size-xs); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                    "{task_short}" }
                span { style: "font-size: var(--font-size-xs); color: {sc}; flex-shrink: 0;", "{sl}" }
                span {
                    style: "color: var(--muted-color); font-size: var(--font-size-xs); flex-shrink: 0;",
                    "{agent.time_ago()}"
                }
                if agent.token_usage > 0 {
                    span {
                        style: "font-size: var(--font-size-xs); color: var(--muted-color); flex-shrink: 0;",
                        title: "tokens used",
                        "{agent.token_usage}t"
                    }
                }
                if inj_count > 0 {
                    span {
                        style: "font-size: var(--font-size-xs); color: var(--accent); flex-shrink: 0;",
                        title: "commands injected",
                        "+{inj_count}"
                    }
                }
            }

            if is_exp {
                div {
                    style: "padding: 0.4em 1ch 0.6em 2.5ch; border-top: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 0.4em;",

                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color);",
                        "task: " span { style: "color: var(--text-color);", "{agent.task}" }
                    }
                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color);",
                        "source: " span { style: "color: var(--text-color);", "{agent.source}" }
                    }
                    if !agent.injected_commands.is_empty() {
                        div {
                            style: "border: 1px solid var(--border-color); padding: 0.3em 0.5ch;",
                            div { style: "font-size: var(--font-size-xs); color: var(--accent); margin-bottom: 0.2em;", "injected commands" }
                            for cmd in agent.injected_commands.iter() {
                                {
                                    let cmd = cmd.clone();
                                    rsx! {
                                        div { style: "font-size: var(--font-size-xs); color: var(--muted-color); display: flex; gap: 0.5ch;",
                                            span { style: "flex-shrink: 0; color: var(--accent);", ">" }
                                            span { style: "color: var(--text-color);", "{cmd.text}" }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    div { style: "display: flex; gap: 0.75ch; flex-wrap: wrap;",
                        if !agent.output.is_empty() {
                            {
                                let tog_out_id = task_id.clone();
                                let is_out_shown = show_output.read().contains(&tog_out_id);
                                rsx! {
                                    button {
                                        style: if is_out_shown { "padding: 0.2em 1.25ch; font-size: var(--font-size-xs); background: var(--accent); border: 1px solid var(--accent); color: var(--background); cursor: pointer;" } else { "{BTN_MUTED}" },
                                        onclick: move |_| {
                                            let mut s = show_output.write();
                                            if s.contains(&tog_out_id) {
                                                s.remove(&tog_out_id);
                                            } else {
                                                s.insert(tog_out_id.clone());
                                            }
                                        },
                                        if is_out_shown { "hide output" } else { "show output ({agent.output.len()} chunks)" }
                                    }
                                }
                            }
                        }
                        if !agent.output.is_empty() {
                            button {
                                style: "{BTN_MUTED}",
                                onclick: move |_| on_open_output.call(output_id.clone()),
                                "open in chat"
                            }
                        }
                        if agent.status.is_active() {
                            button {
                                style: "{BTN_DANGER}",
                                onclick: move |_| on_cancel.call(cancel_id.clone()),
                                "cancel"
                            }
                        }
                    }

                    if show_output.read().contains(&task_id) && !agent.output.is_empty() {
                        div {
                            style: "border: 1px solid var(--border-color); background: var(--background); margin-top: 0.25em; max-height: 16em; overflow-y: auto; display: flex; flex-direction: column;",
                            div { style: "padding: 0.3em 0.75ch; font-size: var(--font-size-xs); color: var(--muted-color); border-bottom: 1px solid var(--border-color); background: var(--card-bg); display: flex; align-items: center; gap: 1ch; position: sticky; top: 0;",
                                span { "output" }
                                span { style: "color: var(--text-color);", "{agent.output.len()} chunks" }
                            }
                            div {
                                style: "padding: 0.4em 0.75ch; display: flex; flex-direction: column; gap: 0.15em; font-size: var(--font-size-xs); font-family: var(--font-mono); white-space: pre-wrap; word-break: break-word;",
                                for chunk in agent.output.iter() {
                                    {
                                        let chunk = chunk.clone();
                                        let text_color = if chunk.is_event {
                                            "var(--accent)"
                                        } else {
                                            "var(--text-color)"
                                        };
                                        let content = chunk.content.clone();
                                        rsx! {
                                            span { style: "color: {text_color};", "{content}" }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if !is_terminal {
                        div {
                            style: "margin-top: 0.25em; border: 1px solid var(--border-color);",
                            div { style: "padding: 0.25em 0.5ch; font-size: var(--font-size-xs); color: var(--muted-color); background: var(--card-bg); border-bottom: 1px solid var(--border-color);",
                                "inject command"
                            }
                            div { style: "display: flex; gap: 0.5ch; padding: 0.4em 0.5ch;",
                                input {
                                    r#type: "text",
                                    style: "flex: 1; {INPUT_STYLE} border-color: transparent;",
                                    placeholder: "inject a command or context update…",
                                    value: "{inject_val}",
                                    oninput: move |e| {
                                        task_inject.write().insert(inject_id.clone(), e.value());
                                    },
                                    onkeydown: {
                                        let iid = agent.id.clone();
                                        move |e: KeyboardEvent| {
                                            if e.key() == Key::Enter {
                                                on_inject.call(iid.clone());
                                            }
                                        }
                                    },
                                }
                                button {
                                    style: "{BTN_ACCENT}",
                                    disabled: inject_val.trim().is_empty(),
                                    onclick: move |_| on_inject.call(task_id.clone()),
                                    "inject"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
