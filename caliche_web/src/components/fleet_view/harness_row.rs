use std::collections::HashMap;

use dioxus::prelude::*;

use super::helpers::{BTN_ACCENT, BTN_DANGER, BTN_MUTED, INPUT_STYLE};
use crate::api::HarnessInfo;
use crate::harness::preset_by_id;

#[derive(Props, Clone, PartialEq)]
pub struct HarnessRowProps {
    pub harness: HarnessInfo,
    pub selected_ids: Signal<std::collections::HashSet<String>>,
    pub dispatch_inp: Signal<HashMap<String, String>>,
    pub op_busy: Signal<Option<String>>,
    pub on_dispatch: EventHandler<String>,
    pub on_delete: EventHandler<String>,
}

#[component]
pub fn HarnessRow(props: HarnessRowProps) -> Element {
    let harness = props.harness;
    let mut selected_ids = props.selected_ids;
    let mut dispatch_inp = props.dispatch_inp;
    let op_busy = props.op_busy;
    let on_dispatch = props.on_dispatch;
    let on_delete = props.on_delete;
    let id = harness.id.clone();
    let selected = selected_ids.read().contains(&id);
    let busy = op_busy.read().as_deref() == Some(id.as_str());
    let preset_label = preset_by_id(&harness.preset_id).label;
    let argv = display_argv(&harness.command, &harness.args);
    let dispatch_value = dispatch_inp.read().get(&id).cloned().unwrap_or_default();

    rsx! {
        div { class: "worker-row",
            div { class: "worker-row-main",
                input {
                    r#type: "checkbox",
                    checked: selected,
                    onchange: {
                        let id = id.clone();
                        move |_| {
                            let mut sel = selected_ids.write();
                            if sel.contains(&id) {
                                sel.remove(&id);
                            } else {
                                sel.insert(id.clone());
                            }
                        }
                    },
                }
                span { class: "worker-name", "{harness.name}" }
                span { class: "worker-kind", "{preset_label}" }
                span { class: "worker-meta", "{argv}" }
                if let Some(cwd) = harness.cwd.as_ref() {
                    span { class: "worker-meta", "{cwd}" }
                }
                if let Some(err) = harness.last_error.as_ref() {
                    span { class: "worker-error", title: "{err}", "error" }
                } else {
                    span { class: "worker-ready", "ready" }
                }
                button {
                    style: "{BTN_DANGER}",
                    disabled: busy,
                    onclick: {
                        let id = id.clone();
                        move |_| on_delete.call(id.clone())
                    },
                    "delete"
                }
            }
            div { class: "worker-row-dispatch",
                input {
                    r#type: "text",
                    style: "{INPUT_STYLE}",
                    placeholder: "task for this harness",
                    value: "{dispatch_value}",
                    oninput: {
                        let id = id.clone();
                        move |e: FormEvent| {
                            dispatch_inp.write().insert(id.clone(), e.value());
                        }
                    },
                    onkeydown: {
                        let id = id.clone();
                        move |e: KeyboardEvent| {
                            if e.key() == Key::Enter {
                                on_dispatch.call(id.clone());
                            }
                        }
                    },
                }
                button {
                    style: "{BTN_ACCENT}",
                    disabled: busy || dispatch_value.trim().is_empty(),
                    onclick: {
                        let id = id.clone();
                        move |_| on_dispatch.call(id.clone())
                    },
                    if busy { "sending…" } else { "send" }
                }
                button {
                    style: "{BTN_MUTED}",
                    onclick: {
                        move |_| {
                            dispatch_inp.write().insert(id.clone(), String::new());
                        }
                    },
                    "clear"
                }
            }
            if let Some(output) = harness.last_output.as_ref().filter(|text| !text.is_empty()) {
                pre { class: "worker-output", "{output}" }
            }
        }
    }
}

fn display_argv(command: &str, args: &[String]) -> String {
    if args.is_empty() {
        command.to_string()
    } else {
        format!("{command} {}", args.join(" "))
    }
}
