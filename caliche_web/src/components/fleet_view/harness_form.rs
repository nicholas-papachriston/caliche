use dioxus::prelude::*;

use super::helpers::INPUT_STYLE;
use crate::components::chrome::FieldRow;
use crate::harness::{HARNESS_PRESETS, HarnessDraft, PromptPlacementKind, preset_by_id};

#[derive(Props, Clone, PartialEq)]
pub struct HarnessFormProps {
    pub draft: Signal<HarnessDraft>,
    pub error: Signal<Option<String>>,
    pub busy: Signal<bool>,
    pub on_submit: EventHandler<()>,
    pub on_cancel: EventHandler<()>,
}

#[component]
pub fn HarnessForm(props: HarnessFormProps) -> Element {
    let mut draft = props.draft;
    let error = props.error;
    let busy = props.busy;
    let on_submit = props.on_submit;
    let on_cancel = props.on_cancel;
    let selected = draft.read().preset_id.clone();

    rsx! {
        div { class: "harness-form",
            div { class: "preset-grid", role: "listbox", aria_label: "Harness presets",
                for preset in HARNESS_PRESETS {
                    button {
                        class: if selected == preset.id { "preset-chip active" } else { "preset-chip" },
                        role: "option",
                        aria_selected: if selected == preset.id { "true" } else { "false" },
                        title: "{preset.summary}",
                        onclick: move |_| {
                            let mut next = draft.read().clone();
                            next.apply_preset(preset_by_id(preset.id));
                            draft.set(next);
                        },
                        span { class: "preset-chip-label", "{preset.label}" }
                        span { class: "preset-chip-help", "{preset.summary}" }
                    }
                }
            }

            FieldRow {
                label: "name".to_string(),
                help: "Shown in the fleet list.".to_string(),
                input {
                    r#type: "text",
                    style: "{INPUT_STYLE}",
                    placeholder: "reviewer",
                    value: "{draft.read().name}",
                    oninput: move |e| {
                        let mut next = draft.read().clone();
                        next.name = e.value();
                        draft.set(next);
                    },
                }
            }

            FieldRow {
                label: "command".to_string(),
                help: "Executable name or path. Not a shell.".to_string(),
                input {
                    r#type: "text",
                    style: "{INPUT_STYLE}",
                    placeholder: "claude",
                    value: "{draft.read().command}",
                    oninput: move |e| {
                        let mut next = draft.read().clone();
                        next.command = e.value();
                        draft.set(next);
                    },
                }
            }

            FieldRow {
                label: "args".to_string(),
                help: "One argument per line, before the task text.".to_string(),
                textarea {
                    style: "{INPUT_STYLE} min-height: 3.5em; resize: vertical;",
                    placeholder: "-p",
                    value: "{draft.read().args_text}",
                    oninput: move |e| {
                        let mut next = draft.read().clone();
                        next.args_text = e.value();
                        draft.set(next);
                    },
                }
            }

            FieldRow {
                label: "working directory".to_string(),
                help: "Optional. Inherited from the server when empty.".to_string(),
                input {
                    r#type: "text",
                    style: "{INPUT_STYLE}",
                    placeholder: "~/src/project",
                    value: "{draft.read().cwd}",
                    oninput: move |e| {
                        let mut next = draft.read().clone();
                        next.cwd = e.value();
                        draft.set(next);
                    },
                }
            }

            FieldRow {
                label: "task text".to_string(),
                help: draft.read().prompt_placement.help().to_string(),
                select {
                    style: "{INPUT_STYLE}",
                    value: "{placement_value(draft.read().prompt_placement)}",
                    onchange: move |e| {
                        let mut next = draft.read().clone();
                        next.prompt_placement = placement_from_value(&e.value());
                        draft.set(next);
                    },
                    for kind in PromptPlacementKind::all() {
                        option {
                            value: "{placement_value(kind)}",
                            "{kind.label()}"
                        }
                    }
                }
            }

            FieldRow {
                label: "capabilities".to_string(),
                help: "Comma-separated tags for coordinator routing.".to_string(),
                input {
                    r#type: "text",
                    style: "{INPUT_STYLE}",
                    placeholder: "coding, review",
                    value: "{draft.read().capabilities_text}",
                    oninput: move |e| {
                        let mut next = draft.read().clone();
                        next.capabilities_text = e.value();
                        draft.set(next);
                    },
                }
            }

            if let Some(err) = error.read().as_ref() {
                div { class: "form-error", "{err}" }
            }

            div { class: "form-actions",
                button {
                    class: "primary",
                    style: "padding: 0.3em 1.5ch;",
                    disabled: *busy.read(),
                    onclick: move |_| on_submit.call(()),
                    if *busy.read() { "adding…" } else { "add harness" }
                }
                button {
                    style: "padding: 0.3em 1.5ch;",
                    onclick: move |_| on_cancel.call(()),
                    "cancel"
                }
            }
        }
    }
}

fn placement_value(kind: PromptPlacementKind) -> &'static str {
    match kind {
        PromptPlacementKind::LastArg => "last_arg",
        PromptPlacementKind::LastArgEoo => "last_arg_eoo",
        PromptPlacementKind::Stdin => "stdin",
    }
}

fn placement_from_value(value: &str) -> PromptPlacementKind {
    match value {
        "last_arg_eoo" => PromptPlacementKind::LastArgEoo,
        "stdin" => PromptPlacementKind::Stdin,
        _ => PromptPlacementKind::LastArg,
    }
}
