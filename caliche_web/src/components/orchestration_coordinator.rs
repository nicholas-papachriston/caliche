//! Coordinator pane: route a task by capability, using registered CLI harness tags.

use crate::api::{ApiClient, CoordinatorRequest, CoordinatorResponse, HarnessInfo};
use crate::components::chrome::{ChipButton, FieldRow};
use crate::harness::{preset_by_id, toggle_csv_tag, unique_capabilities};
use dioxus::prelude::*;

#[component]
pub fn OrchestrationCoordinator(api_client: ApiClient) -> Element {
    let mut coord_task = use_signal(String::new);
    let mut coord_capabilities = use_signal(String::new);
    let mut coord_result: Signal<Option<CoordinatorResponse>> = use_signal(|| None);
    let mut coord_error: Signal<Option<String>> = use_signal(|| None);
    let mut coord_running = use_signal(|| false);
    let mut harnesses = use_signal(Vec::<HarnessInfo>::new);
    let mut poll_started = use_signal(|| false);

    {
        let client = api_client.clone();
        use_effect(move || {
            if *poll_started.read() {
                return;
            }
            poll_started.set(true);
            let client = client.clone();
            spawn(async move {
                if let Ok(list) = client.list_harnesses().await {
                    harnesses.set(list);
                }
            });
        });
    }

    let run_coordinator = {
        let api = api_client.clone();
        move |_| {
            if *coord_running.read() {
                return;
            }
            let task_val = coord_task.read().trim().to_string();
            if task_val.is_empty() {
                return;
            }
            let caps_str = coord_capabilities.read().clone();
            let required_capabilities: Vec<String> = caps_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let req = CoordinatorRequest {
                task: task_val,
                required_capabilities,
            };
            coord_running.set(true);
            coord_result.set(None);
            coord_error.set(None);
            let api = api.clone();
            spawn(async move {
                match api.coordinator_route(&req).await {
                    Ok(r) => coord_result.set(Some(r)),
                    Err(e) => coord_error.set(Some(e.to_string())),
                }
                coord_running.set(false);
            });
        }
    };

    let list = harnesses.read().clone();
    let tags = unique_capabilities(list.iter().map(|item| item.capabilities.as_slice()));
    let selected_caps = coord_capabilities.read().clone();

    rsx! {
        div { class: "orch-scroll",
            p { class: "hint-banner",
                "Pick capability chips from registered harnesses. The coordinator routes the task."
            }

            if list.is_empty() {
                p { class: "hint-banner",
                    "Add a CLI harness on the fleet pane. Then route by capability."
                }
            } else {
                div { class: "chip-row",
                    for harness in list.iter() {
                        span { class: "worker-kind", "{harness.name}" }
                        span { class: "worker-meta", "{preset_by_id(&harness.preset_id).label}" }
                    }
                }
                if !tags.is_empty() {
                    FieldRow {
                        label: "capabilities".to_string(),
                        help: "Tap a tag to add or remove it.".to_string(),
                        div { class: "chip-row",
                            for tag in tags.iter() {
                                {
                                    let tag = tag.clone();
                                    let active = selected_caps
                                        .split(',')
                                        .map(str::trim)
                                        .any(|item| item == tag);
                                    rsx! {
                                        ChipButton {
                                            label: tag.clone(),
                                            active,
                                            on_click: move |_| {
                                                let next = toggle_csv_tag(
                                                    coord_capabilities.read().as_str(),
                                                    &tag,
                                                );
                                                coord_capabilities.set(next);
                                            },
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            FieldRow {
                label: "task".to_string(),
                help: "Sent to the selected agent.".to_string(),
                textarea {
                    style: "height: 6ch; resize: vertical; width: 100%; box-sizing: border-box;",
                    placeholder: "Describe the task to route",
                    value: "{coord_task}",
                    oninput: move |e| coord_task.set(e.value()),
                }
            }

            button {
                class: "primary",
                style: "padding: 0.4ch 2ch; align-self: flex-start;",
                disabled: *coord_running.read() || coord_task.read().trim().is_empty(),
                onclick: run_coordinator,
                if *coord_running.read() { "routing…" } else { "route task" }
            }

            if let Some(ref err) = *coord_error.read() {
                div { class: "form-error", "error: {err}" }
            }

            if let Some(ref result) = *coord_result.read() {
                div { class: "result-block final",
                    div { class: "worker-row-main",
                        span { class: "worker-kind", "selected: {result.selected_agent}" }
                        if !result.capabilities_matched.is_empty() {
                            span { class: "worker-meta", "matched: {result.capabilities_matched.join(\", \")}" }
                        }
                    }
                    pre { class: "result-pre", "{result.result}" }
                }
            }
        }
    }
}
