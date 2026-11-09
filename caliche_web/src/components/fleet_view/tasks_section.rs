use std::collections::{HashMap, HashSet};

use dioxus::prelude::*;

use super::helpers::{BTN_ACCENT, BTN_MUTED, default_provider_and_model};
use super::spawn_form::SpawnTaskForm;
use super::task_row::TaskRow;
use super::types::SpawnRequest;
use crate::api::{ModelInfo, ProviderCatalogEntry};
use crate::components::chrome::{EMPTY_SENTENCE, EmptyHint, SectionHeader};
use crate::state::agents::SubAgent;

#[derive(Props, Clone, PartialEq)]
pub struct TasksSectionProps {
    pub agents: Signal<Vec<SubAgent>>,
    pub task_label: String,
    pub filtered_tasks: Vec<SubAgent>,
    pub task_list_empty: bool,
    pub completed_count: usize,
    pub provider_options: Signal<Vec<ProviderCatalogEntry>>,
    pub model_options: Signal<Vec<ModelInfo>>,
    pub current_provider_setting: Signal<Option<String>>,
    pub current_model_setting: Signal<Option<String>>,
    pub task_filter: Signal<String>,
    pub show_spawn: Signal<bool>,
    pub spawn_task_str: Signal<String>,
    pub spawn_provider: Signal<String>,
    pub spawn_model: Signal<String>,
    pub task_expanded: Signal<Option<String>>,
    pub selected_ids: Signal<HashSet<String>>,
    pub task_inject: Signal<HashMap<String, String>>,
    pub show_output: Signal<HashSet<String>>,
    pub on_spawn: EventHandler<SpawnRequest>,
    pub on_cancel: EventHandler<String>,
    pub on_open_output: EventHandler<String>,
    pub on_inject: EventHandler<String>,
}

#[component]
pub fn TasksSection(props: TasksSectionProps) -> Element {
    let mut agents = props.agents;
    let task_label = props.task_label;
    let filtered_tasks = props.filtered_tasks;
    let task_list_empty = props.task_list_empty;
    let completed_count = props.completed_count;
    let mut task_filter = props.task_filter;
    let mut show_spawn = props.show_spawn;
    let mut spawn_provider = props.spawn_provider;
    let mut spawn_model = props.spawn_model;
    let provider_options = props.provider_options;
    let model_options = props.model_options;
    let current_provider_setting = props.current_provider_setting;
    let current_model_setting = props.current_model_setting;

    rsx! {
        div { class: "workers-section grow",
            SectionHeader {
                title: "TASKS".to_string(),
                summary: task_label,
                input {
                    r#type: "text",
                    style: "flex: 1; min-width: 8ch; background: var(--background); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.15em 0.5ch; font-size: var(--font-size-xs); font-family: inherit;",
                    placeholder: "filter…",
                    value: "{task_filter}",
                    oninput: move |e| task_filter.set(e.value()),
                }
                if completed_count > 0 {
                    button {
                        style: "{BTN_MUTED}",
                        onclick: move |_| {
                            agents.write().retain(|a| !a.status.is_terminal());
                        },
                        "clear {completed_count} done"
                    }
                }
                button {
                    style: "{BTN_ACCENT}",
                    onclick: move |_| {
                        let next = !*show_spawn.read();
                        if next {
                            let providers = provider_options.read();
                            let models = model_options.read();
                            let current_provider = current_provider_setting.read();
                            let current_model = current_model_setting.read();
                            let (provider, model) = default_provider_and_model(
                                &providers,
                                &models,
                                current_provider.as_deref(),
                                current_model.as_deref(),
                            );
                            spawn_provider.set(provider);
                            spawn_model.set(model);
                        }
                        show_spawn.set(next);
                    },
                    if *show_spawn.read() { "✕ cancel" } else { "+ new task" }
                }
            }

            if *show_spawn.read() {
                SpawnTaskForm {
                    provider_options,
                    model_options,
                    spawn_task_str: props.spawn_task_str,
                    spawn_provider: props.spawn_provider,
                    spawn_model: props.spawn_model,
                    show_spawn,
                    on_spawn: props.on_spawn,
                }
            }

            div { class: "worker-list",
                if task_list_empty && !*show_spawn.read() {
                    EmptyHint {
                        body: EMPTY_SENTENCE.to_string(),
                        action: "add a task".to_string(),
                        on_action: move |_| show_spawn.set(true),
                    }
                }

                for agent in filtered_tasks.iter() {
                    TaskRow {
                        key: "{agent.id}",
                        agent: agent.clone(),
                        task_expanded: props.task_expanded,
                        selected_ids: props.selected_ids,
                        task_inject: props.task_inject,
                        show_output: props.show_output,
                        on_cancel: props.on_cancel,
                        on_open_output: props.on_open_output,
                        on_inject: props.on_inject,
                    }
                }
            }
        }
    }
}
