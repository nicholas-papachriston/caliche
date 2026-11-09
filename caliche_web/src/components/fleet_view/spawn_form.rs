use dioxus::prelude::*;

use super::helpers::{INPUT_STYLE, models_for_provider};
use super::types::SpawnRequest;
use crate::api::{ModelInfo, ProviderCatalogEntry};
use crate::state::agents::AgentSource;

#[derive(Props, Clone, PartialEq)]
pub struct SpawnTaskFormProps {
    pub provider_options: Signal<Vec<ProviderCatalogEntry>>,
    pub model_options: Signal<Vec<ModelInfo>>,
    pub spawn_task_str: Signal<String>,
    pub spawn_provider: Signal<String>,
    pub spawn_model: Signal<String>,
    pub show_spawn: Signal<bool>,
    pub on_spawn: EventHandler<SpawnRequest>,
}

#[component]
pub fn SpawnTaskForm(props: SpawnTaskFormProps) -> Element {
    let provider_options = props.provider_options;
    let model_options = props.model_options;
    let mut spawn_task_str = props.spawn_task_str;
    let mut spawn_provider = props.spawn_provider;
    let mut spawn_model = props.spawn_model;
    let mut show_spawn = props.show_spawn;
    let on_spawn = props.on_spawn;

    rsx! {
        div {
            style: "padding: 0.75em 1ch; background: var(--card-bg); border-bottom: 1px solid var(--border-color); flex-shrink: 0;",
            div { style: "margin-bottom: 0.5em;",
                div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "task description" }
                textarea {
                    style: "width: 100%; min-height: 2.5em; resize: vertical; {INPUT_STYLE} box-sizing: border-box;",
                    placeholder: "Describe the task…",
                    value: "{spawn_task_str}",
                    oninput: move |e| spawn_task_str.set(e.value()),
                }
            }
            div { style: "display: flex; gap: 1ch; margin-bottom: 0.6em;",
                div { style: "flex: 1;",
                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "provider" }
                    select {
                        style: "{INPUT_STYLE}",
                        value: "{spawn_provider}",
                        onchange: move |e| {
                            let provider = e.value();
                            let models = model_options.read();
                            let next_model = models_for_provider(&models, &provider)
                                .first()
                                .map(|model| model.id.clone())
                                .unwrap_or_default();
                            spawn_provider.set(provider);
                            spawn_model.set(next_model);
                        },
                        if provider_options.read().is_empty() {
                            option { value: "", "server default" }
                        }
                        for provider in provider_options.read().iter() {
                            option { value: "{provider.id}", "{provider.display_name}" }
                        }
                    }
                }
                div { style: "flex: 1;",
                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "model" }
                    select {
                        style: "{INPUT_STYLE}",
                        value: "{spawn_model}",
                        onchange: move |e| spawn_model.set(e.value()),
                        {
                            let models = model_options.read();
                            let filtered_models = models_for_provider(&models, &spawn_provider.read());
                            rsx! {
                                if filtered_models.is_empty() {
                                    option { value: "", "server default" }
                                }
                                for model in filtered_models {
                                    option { value: "{model.id}", "{model.id}" }
                                }
                            }
                        }
                    }
                }
            }
            button {
                class: "primary",
                style: "padding: 0.3em 1.5ch;",
                disabled: spawn_task_str.read().trim().is_empty(),
                onclick: move |_| {
                    let task = spawn_task_str.read().trim().to_string();
                    if task.is_empty() {
                        return;
                    }
                    let prov = {
                        let v = spawn_provider.read();
                        let t = v.trim();
                        if t.is_empty() { None } else { Some(t.to_string()) }
                    };
                    let model = {
                        let v = spawn_model.read();
                        let t = v.trim();
                        if t.is_empty() { None } else { Some(t.to_string()) }
                    };
                    on_spawn.call(SpawnRequest {
                        task,
                        source: AgentSource::Local,
                        provider: prov,
                        model,
                    });
                    spawn_task_str.set(String::new());
                    show_spawn.set(false);
                },
                "spawn task"
            }
        }
    }
}
