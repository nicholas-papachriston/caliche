use dioxus::prelude::*;

use super::helpers::{INPUT_STYLE, default_provider_and_model, models_for_provider};
use crate::api::{ApiClient, CreateEndpointRequest, ModelInfo, ProviderCatalogEntry};

#[derive(Props, Clone, PartialEq)]
pub struct EndpointLaunchFormProps {
    pub api_client: ApiClient,
    pub provider_options: Signal<Vec<ProviderCatalogEntry>>,
    pub model_options: Signal<Vec<ModelInfo>>,
    pub current_provider_setting: Signal<Option<String>>,
    pub current_model_setting: Signal<Option<String>>,
    pub form_name: Signal<String>,
    pub form_workspace: Signal<String>,
    pub form_port: Signal<String>,
    pub form_provider: Signal<String>,
    pub form_model: Signal<String>,
    pub form_auto_start: Signal<bool>,
    pub form_error: Signal<Option<String>>,
    pub form_busy: Signal<bool>,
    pub show_launch: Signal<bool>,
    pub on_launched: EventHandler<()>,
}

#[component]
pub fn EndpointLaunchForm(props: EndpointLaunchFormProps) -> Element {
    let api_client = props.api_client;
    let provider_options = props.provider_options;
    let model_options = props.model_options;
    let current_provider_setting = props.current_provider_setting;
    let current_model_setting = props.current_model_setting;
    let mut form_name = props.form_name;
    let mut form_workspace = props.form_workspace;
    let mut form_port = props.form_port;
    let mut form_provider = props.form_provider;
    let mut form_model = props.form_model;
    let mut form_auto_start = props.form_auto_start;
    let mut form_error = props.form_error;
    let mut form_busy = props.form_busy;
    let mut show_launch = props.show_launch;
    let on_launched = props.on_launched;

    rsx! {
        div {
            style: "padding: 0.75em 1ch; background: var(--card-bg); border-bottom: 1px solid var(--border-color); flex-shrink: 0;",
            div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.5em;",
                "launch caliche_server in another workspace directory"
            }
            div { style: "display: flex; gap: 1ch; margin-bottom: 0.5em;",
                div { style: "flex: 2;",
                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "name" }
                    input { r#type: "text", style: "{INPUT_STYLE}", placeholder: "my-project",
                        value: "{form_name}", oninput: move |e| form_name.set(e.value()) }
                }
                div { style: "flex: 1;",
                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "port" }
                    input { r#type: "number", style: "{INPUT_STYLE}", placeholder: "3001", min: "1024", max: "65535",
                        value: "{form_port}", oninput: move |e| form_port.set(e.value()) }
                }
            }
            div { style: "margin-bottom: 0.5em;",
                div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "workspace path (~ supported)" }
                input { r#type: "text", style: "{INPUT_STYLE}", placeholder: "~/projects/my-project",
                    value: "{form_workspace}", oninput: move |e| form_workspace.set(e.value()) }
            }
            div { style: "display: flex; gap: 1ch; margin-bottom: 0.5em;",
                div { style: "flex: 1;",
                    div { style: "font-size: var(--font-size-xs); color: var(--muted-color); margin-bottom: 0.2em;", "provider" }
                    select {
                        style: "{INPUT_STYLE}",
                        value: "{form_provider}",
                        onchange: move |e| {
                            let provider = e.value();
                            let models = model_options.read();
                            let next_model = models_for_provider(&models, &provider)
                                .first()
                                .map(|model| model.id.clone())
                                .unwrap_or_default();
                            form_provider.set(provider);
                            form_model.set(next_model);
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
                        value: "{form_model}",
                        onchange: move |e| form_model.set(e.value()),
                        {
                            let models = model_options.read();
                            let filtered_models = models_for_provider(&models, &form_provider.read());
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
            div { style: "display: flex; align-items: center; gap: 0.5ch; margin-bottom: 0.6em;",
                input { r#type: "checkbox", id: "ep-auto-start", style: "accent-color: var(--accent);",
                    checked: *form_auto_start.read(),
                    onchange: move |e| form_auto_start.set(e.checked()) }
                label { r#for: "ep-auto-start", style: "font-size: var(--font-size-xs); color: var(--muted-color); cursor: pointer;",
                    "start immediately" }
            }
            div { style: "display: flex; gap: 1ch; align-items: center;",
                button {
                    class: "primary",
                    style: "padding: 0.3em 1.5ch;",
                    disabled: *form_busy.read(),
                    onclick: move |_| {
                        let name = form_name.read().trim().to_string();
                        let workspace = form_workspace.read().trim().to_string();
                        let port_str = form_port.read().trim().to_string();
                        if name.is_empty() {
                            form_error.set(Some("Name required.".into()));
                            return;
                        }
                        if workspace.is_empty() {
                            form_error.set(Some("Workspace required.".into()));
                            return;
                        }
                        let port: u16 = match port_str.parse() {
                            Ok(p) => p,
                            Err(_) => {
                                form_error.set(Some("Port must be 1024–65535.".into()));
                                return;
                            }
                        };
                        let provider = {
                            let v = form_provider.read();
                            let t = v.trim();
                            if t.is_empty() { None } else { Some(t.to_string()) }
                        };
                        let model = {
                            let v = form_model.read();
                            let t = v.trim();
                            if t.is_empty() { None } else { Some(t.to_string()) }
                        };
                        let auto_start = *form_auto_start.read();
                        let client = api_client.clone();
                        form_busy.set(true);
                        form_error.set(None);
                        spawn(async move {
                            match client
                                .create_endpoint(CreateEndpointRequest {
                                    name,
                                    workspace,
                                    port,
                                    provider,
                                    model,
                                    auto_start,
                                })
                                .await
                            {
                                Ok(_) => {
                                    form_name.set(String::new());
                                    form_workspace.set(String::new());
                                    form_port.set("3001".to_string());
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
                                    form_provider.set(provider);
                                    form_model.set(model);
                                    show_launch.set(false);
                                    on_launched.call(());
                                }
                                Err(e) => form_error.set(Some(e.to_string())),
                            }
                            form_busy.set(false);
                        });
                    },
                    if *form_busy.read() { "launching…" } else { "launch" }
                }
                if let Some(ref e) = *form_error.read() {
                    span { style: "color: var(--error); font-size: var(--font-size-xs);", "{e}" }
                }
            }
        }
    }
}
