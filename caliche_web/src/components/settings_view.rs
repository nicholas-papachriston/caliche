//! Settings tab — edit and persist agent settings via PUT /api/settings.

use dioxus::prelude::*;

use crate::api::{
    AgentSettings, ApiClient, McpServerConfig, ModelInfo, ProviderCatalogEntry, ToolInfo,
};
use crate::app::settings_merge::show_uncommitted_option;
use crate::catalog_select::{first_selectable_model_id, models_for_picker};
use crate::components::permissions_view::PermissionsView;
use crate::components::tunnel_view::TunnelView;
use crate::state::ConfigState;

fn blank_mcp_server() -> McpServerConfig {
    McpServerConfig {
        name: String::new(),
        transport: "stdio".to_string(),
        command: Some(String::new()),
        args: Vec::new(),
        url: None,
        enabled: true,
    }
}

const EMBEDDING_PROVIDERS: [(&str, &str); 5] = [
    ("openai", "OpenAI"),
    ("bedrock", "Bedrock"),
    ("openrouter", "OpenRouter"),
    ("cloudflare-ai-gateway", "Cloudflare AI Gateway"),
    ("lmstudio", "LM Studio"),
];

fn retrieval_status_caption(state: &str) -> &'static str {
    match state {
        "configured" => {
            "VECTOR_BACKEND names a backend this binary can construct because the feature is compiled in. That does not mean a remote store is reachable or healthy. Dripstone stores and searches vectors for this process. Graph search is optional in the same library."
        }
        "unavailable" => {
            "VECTOR_BACKEND is an unknown name, or the feature is off in this binary. Dripstone stores and searches vectors when that feature is compiled in."
        }
        _ => {
            "VECTOR_BACKEND is unset, so retrieval is unconfigured. Unset is not an in-memory tenant store, and Dripstone is not an isolation boundary."
        }
    }
}

fn default_embedding_model_for(provider: &str) -> &'static str {
    match provider {
        "openai" => "text-embedding-3-small",
        "bedrock" => "amazon.titan-embed-text-v2:0",
        "openrouter" => "openai/text-embedding-3-small",
        "cloudflare-ai-gateway" => "workers-ai/@cf/baai/bge-base-en-v1.5",
        "lmstudio" => "default",
        _ => "",
    }
}

/// Props for the SettingsView component.
#[derive(Props, Clone, PartialEq)]
pub struct SettingsViewProps {
    pub config: Signal<ConfigState>,
    pub providers: Vec<ProviderCatalogEntry>,
    pub all_models: Vec<ModelInfo>,
    pub api_client: ApiClient,
    /// API base URL shown in the tunnel notice.
    pub api_base_url: String,
    /// Same write path as the header provider/model selects.
    #[props(optional)]
    pub on_settings_changed: Option<EventHandler<()>>,
}

/// Full-page settings editor. Loads from GET /api/settings on mount, saves via PUT /api/settings.
#[component]
pub fn SettingsView(props: SettingsViewProps) -> Element {
    let mut config_signal = props.config;
    let on_settings_changed = props.on_settings_changed.clone();
    let mut model_query = use_signal(String::new);
    let mut local_temperature = use_signal(String::new);
    let mut local_max_tokens = use_signal(String::new);
    let mut local_system_prompt = use_signal(String::new);
    let mut local_tools = use_signal(|| props.config.read().selected_tools.clone());
    let mut local_embedding_provider = use_signal(|| {
        props
            .config
            .read()
            .embedding_provider
            .clone()
            .unwrap_or_default()
    });
    let mut local_embedding_model = use_signal(|| {
        props
            .config
            .read()
            .embedding_model
            .clone()
            .unwrap_or_default()
    });
    let mut local_show_sources = use_signal(|| props.config.read().show_sources);
    let mut local_vector_backend = use_signal(String::new);
    let mut local_vector_backend_state = use_signal(|| "unconfigured".to_string());
    let mut local_mcp_servers = use_signal(Vec::<McpServerConfig>::new);
    let mut mcp_ready = use_signal(|| false);
    let mut mcp_baseline = use_signal(Vec::<McpServerConfig>::new);
    let mut available_tools = use_signal(Vec::<ToolInfo>::new);
    let mut mcp_test_status = use_signal(|| None::<(usize, bool, String)>);
    let mut save_status = use_signal(|| Option::<(bool, String)>::None);
    let mut is_saving = use_signal(|| false);
    let mut loaded = use_signal(|| false);
    let mut local_api_base = use_signal(|| props.config.read().api_base_url.clone());

    // Fetch settings + tools from server on first render.
    {
        let client = props.api_client.clone();
        let mut config_for_load = props.config;
        use_effect(move || {
            if *loaded.read() {
                return;
            }
            loaded.set(true);
            let client = client.clone();
            spawn(async move {
                if let Ok(s) = client.get_settings().await {
                    config_for_load.write().apply_from_settings(&s);
                    if let Some(t) = s.temperature {
                        local_temperature.set(format!("{:.2}", t));
                    }
                    if let Some(mt) = s.max_tokens {
                        local_max_tokens.set(mt.to_string());
                    }
                    if let Some(sp) = s.system_prompt {
                        local_system_prompt.set(sp);
                    }
                    local_tools.set(s.tools);
                    if let Some(provider) = s.embedding_provider {
                        local_embedding_provider.set(provider);
                    }
                    if let Some(model) = s.embedding_model {
                        local_embedding_model.set(model);
                    }
                    if let Some(show_sources) = s.show_sources {
                        local_show_sources.set(show_sources);
                    }
                    local_vector_backend.set(s.vector_backend.unwrap_or_default());
                    local_vector_backend_state.set(
                        s.vector_backend_state
                            .unwrap_or_else(|| "unconfigured".to_string()),
                    );
                }
                if let Ok(tools) = client.get_tools().await {
                    available_tools.set(tools);
                }
                if let Ok(mcp) = client.get_mcp_config().await {
                    mcp_baseline.set(mcp.servers.clone());
                    local_mcp_servers.set(mcp.servers);
                    mcp_ready.set(true);
                }
            });
        });
    }

    let all_models = props.all_models.clone();
    let models_for_change = all_models.clone();
    let current_provider = props.config.read().provider.clone().unwrap_or_default();
    let current_model = props.config.read().model.clone().unwrap_or_default();
    let show_unset_provider = show_uncommitted_option(&current_provider);
    let query = model_query.read().clone();
    let filtered_models = models_for_picker(
        &all_models,
        &current_provider,
        &query,
        Some(current_model.as_str()),
    );

    // Save handler — builds AgentSettings and PUTs to server.
    let handle_save = {
        let client = props.api_client.clone();
        let mut config = props.config;
        move |_| {
            let provider = config.read().provider.clone().unwrap_or_default();
            let model = config.read().model.clone().unwrap_or_default();
            if provider.trim().is_empty() {
                save_status.set(Some((false, "choose a provider before saving".to_string())));
                return;
            }
            let temp_str = local_temperature.read().clone();
            let tok_str = local_max_tokens.read().clone();
            let tools = local_tools.read().clone();
            let embedding_provider = Some(local_embedding_provider.read().clone());
            let embedding_model = Some(local_embedding_model.read().clone());
            let show_sources = Some(*local_show_sources.read());
            let mcp_servers = local_mcp_servers.read().clone();
            let write_mcp = *mcp_ready.read() && mcp_servers != *mcp_baseline.read();
            let allow_empty_mcp = mcp_servers.is_empty();
            let client = client.clone();
            let mut mcp_baseline = mcp_baseline;

            spawn(async move {
                is_saving.set(true);
                save_status.set(None);

                let temperature = temp_str.parse::<f32>().ok();
                let max_tokens = tok_str.parse::<u32>().ok();
                let settings = AgentSettings {
                    provider: provider.clone(),
                    model: model.clone(),
                    system_prompt: None,
                    temperature,
                    max_tokens,
                    context_window: None,
                    tools: tools.clone(),
                    mode: None,
                    embedding_provider: embedding_provider.clone(),
                    embedding_model: embedding_model.clone(),
                    show_sources,
                    vector_backend: None,
                    vector_backend_state: None,
                };

                match client.put_settings(&settings).await {
                    Ok(updated) => {
                        config.write().apply_from_settings(&updated);
                        if write_mcp {
                            match client
                                .put_mcp_config(mcp_servers.clone(), allow_empty_mcp)
                                .await
                            {
                                Ok(_) => {
                                    mcp_baseline.set(mcp_servers);
                                    save_status.set(Some((
                                        true,
                                        "settings and MCP config saved".to_string(),
                                    )));
                                }
                                Err(e) => {
                                    save_status.set(Some((
                                        false,
                                        format!("saved settings, but MCP config failed: {}", e),
                                    )));
                                }
                            }
                        } else {
                            save_status.set(Some((true, "settings saved".to_string())));
                        }
                    }
                    Err(e) => {
                        save_status.set(Some((false, format!("error: {}", e))));
                    }
                }
                is_saving.set(false);
            });
        }
    };

    // Row style helpers
    let row = "display: flex; align-items: flex-start; gap: 2ch; margin-bottom: 1.25em;";
    let label_style = "color: var(--muted-color); width: 16ch; flex-shrink: 0; padding-top: 0.2em;";
    let input_style = "background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.25em 1ch; flex: 1; min-width: 0;";
    let vector_backend_label = {
        let name = local_vector_backend.read().clone();
        if name.is_empty() {
            "(not set)".to_string()
        } else {
            name
        }
    };
    let vector_status_caption =
        retrieval_status_caption(local_vector_backend_state.read().as_str());

    rsx! {
        div { class: "settings-view",
            style: "flex: 1; min-height: 0; overflow-y: auto; padding: 2ch; max-width: 80ch; margin: 0 auto; width: 100%;",

            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "connection"
                }
                div { style: "{row}",
                    span { style: "{label_style}", "api base url" }
                    div { style: "display: flex; flex-direction: column; gap: 0.25em; flex: 1;",
                        input {
                            style: "{input_style}",
                            value: "{local_api_base}",
                            aria_label: "API base URL",
                            oninput: move |evt| local_api_base.set(evt.value()),
                            onchange: move |evt| {
                                let next = evt.value().trim().to_string();
                                if !next.is_empty() {
                                    config_signal.write().api_base_url = next;
                                }
                            },
                        }
                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                            "Applied when you leave the field. The console retries this address every 5s."
                        }
                    }
                }
            }

            // Section: model
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "model"
                }

                // Provider and model share ConfigState with the header.
                div { style: "{row}",
                    span { style: "{label_style}", "provider" }
                    select {
                        style: "{input_style}",
                        value: "{current_provider}",
                        onchange: move |evt| {
                            let v = evt.value();
                            let new_model = if v.is_empty() {
                                None
                            } else {
                                first_selectable_model_id(&models_for_change, &v)
                            };
                            {
                                let mut config = config_signal.write();
                                config.user_edited_live_settings = true;
                                config.set_provider(if v.is_empty() { None } else { Some(v) });
                                config.set_model(new_model);
                            }
                            model_query.set(String::new());
                            if let Some(handler) = on_settings_changed.clone() {
                                handler.call(());
                            }
                        },
                        if show_unset_provider {
                            option { value: "", selected: current_provider.is_empty(), "(not set)" }
                        }
                        for p in &props.providers {
                            option {
                                value: "{p.id}",
                                selected: p.id == current_provider,
                                "{p.display_name}"
                            }
                        }
                    }
                }

                div { style: "{row}",
                    span { style: "{label_style}", "model" }
                    div { style: "display: flex; gap: 1ch; flex: 1; min-width: 0;",
                        input {
                            style: "{input_style}",
                            placeholder: "filter",
                            aria_label: "filter models",
                            value: "{model_query}",
                            oninput: move |evt| model_query.set(evt.value()),
                        }
                        select {
                            style: "{input_style}",
                            value: "{current_model}",
                            onchange: move |evt| {
                                let value = evt.value();
                                {
                                    let mut config = config_signal.write();
                                    config.user_edited_live_settings = true;
                                    config.set_model(if value.is_empty() { None } else { Some(value) });
                                }
                                if let Some(handler) = on_settings_changed.clone() {
                                    handler.call(());
                                }
                            },
                            option { value: "", selected: current_model.is_empty(), "(not set)" }
                            if !current_model.is_empty()
                                && !filtered_models.iter().any(|model| model.id == current_model)
                            {
                                option { value: "{current_model}", selected: true, "{current_model}" }
                            }
                            for m in &filtered_models {
                                option {
                                    value: "{m.id}",
                                    selected: m.id == current_model,
                                    "{m.id}"
                                }
                            }
                        }
                    }
                }
                p { style: "color: var(--muted-color); font-size: var(--font-size-xs); margin: -0.5em 0 1em 18ch;",
                    "Mode is the header control. It is sent with sessions.setMode after a session exists."
                }
            }

            // Section: Dripstone retrieval (read-only process status)
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "retrieval"
                }
                div { style: "{row}",
                    span { style: "{label_style}", "backend" }
                    span { style: "color: var(--text-color);", "{vector_backend_label}" }
                }
                div { style: "{row}",
                    span { style: "{label_style}", "state" }
                    span { style: "color: var(--text-color);", "{local_vector_backend_state}" }
                }
                div { style: "color: var(--muted-color); font-size: var(--font-size-xs); max-width: 72ch;",
                    "{vector_status_caption}"
                }
            }

            // Section: embeddings
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "embeddings"
                }

                div { style: "{row}",
                    span { style: "{label_style}", "provider" }
                    select {
                        style: "{input_style}",
                        value: "{local_embedding_provider}",
                        onchange: move |evt| {
                            let v = evt.value();
                            let next_model = default_embedding_model_for(&v).to_string();
                            local_embedding_provider.set(v);
                            local_embedding_model.set(next_model);
                        },
                        option { value: "", "(none)" }
                        for (id, label) in EMBEDDING_PROVIDERS {
                            option { value: "{id}", selected: id == *local_embedding_provider.read(), "{label}" }
                        }
                    }
                }

                div { style: "{row}",
                    span { style: "{label_style}", "model" }
                    input {
                        style: "{input_style}",
                        value: "{local_embedding_model}",
                        placeholder: "embedding model id",
                        oninput: move |evt| local_embedding_model.set(evt.value()),
                    }
                }
            }

            // Section: display
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "display"
                }
                div { style: "{row}",
                    span { style: "{label_style}", "sources" }
                    label { style: "display: flex; gap: 0.5ch; align-items: center; color: var(--text-color);",
                        input {
                            r#type: "checkbox",
                            checked: *local_show_sources.read(),
                            onchange: move |evt| local_show_sources.set(evt.checked()),
                        }
                        "show source citations"
                    }
                }
            }

            // Section: parameters
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "parameters"
                }

                div { style: "{row}",
                    span { style: "{label_style}", "temperature" }
                    div { style: "display: flex; flex-direction: column; gap: 0.25em; flex: 1;",
                        input {
                            r#type: "number",
                            style: "{input_style} flex: unset;",
                            value: "{local_temperature}",
                            min: "0.0",
                            max: "2.0",
                            step: "0.05",
                            placeholder: "default",
                            oninput: move |evt| local_temperature.set(evt.value()),
                        }
                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                            "0.0 – 2.0  (blank = provider default)"
                        }
                    }
                }

                div { style: "{row}",
                    span { style: "{label_style}", "max tokens" }
                    div { style: "display: flex; flex-direction: column; gap: 0.25em; flex: 1;",
                        input {
                            r#type: "number",
                            style: "{input_style} flex: unset;",
                            value: "{local_max_tokens}",
                            min: "1",
                            step: "256",
                            placeholder: "default",
                            oninput: move |evt| local_max_tokens.set(evt.value()),
                        }
                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                            "blank = provider default"
                        }
                    }
                }
            }

            // Section: MCP
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "mcp"
                }

                div { style: "{row}",
                    span { style: "{label_style}", "remote servers" }
                    div { style: "display: flex; flex-direction: column; gap: 1em; flex: 1;",
                        span { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                            "Configure reusable remote MCP servers. These are persisted and applied live when the server is CLI-hosted."
                        }
                        for (idx, server) in local_mcp_servers.read().iter().cloned().enumerate() {
                            div { key: "{idx}", style: "border: 1px solid var(--border-color); padding: 0.75em 1ch; display: flex; flex-direction: column; gap: 0.75em;",
                                div { style: "display: flex; gap: 1ch; align-items: center;",
                                    input {
                                        style: "{input_style}",
                                        placeholder: "server name",
                                        value: "{server.name}",
                                        oninput: move |evt| {
                                            let mut next = local_mcp_servers.read().clone();
                                            if let Some(item) = next.get_mut(idx) {
                                                item.name = evt.value();
                                            }
                                            local_mcp_servers.set(next);
                                        }
                                    }
                                    select {
                                        style: "{input_style} max-width: 12ch;",
                                        value: "{server.transport}",
                                        onchange: move |evt| {
                                            let mut next = local_mcp_servers.read().clone();
                                            if let Some(item) = next.get_mut(idx) {
                                                item.transport = evt.value();
                                                if item.transport == "stdio" {
                                                    item.url = None;
                                                    item.command.get_or_insert_with(String::new);
                                                } else {
                                                    item.command = None;
                                                    item.url.get_or_insert_with(String::new);
                                                }
                                            }
                                            local_mcp_servers.set(next);
                                        },
                                        option { value: "stdio", "stdio" }
                                        option { value: "streamable-http", "http" }
                                    }
                                    label { style: "display: flex; gap: 0.5ch; align-items: center; color: var(--muted-color);",
                                        input {
                                            r#type: "checkbox",
                                            checked: server.enabled,
                                            onchange: move |evt| {
                                                let mut next = local_mcp_servers.read().clone();
                                                if let Some(item) = next.get_mut(idx) {
                                                    item.enabled = evt.checked();
                                                }
                                                local_mcp_servers.set(next);
                                            }
                                        }
                                        "enabled"
                                    }
                                    button {
                                        style: "background: transparent; border: 1px solid var(--border-color); color: var(--error); padding: 0.25em 1ch;",
                                        onclick: move |_| {
                                            let mut next = local_mcp_servers.read().clone();
                                            if idx < next.len() {
                                                next.remove(idx);
                                            }
                                            local_mcp_servers.set(next);
                                        },
                                        "remove"
                                    }
                                }

                                if server.transport == "stdio" {
                                    div { style: "display: flex; flex-direction: column; gap: 0.5em;",
                                        input {
                                            style: "{input_style}",
                                            placeholder: "command",
                                            value: "{server.command.clone().unwrap_or_default()}",
                                            oninput: move |evt| {
                                                let mut next = local_mcp_servers.read().clone();
                                                if let Some(item) = next.get_mut(idx) {
                                                    item.command = Some(evt.value());
                                                }
                                                local_mcp_servers.set(next);
                                            }
                                        }
                                        input {
                                            style: "{input_style}",
                                            placeholder: "args (space separated)",
                                            value: "{server.args.join(\" \")}",
                                            oninput: move |evt| {
                                                let mut next = local_mcp_servers.read().clone();
                                                if let Some(item) = next.get_mut(idx) {
                                                    item.args = evt
                                                        .value()
                                                        .split_whitespace()
                                                        .map(ToString::to_string)
                                                        .collect();
                                                }
                                                local_mcp_servers.set(next);
                                            }
                                        }
                                    }
                                } else {
                                    input {
                                        style: "{input_style}",
                                        placeholder: "sse url",
                                        value: "{server.url.clone().unwrap_or_default()}",
                                        oninput: move |evt| {
                                            let mut next = local_mcp_servers.read().clone();
                                            if let Some(item) = next.get_mut(idx) {
                                                item.url = Some(evt.value());
                                            }
                                            local_mcp_servers.set(next);
                                        }
                                    }
                                }

                                div { style: "display: flex; gap: 1ch; align-items: center;",
                                    button {
                                        style: "background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0.25em 1ch;",
                                        onclick: {
                                            let client = props.api_client.clone();
                                            let server = server.clone();
                                            move |_| {
                                                let client = client.clone();
                                                let server = server.clone();
                                                spawn(async move {
                                                    match client.test_mcp_server(server).await {
                                                        Ok(resp) => {
                                                            if resp.ok {
                                                                let summary = if resp.tools.is_empty() {
                                                                    "connected; no tools returned".to_string()
                                                                } else {
                                                                    format!("connected; tools: {}", resp.tools.join(", "))
                                                                };
                                                                mcp_test_status.set(Some((idx, true, summary)));
                                                            } else {
                                                                mcp_test_status.set(Some((idx, false, resp.error.unwrap_or_else(|| "unknown MCP error".to_string()))));
                                                            }
                                                        }
                                                        Err(err) => {
                                                            mcp_test_status.set(Some((idx, false, err.to_string())));
                                                        }
                                                    }
                                                });
                                            }
                                        },
                                        "test"
                                    }
                                    if let Some((test_idx, ok, message)) = mcp_test_status.read().clone()
                                        && test_idx == idx {
                                        span {
                                            style: if ok { "color: var(--accent);" } else { "color: var(--error);" },
                                            "{message}"
                                        }
                                    }
                                }
                            }
                        }

                        button {
                            style: "align-self: flex-start; background: transparent; border: 1px solid var(--border-color); color: var(--text-color); padding: 0.35em 1ch;",
                            onclick: move |_| {
                                let mut next = local_mcp_servers.read().clone();
                                next.push(blank_mcp_server());
                                local_mcp_servers.set(next);
                            },
                            "add mcp server"
                        }
                    }
                }
            }

            // Section: system prompt (inspection only; startup builds it)
            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "system prompt"
                }
                p { style: "color: var(--muted-color); font-size: var(--font-size-sm); margin: 0 0 0.75em 0;",
                    "Shown for inspection. Built at startup from the built-in prompt and AGENTS.md. Save does not store this text. Restart the server to pick up AGENTS.md edits."
                }
                textarea {
                    style: "width: 100%; min-height: 8em; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5em 1ch; resize: vertical; font-family: var(--font-mono); font-size: var(--font-size-sm);",
                    readonly: true,
                    aria_label: "Effective system prompt",
                    value: "{local_system_prompt}",
                }
            }

            // Section: tools
            if !available_tools.read().is_empty() {
                div { style: "margin-bottom: 2em;",
                    div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                        "tools"
                    }
                    div { style: "display: flex; flex-wrap: wrap; gap: 0.5em 2ch;",
                        for tool in available_tools.read().clone() {
                            {
                                let tool_name = tool.name.clone();
                                let name_for_toggle = tool_name.clone();
                                let is_checked = local_tools.read().contains(&tool_name);
                                rsx! {
                                    label { style: "display: flex; align-items: center; gap: 0.5ch; cursor: pointer; color: var(--text-color); min-width: 16ch;",
                                        input {
                                            r#type: "checkbox",
                                            style: "accent-color: var(--accent); cursor: pointer;",
                                            checked: is_checked,
                                            onchange: move |evt| {
                                                let mut tools = local_tools.write();
                                                if evt.checked() {
                                                    if !tools.contains(&name_for_toggle) {
                                                        tools.push(name_for_toggle.clone());
                                                    }
                                                } else {
                                                    tools.retain(|t| t != &name_for_toggle);
                                                }
                                            },
                                        }
                                        span { style: "font-size: var(--font-size-sm);", "{tool_name}" }
                                        if let Some(ref desc) = tool.description.split('.').next() {
                                            span { style: "color: var(--muted-color); font-size: var(--font-size-xs); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;",
                                                " – {desc}"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            PermissionsView { api_client: props.api_client.clone() }

            div { style: "margin-bottom: 2em;",
                div { style: "color: var(--muted-color); margin-bottom: 1em; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5em;",
                    "cloudflare tunnels"
                }
                TunnelView {
                    api_base_url: props.api_base_url,
                }
            }

            // Save row
            div { style: "display: flex; align-items: center; gap: 2ch; padding-top: 1em; border-top: 1px solid var(--border-color);",
                button {
                    class: "primary",
                    style: "padding: 0.4em 2ch;",
                    disabled: *is_saving.read(),
                    onclick: handle_save,
                    if *is_saving.read() { "saving..." } else { "save settings" }
                }
                if let Some((ok, ref msg)) = *save_status.read() {
                    span {
                        style: if ok {
                            "color: var(--accent); font-size: var(--font-size-sm);"
                        } else {
                            "color: var(--error-color); font-size: var(--font-size-sm);"
                        },
                        "{msg}"
                    }
                }
            }
        }
    }
}
