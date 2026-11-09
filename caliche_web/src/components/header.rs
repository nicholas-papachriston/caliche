//! Header component with provider/model selection - monospace terminal style.

use dioxus::prelude::*;

use crate::api::types::{InteractionMode, ReasoningPatternInfo};
use crate::api::{ModelInfo, ProviderCatalogEntry};
use crate::app::settings_merge::{accept_header_selection, show_uncommitted_option};
use crate::catalog_select::{first_selectable_model_id, models_for_picker};
use crate::state::ConfigState;

/// Props for the Header component
#[derive(Props, Clone, PartialEq)]
pub struct HeaderProps {
    /// Config state signal
    pub config: Signal<ConfigState>,
    /// Available providers
    pub providers: Vec<ProviderCatalogEntry>,
    /// All available models (will be filtered by selected provider)
    pub all_models: Vec<ModelInfo>,
    /// Current interaction mode
    pub mode: InteractionMode,
    /// Called when the user changes interaction mode
    pub on_mode_change: EventHandler<InteractionMode>,
    /// Called after user changes provider or model (e.g. to send UpdateSettings over WebSocket)
    #[props(optional)]
    pub on_settings_changed: Option<EventHandler<()>>,
    /// Available reasoning patterns (empty = hide selector)
    #[props(default)]
    pub reasoning_patterns: Vec<ReasoningPatternInfo>,
}

/// Header component with app title and configuration controls
#[component]
pub fn Header(props: HeaderProps) -> Element {
    let mut config_signal = props.config;
    let all_models = props.all_models.clone();
    let all_models_desktop = all_models.clone();
    let all_models_mobile = all_models.clone();
    let providers = props.providers.clone();
    let on_settings_changed = props.on_settings_changed.clone();
    // Start with settings collapsed on mobile
    let mut show_settings = use_signal(|| false);
    let mut model_query = use_signal(String::new);
    // Arm after paint so a select that mounts with the server pair cannot
    // persist the first catalog option from its own change event.
    let mut selection_armed = use_signal(|| false);
    use_effect(move || {
        if config_signal.read().server_settings_seen {
            selection_armed.set(true);
        }
    });

    let (current_provider, current_model, current_reasoning_pattern) = {
        let config = config_signal.read();
        (
            config.provider.clone().unwrap_or_default(),
            config.model.clone().unwrap_or_default(),
            config.reasoning_pattern.clone().unwrap_or_default(),
        )
    };

    let query = model_query.read().clone();
    let filtered_models = models_for_picker(
        &all_models,
        &current_provider,
        &query,
        Some(current_model.as_str()),
    );

    // Get display name for current model (truncate if too long)
    let show_unset_provider = show_uncommitted_option(&current_provider);
    let model_display = if current_model.is_empty() {
        "(no model)".to_string()
    } else if current_model.len() > 20 {
        format!(
            "{}...",
            caliche_contracts::text::truncate_utf8(&current_model, 17)
        )
    } else {
        current_model.clone()
    };

    let settings_icon = if *show_settings.read() { "x" } else { "=" };

    rsx! {
        header { class: "app-header",
            style: "display: flex; justify-content: space-between; align-items: center; padding: 0.5em 1ch; background: var(--card-bg); border-bottom: 1px solid var(--border-color); gap: 1ch; flex-shrink: 0; min-height: 44px;",

            span { style: "color: var(--accent); font-weight: bold; white-space: nowrap; letter-spacing: 0.05em; min-width: 0;",
                "caliche"
            }

            // Desktop: inline controls
            div { class: "desktop-controls",
                style: "display: flex; align-items: center; gap: 2ch;",

                // Provider selector
                div { style: "display: flex; align-items: center; gap: 1ch;",
                    span { style: "color: var(--muted-color);", "provider:" }
                    select {
                        style: "background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.25em 1ch;",
                        value: "{current_provider}",
                        onchange: move |evt| {
                            let value = evt.value();
                            let (seen, current) = {
                                let config = config_signal.read();
                                (
                                    config.server_settings_seen,
                                    config.provider.clone().unwrap_or_default(),
                                )
                            };
                            if !accept_header_selection(seen, *selection_armed.read(), &current, &value)
                            {
                                return;
                            }
                            let next_model = if value.is_empty() {
                                None
                            } else {
                                first_selectable_model_id(&all_models_desktop, &value)
                            };
                            {
                                let mut config = config_signal.write();
                                config.user_edited_live_settings = true;
                                config.set_provider(if value.is_empty() { None } else { Some(value) });
                                config.set_model(next_model);
                            }
                            model_query.set(String::new());
                            if let Some(ref h) = on_settings_changed {
                                h.call(());
                            }
                        },
                        if show_unset_provider {
                            option { value: "", selected: current_provider.is_empty(), "(not set)" }
                        }
                        for provider in &providers {
                            option {
                                value: "{provider.id}",
                                selected: provider.id == current_provider,
                                "{provider.display_name}"
                            }
                        }
                    }
                }

                // Model selector
                div { style: "display: flex; align-items: center; gap: 1ch;",
                    span { style: "color: var(--muted-color);", "model:" }
                    input {
                        style: "background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.25em 1ch; width: 12ch;",
                        placeholder: "filter",
                        aria_label: "filter models",
                        value: "{model_query}",
                        oninput: move |evt| model_query.set(evt.value()),
                    }
                    select {
                        style: "background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.25em 1ch; max-width: 24ch;",
                        value: "{current_model}",
                        onchange: move |evt| {
                            let value = evt.value();
                            let (seen, current) = {
                                let config = config_signal.read();
                                (
                                    config.server_settings_seen,
                                    config.model.clone().unwrap_or_default(),
                                )
                            };
                            if !accept_header_selection(
                                seen,
                                *selection_armed.read(),
                                &current,
                                &value,
                            ) {
                                return;
                            }
                            {
                                let mut config = config_signal.write();
                                config.user_edited_live_settings = true;
                                config.set_model(if value.is_empty() { None } else { Some(value) });
                            }
                            if let Some(ref h) = on_settings_changed {
                                h.call(());
                            }
                        },
                        option { value: "", selected: current_model.is_empty(), "(not set)" }
                        if !current_model.is_empty()
                            && !filtered_models.iter().any(|model| model.id == current_model)
                        {
                            option { value: "{current_model}", selected: true, "{current_model}" }
                        }
                        for model in &filtered_models {
                            {
                                let ctx_hint = model.context_length
                                    .map(|n| format!(" | {}k ctx", n / 1000))
                                    .unwrap_or_default();
                                let desc = model.description.as_deref().unwrap_or("");
                                let title_text = format!("{}{}{}", desc, if desc.is_empty() { "" } else { " " }, ctx_hint);
                                rsx! {
                                    option {
                                        value: "{model.id}",
                                        selected: model.id == current_model,
                                        title: "{title_text}",
                                        "{model.id}"
                                    }
                                }
                            }
                        }
                    }
                }

                // Mode toggle buttons
                div { style: "display: flex; align-items: center; gap: 0.5ch;",
                    span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "mode:" }
                    {
                        let agent_active = props.mode == InteractionMode::Agent;
                        let ask_active = props.mode == InteractionMode::Ask;
                        let plan_active = props.mode == InteractionMode::Plan;
                        let on_mode = props.on_mode_change.clone();
                        rsx! {
                            button {
                                class: if agent_active { "mode-btn active" } else { "mode-btn" },
                                title: "Agent mode — full capabilities, can read and write files",
                                onclick: {
                                    let h = on_mode.clone();
                                    move |_| h.call(InteractionMode::Agent)
                                },
                                "agent"
                            }
                            button {
                                class: if ask_active { "mode-btn active" } else { "mode-btn" },
                                title: "Ask mode — read-only, can only answer questions",
                                onclick: {
                                    let h = on_mode.clone();
                                    move |_| h.call(InteractionMode::Ask)
                                },
                                "ask"
                            }
                            button {
                                class: if plan_active { "mode-btn active" } else { "mode-btn" },
                                title: "Plan mode — interview and document requirements",
                                onclick: {
                                    let h = on_mode.clone();
                                    move |_| h.call(InteractionMode::Plan)
                                },
                                "plan"
                            }
                        }
                    }
                }

                // Reasoning pattern selector (only shown when patterns are available)
                if !props.reasoning_patterns.is_empty() {
                    {
                        let patterns = props.reasoning_patterns.clone();
                        let current_pattern = current_reasoning_pattern.clone();
                        rsx! {
                            div { style: "display: flex; align-items: center; gap: 1ch;",
                                span { style: "color: var(--muted-color); font-size: var(--font-size-xs);", "pattern:" }
                                select {
                                    style: "background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.25em 1ch; font-size: var(--font-size-xs);",
                                    value: "{current_pattern}",
                                    onchange: move |evt| {
                                        let v = evt.value();
                                        config_signal.write().reasoning_pattern =
                                            if v.is_empty() { None } else { Some(v) };
                                    },
                                    option { value: "", "(default)" }
                                    for p in &patterns {
                                        option {
                                            value: "{p.name}",
                                            title: "{p.description}",
                                            "{p.name}"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Mobile: current model display + settings button
            div { class: "mobile-settings-btn",
                style: "display: none; align-items: center; gap: 1ch;",

                // Show current model (truncated)
                span {
                    title: "{current_model}",
                    style: "color: var(--muted-color); font-size: var(--font-size-xs); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 15ch;",
                    "{model_display}"
                }

                button {
                    style: "background: var(--secondary-bg); border: 1px solid var(--border-color); color: var(--text-color); padding: 0.5em; min-width: 44px; min-height: 44px; display: flex; align-items: center; justify-content: center;",
                    onclick: move |_| {
                        let current = *show_settings.read();
                        show_settings.set(!current);
                    },
                    "[{settings_icon}]"
                }
            }
        }

        // Mobile settings panel (expandable)
        if *show_settings.read() {
            div { class: "mobile-settings-panel",
                style: "background: var(--card-bg); border-bottom: 1px solid var(--border-color); padding: 1ch; display: flex; flex-direction: column; gap: 0.75em; flex-shrink: 0; overflow: hidden;",

                // Provider row
                div { style: "display: flex; align-items: center; gap: 1ch; min-width: 0;",
                    span { style: "color: var(--muted-color); width: 8ch; flex-shrink: 0;", "provider" }
                    select {
                        style: "flex: 1; min-width: 0; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5em 1ch; font-size: 16px; min-height: 44px;",
                        value: "{current_provider}",
                        onchange: move |evt| {
                            let value = evt.value();
                            let (seen, current) = {
                                let config = config_signal.read();
                                (
                                    config.server_settings_seen,
                                    config.provider.clone().unwrap_or_default(),
                                )
                            };
                            if !accept_header_selection(
                                seen,
                                *selection_armed.read(),
                                &current,
                                &value,
                            ) {
                                return;
                            }
                            let next_model = if value.is_empty() {
                                None
                            } else {
                                first_selectable_model_id(&all_models_mobile, &value)
                            };
                            {
                                let mut config = config_signal.write();
                                config.user_edited_live_settings = true;
                                config.set_provider(if value.is_empty() { None } else { Some(value) });
                                config.set_model(next_model);
                            }
                            model_query.set(String::new());
                            if let Some(ref h) = on_settings_changed {
                                h.call(());
                            }
                        },
                        if show_unset_provider {
                            option { value: "", selected: current_provider.is_empty(), "(not set)" }
                        }
                        for provider in &providers {
                            option {
                                value: "{provider.id}",
                                selected: provider.id == current_provider,
                                "{provider.display_name}"
                            }
                        }
                    }
                }

                // Model row
                div { style: "display: flex; align-items: center; gap: 1ch; min-width: 0;",
                    span { style: "color: var(--muted-color); width: 8ch; flex-shrink: 0;", "model" }
                    input {
                        style: "width: 12ch; flex-shrink: 0; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5em 1ch; font-size: 16px; min-height: 44px;",
                        placeholder: "filter",
                        aria_label: "filter models",
                        value: "{model_query}",
                        oninput: move |evt| model_query.set(evt.value()),
                    }
                    select {
                        style: "flex: 1; min-width: 0; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5em 1ch; font-size: 16px; min-height: 44px;",
                        value: "{current_model}",
                        onchange: move |evt| {
                            let value = evt.value();
                            let (seen, current) = {
                                let config = config_signal.read();
                                (
                                    config.server_settings_seen,
                                    config.model.clone().unwrap_or_default(),
                                )
                            };
                            if !accept_header_selection(
                                seen,
                                *selection_armed.read(),
                                &current,
                                &value,
                            ) {
                                return;
                            }
                            {
                                let mut config = config_signal.write();
                                config.user_edited_live_settings = true;
                                config.set_model(if value.is_empty() { None } else { Some(value) });
                            }
                            if let Some(ref h) = on_settings_changed {
                                h.call(());
                            }
                        },
                        option { value: "", selected: current_model.is_empty(), "(not set)" }
                        if !current_model.is_empty()
                            && !filtered_models.iter().any(|model| model.id == current_model)
                        {
                            option { value: "{current_model}", selected: true, "{current_model}" }
                        }
                        for model in &filtered_models {
                            option {
                                value: "{model.id}",
                                selected: model.id == current_model,
                                "{model.id}"
                            }
                        }
                    }
                }

                // Mode row
                div { style: "display: flex; align-items: center; gap: 1ch;",
                    span { style: "color: var(--muted-color); width: 8ch; flex-shrink: 0;", "mode" }
                    div { style: "display: flex; gap: 0.5ch;",
                        {
                            let agent_active = props.mode == InteractionMode::Agent;
                            let ask_active = props.mode == InteractionMode::Ask;
                            let plan_active = props.mode == InteractionMode::Plan;
                            let on_mode = props.on_mode_change.clone();
                            rsx! {
                                button {
                                    style: if agent_active { "flex: 1; min-height: 44px; background: var(--accent-dim); color: #000; border: 1px solid var(--accent); font-size: 16px;" } else { "flex: 1; min-height: 44px; background: transparent; color: var(--muted-color); border: 1px solid var(--border-color); font-size: 16px;" },
                                    onclick: { let h = on_mode.clone(); move |_| h.call(InteractionMode::Agent) },
                                    "agent"
                                }
                                button {
                                    style: if ask_active { "flex: 1; min-height: 44px; background: var(--accent-dim); color: #000; border: 1px solid var(--accent); font-size: 16px;" } else { "flex: 1; min-height: 44px; background: transparent; color: var(--muted-color); border: 1px solid var(--border-color); font-size: 16px;" },
                                    onclick: { let h = on_mode.clone(); move |_| h.call(InteractionMode::Ask) },
                                    "ask"
                                }
                                button {
                                    style: if plan_active { "flex: 1; min-height: 44px; background: var(--accent-dim); color: #000; border: 1px solid var(--accent); font-size: 16px;" } else { "flex: 1; min-height: 44px; background: transparent; color: var(--muted-color); border: 1px solid var(--border-color); font-size: 16px;" },
                                    onclick: { let h = on_mode.clone(); move |_| h.call(InteractionMode::Plan) },
                                    "plan"
                                }
                            }
                        }
                    }
                }

                // Reasoning pattern row (only when patterns loaded)
                if !props.reasoning_patterns.is_empty() {
                    {
                        let patterns = props.reasoning_patterns.clone();
                        let current_pattern = current_reasoning_pattern.clone();
                        rsx! {
                            div { style: "display: flex; align-items: center; gap: 1ch; min-width: 0;",
                                span { style: "color: var(--muted-color); width: 8ch; flex-shrink: 0;", "pattern" }
                                select {
                                    style: "flex: 1; min-width: 0; background: var(--input-bg); color: var(--text-color); border: 1px solid var(--border-color); padding: 0.5em 1ch; font-size: 16px; min-height: 44px;",
                                    value: "{current_pattern}",
                                    onchange: move |evt| {
                                        let v = evt.value();
                                        config_signal.write().reasoning_pattern =
                                            if v.is_empty() { None } else { Some(v) };
                                    },
                                    option { value: "", "(default)" }
                                    for p in &patterns {
                                        option {
                                            value: "{p.name}",
                                            title: "{p.description}",
                                            "{p.name}"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Simplified header for when API info isn't loaded yet
#[component]
pub fn SimpleHeader() -> Element {
    rsx! {
        header { class: "app-header",
            style: "display: flex; justify-content: space-between; align-items: center; padding: 0.5em 1ch; background: var(--card-bg); border-bottom: 1px solid var(--border-color); flex-shrink: 0; min-height: 44px;",

            span { style: "color: var(--accent); font-weight: bold; letter-spacing: 0.05em;",
                "caliche"
            }
        }
    }
}
