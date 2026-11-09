//! Application chrome: tab bar, view router, and session inspector layout.

use dioxus::prelude::*;

use crate::api::types::InteractionMode;
use crate::api::{
    ApiClient, EndpointInfo, LiveSessionInfo, ModelInfo, ProviderCatalogEntry, ReasoningPatternInfo,
};
use crate::components::chat_view::AttachedFileContext;
use crate::components::chrome::{DISCONNECTED_DETAIL, EmptyHint, PANE_UNAVAILABLE_SENTENCE};
use crate::components::history_view::OpenedSession;
use crate::components::nav::{
    InspectPaneFlags, is_primary_active, pane_label, select_primary, tab_group, visible_panes,
};
use crate::components::{
    ChainsView, ChatView, DocumentsView, FleetView, Header, HistoryView, MetricsView,
    OrchestrationView, PluginsView, ReasoningView, SessionPanel, SettingsView, SimpleHeader,
    SpawnRequest, StatusBar, WorkflowView,
};
use crate::state::agents::SubAgent;
use crate::state::streaming::EpochCounter;
use crate::state::{ChatState, ConfigState, TokenTotals};

/// Which top-level tab is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveTab {
    Chat,
    History,
    Agents,
    Reasoning,
    Documents,
    Chains,
    Orchestration,
    Workflow,
    Metrics,
    Plugins,
    Settings,
}

/// Props for the application shell (header, tabs, main view router).
#[derive(Props, Clone, PartialEq)]
pub struct AppShellProps {
    pub is_connected: bool,
    pub config: Signal<ConfigState>,
    pub chat: Signal<ChatState>,
    pub active_tab: Signal<ActiveTab>,
    /// Live sessions from `GET /api/agent/sessions`.
    pub live_sessions: Signal<Vec<LiveSessionInfo>>,
    pub live_sessions_error: Signal<Option<String>>,
    pub live_sessions_loading: Signal<bool>,
    pub agents: Signal<Vec<SubAgent>>,
    pub endpoints: Signal<Vec<EndpointInfo>>,
    pub show_session_panel: Signal<bool>,
    pub turn_in_flight: Signal<bool>,
    pub stream_epoch: Signal<EpochCounter>,
    pub providers: Vec<ProviderCatalogEntry>,
    pub all_models: Vec<ModelInfo>,
    pub api_client: ApiClient,
    pub file_browsing_enabled: bool,
    pub current_mode: InteractionMode,
    pub fleet_label: String,
    pub inspect_panes: InspectPaneFlags,
    pub reasoning_patterns: Vec<ReasoningPatternInfo>,
    pub on_mode_change: EventHandler<InteractionMode>,
    #[props(optional)]
    pub on_settings_changed: Option<EventHandler<()>>,
    pub on_send: EventHandler<(String, Vec<AttachedFileContext>)>,
    pub on_stop: EventHandler<()>,
    pub on_resume: EventHandler<OpenedSession>,
    pub on_reload_sessions: EventHandler<()>,
    pub on_spawn_task: EventHandler<SpawnRequest>,
    pub on_cancel_task: EventHandler<String>,
    pub on_open_task_output: EventHandler<String>,
    pub on_connect_endpoint: EventHandler<String>,
}

/// Tab bar, disconnected screen, per-tab views, and optional session inspector.
#[component]
pub fn AppShell(props: AppShellProps) -> Element {
    let mut active_tab = props.active_tab;
    let mut show_session_panel = props.show_session_panel;
    let chat_state = props.chat;
    let config_state = props.config;
    let live_sessions = props.live_sessions;
    let is_connected = props.is_connected;
    let settings_open = *active_tab.read() == ActiveTab::Settings;
    let show_main = is_connected || settings_open;
    let api_client = props.api_client.clone();
    let providers = props.providers.clone();
    let all_models = props.all_models.clone();
    let fleet_label = props.fleet_label.clone();

    rsx! {
        div { class: "app",
            style: "height: 100%; max-height: 100%; background: var(--background); color: var(--text-color); display: flex; flex-direction: column; overflow: hidden;",

            a {
                class: "skip-link",
                href: "#main-content",
                "skip to main content"
            }

            if is_connected {
                Header {
                    config: config_state,
                    providers: providers.clone(),
                    all_models: all_models.clone(),
                    mode: props.current_mode,
                    on_mode_change: props.on_mode_change,
                    on_settings_changed: props.on_settings_changed,
                    reasoning_patterns: props.reasoning_patterns.clone(),
                }
            } else {
                SimpleHeader {}
            }

            if is_connected {
                nav { class: "tab-bar",
                    aria_label: "Primary",
                    role: "tablist",
                    button {
                        class: if *active_tab.read() == ActiveTab::Chat { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if *active_tab.read() == ActiveTab::Chat { "true" } else { "false" },
                        onclick: move |_| active_tab.set(ActiveTab::Chat),
                        "chat"
                        if chat_state.read().is_streaming && *active_tab.read() != ActiveTab::Chat {
                            span { class: "tab-streaming-dot", title: "streaming in progress", aria_label: "streaming" }
                        }
                    }
                    button {
                        class: if *active_tab.read() == ActiveTab::History { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if *active_tab.read() == ActiveTab::History { "true" } else { "false" },
                        onclick: move |_| active_tab.set(ActiveTab::History),
                        {
                            let count = live_sessions.read().len();
                            if count > 0 { format!("history ({count})") } else { "history".to_string() }
                        }
                    }

                    span { class: "tab-divider", role: "presentation" }

                    button {
                        class: if is_primary_active(ActiveTab::Agents, *active_tab.read()) { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if is_primary_active(ActiveTab::Agents, *active_tab.read()) { "true" } else { "false" },
                        onclick: move |_| {
                            let next = select_primary(*active_tab.read(), ActiveTab::Agents);
                            active_tab.set(next);
                        },
                        "{fleet_label}"
                    }
                    button {
                        class: if is_primary_active(ActiveTab::Metrics, *active_tab.read()) { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if is_primary_active(ActiveTab::Metrics, *active_tab.read()) { "true" } else { "false" },
                        onclick: move |_| {
                            let next = select_primary(*active_tab.read(), ActiveTab::Metrics);
                            active_tab.set(next);
                        },
                        "inspect"
                    }

                    span { class: "tab-divider", role: "presentation" }

                    button {
                        class: if *active_tab.read() == ActiveTab::Plugins { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if *active_tab.read() == ActiveTab::Plugins { "true" } else { "false" },
                        onclick: move |_| active_tab.set(ActiveTab::Plugins),
                        "plugins"
                    }
                    button {
                        class: if *active_tab.read() == ActiveTab::Settings { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if *active_tab.read() == ActiveTab::Settings { "true" } else { "false" },
                        onclick: move |_| active_tab.set(ActiveTab::Settings),
                        "settings"
                    }
                }

                {
                    let current = *active_tab.read();
                    let panes = visible_panes(tab_group(current), props.inspect_panes);
                    rsx! {
                        if !panes.is_empty() {
                            nav { class: "pane-bar",
                                aria_label: "Section",
                                role: "tablist",
                                for pane in panes.iter().copied() {
                                    button {
                                        class: if current == pane { "pane-btn active" } else { "pane-btn" },
                                        role: "tab",
                                        aria_selected: if current == pane { "true" } else { "false" },
                                        onclick: move |_| active_tab.set(pane),
                                        "{pane_label(pane)}"
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                nav { class: "tab-bar",
                    aria_label: "Primary",
                    role: "tablist",
                    button {
                        class: if settings_open { "tab-btn active" } else { "tab-btn" },
                        role: "tab",
                        aria_selected: if settings_open { "true" } else { "false" },
                        onclick: move |_| active_tab.set(ActiveTab::Settings),
                        "settings"
                    }
                }
            }

            if is_connected && *active_tab.read() == ActiveTab::Chat {
                div { style: "display: flex; align-items: stretch; flex-shrink: 0;",
                    div { style: "flex: 1; min-width: 0;",
                        StatusBar {
                            session_id: chat_state.read().current_session_id.clone(),
                            api_client: api_client.clone(),
                        }
                    }
                    if chat_state.read().current_session_id.is_some() {
                        button {
                            class: if *show_session_panel.read() { "session-panel-btn active" } else { "session-panel-btn" },
                            title: "Toggle session inspector (memory, checkpoints, diff, plan)",
                            onclick: move |_| {
                                let cur = *show_session_panel.read();
                                show_session_panel.set(!cur);
                            },
                            if *show_session_panel.read() { "inspector ◂" } else { "inspector ▸" }
                        }
                    }
                }
            }

            main {
                id: "main-content",
                role: "main",
                style: "flex: 1; display: flex; flex-direction: row; width: 100%; min-height: 0; overflow: hidden;",

                if !show_main {
                    div { class: "connecting-screen",
                        div { class: "connecting-screen-box",
                            div { style: "color: var(--text-color);", "API unreachable or still starting" }
                            div { style: "color: var(--muted-color); font-size: var(--font-size-xs);",
                                "{config_state.read().api_base_url}"
                            }
                            div { style: "color: var(--border-color); font-size: var(--font-size-xs); margin-top: 0.5em;",
                                "retrying every 5s"
                            }
                            div { style: "color: var(--muted-color); font-size: var(--font-size-xs); margin-top: 0.75em; max-width: 56ch; line-height: 1.35;",
                                "{DISCONNECTED_DETAIL}"
                            }
                            button {
                                class: "tab-btn",
                                r#type: "button",
                                onclick: move |_| active_tab.set(ActiveTab::Settings),
                                "settings"
                            }
                        }
                    }
                }

                div { style: if show_main { "flex: 1; display: flex; flex-direction: column; min-width: 0; overflow: hidden;" } else { "display: none;" },
                    match *active_tab.read() {
                        ActiveTab::Chat => rsx! {
                            ChatView {
                                chat: chat_state,
                                config: config_state,
                                api_client: api_client.clone(),
                                file_browsing_enabled: props.file_browsing_enabled,
                                turn_locked: *props.turn_in_flight.read(),
                                on_send: props.on_send,
                                on_stop: props.on_stop,
                            }
                        },
                        ActiveTab::History => rsx! {
                            HistoryView {
                                sessions: live_sessions,
                                error: props.live_sessions_error,
                                loading: props.live_sessions_loading,
                                address: config_state.read().api_base_url.clone(),
                                on_open: props.on_resume,
                                on_retry: props.on_reload_sessions,
                                on_start_chat: move |_| active_tab.set(ActiveTab::Chat),
                            }
                        },
                        ActiveTab::Agents => rsx! {
                            FleetView {
                                api_client: api_client.clone(),
                                api_base_url: config_state.read().api_base_url.clone(),
                                agents: props.agents,
                                endpoints: props.endpoints,
                                providers: providers.clone(),
                                all_models: all_models.clone(),
                                current_provider: config_state.read().provider.clone(),
                                current_model: config_state.read().model.clone(),
                                on_spawn_task: props.on_spawn_task,
                                on_cancel_task: props.on_cancel_task,
                                on_open_task_output: props.on_open_task_output,
                                on_connect_endpoint: props.on_connect_endpoint,
                            }
                        },
                        ActiveTab::Metrics => rsx! {
                            MetricsView {
                                api_client: api_client.clone(),
                                session_id: chat_state.read().current_session_id.clone(),
                                api_base_url: config_state.read().api_base_url.clone(),
                                on_start_chat: move |_| active_tab.set(ActiveTab::Chat),
                            }
                        },
                        ActiveTab::Reasoning => {
                            if props.inspect_panes.reasoning {
                                rsx! {
                                    ReasoningView {
                                        api_client: api_client.clone(),
                                        reasoning_patterns: props.reasoning_patterns.clone(),
                                    }
                                }
                            } else {
                                unavailable_pane(active_tab)
                            }
                        },
                        ActiveTab::Documents => {
                            if props.inspect_panes.documents {
                                rsx! {
                                    DocumentsView {
                                        api_client: api_client.clone(),
                                        config: config_state,
                                    }
                                }
                            } else {
                                unavailable_pane(active_tab)
                            }
                        },
                        ActiveTab::Chains => {
                            if props.inspect_panes.chains {
                                rsx! {
                                    ChainsView {
                                        api_client: api_client.clone(),
                                    }
                                }
                            } else {
                                unavailable_pane(active_tab)
                            }
                        },
                        ActiveTab::Orchestration => rsx! {
                            OrchestrationView {
                                api_client: api_client.clone(),
                            }
                        },
                        ActiveTab::Workflow => rsx! {
                            WorkflowView {
                                api_client: api_client.clone(),
                            }
                        },
                        ActiveTab::Plugins => rsx! {
                            PluginsView {
                                api_client: api_client.clone(),
                            }
                        },
                        ActiveTab::Settings => rsx! {
                            SettingsView {
                                config: config_state,
                                providers: providers,
                                all_models: all_models,
                                api_client: api_client.clone(),
                                api_base_url: config_state.read().api_base_url.clone(),
                                on_settings_changed: props.on_settings_changed.clone(),
                            }
                        },
                    }
                }

                if is_connected && *active_tab.read() == ActiveTab::Chat && *show_session_panel.read() {
                    SessionPanel {
                        key: "{chat_state.read().current_session_id.clone().unwrap_or_default()}",
                        session_id: chat_state.read().current_session_id.clone(),
                        api_client: api_client.clone(),
                        on_close: move |_| show_session_panel.set(false),
                        on_end_session: {
                            let mut chat = chat_state;
                            let mut panel = show_session_panel;
                            let mut stream_epoch = props.stream_epoch;
                            move |_sid: String| {
                                stream_epoch.write().bump();
                                let mut c = chat.write();
                                c.clear_conversation();
                                c.current_session_id = None;
                                c.session_created = false;
                                c.is_streaming = false;
                                c.streaming_message_id = None;
                                c.total_tokens = TokenTotals::default();
                                panel.set(false);
                            }
                        },
                    }
                }
            }
        }
    }
}

fn unavailable_pane(mut active_tab: Signal<ActiveTab>) -> Element {
    rsx! {
        EmptyHint {
            body: PANE_UNAVAILABLE_SENTENCE.to_string(),
            action: "metrics".to_string(),
            on_action: move |_| active_tab.set(ActiveTab::Metrics),
        }
    }
}
