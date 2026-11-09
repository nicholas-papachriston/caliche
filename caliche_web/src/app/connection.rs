//! API connection lifecycle: health checks, catalog merge, and reconnect loops.

use dioxus::prelude::*;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use crate::api::{AgentSettings, ApiClient, ModelInfo, ProviderCatalogEntry};
use crate::app::settings_merge::{
    merge_current_settings_into_model_options, should_adopt_server_settings,
};
use crate::state::ConfigState;

/// Reconnection interval in seconds.
pub const RECONNECT_INTERVAL_SECS: u64 = 5;

/// True when `url` looks like a loopback HTTP(S) URL (`localhost`, `127.0.0.1`,
/// or `[::1]`). Used to decide whether a stale persisted `api_base_url` is safe
/// to auto-rewrite to the page origin without disturbing remote API setups.
#[cfg(target_arch = "wasm32")]
pub fn is_loopback_url(url: &str) -> bool {
    let trimmed = url
        .trim()
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    let host = trimmed
        .split('/')
        .next()
        .unwrap_or(trimmed)
        .rsplit_once(':')
        .map(|(host, _port)| host)
        .unwrap_or(trimmed);
    matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1")
}

/// Initial config: in the browser use current origin so API/WS match the page host and port.
pub fn initial_config() -> ConfigState {
    #[cfg(target_arch = "wasm32")]
    {
        let base = web_sys::window()
            .and_then(|w| w.location().origin().ok())
            .unwrap_or_else(|| "http://localhost:3000".to_string());
        ConfigState {
            api_base_url: base,
            ..ConfigState::default()
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    ConfigState::default()
}

/// Merge persisted or restored config with platform defaults and loopback self-heal on WASM.
pub fn resolve_effective_initial_config(restored_config: Option<ConfigState>) -> ConfigState {
    #[cfg(target_arch = "wasm32")]
    let mut effective = restored_config.unwrap_or_else(initial_config);
    #[cfg(not(target_arch = "wasm32"))]
    let effective = restored_config.unwrap_or_else(initial_config);

    #[cfg(target_arch = "wasm32")]
    {
        if let Some(origin) = web_sys::window().and_then(|w| w.location().origin().ok())
            && origin != effective.api_base_url
            && is_loopback_url(&origin)
            && is_loopback_url(&effective.api_base_url)
        {
            tracing::info!(
                stored = %effective.api_base_url,
                origin = %origin,
                "rewriting stored api_base_url to match page origin (loopback self-heal)",
            );
            effective.api_base_url = origin;
        }
    }

    effective
}

/// Application connection state (API client and discovered catalog).
#[derive(Clone)]
pub struct AppState {
    pub api_client: ApiClient,
    pub providers: Vec<ProviderCatalogEntry>,
    pub all_models: Vec<ModelInfo>,
    pub is_connected: bool,
    pub file_browsing_enabled: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            api_client: ApiClient::default(),
            providers: Vec::new(),
            all_models: Vec::new(),
            is_connected: false,
            file_browsing_enabled: false,
        }
    }
}

/// Attempt to connect to the API and fetch models and settings.
pub async fn try_connect(
    client: &ApiClient,
) -> Option<(
    Vec<ProviderCatalogEntry>,
    Vec<ModelInfo>,
    bool,
    Option<crate::api::AgentSettings>,
)> {
    let is_connected = client.health_check().await.unwrap_or(false);

    if !is_connected {
        return None;
    }

    let catalog = client.get_model_catalog().await.unwrap_or_default();
    let mut providers = catalog.providers;
    let mut models = catalog.models;

    let mut settings = client.get_settings().await.ok();
    merge_current_settings_into_model_options(&mut providers, &mut models, &mut settings);

    let file_browsing = client
        .is_file_browsing_enabled()
        .await
        .map(|status| status.enabled)
        .unwrap_or(false);

    Some((providers, models, file_browsing, settings))
}

/// Reset the WASM WebSocket client when the API base URL changes.
#[cfg(target_arch = "wasm32")]
pub(crate) fn spawn_ws_url_watch(
    config_state: Signal<ConfigState>,
    mut ws_client_ref: Signal<Option<crate::api::WsClient>>,
    mut stream_gate: Signal<Option<super::streaming::TurnGate>>,
) {
    use dioxus::prelude::*;
    let mut last_url = use_signal(|| config_state.read().api_base_url.clone());
    use_effect(move || {
        let url = config_state.read().api_base_url.clone();
        if url == *last_url.read() {
            return;
        }
        tracing::info!(from = %last_url.read(), to = %url, "API URL changed — resetting WebSocket client");
        stream_gate.set(None);
        if let Some(client) = ws_client_ref.write().take() {
            client.close();
        }
        last_url.set(url);
    });
}

/// Tear down the WASM WebSocket when HTTP health checks fail (mirrors API reconnect loop).
#[cfg(target_arch = "wasm32")]
pub(crate) fn spawn_ws_connection_loop(
    app_state: Signal<AppState>,
    mut ws_client_ref: Signal<Option<crate::api::WsClient>>,
    mut stream_gate: Signal<Option<super::streaming::TurnGate>>,
    mut ws_loop_started: Signal<bool>,
) {
    use dioxus::prelude::*;

    use_effect(move || {
        if *ws_loop_started.read() {
            return;
        }
        ws_loop_started.set(true);

        spawn(async move {
            let mut was_connected = app_state.read().is_connected;
            loop {
                let api_connected = app_state.read().is_connected;
                if was_connected && !api_connected {
                    tracing::warn!("API disconnected — closing WebSocket");
                    stream_gate.set(None);
                    if let Some(client) = ws_client_ref.write().take() {
                        client.close();
                    }
                }
                was_connected = api_connected;

                gloo_timers::future::TimeoutFuture::new(
                    u32::try_from(RECONNECT_INTERVAL_SECS * 1000).unwrap_or(u32::MAX),
                )
                .await;
            }
        });
    });
}

/// Copy a non-empty server provider/model into the header until the operator edits.
fn adopt_server_settings(config_state: &mut Signal<ConfigState>, server: &AgentSettings) {
    if server.provider.trim().is_empty() && server.model.trim().is_empty() {
        return;
    }
    let (adopt, seen) = {
        let config = config_state.read();
        (
            should_adopt_server_settings(
                config.provider.as_deref().unwrap_or(""),
                config.model.as_deref().unwrap_or(""),
                &server.provider,
                &server.model,
                config.user_edited_live_settings,
            ),
            config.server_settings_seen,
        )
    };
    if !adopt && seen {
        return;
    }
    let mut config = config_state.write();
    if adopt {
        config.apply_from_settings(server);
    }
    config.server_settings_seen = true;
}

/// Start the outer/inner API reconnection loop (once per app mount).
pub fn spawn_connection_loop(
    mut config_state: Signal<ConfigState>,
    mut app_state: Signal<AppState>,
    mut connection_loop_started: Signal<bool>,
) {
    use_effect(move || {
        if *connection_loop_started.read() {
            return;
        }
        connection_loop_started.set(true);

        spawn(async move {
            loop {
                let target_url = config_state.read().api_base_url.clone();
                let client = ApiClient::new(&target_url);
                app_state.write().api_client = client.clone();

                loop {
                    let was_connected = app_state.read().is_connected;
                    let connection_result = try_connect(&client).await;

                    match connection_result {
                        Some((providers, models, file_browsing, settings)) => {
                            if !was_connected {
                                tracing::info!(
                                    "API connected to {} ({} models, files: {})",
                                    target_url,
                                    models.len(),
                                    file_browsing
                                );
                            }
                            if let Some(ref server) = settings {
                                adopt_server_settings(&mut config_state, server);
                            }
                            app_state.write().is_connected = true;
                            app_state.write().providers = providers;
                            app_state.write().all_models = models;
                            app_state.write().file_browsing_enabled = file_browsing;
                        }
                        None => {
                            if was_connected {
                                tracing::warn!("API connection lost ({})", target_url);
                            }
                            app_state.write().is_connected = false;
                        }
                    }

                    #[cfg(target_arch = "wasm32")]
                    {
                        gloo_timers::future::TimeoutFuture::new(
                            (RECONNECT_INTERVAL_SECS * 1000) as u32,
                        )
                        .await;
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        tokio::time::sleep(Duration::from_secs(RECONNECT_INTERVAL_SECS)).await;
                    }

                    let current_url = config_state.read().api_base_url.clone();
                    if current_url != target_url {
                        tracing::info!(
                            "API target changed from {} to {} — reconnecting",
                            target_url,
                            current_url
                        );
                        app_state.write().is_connected = false;
                        break;
                    }
                }
            }
        });
    });
}
