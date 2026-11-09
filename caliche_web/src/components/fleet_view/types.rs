use dioxus::prelude::*;

use crate::api::{ApiClient, EndpointInfo, ModelInfo, ProviderCatalogEntry};
use crate::state::agents::{AgentSource, SubAgent};

#[derive(Props, Clone, PartialEq)]
pub struct FleetViewProps {
    pub api_client: ApiClient,
    pub agents: Signal<Vec<SubAgent>>,
    /// Shared endpoint list — written here, read by the parent for the tab label.
    pub endpoints: Signal<Vec<EndpointInfo>>,
    /// Available providers from the connected API.
    pub providers: Vec<ProviderCatalogEntry>,
    /// Available models from the connected API.
    pub all_models: Vec<ModelInfo>,
    /// Currently configured UI provider.
    pub current_provider: Option<String>,
    /// Currently configured UI model.
    pub current_model: Option<String>,
    pub on_spawn_task: EventHandler<SpawnRequest>,
    pub on_cancel_task: EventHandler<String>,
    pub on_open_task_output: EventHandler<String>,
    /// Called when user clicks "connect" — switches the active API URL.
    pub on_connect_endpoint: EventHandler<String>,
    /// API address included in shared failure sentences.
    pub api_base_url: String,
}

/// Parameters for spawning a new sub-agent task.
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnRequest {
    pub task: String,
    pub source: AgentSource,
    pub provider: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone, PartialEq)]
pub(crate) struct ChatMsg {
    pub role: String,
    pub text: String,
}

#[derive(Clone, PartialEq)]
pub(crate) struct BroadcastResult {
    pub name: String,
    pub kind: String,
    pub ok: bool,
    pub text: String,
}
