//! API client module for communicating with caliche_server.

pub mod client;
pub mod types;
pub mod ws_client;

pub use client::ApiClient;
pub use types::*;
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use ws_client::{
    AgentSettings as WsAgentSettings, ClientMessage as WsClientMessage,
    ServerMessage as WsServerMessage, ws_url_from_base,
};
#[allow(unused_imports)]
pub use ws_client::{WsClient, WsConnectionState};
