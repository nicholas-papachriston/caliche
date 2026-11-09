use crate::gateway::types::{ChannelKind, GatewayIngressEnvelope, GatewayOutboundEnvelope};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

mod discord;
mod generic_webhook;
mod slack;
mod teams;
mod telegram;
mod twilio;

pub use discord::{DiscordAdapter, DiscordAdapterConfig};
pub use generic_webhook::GenericWebhookAdapter;
pub use slack::{SlackAdapter, SlackAdapterConfig};
pub use teams::{TeamsAdapter, TeamsAdapterConfig};
pub use telegram::{TelegramAdapter, TelegramAdapterConfig};
pub use twilio::{TwilioAdapter, TwilioAdapterConfig};

/// Transport mode used by adapter ingress.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdapterTransport {
    Poll,
    Webhook,
    Socket,
}

/// Adapter capability metadata used for routing and policy selection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdapterCapabilities {
    pub supports_text: bool,
    pub supports_files: bool,
    pub supports_voice: bool,
    pub supports_threading: bool,
    pub transport: AdapterTransport,
    pub supports_ack: bool,
}

/// Trait for channel-specific ingress/egress implementations.
#[async_trait]
pub trait GatewayAdapter: Send + Sync {
    fn id(&self) -> &str;
    fn channel_kind(&self) -> ChannelKind;
    fn capabilities(&self) -> AdapterCapabilities;

    /// Normalize channel-native payload into gateway ingress envelope.
    async fn normalize_ingress(
        &self,
        raw_payload: serde_json::Value,
    ) -> Result<GatewayIngressEnvelope>;

    /// Send a normalized outbound payload to this channel.
    async fn deliver(&self, envelope: &GatewayOutboundEnvelope) -> Result<()>;
}
