use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Routing context that can carry policy hints (used by router and routing rules).
#[derive(Debug, Clone, Default)]
pub struct RouteContext {
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Supported gateway channel categories.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ChannelKind {
    Telegram,
    Discord,
    Teams,
    Twilio,
    Slack,
    WhatsApp,
    Webhook,
    Cli,
    Api,
    Custom,
}

/// Optional reference to an existing session context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatewaySessionRef {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub conversation_id: Option<String>,
}

/// Inbound payload normalized from channel-specific events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IngressPayload {
    pub text: String,
    pub attachments: Vec<String>,
    pub is_command: bool,
    pub reply_to_message_id: Option<String>,
}

/// Canonical ingress envelope used by router and session-binding layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayIngressEnvelope {
    pub event_id: String,
    pub idempotency_key: Option<String>,
    pub channel_kind: ChannelKind,
    pub channel_id: String,
    pub adapter_id: String,
    pub tenant_id: Option<String>,
    pub user_id: String,
    pub user_display_name: Option<String>,
    pub received_at: DateTime<Utc>,
    pub session_ref: GatewaySessionRef,
    pub payload: IngressPayload,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Outbound payload independent from channel transport details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutboundPayload {
    pub text: String,
    pub attachments: Vec<String>,
    pub reply_to_message_id: Option<String>,
}

/// Routing/delivery envelope emitted after agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayOutboundEnvelope {
    pub outbound_id: String,
    pub channel_kind: ChannelKind,
    pub channel_id: String,
    pub adapter_id: String,
    pub tenant_id: Option<String>,
    pub user_id: String,
    pub created_at: DateTime<Utc>,
    pub payload: OutboundPayload,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Operational events emitted by the gateway lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayEventEnvelope {
    pub event_id: String,
    pub stage: String,
    pub timestamp: DateTime<Utc>,
    pub tenant_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub details: HashMap<String, serde_json::Value>,
}

/// Delivery policy controlling retries, backoff, and dead-letter behavior.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeliveryPolicy {
    pub max_attempts: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub jitter: bool,
    pub dead_letter_on_failure: bool,
}

impl Default for DeliveryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 250,
            max_backoff_ms: 5_000,
            jitter: true,
            dead_letter_on_failure: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeliveryPolicy;

    #[test]
    fn delivery_policy_default_matches_contract() {
        let p = DeliveryPolicy::default();
        assert_eq!(p.max_attempts, 3);
        assert_eq!(p.initial_backoff_ms, 250);
        assert_eq!(p.max_backoff_ms, 5_000);
        assert!(p.jitter);
        assert!(p.dead_letter_on_failure);
    }
}
