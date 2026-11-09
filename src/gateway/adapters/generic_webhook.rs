use crate::gateway::adapters::{AdapterCapabilities, AdapterTransport, GatewayAdapter};
use crate::gateway::types::{ChannelKind, GatewayIngressEnvelope, GatewayOutboundEnvelope};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use std::sync::Arc;

/// Generic webhook adapter used as universal fallback integration.
///
/// Delivers outbound envelopes via HTTP POST to the `channel_id` URL (must be
/// an `http://` or `https://` URL). The envelope is serialized as JSON in the
/// request body.
#[derive(Debug, Clone)]
pub struct GenericWebhookAdapter {
    id: String,
    #[expect(
        dead_code,
        reason = "retained for with_client; deliver uses a DNS-pinned client"
    )]
    http_client: Arc<reqwest::Client>,
}

impl GenericWebhookAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            http_client: Arc::new(reqwest::Client::new()),
        }
    }

    /// Create with a shared HTTP client (avoids creating a new connection pool per adapter).
    pub fn with_client(id: impl Into<String>, http_client: Arc<reqwest::Client>) -> Self {
        Self {
            id: id.into(),
            http_client,
        }
    }
}

#[async_trait]
impl GatewayAdapter for GenericWebhookAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Webhook
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_text: true,
            supports_files: true,
            supports_voice: false,
            supports_threading: true,
            transport: AdapterTransport::Webhook,
            supports_ack: true,
        }
    }

    async fn normalize_ingress(
        &self,
        raw_payload: serde_json::Value,
    ) -> Result<GatewayIngressEnvelope> {
        serde_json::from_value(raw_payload)
            .map_err(|e| anyhow!("failed to decode webhook ingress envelope: {e}"))
    }

    async fn deliver(&self, envelope: &GatewayOutboundEnvelope) -> Result<()> {
        #[cfg(feature = "reqwest")]
        {
            use crate::security::http_policy::HttpPolicy;
            if !envelope.channel_id.starts_with("http://")
                && !envelope.channel_id.starts_with("https://")
            {
                return Err(anyhow!(
                    "webhook adapter expects a URL channel_id (http:// or https://), got '{}'",
                    envelope.channel_id
                ));
            }

            let validated = HttpPolicy::from_env()?.validate_resolved_url(&envelope.channel_id)?;
            let client = validated.pinned_client()?;
            let resp = client
                .post(validated.url.as_str())
                .json(envelope)
                .send()
                .await
                .map_err(|e| anyhow!("webhook delivery failed: {e}"))?;

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(anyhow!(
                    "webhook delivery returned non-success status {status}: {body}"
                ));
            }

            tracing::debug!(
                adapter_id = %self.id,
                url = %envelope.channel_id,
                status = %status,
                "webhook delivered"
            );

            Ok(())
        }
        #[cfg(not(feature = "reqwest"))]
        {
            let _ = envelope;
            Err(anyhow!(
                "generic webhook delivery requires the reqwest feature (HTTP SSRF policy)"
            ))
        }
    }
}

#[cfg(all(test, feature = "reqwest"))]
mod tests {
    use super::GenericWebhookAdapter;
    use crate::gateway::adapters::GatewayAdapter;
    use crate::gateway::types::{ChannelKind, GatewayOutboundEnvelope, OutboundPayload};
    use chrono::Utc;
    use std::collections::HashMap;

    #[tokio::test]
    async fn deliver_rejects_loopback_channel_id() {
        let adapter = GenericWebhookAdapter::new("webhook-test");
        let envelope = GatewayOutboundEnvelope {
            outbound_id: "out-1".into(),
            channel_kind: ChannelKind::Webhook,
            channel_id: "http://127.0.0.1/ssrf".into(),
            adapter_id: "webhook-test".into(),
            tenant_id: None,
            user_id: "user".into(),
            created_at: Utc::now(),
            payload: OutboundPayload {
                text: "hi".into(),
                attachments: Vec::new(),
                reply_to_message_id: None,
            },
            metadata: HashMap::new(),
        };
        let err = adapter
            .deliver(&envelope)
            .await
            .expect_err("loopback webhook delivery");
        assert!(
            err.to_string().contains("blocked") || err.to_string().contains("allowlist"),
            "unexpected error: {err}"
        );
    }
}
