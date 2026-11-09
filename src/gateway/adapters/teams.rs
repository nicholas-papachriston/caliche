use crate::gateway::adapters::{AdapterCapabilities, AdapterTransport, GatewayAdapter};
use crate::gateway::types::{
    ChannelKind, GatewayIngressEnvelope, GatewayOutboundEnvelope, GatewaySessionRef, IngressPayload,
};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamsAdapterConfig {
    /// Incoming webhook URL used for outbound delivery.
    pub webhook_url: Option<String>,
}

impl Default for TeamsAdapterConfig {
    fn default() -> Self {
        Self {
            webhook_url: caliche_contracts::credential_env::var("TEAMS_WEBHOOK_URL").ok(),
        }
    }
}

/// Microsoft Teams adapter for Bot Framework-style ingress and webhook delivery.
#[derive(Debug, Clone)]
pub struct TeamsAdapter {
    id: String,
    config: TeamsAdapterConfig,
}

impl TeamsAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self::with_config(id, TeamsAdapterConfig::default())
    }

    pub fn with_config(id: impl Into<String>, config: TeamsAdapterConfig) -> Self {
        Self {
            id: id.into(),
            config,
        }
    }

    pub fn from_env(id: impl Into<String>) -> Self {
        Self::with_config(id, TeamsAdapterConfig::default())
    }

    pub fn set_webhook_url(&mut self, webhook_url: Option<String>) {
        self.config.webhook_url = webhook_url;
    }

    pub fn config(&self) -> &TeamsAdapterConfig {
        &self.config
    }
}

#[async_trait]
impl GatewayAdapter for TeamsAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Teams
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
        if let Ok(env) = GatewayIngressEnvelope::deserialize(&raw_payload) {
            return Ok(env);
        }

        let event_type = raw_payload
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("message");
        if event_type != "message" {
            return Err(anyhow!("unsupported teams activity type: {event_type}"));
        }

        let event_id = raw_payload
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(|v| format!("teams:{v}"))
            .unwrap_or_else(|| format!("teams:{}", Uuid::now_v7()));
        let conversation_id = raw_payload
            .get("conversation")
            .and_then(|v| v.get("id"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("teams payload missing conversation.id"))?
            .to_string();
        let user = raw_payload
            .get("from")
            .ok_or_else(|| anyhow!("teams payload missing from"))?;
        let user_id = user
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let user_display_name = user
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);
        let text = raw_payload
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        let tenant_id = raw_payload
            .get("conversation")
            .and_then(|v| v.get("tenantId"))
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .or_else(|| {
                raw_payload
                    .get("channelData")
                    .and_then(|v| v.get("tenant"))
                    .and_then(|v| v.get("id"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
            });

        let attachments = raw_payload
            .get("attachments")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        v.get("contentUrl")
                            .and_then(serde_json::Value::as_str)
                            .or_else(|| v.get("content").and_then(serde_json::Value::as_str))
                    })
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut metadata = HashMap::new();
        if let Some(reply_to) = raw_payload
            .get("replyToId")
            .and_then(serde_json::Value::as_str)
        {
            metadata.insert(
                "reply_to_id".to_string(),
                serde_json::Value::String(reply_to.to_string()),
            );
        }
        if let Some(service_url) = raw_payload
            .get("serviceUrl")
            .and_then(serde_json::Value::as_str)
        {
            metadata.insert(
                "service_url".to_string(),
                serde_json::Value::String(service_url.to_string()),
            );
        }

        Ok(GatewayIngressEnvelope {
            event_id: event_id.clone(),
            idempotency_key: Some(event_id),
            channel_kind: ChannelKind::Teams,
            channel_id: conversation_id.clone(),
            adapter_id: self.id.clone(),
            tenant_id,
            user_id,
            user_display_name,
            received_at: Utc::now(),
            session_ref: GatewaySessionRef {
                session_id: None,
                thread_id: Some(conversation_id),
                conversation_id: None,
            },
            payload: IngressPayload {
                text: text.clone(),
                attachments,
                is_command: text.starts_with('/'),
                reply_to_message_id: raw_payload
                    .get("replyToId")
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string),
            },
            metadata,
        })
    }

    async fn deliver(&self, envelope: &GatewayOutboundEnvelope) -> Result<()> {
        #[cfg(feature = "reqwest")]
        {
            let webhook_url = envelope
                .metadata
                .get("webhook_url")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
                .or_else(|| self.config.webhook_url.clone())
                .or_else(|| {
                    if envelope.channel_id.starts_with("http://")
                        || envelope.channel_id.starts_with("https://")
                    {
                        Some(envelope.channel_id.clone())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| anyhow!("teams webhook url is not configured"))?;

            let body = serde_json::json!({
                "text": envelope.payload.text,
            });
            let validated = crate::security::http_policy::HttpPolicy::from_env()?
                .validate_resolved_url(&webhook_url)?;
            let client = validated.pinned_client()?;
            let resp = client
                .post(validated.url.as_str())
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .context("teams webhook post failed")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("teams webhook post failed: {status} {text}"));
            }
            Ok(())
        }
        #[cfg(not(feature = "reqwest"))]
        {
            let _ = envelope;
            Err(anyhow!("teams delivery requires reqwest feature"))
        }
    }
}
