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
pub struct DiscordAdapterConfig {
    pub bot_token: Option<String>,
    pub api_base_url: String,
}

impl Default for DiscordAdapterConfig {
    fn default() -> Self {
        Self {
            bot_token: None,
            api_base_url: "https://discord.com/api/v10".to_owned(),
        }
    }
}

/// Discord adapter with programmatic config support.
#[derive(Debug, Clone)]
pub struct DiscordAdapter {
    id: String,
    config: DiscordAdapterConfig,
}

impl DiscordAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self::with_config(id, DiscordAdapterConfig::default())
    }

    pub fn with_config(id: impl Into<String>, config: DiscordAdapterConfig) -> Self {
        Self {
            id: id.into(),
            config,
        }
    }

    pub fn from_env(id: impl Into<String>) -> Self {
        Self::with_config(
            id,
            DiscordAdapterConfig {
                bot_token: caliche_contracts::credential_env::var("DISCORD_BOT_TOKEN").ok(),
                api_base_url: caliche_contracts::credential_env::var("DISCORD_API_BASE_URL")
                    .unwrap_or_else(|_| "https://discord.com/api/v10".to_owned()),
            },
        )
    }

    pub fn set_bot_token(&mut self, token: impl Into<String>) {
        self.config.bot_token = Some(token.into());
    }

    pub fn set_api_base_url(&mut self, api_base_url: impl Into<String>) {
        self.config.api_base_url = api_base_url.into();
    }

    pub fn config(&self) -> &DiscordAdapterConfig {
        &self.config
    }
}

#[async_trait]
impl GatewayAdapter for DiscordAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Discord
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_text: true,
            supports_files: true,
            supports_voice: true,
            supports_threading: true,
            transport: AdapterTransport::Socket,
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

        // Discord gateway/message create payloads.
        let id = raw_payload
            .get("id")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| Uuid::now_v7().to_string());
        let channel_id = raw_payload
            .get("channel_id")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .ok_or_else(|| anyhow!("discord payload missing channel_id"))?;
        let user_id = raw_payload
            .get("author")
            .and_then(|v| v.get("id"))
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .ok_or_else(|| anyhow!("discord payload missing author.id"))?;
        let user_display_name = raw_payload
            .get("author")
            .and_then(|v| v.get("username"))
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);
        let text = raw_payload
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        let thread_id = raw_payload
            .get("thread")
            .and_then(|v| v.get("id"))
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .or_else(|| {
                raw_payload
                    .get("message_reference")
                    .and_then(|v| v.get("channel_id"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
            });
        let attachments = raw_payload
            .get("attachments")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.get("url").and_then(serde_json::Value::as_str))
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Ok(GatewayIngressEnvelope {
            event_id: format!("discord:{id}"),
            idempotency_key: Some(format!("discord:{id}")),
            channel_kind: ChannelKind::Discord,
            channel_id,
            adapter_id: self.id.clone(),
            tenant_id: None,
            user_id,
            user_display_name,
            received_at: Utc::now(),
            session_ref: GatewaySessionRef {
                session_id: None,
                thread_id,
                conversation_id: None,
            },
            payload: IngressPayload {
                text: text.clone(),
                attachments,
                is_command: text.starts_with('/'),
                reply_to_message_id: raw_payload
                    .get("message_reference")
                    .and_then(|v| v.get("message_id"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string),
            },
            metadata: HashMap::new(),
        })
    }

    async fn deliver(&self, envelope: &GatewayOutboundEnvelope) -> Result<()> {
        #[cfg(feature = "reqwest")]
        {
            let token = self
                .config
                .bot_token
                .as_ref()
                .ok_or_else(|| anyhow!("DISCORD_BOT_TOKEN is not configured"))?;
            let channel_id = envelope.channel_id.as_str();
            let url = format!(
                "{}/channels/{}/messages",
                self.config.api_base_url.trim_end_matches('/'),
                channel_id
            );
            let mut body = serde_json::json!({
                "content": envelope.payload.text,
            });
            if let Some(reply_to) = envelope.payload.reply_to_message_id.as_deref() {
                body["message_reference"] = serde_json::json!({ "message_id": reply_to });
            }

            let client = reqwest::Client::new();
            let resp = client
                .post(url)
                .header("Authorization", format!("Bot {token}"))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .context("discord create message request failed")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("discord create message failed: {status} {text}"));
            }
            Ok(())
        }
        #[cfg(not(feature = "reqwest"))]
        {
            let _ = envelope;
            Err(anyhow!("discord delivery requires reqwest feature"))
        }
    }
}
