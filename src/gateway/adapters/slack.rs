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
pub struct SlackAdapterConfig {
    pub bot_token: Option<String>,
    pub api_base_url: String,
}

impl Default for SlackAdapterConfig {
    fn default() -> Self {
        Self {
            bot_token: None,
            api_base_url: "https://slack.com/api".to_owned(),
        }
    }
}

/// Slack adapter for Event API ingress and chat.postMessage delivery.
#[derive(Debug, Clone)]
pub struct SlackAdapter {
    id: String,
    config: SlackAdapterConfig,
}

impl SlackAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self::with_config(id, SlackAdapterConfig::default())
    }

    pub fn with_config(id: impl Into<String>, config: SlackAdapterConfig) -> Self {
        Self {
            id: id.into(),
            config,
        }
    }

    pub fn from_env(id: impl Into<String>) -> Self {
        Self::with_config(
            id,
            SlackAdapterConfig {
                bot_token: caliche_contracts::credential_env::var("SLACK_BOT_TOKEN").ok(),
                api_base_url: caliche_contracts::credential_env::var("SLACK_API_BASE_URL")
                    .unwrap_or_else(|_| "https://slack.com/api".to_owned()),
            },
        )
    }

    pub fn set_bot_token(&mut self, token: impl Into<String>) {
        self.config.bot_token = Some(token.into());
    }

    pub fn set_api_base_url(&mut self, api_base_url: impl Into<String>) {
        self.config.api_base_url = api_base_url.into();
    }

    pub fn config(&self) -> &SlackAdapterConfig {
        &self.config
    }
}

#[async_trait]
impl GatewayAdapter for SlackAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Slack
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

        let event = raw_payload
            .get("event")
            .ok_or_else(|| anyhow!("slack payload missing event"))?;
        let channel_id = event
            .get("channel")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow!("slack payload missing event.channel"))?
            .to_string();
        let user_id = event
            .get("user")
            .and_then(serde_json::Value::as_str)
            .or_else(|| event.get("bot_id").and_then(serde_json::Value::as_str))
            .unwrap_or("unknown")
            .to_string();
        let text = event
            .get("text")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                event
                    .get("message")
                    .and_then(|v| v.get("text"))
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("")
            .to_string();
        let thread_ts = event
            .get("thread_ts")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string)
            .or_else(|| {
                event
                    .get("ts")
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
            });

        let attachments = event
            .get("files")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|file| {
                        file.get("url_private_download")
                            .and_then(serde_json::Value::as_str)
                            .or_else(|| file.get("url_private").and_then(serde_json::Value::as_str))
                    })
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let event_id = raw_payload
            .get("event_id")
            .and_then(serde_json::Value::as_str)
            .map(|v| format!("slack:{v}"))
            .unwrap_or_else(|| format!("slack:{}", Uuid::now_v7()));

        let team_id = raw_payload
            .get("team_id")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        let mut metadata = HashMap::new();
        if let Some(event_type) = event.get("type").and_then(serde_json::Value::as_str) {
            metadata.insert(
                "event_type".to_string(),
                serde_json::Value::String(event_type.to_string()),
            );
        }
        if let Some(subtype) = event.get("subtype").and_then(serde_json::Value::as_str) {
            metadata.insert(
                "event_subtype".to_string(),
                serde_json::Value::String(subtype.to_string()),
            );
        }
        if let Some(ts) = event.get("ts").and_then(serde_json::Value::as_str) {
            metadata.insert(
                "event_ts".to_string(),
                serde_json::Value::String(ts.to_string()),
            );
        }

        Ok(GatewayIngressEnvelope {
            event_id: event_id.clone(),
            idempotency_key: Some(event_id),
            channel_kind: ChannelKind::Slack,
            channel_id,
            adapter_id: self.id.clone(),
            tenant_id: team_id,
            user_id,
            user_display_name: None,
            received_at: Utc::now(),
            session_ref: GatewaySessionRef {
                session_id: None,
                thread_id: thread_ts,
                conversation_id: None,
            },
            payload: IngressPayload {
                text: text.clone(),
                attachments,
                is_command: text.starts_with('/'),
                reply_to_message_id: None,
            },
            metadata,
        })
    }

    async fn deliver(&self, envelope: &GatewayOutboundEnvelope) -> Result<()> {
        #[cfg(feature = "reqwest")]
        {
            let token = self
                .config
                .bot_token
                .as_ref()
                .ok_or_else(|| anyhow!("SLACK_BOT_TOKEN is not configured"))?;
            let url = format!(
                "{}/chat.postMessage",
                self.config.api_base_url.trim_end_matches('/')
            );

            let mut body = serde_json::json!({
                "channel": envelope.channel_id,
                "text": envelope.payload.text,
                "unfurl_links": false,
                "unfurl_media": false
            });

            if let Some(thread_ts) = envelope
                .metadata
                .get("thread_ts")
                .and_then(serde_json::Value::as_str)
            {
                body["thread_ts"] = serde_json::Value::String(thread_ts.to_string());
            }

            let client = reqwest::Client::new();
            let resp = client
                .post(url)
                .bearer_auth(token)
                .header("Content-Type", "application/json; charset=utf-8")
                .json(&body)
                .send()
                .await
                .context("slack chat.postMessage request failed")?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("slack chat.postMessage failed: {status} {text}"));
            }

            let payload: serde_json::Value = resp
                .json()
                .await
                .context("failed to parse slack chat.postMessage response")?;
            if payload
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                Ok(())
            } else {
                let err = payload
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown_error");
                Err(anyhow!("slack chat.postMessage returned ok=false: {err}"))
            }
        }
        #[cfg(not(feature = "reqwest"))]
        {
            let _ = envelope;
            Err(anyhow!("slack delivery requires reqwest feature"))
        }
    }
}
