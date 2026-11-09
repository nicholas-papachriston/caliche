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
pub struct TelegramAdapterConfig {
    pub bot_token: Option<String>,
    pub api_base_url: String,
    pub default_parse_mode: Option<String>,
}

impl Default for TelegramAdapterConfig {
    fn default() -> Self {
        Self {
            bot_token: None,
            api_base_url: "https://api.telegram.org".to_owned(),
            default_parse_mode: None,
        }
    }
}

/// Telegram adapter with programmatic config support.
#[derive(Debug, Clone)]
pub struct TelegramAdapter {
    id: String,
    config: TelegramAdapterConfig,
}

impl TelegramAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self::with_config(id, TelegramAdapterConfig::default())
    }

    pub fn with_config(id: impl Into<String>, config: TelegramAdapterConfig) -> Self {
        Self {
            id: id.into(),
            config,
        }
    }

    pub fn from_env(id: impl Into<String>) -> Self {
        Self::with_config(
            id,
            TelegramAdapterConfig {
                bot_token: caliche_contracts::credential_env::var("TELEGRAM_BOT_TOKEN").ok(),
                api_base_url: caliche_contracts::credential_env::var("TELEGRAM_API_BASE_URL")
                    .unwrap_or_else(|_| "https://api.telegram.org".to_owned()),
                default_parse_mode: caliche_contracts::credential_env::var("TELEGRAM_PARSE_MODE")
                    .ok(),
            },
        )
    }

    pub fn set_bot_token(&mut self, token: impl Into<String>) {
        self.config.bot_token = Some(token.into());
    }

    pub fn set_api_base_url(&mut self, api_base_url: impl Into<String>) {
        self.config.api_base_url = api_base_url.into();
    }

    pub fn set_default_parse_mode(&mut self, parse_mode: Option<String>) {
        self.config.default_parse_mode = parse_mode;
    }

    pub fn config(&self) -> &TelegramAdapterConfig {
        &self.config
    }
}

#[async_trait]
impl GatewayAdapter for TelegramAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Telegram
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            supports_text: true,
            supports_files: true,
            supports_voice: true,
            supports_threading: false,
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

        let message = raw_payload
            .get("message")
            .or_else(|| raw_payload.get("edited_message"))
            .or_else(|| raw_payload.get("channel_post"))
            .or_else(|| {
                raw_payload
                    .get("callback_query")
                    .and_then(|v| v.get("message"))
            })
            .ok_or_else(|| anyhow!("telegram update missing message payload"))?;

        let chat_id = message
            .get("chat")
            .and_then(|v| v.get("id"))
            .and_then(serde_json::Value::as_i64)
            .map(|v| v.to_string())
            .or_else(|| {
                message
                    .get("chat")
                    .and_then(|v| v.get("id"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
            })
            .ok_or_else(|| anyhow!("telegram message missing chat.id"))?;

        let user = message.get("from").or_else(|| {
            raw_payload
                .get("callback_query")
                .and_then(|v| v.get("from"))
        });
        let user_id = user
            .and_then(|v| v.get("id"))
            .and_then(serde_json::Value::as_i64)
            .map(|v| v.to_string())
            .or_else(|| {
                user.and_then(|v| v.get("id"))
                    .and_then(serde_json::Value::as_str)
                    .map(ToString::to_string)
            })
            .unwrap_or_else(|| "unknown".to_owned());

        let user_display_name = user.and_then(|u| {
            u.get("username")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
                .or_else(|| {
                    let first = u.get("first_name").and_then(serde_json::Value::as_str);
                    let last = u.get("last_name").and_then(serde_json::Value::as_str);
                    match (first, last) {
                        (Some(f), Some(l)) => Some(format!("{f} {l}")),
                        (Some(f), None) => Some(f.to_string()),
                        (None, Some(l)) => Some(l.to_string()),
                        _ => None,
                    }
                })
        });

        let text = message
            .get("text")
            .and_then(serde_json::Value::as_str)
            .or_else(|| message.get("caption").and_then(serde_json::Value::as_str))
            .or_else(|| {
                raw_payload
                    .get("callback_query")
                    .and_then(|v| v.get("data"))
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("")
            .to_string();

        let attachments = collect_telegram_attachments(message);

        let event_id = raw_payload
            .get("update_id")
            .and_then(serde_json::Value::as_i64)
            .map(|v| format!("telegram:{v}"))
            .unwrap_or_else(|| format!("telegram:{}", Uuid::now_v7()));

        let thread_id = message
            .get("message_thread_id")
            .and_then(serde_json::Value::as_i64)
            .map(|v| v.to_string());

        let reply_to = message
            .get("reply_to_message")
            .and_then(|v| v.get("message_id"))
            .and_then(serde_json::Value::as_i64)
            .map(|v| v.to_string());

        Ok(GatewayIngressEnvelope {
            event_id: event_id.clone(),
            idempotency_key: Some(event_id),
            channel_kind: ChannelKind::Telegram,
            channel_id: chat_id,
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
                is_command: text.starts_with('/'),
                text,
                attachments,
                reply_to_message_id: reply_to,
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
                .ok_or_else(|| anyhow!("TELEGRAM_BOT_TOKEN is not configured"))?;
            let url = format!(
                "{}/bot{}/sendMessage",
                self.config.api_base_url.trim_end_matches('/'),
                token
            );
            let client = reqwest::Client::new();

            let mut body = serde_json::json!({
                "chat_id": envelope.channel_id,
                "text": envelope.payload.text,
                "disable_web_page_preview": true
            });

            if let Some(thread_id) = envelope
                .metadata
                .get("thread_id")
                .and_then(serde_json::Value::as_str)
            {
                body["message_thread_id"] = serde_json::Value::String(thread_id.to_string());
            }
            if let Some(reply_to) = envelope.payload.reply_to_message_id.as_deref() {
                body["reply_to_message_id"] = serde_json::Value::String(reply_to.to_string());
            }
            if let Some(parse_mode) = self.config.default_parse_mode.as_deref() {
                body["parse_mode"] = serde_json::Value::String(parse_mode.to_string());
            }

            let resp = client
                .post(url)
                .json(&body)
                .send()
                .await
                .context("telegram sendMessage request failed")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("telegram sendMessage failed: {status} {text}"));
            }
            Ok(())
        }
        #[cfg(not(feature = "reqwest"))]
        {
            let _ = envelope;
            Err(anyhow!("telegram delivery requires reqwest feature"))
        }
    }
}

fn collect_telegram_attachments(message: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(photo_arr) = message.get("photo").and_then(serde_json::Value::as_array) {
        for photo in photo_arr {
            if let Some(file_id) = photo.get("file_id").and_then(serde_json::Value::as_str) {
                out.push(format!("telegram://file/{file_id}"));
            }
        }
    }
    if let Some(doc) = message.get("document")
        && let Some(file_id) = doc.get("file_id").and_then(serde_json::Value::as_str)
    {
        out.push(format!("telegram://file/{file_id}"));
    }
    if let Some(audio) = message.get("audio")
        && let Some(file_id) = audio.get("file_id").and_then(serde_json::Value::as_str)
    {
        out.push(format!("telegram://file/{file_id}"));
    }
    if let Some(voice) = message.get("voice")
        && let Some(file_id) = voice.get("file_id").and_then(serde_json::Value::as_str)
    {
        out.push(format!("telegram://file/{file_id}"));
    }
    out
}
