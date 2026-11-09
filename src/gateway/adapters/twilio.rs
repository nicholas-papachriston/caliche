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
pub struct TwilioAdapterConfig {
    pub account_sid: Option<String>,
    pub auth_token: Option<String>,
    pub api_base_url: String,
    pub from_address: Option<String>,
}

impl Default for TwilioAdapterConfig {
    fn default() -> Self {
        Self {
            account_sid: None,
            auth_token: None,
            api_base_url: "https://api.twilio.com".to_owned(),
            from_address: None,
        }
    }
}

/// Twilio adapter for SMS/WhatsApp style channels.
#[derive(Debug, Clone)]
pub struct TwilioAdapter {
    id: String,
    config: TwilioAdapterConfig,
}

impl TwilioAdapter {
    pub fn new(id: impl Into<String>) -> Self {
        Self::with_config(id, TwilioAdapterConfig::default())
    }

    pub fn with_config(id: impl Into<String>, config: TwilioAdapterConfig) -> Self {
        Self {
            id: id.into(),
            config,
        }
    }

    pub fn from_env(id: impl Into<String>) -> Self {
        Self::with_config(
            id,
            TwilioAdapterConfig {
                account_sid: caliche_contracts::credential_env::var("TWILIO_ACCOUNT_SID").ok(),
                auth_token: caliche_contracts::credential_env::var("TWILIO_AUTH_TOKEN").ok(),
                api_base_url: caliche_contracts::credential_env::var("TWILIO_API_BASE_URL")
                    .unwrap_or_else(|_| "https://api.twilio.com".to_owned()),
                from_address: caliche_contracts::credential_env::var("TWILIO_FROM").ok(),
            },
        )
    }

    pub fn set_account_sid(&mut self, sid: impl Into<String>) {
        self.config.account_sid = Some(sid.into());
    }

    pub fn set_auth_token(&mut self, token: impl Into<String>) {
        self.config.auth_token = Some(token.into());
    }

    pub fn set_api_base_url(&mut self, api_base_url: impl Into<String>) {
        self.config.api_base_url = api_base_url.into();
    }

    pub fn set_from_address(&mut self, from: Option<String>) {
        self.config.from_address = from;
    }

    pub fn config(&self) -> &TwilioAdapterConfig {
        &self.config
    }
}

#[async_trait]
impl GatewayAdapter for TwilioAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Twilio
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

        // Twilio webhook payloads are typically form fields projected into JSON by our API.
        let event_id = raw_payload
            .get("MessageSid")
            .or_else(|| raw_payload.get("SmsSid"))
            .and_then(serde_json::Value::as_str)
            .map(|v| format!("twilio:{v}"))
            .unwrap_or_else(|| format!("twilio:{}", Uuid::now_v7()));
        let from = raw_payload
            .get("From")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown");
        let to = raw_payload
            .get("To")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        let body = raw_payload
            .get("Body")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();

        let thread_id = raw_payload
            .get("ConversationSid")
            .and_then(serde_json::Value::as_str)
            .map(ToString::to_string);

        let mut attachments = Vec::new();
        let num_media = raw_payload
            .get("NumMedia")
            .and_then(serde_json::Value::as_str)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        for idx in 0..num_media {
            let media_key = format!("MediaUrl{idx}");
            if let Some(url) = raw_payload
                .get(&media_key)
                .and_then(serde_json::Value::as_str)
            {
                attachments.push(url.to_string());
            }
        }

        Ok(GatewayIngressEnvelope {
            event_id: event_id.clone(),
            idempotency_key: Some(event_id),
            channel_kind: ChannelKind::Twilio,
            channel_id: to.to_string(),
            adapter_id: self.id.clone(),
            tenant_id: None,
            user_id: from.to_string(),
            user_display_name: raw_payload
                .get("ProfileName")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string),
            received_at: Utc::now(),
            session_ref: GatewaySessionRef {
                session_id: None,
                thread_id,
                conversation_id: None,
            },
            payload: IngressPayload {
                text: body.clone(),
                attachments,
                is_command: body.starts_with('/'),
                reply_to_message_id: None,
            },
            metadata: HashMap::new(),
        })
    }

    async fn deliver(&self, envelope: &GatewayOutboundEnvelope) -> Result<()> {
        #[cfg(feature = "reqwest")]
        {
            let account_sid = self
                .config
                .account_sid
                .as_ref()
                .ok_or_else(|| anyhow!("TWILIO_ACCOUNT_SID is not configured"))?;
            let auth_token = self
                .config
                .auth_token
                .as_ref()
                .ok_or_else(|| anyhow!("TWILIO_AUTH_TOKEN is not configured"))?;
            let from = envelope
                .metadata
                .get("from")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
                .or_else(|| self.config.from_address.clone())
                .ok_or_else(|| anyhow!("twilio from address is not configured"))?;
            let to = envelope
                .metadata
                .get("to")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(envelope.channel_id.as_str())
                .to_string();

            let url = format!(
                "{}/2010-04-01/Accounts/{}/Messages.json",
                self.config.api_base_url.trim_end_matches('/'),
                account_sid
            );

            let mut form = HashMap::new();
            form.insert("To".to_string(), to);
            form.insert("From".to_string(), from);
            form.insert("Body".to_string(), envelope.payload.text.clone());

            let client = reqwest::Client::new();
            let resp = client
                .post(url)
                .basic_auth(account_sid, Some(auth_token))
                .form(&form)
                .send()
                .await
                .context("twilio message create request failed")?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(anyhow!("twilio message create failed: {status} {text}"));
            }
            Ok(())
        }
        #[cfg(not(feature = "reqwest"))]
        {
            let _ = envelope;
            Err(anyhow!("twilio delivery requires reqwest feature"))
        }
    }
}
