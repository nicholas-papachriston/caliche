//! `chat` API client methods.

use anyhow::{Context, Result};
use caliche_contracts::rpc::methods as rpc_methods;
use futures_util::StreamExt;
use reqwest_eventsource::EventSource;
use std::sync::atomic::Ordering;

use super::{
    ApiClient, done_stream_event, error_stream_event, stream_event_from_envelope_str,
    stream_event_from_envelope_value,
};
use crate::api::types::{
    ChatRequest, ChatResponse, MediaAttachment, StreamingEvent, StreamingEventType,
};

impl ApiClient {
    /// Send a chat message with media attachments, streaming the response via SSE.
    /// Uses `POST /api/chat/stream/multipart` with multipart/form-data.
    pub async fn chat_multipart_stream(
        &self,
        message: &str,
        attachments: Vec<&MediaAttachment>,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<impl futures_util::Stream<Item = Result<StreamingEvent>>> {
        let mut form = reqwest::multipart::Form::new().text("message", message.to_string());

        if let Some(p) = provider {
            form = form.text("provider", p.to_string());
        }
        if let Some(m) = model {
            form = form.text("model", m.to_string());
        }

        for att in attachments {
            let part = reqwest::multipart::Part::bytes(att.bytes.clone())
                .file_name(att.filename.clone())
                .mime_str(&att.media_type)
                .unwrap_or_else(|_| {
                    reqwest::multipart::Part::bytes(att.bytes.clone())
                        .file_name(att.filename.clone())
                });
            form = form.part("attachments", part);
        }

        let request_builder = self
            .client
            .post(format!("{}/api/chat/stream/multipart", self.base_url))
            .multipart(form);

        let es =
            EventSource::new(request_builder).context("Failed to create multipart event source")?;

        Ok(futures_util::stream::unfold(es, |mut es| async move {
            use reqwest_eventsource::Event;
            loop {
                match es.next().await {
                    Some(Ok(Event::Message(msg))) => {
                        if msg.data == "[DONE]" {
                            es.close();
                            return None;
                        }
                        match stream_event_from_envelope_str(&msg.data) {
                            Ok(event) => {
                                let is_done = event.event_type == StreamingEventType::Done;
                                if is_done {
                                    es.close();
                                }
                                return Some((Ok(event), es));
                            }
                            Err(_) => continue,
                        }
                    }
                    Some(Ok(Event::Open)) => continue,
                    Some(Err(reqwest_eventsource::Error::StreamEnded)) | None => {
                        es.close();
                        return None;
                    }
                    Some(Err(e)) => {
                        es.close();
                        return Some((Err(anyhow::anyhow!("SSE error: {}", e)), es));
                    }
                }
            }
        }))
    }

    /// Send a chat message with media attachments (non-streaming fallback).
    /// Uses `POST /api/chat/multipart` with multipart/form-data.
    pub async fn chat_multipart(
        &self,
        message: &str,
        attachments: Vec<&MediaAttachment>,
        provider: Option<&str>,
        model: Option<&str>,
    ) -> Result<ChatResponse> {
        let mut form = reqwest::multipart::Form::new().text("message", message.to_string());

        if let Some(p) = provider {
            form = form.text("provider", p.to_string());
        }
        if let Some(m) = model {
            form = form.text("model", m.to_string());
        }

        for att in attachments {
            let part = reqwest::multipart::Part::bytes(att.bytes.clone())
                .file_name(att.filename.clone())
                .mime_str(&att.media_type)
                .unwrap_or_else(|_| {
                    reqwest::multipart::Part::bytes(att.bytes.clone())
                        .file_name(att.filename.clone())
                });
            form = form.part("attachments", part);
        }

        let response = self
            .client
            .post(format!("{}/api/chat/multipart", self.base_url))
            .multipart(form)
            .send()
            .await
            .context("Failed to send multipart chat request")?;

        if !response.status().is_success() {
            let error: serde_json::Value = response.json().await.unwrap_or_default();
            let msg = error
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("Failed to send multipart chat request");
            anyhow::bail!("{}", msg);
        }

        response
            .json()
            .await
            .context("Failed to parse multipart chat response")
    }

    /// Send a chat message with streaming response
    #[cfg_attr(
        target_arch = "wasm32",
        allow(
            dead_code,
            reason = "the wasm build does not call it; lib.rs and main.rs both compile this module, so expect cannot hold in both"
        )
    )]
    pub async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> Result<impl futures_util::Stream<Item = Result<StreamingEvent>>> {
        let request_id = self.rpc_id.fetch_add(1, Ordering::Relaxed);
        let rpc_request = serde_json::json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "id": request_id,
            "method": rpc_methods::CHAT_STREAM,
            "params": request,
        });
        let request_builder = self
            .client
            .post(format!("{}/rpc", self.base_url))
            .json(&rpc_request);

        let es = EventSource::new(request_builder).context("Failed to create event source")?;

        // Use unfold to process events and stop when we get StreamEnded, [DONE], or Done event
        Ok(futures_util::stream::unfold(es, |mut es| async move {
            use reqwest_eventsource::Event;

            loop {
                match es.next().await {
                    Some(Ok(Event::Message(msg))) => {
                        // Check for raw [DONE] marker
                        if msg.data == "[DONE]" {
                            es.close();
                            return None;
                        }

                        match serde_json::from_str::<serde_json::Value>(&msg.data) {
                            Ok(value) => {
                                if let Some(event_value) =
                                    value.get("params").and_then(|p| p.get("event")).cloned()
                                {
                                    match stream_event_from_envelope_value(event_value) {
                                        Ok(event) => {
                                            let is_done =
                                                event.event_type == StreamingEventType::Done;
                                            if is_done {
                                                es.close();
                                            }
                                            return Some((Ok(event), es));
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "Failed to parse RPC stream event: {} - data: {}",
                                                e,
                                                msg.data
                                            );
                                            continue;
                                        }
                                    }
                                }
                                // Final JSON-RPC response for the stream; synthesize done.
                                if value.get("result").is_some() {
                                    let done = done_stream_event();
                                    es.close();
                                    return Some((Ok(done), es));
                                }
                                if let Some(err) = value.get("error") {
                                    let message = err
                                        .get("message")
                                        .and_then(serde_json::Value::as_str)
                                        .unwrap_or("Unknown stream error")
                                        .to_string();
                                    let event = error_stream_event(message);
                                    es.close();
                                    return Some((Ok(event), es));
                                }
                                continue;
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse SSE event payload: {} - data: {}",
                                    e,
                                    msg.data
                                );
                                continue;
                            }
                        }
                    }
                    Some(Ok(Event::Open)) => {
                        continue;
                    }
                    Some(Err(reqwest_eventsource::Error::StreamEnded)) => {
                        es.close();
                        return None;
                    }
                    Some(Err(e)) => {
                        tracing::error!("SSE error: {}", e);
                        es.close();
                        return Some((Err(anyhow::anyhow!("SSE error: {}", e)), es));
                    }
                    None => {
                        es.close();
                        return None;
                    }
                }
            }
        }))
    }
}
