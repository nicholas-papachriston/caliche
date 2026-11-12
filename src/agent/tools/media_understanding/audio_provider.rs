//! Audio transcription provider backed by OpenAI-compatible API.

use super::config::MediaConfig;
use super::provider::{MediaCapabilities, MediaUnderstandingProvider};
use anyhow::Context;
use async_trait::async_trait;
use reqwest::multipart::{Form, Part};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct WhisperApiProvider {
    api_key: String,
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl WhisperApiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "whisper-1".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[async_trait]
impl MediaUnderstandingProvider for WhisperApiProvider {
    fn capabilities(&self) -> MediaCapabilities {
        MediaCapabilities {
            audio_transcription: true,
            image_description: false,
            video_description: false,
        }
    }

    async fn transcribe_audio(
        &self,
        audio_path: &Path,
        config: &MediaConfig,
    ) -> anyhow::Result<String> {
        let audio_path = &crate::agent::tools::common::path::canonical_path_async(audio_path)
            .await
            .with_context(|| format!("Failed to resolve audio file '{}'", audio_path.display()))?;
        let metadata = crate::agent::tools::common::path::resolved_metadata_async(audio_path)
            .await
            .with_context(|| format!("Failed to stat audio file '{}'", audio_path.display()))?;
        if metadata.len() > (config.max_file_size_mb as u64 * 1024 * 1024) {
            return Err(anyhow::anyhow!(
                "Audio file too large: {} bytes (limit {} MB)",
                metadata.len(),
                config.max_file_size_mb
            ));
        }

        let bytes = crate::agent::tools::common::path::read_resolved_bytes_async(audio_path)
            .await
            .with_context(|| format!("Failed to read audio file '{}'", audio_path.display()))?;
        let filename = audio_path
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("audio.wav")
            .to_string();
        let part = Part::bytes(bytes).file_name(filename);
        let form = Form::new()
            .part("file", part)
            .text("model", self.model.clone());

        let endpoint = format!(
            "{}/audio/transcriptions",
            self.base_url.trim_end_matches('/')
        );

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(config.timeout_seconds),
            self.client
                .post(endpoint)
                .bearer_auth(&self.api_key)
                .multipart(form)
                .send(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("Audio transcription request timed out"))?
        .context("Failed to call transcription API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Transcription API failed: {} {}",
                status,
                body
            ));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse transcription response")?;
        let text = body
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        Ok(text)
    }
}
