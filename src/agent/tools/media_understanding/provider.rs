//! Media understanding provider trait.

use super::config::MediaConfig;
use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

/// Capabilities a media understanding provider can offer.
#[derive(Debug, Clone)]
pub struct MediaCapabilities {
    pub audio_transcription: bool,
    pub image_description: bool,
    pub video_description: bool,
}

#[async_trait]
pub trait MediaUnderstandingProvider: Send + Sync + std::fmt::Debug {
    fn capabilities(&self) -> MediaCapabilities;

    async fn transcribe_audio(&self, _audio_path: &Path, _config: &MediaConfig) -> Result<String> {
        Err(anyhow::anyhow!("Audio transcription not supported"))
    }

    async fn describe_image(&self, _image_path: &Path, _config: &MediaConfig) -> Result<String> {
        Err(anyhow::anyhow!("Image description not supported"))
    }

    async fn describe_video(&self, _video_path: &Path, _config: &MediaConfig) -> Result<String> {
        Err(anyhow::anyhow!("Video description not supported"))
    }
}
