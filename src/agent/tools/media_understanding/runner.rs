//! Runner that dispatches to media understanding providers.

use super::config::MediaConfig;
use super::provider::MediaUnderstandingProvider;
use std::path::Path;
use std::sync::Arc;

/// Type of media for attachment handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    Audio,
    Image,
    Video,
}

/// Result of processing one media attachment.
#[derive(Debug)]
pub struct MediaOutput {
    pub media_type: MediaType,
    pub description: String,
    pub provider_used: String,
}

/// An attachment to process (path + type).
#[derive(Debug)]
pub struct MediaAttachment {
    pub path: std::path::PathBuf,
    pub media_type: MediaType,
}

/// Runs media understanding across a list of providers.
pub struct MediaUnderstandingRunner {
    providers: Vec<Arc<dyn MediaUnderstandingProvider>>,
    config: MediaConfig,
}

impl MediaUnderstandingRunner {
    pub fn new(providers: Vec<Arc<dyn MediaUnderstandingProvider>>, config: MediaConfig) -> Self {
        Self { providers, config }
    }

    /// Process all attachments and return descriptions to inject as context.
    pub async fn process_attachments(&self, attachments: &[MediaAttachment]) -> Vec<MediaOutput> {
        let mut results = Vec::new();
        for att in attachments {
            let desc = match att.media_type {
                MediaType::Audio => self.transcribe_audio(&att.path).await,
                MediaType::Image => self.describe_image(&att.path).await,
                MediaType::Video => self.describe_video(&att.path).await,
            };
            if let Some((description, provider_name)) = desc {
                results.push(MediaOutput {
                    media_type: att.media_type,
                    description,
                    provider_used: provider_name,
                });
            }
        }
        results
    }

    async fn transcribe_audio(&self, path: &Path) -> Option<(String, String)> {
        if !self.config.audio_enabled {
            return None;
        }
        for p in &self.providers {
            if p.capabilities().audio_transcription
                && let Ok(s) = p.transcribe_audio(path, &self.config).await
            {
                return Some((s, format!("{:?}", p)));
            }
        }
        None
    }

    async fn describe_image(&self, path: &Path) -> Option<(String, String)> {
        if !self.config.image_enabled {
            return None;
        }
        for p in &self.providers {
            if p.capabilities().image_description
                && let Ok(s) = p.describe_image(path, &self.config).await
            {
                return Some((s, format!("{:?}", p)));
            }
        }
        None
    }

    async fn describe_video(&self, path: &Path) -> Option<(String, String)> {
        if !self.config.video_enabled {
            return None;
        }
        for p in &self.providers {
            if p.capabilities().video_description
                && let Ok(s) = p.describe_video(path, &self.config).await
            {
                return Some((s, format!("{:?}", p)));
            }
        }
        None
    }
}
