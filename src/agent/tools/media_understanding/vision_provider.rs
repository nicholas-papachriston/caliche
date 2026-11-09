//! Vision-backed media provider using the configured chat LLM.

use super::config::MediaConfig;
use super::provider::{MediaCapabilities, MediaUnderstandingProvider};
use anyhow::Context;
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub struct VisionMediaProvider {
    llm: Arc<dyn crate::llm::Provider>,
}

impl VisionMediaProvider {
    pub fn new(llm: Arc<dyn crate::llm::Provider>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl MediaUnderstandingProvider for VisionMediaProvider {
    fn capabilities(&self) -> MediaCapabilities {
        MediaCapabilities {
            audio_transcription: false,
            image_description: self.llm.supports_vision(),
            video_description: self.llm.supports_vision(),
        }
    }

    async fn describe_image(
        &self,
        image_path: &Path,
        _config: &MediaConfig,
    ) -> anyhow::Result<String> {
        let image = crate::messages::Attachment::from_file(image_path)
            .await
            .with_context(|| format!("Failed to read image '{}'", image_path.display()))?;
        let req = crate::messages::CompletionRequest::new(Arc::new(vec![
            crate::messages::Message::user_with_attachments(
                "Describe this image concisely and include key entities/text.",
                vec![image],
            ),
        ]));
        let resp = self.llm.complete(req).await?;
        Ok(resp.content.extract_text())
    }

    async fn describe_video(
        &self,
        video_path: &Path,
        _config: &MediaConfig,
    ) -> anyhow::Result<String> {
        let bytes = tokio::fs::read(video_path)
            .await
            .with_context(|| format!("Failed to read video '{}'", video_path.display()))?;
        let message = crate::messages::Message::user_with_attachments(
            "Describe this video concisely and include salient moments.",
            vec![crate::messages::Attachment {
                data: bytes,
                media_type: crate::messages::detect_media_type(video_path),
                filename: video_path
                    .file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .map(std::string::ToString::to_string),
                attachment_type: crate::messages::AttachmentType::Video,
            }],
        );
        let req = crate::messages::CompletionRequest::new(Arc::new(vec![message]));
        let resp = self.llm.complete(req).await?;
        Ok(resp.content.extract_text())
    }
}
