use crate::llm::provider::Provider;
use crate::messages::ContentPart;

#[cfg(feature = "tools-media")]
use crate::agent::tools::media_understanding::MediaUnderstandingRunner;
#[cfg(feature = "document-loaders")]
use crate::data::loaders::DocumentProcessor;

/// Normalize content parts for providers that may not support all modalities.
pub async fn prepare_content_for_provider(
    parts: &[ContentPart],
    provider: &dyn Provider,
    #[cfg(feature = "tools-media")] media_runner: Option<&MediaUnderstandingRunner>,
    #[cfg(not(feature = "tools-media"))] _media_runner: Option<&()>,
    #[cfg(feature = "document-loaders")] _doc_processor: Option<&DocumentProcessor>,
    #[cfg(not(feature = "document-loaders"))] _doc_processor: Option<&()>,
) -> Vec<ContentPart> {
    let mut normalized = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ContentPart::Image { .. } | ContentPart::ImageUrl { .. } => {
                normalized.push(part.clone());
            }
            ContentPart::Video { .. } | ContentPart::VideoUrl { .. } => {
                if provider.supports_vision() {
                    normalized.push(part.clone());
                } else {
                    normalized.push(ContentPart::Text {
                        text: beef::lean::Cow::borrowed(
                            "[Video attachment omitted for provider without vision support]",
                        ),
                    });
                }
            }
            ContentPart::Audio { .. } | ContentPart::AudioUrl { .. } => {
                if provider.supports_audio_input() {
                    normalized.push(part.clone());
                } else {
                    #[cfg(feature = "tools-media")]
                    {
                        if media_runner.is_some() {
                            normalized.push(ContentPart::Text {
                                text: beef::lean::Cow::borrowed(
                                    "[Audio attachment received. Configure media transcription pipeline to inject transcript.]",
                                ),
                            });
                            continue;
                        }
                    }
                    normalized.push(ContentPart::Text {
                        text: beef::lean::Cow::borrowed(
                            "[Audio attachment omitted for provider without native audio input]",
                        ),
                    });
                }
            }
            ContentPart::Document { .. } | ContentPart::DocumentUrl { .. } => {
                if provider.supports_document_input() {
                    normalized.push(part.clone());
                } else {
                    normalized.push(ContentPart::Text {
                        text: beef::lean::Cow::borrowed(
                            "[Document attachment received. Convert to extracted text for this provider.]",
                        ),
                    });
                }
            }
            ContentPart::Text { .. } => normalized.push(part.clone()),
        }
    }
    normalized
}
