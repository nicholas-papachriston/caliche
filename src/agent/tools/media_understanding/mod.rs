//! Media understanding: audio transcription, image/video description.
//!
//! Providers implement [MediaUnderstandingProvider]; the [MediaUnderstandingRunner]
//! dispatches attachments to the first capable provider.

pub mod audio_provider;
pub mod config;
pub mod provider;
pub mod runner;
pub mod vision_provider;

pub use audio_provider::WhisperApiProvider;
pub use config::MediaConfig;
pub use provider::{MediaCapabilities, MediaUnderstandingProvider};
pub use runner::{MediaAttachment, MediaOutput, MediaType, MediaUnderstandingRunner};
pub use vision_provider::VisionMediaProvider;
