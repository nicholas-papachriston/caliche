//! Multimodal embedding generation for images and videos.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;

pub mod config;
pub mod media;

#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
mod image_utils;

#[cfg(feature = "vision-candle")]
pub mod candle;

#[cfg(feature = "vision-ort")]
pub mod ort;

#[cfg(feature = "video-processing")]
pub mod video;

pub use config::{ImagePreprocessor, VisionConfig};
pub use media::{ImageInput, VideoInput};

/// Trait for generating embeddings from media (images and videos).
#[async_trait::async_trait]
pub trait MediaGenerator: Send + Sync {
    /// Generate embeddings for the given images.
    async fn generate_image(
        &self,
        images: &[ImageInput],
    ) -> Result<Vec<EmbeddingVector>, DripstoneError>;

    /// Generate embeddings for the given video.
    ///
    /// Returns a vector of embeddings - one per frame if using frame extraction,
    /// or a single embedding if using a dedicated video model.
    #[cfg(feature = "video-processing")]
    async fn generate_video(
        &self,
        video: &VideoInput,
        opts: crate::multimodal::video::VideoOptions,
    ) -> Result<Vec<EmbeddingVector>, DripstoneError>;

    /// Get the dimension of embeddings produced by this generator.
    fn dimension(&self) -> usize;

    /// Get the name/identifier of this generator.
    fn name(&self) -> &str;
}

#[cfg(feature = "vision-candle")]
pub use candle::CandleVisionEmbedder;

#[cfg(feature = "vision-ort")]
pub use ort::OrtVisionEmbedder;

#[cfg(feature = "video-processing")]
pub use video::{TemporalAggregation, VideoOptions, VideoProcessor, VideoStrategy};
