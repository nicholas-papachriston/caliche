//! Vision embedding providers for images and videos.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Request for generating vision embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEmbeddingRequest {
    /// Image paths to embed
    pub images: Vec<PathBuf>,
    /// Optional model name (if not provided, provider will use default)
    pub model: Option<String>,
}

/// Request for generating video embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEmbeddingRequest {
    /// Video path to embed
    pub video: PathBuf,
    /// Optional model name (if not provided, provider will use default)
    pub model: Option<String>,
    /// Maximum frames to extract
    pub max_frames: Option<usize>,
    /// FPS sampling rate (None = extract all frames up to max_frames)
    pub fps_sample: Option<f32>,
}

/// Response containing vision embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionEmbeddingResponse {
    /// Vector embeddings, one per input image
    pub embeddings: Vec<Vec<f32>>,
    /// Token usage information (if applicable)
    pub usage: Option<crate::messages::TokenUsage>,
    /// Model used for generating embeddings
    pub model: String,
}

/// Trait for providers that support vision embeddings
#[async_trait]
pub trait VisionEmbeddingProvider: Send + Sync + std::fmt::Debug {
    /// Generate embeddings for the given images
    async fn embed_images(
        &self,
        request: VisionEmbeddingRequest,
    ) -> anyhow::Result<VisionEmbeddingResponse>;

    /// Generate embeddings for the given video
    #[cfg(feature = "video-processing")]
    async fn embed_video(
        &self,
        request: VideoEmbeddingRequest,
    ) -> anyhow::Result<VisionEmbeddingResponse>;

    /// Get the name of this vision embedding provider
    fn name(&self) -> &str;

    /// Get the dimension of embeddings produced by this provider
    fn dimension(&self) -> usize;
}

#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
pub mod local;
