//! Local vision embedding providers using Candle or ONNX Runtime.

use crate::llm::config::CommonProviderOptions;
#[cfg(feature = "video-processing")]
use crate::llm::vision::VideoEmbeddingRequest;
use crate::llm::vision::{
    VisionEmbeddingProvider, VisionEmbeddingRequest, VisionEmbeddingResponse,
};
use crate::messages::TokenUsage;
use async_trait::async_trait;
use std::sync::Arc;

#[cfg(any(feature = "embedding-candle", feature = "embedding-ort"))]
use dripstone::multimodal::{ImageInput, MediaGenerator, VisionConfig};

#[cfg(feature = "embedding-candle")]
use dripstone::CandleVisionEmbedder;

#[cfg(feature = "embedding-ort")]
use dripstone::OrtVisionEmbedder;

#[cfg(feature = "video-processing")]
use dripstone::{TemporalAggregation, VideoInput, VideoOptions, VideoStrategy};

/// Configuration for LocalVisionCandleProvider
#[cfg(feature = "embedding-candle")]
#[derive(Debug, Clone)]
pub struct LocalVisionCandleProviderConfig {
    /// Vision model configuration
    pub vision_config: VisionConfig,
    /// Model name identifier
    pub model_name: String,
    /// Common provider options
    pub common: CommonProviderOptions,
}

#[cfg(feature = "embedding-candle")]
impl LocalVisionCandleProviderConfig {
    /// Create a new LocalVisionCandleProviderConfig with default settings
    pub fn new() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            model_name: "local-vision-candle".to_string(),
            common: CommonProviderOptions::default(),
        }
    }

    /// Set the model repository ID
    pub fn with_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.vision_config = self.vision_config.with_model_repo(repo);
        self.model_name = self.vision_config.model_repo.clone();
        self
    }

    /// Set the revision/branch/tag
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.vision_config = self.vision_config.with_revision(revision);
        self
    }

    /// Set the image size
    pub fn with_image_size(mut self, size: usize) -> Self {
        self.vision_config = self.vision_config.with_image_size(size);
        self
    }
}

#[cfg(feature = "embedding-candle")]
impl Default for LocalVisionCandleProviderConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "embedding-candle")]
crate::impl_provider_config!(LocalVisionCandleProviderConfig);

/// Local vision embedding provider using Candle
#[cfg(feature = "embedding-candle")]
#[derive(Debug)]
pub struct LocalVisionCandleProvider {
    embedder: Arc<CandleVisionEmbedder>,
    model_name: String,
}

#[cfg(feature = "embedding-candle")]
impl LocalVisionCandleProvider {
    /// Create a new LocalVisionCandleProvider with configuration
    pub fn with_config(config: LocalVisionCandleProviderConfig) -> anyhow::Result<Self> {
        let embedder = CandleVisionEmbedder::new(config.vision_config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create Candle vision embedder: {}", e))?;

        Ok(Self {
            embedder: Arc::new(embedder),
            model_name: config.model_name,
        })
    }

    /// Create a new LocalVisionCandleProvider from a pre-loaded model
    pub fn from_model(
        model: Arc<parking_lot::Mutex<candle_transformers::models::clip::ClipModel>>,
        config: VisionConfig,
        device: candle_core::Device,
        dimension: usize,
    ) -> Self {
        let embedder = CandleVisionEmbedder::from_model(model, config, device, dimension);
        Self {
            embedder: Arc::new(embedder),
            model_name: "local-vision-candle".to_string(),
        }
    }

    /// Create a new LocalVisionCandleProvider with default settings
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(LocalVisionCandleProviderConfig::default())
    }
}

#[cfg(feature = "embedding-candle")]
#[async_trait]
impl VisionEmbeddingProvider for LocalVisionCandleProvider {
    async fn embed_images(
        &self,
        request: VisionEmbeddingRequest,
    ) -> anyhow::Result<VisionEmbeddingResponse> {
        if request.images.is_empty() {
            return Ok(VisionEmbeddingResponse {
                embeddings: Vec::new(),
                usage: None,
                model: self.model_name.clone(),
            });
        }

        let model_name = request.model.as_ref().unwrap_or(&self.model_name).clone();

        // Convert paths to ImageInput
        let image_inputs: Vec<ImageInput> = request
            .images
            .iter()
            .map(|p| ImageInput::from_path(p.clone()))
            .collect();

        // Generate embeddings
        let embeddings = self
            .embedder
            .generate_image(&image_inputs)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to generate vision embeddings: {}", e))?;
        let embeddings = crate::llm::embedding::owned_embedding_rows(embeddings);

        Ok(VisionEmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage::from_embedding_tokens(
                request.images.len() as u32,
                Some(request.images.len() as u32),
            )),
            model: model_name,
        })
    }

    #[cfg(feature = "video-processing")]
    async fn embed_video(
        &self,
        request: VideoEmbeddingRequest,
    ) -> anyhow::Result<VisionEmbeddingResponse> {
        let model_name = request.model.as_ref().unwrap_or(&self.model_name).clone();

        let video_input = VideoInput::from_path(request.video);

        let video_options = VideoOptions::new()
            .with_strategy(VideoStrategy::FrameExtraction {
                aggregation: TemporalAggregation::Mean,
            })
            .with_max_frames(request.max_frames.unwrap_or(32))
            .with_fps_sample(request.fps_sample);

        let embeddings = self
            .embedder
            .generate_video(&video_input, video_options)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to generate video embeddings: {}", e))?;
        let embeddings = crate::llm::embedding::owned_embedding_rows(embeddings);

        Ok(VisionEmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage::from_embedding_tokens(1, Some(1))),
            model: model_name,
        })
    }

    fn name(&self) -> &str {
        "local-vision-candle"
    }

    fn dimension(&self) -> usize {
        self.embedder.dimension()
    }
}

/// Configuration for LocalVisionOrtProvider
#[cfg(feature = "embedding-ort")]
#[derive(Debug, Clone)]
pub struct LocalVisionOrtProviderConfig {
    /// Vision model configuration
    pub vision_config: VisionConfig,
    /// Model name identifier
    pub model_name: String,
    /// Common provider options
    pub common: CommonProviderOptions,
}

#[cfg(feature = "embedding-ort")]
impl LocalVisionOrtProviderConfig {
    /// Create a new LocalVisionOrtProviderConfig with default settings
    pub fn new() -> Self {
        Self {
            vision_config: VisionConfig::default(),
            model_name: "local-vision-ort".to_string(),
            common: CommonProviderOptions::default(),
        }
    }

    /// Set the model repository ID
    pub fn with_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.vision_config = self.vision_config.with_model_repo(repo);
        self.model_name = self.vision_config.model_repo.clone();
        self
    }

    /// Set the revision/branch/tag
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.vision_config = self.vision_config.with_revision(revision);
        self
    }

    /// Set the image size
    pub fn with_image_size(mut self, size: usize) -> Self {
        self.vision_config = self.vision_config.with_image_size(size);
        self
    }
}

#[cfg(feature = "embedding-ort")]
impl Default for LocalVisionOrtProviderConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "embedding-ort")]
crate::impl_provider_config!(LocalVisionOrtProviderConfig);

/// Local vision embedding provider using ONNX Runtime
#[cfg(feature = "embedding-ort")]
#[derive(Debug)]
pub struct LocalVisionOrtProvider {
    embedder: Arc<OrtVisionEmbedder>,
    model_name: String,
}

#[cfg(feature = "embedding-ort")]
impl LocalVisionOrtProvider {
    /// Create a new LocalVisionOrtProvider with configuration
    pub fn with_config(config: LocalVisionOrtProviderConfig) -> anyhow::Result<Self> {
        let embedder = OrtVisionEmbedder::new(config.vision_config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to create ORT vision embedder: {}", e))?;

        Ok(Self {
            embedder: Arc::new(embedder),
            model_name: config.model_name,
        })
    }

    /// Create a new LocalVisionOrtProvider from a pre-loaded embedder
    ///
    /// This allows the agent framework to manage model loading separately
    /// and pass in a pre-configured embedder.
    pub fn from_embedder(embedder: Arc<OrtVisionEmbedder>) -> Self {
        Self {
            embedder,
            model_name: "local-vision-ort".to_string(),
        }
    }

    /// Create a new LocalVisionOrtProvider with default settings
    pub fn new() -> anyhow::Result<Self> {
        Self::with_config(LocalVisionOrtProviderConfig::default())
    }
}

#[cfg(feature = "embedding-ort")]
#[async_trait]
impl VisionEmbeddingProvider for LocalVisionOrtProvider {
    async fn embed_images(
        &self,
        request: VisionEmbeddingRequest,
    ) -> anyhow::Result<VisionEmbeddingResponse> {
        if request.images.is_empty() {
            return Ok(VisionEmbeddingResponse {
                embeddings: Vec::new(),
                usage: None,
                model: self.model_name.clone(),
            });
        }

        let model_name = request.model.as_ref().unwrap_or(&self.model_name).clone();

        // Convert paths to ImageInput
        let image_inputs: Vec<ImageInput> = request
            .images
            .iter()
            .map(|p| ImageInput::from_path(p.clone()))
            .collect();

        // Generate embeddings
        let embeddings = self
            .embedder
            .generate_image(&image_inputs)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to generate vision embeddings: {}", e))?;
        let embeddings = crate::llm::embedding::owned_embedding_rows(embeddings);

        Ok(VisionEmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage::from_embedding_tokens(
                request.images.len() as u32,
                Some(request.images.len() as u32),
            )),
            model: model_name,
        })
    }

    #[cfg(feature = "video-processing")]
    async fn embed_video(
        &self,
        request: VideoEmbeddingRequest,
    ) -> anyhow::Result<VisionEmbeddingResponse> {
        let model_name = request.model.as_ref().unwrap_or(&self.model_name).clone();

        let video_input = VideoInput::from_path(request.video);

        let video_options = VideoOptions::new()
            .with_strategy(VideoStrategy::FrameExtraction {
                aggregation: TemporalAggregation::Mean,
            })
            .with_max_frames(request.max_frames.unwrap_or(32))
            .with_fps_sample(request.fps_sample);

        let embeddings = self
            .embedder
            .generate_video(&video_input, video_options)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to generate video embeddings: {}", e))?;
        let embeddings = crate::llm::embedding::owned_embedding_rows(embeddings);

        Ok(VisionEmbeddingResponse {
            embeddings,
            usage: Some(TokenUsage::from_embedding_tokens(1, Some(1))),
            model: model_name,
        })
    }

    fn name(&self) -> &str {
        "local-vision-ort"
    }

    fn dimension(&self) -> usize {
        self.embedder.dimension()
    }
}
