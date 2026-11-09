//! Candle-based vision embedding generation.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::loader::HuggingFaceLoader;
use crate::multimodal::MediaGenerator;
use crate::multimodal::config::{ImagePreprocessor, VisionConfig};
use crate::multimodal::image_utils::{
    CLIP_MEAN, CLIP_STD, SIGLIP_MEAN, SIGLIP_STD, load_and_preprocess_image,
};
use crate::multimodal::media::ImageInput;
#[cfg(feature = "video-processing")]
use crate::multimodal::media::VideoInput;
#[cfg(feature = "video-processing")]
use crate::multimodal::video::{VideoOptions, VideoProcessor, VideoStrategy};
use parking_lot::Mutex;
use std::sync::Arc;
#[cfg(feature = "video-processing")]
use std::time::SystemTime;

#[cfg(feature = "vision-candle")]
use candle_core::{Device, Tensor};
#[cfg(feature = "vision-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "vision-candle")]
use candle_transformers::models::clip::{
    ClipConfig, ClipModel,
    text_model::{Activation as ClipActivation, ClipTextConfig},
    vision_model::ClipVisionConfig,
};

#[cfg(feature = "vision-candle")]
fn candle_device() -> Device {
    #[cfg(feature = "candle-metal")]
    if let Ok(d) = Device::new_metal(0) {
        return d;
    }
    #[cfg(feature = "candle-cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

/// Build a `ClipConfig` from a parsed Hugging Face `config.json`.
///
/// CLIP configs are not `Deserialize`-ready in candle-transformers 0.9.x, so
/// we parse the interesting fields out of the JSON and fall back to
/// `vit_base_patch32` defaults for anything that's missing. Consumers should
/// treat any warning logs as a signal that the loaded weights may not match
/// the selected configuration exactly.
#[cfg(feature = "vision-candle")]
fn build_clip_config(
    config_json: &serde_json::Value,
    image_size_hint: usize,
) -> Result<ClipConfig, DripstoneError> {
    let defaults = ClipConfig::vit_base_patch32();

    let vision_json = config_json.get("vision_config");
    let text_json = config_json.get("text_config");

    let vision_config = match vision_json {
        Some(v) => ClipVisionConfig {
            embed_dim: v
                .get("hidden_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.embed_dim),
            activation: ClipActivation::QuickGelu,
            intermediate_size: v
                .get("intermediate_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.intermediate_size),
            num_hidden_layers: v
                .get("num_hidden_layers")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.num_hidden_layers),
            num_attention_heads: v
                .get("num_attention_heads")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.num_attention_heads),
            projection_dim: config_json
                .get("projection_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.projection_dim),
            num_channels: v
                .get("num_channels")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.num_channels),
            image_size: v
                .get("image_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(image_size_hint),
            patch_size: v
                .get("patch_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.vision_config.patch_size),
        },
        None => defaults.vision_config.clone(),
    };

    let text_config = match text_json {
        Some(t) => ClipTextConfig {
            vocab_size: t
                .get("vocab_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.vocab_size),
            embed_dim: t
                .get("hidden_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.embed_dim),
            activation: ClipActivation::QuickGelu,
            intermediate_size: t
                .get("intermediate_size")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.intermediate_size),
            max_position_embeddings: t
                .get("max_position_embeddings")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.max_position_embeddings),
            pad_with: None,
            num_hidden_layers: t
                .get("num_hidden_layers")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.num_hidden_layers),
            num_attention_heads: t
                .get("num_attention_heads")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.num_attention_heads),
            projection_dim: config_json
                .get("projection_dim")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(defaults.text_config.projection_dim),
        },
        None => defaults.text_config.clone(),
    };

    let logit_scale_init_value = config_json
        .get("logit_scale_init_value")
        .and_then(serde_json::Value::as_f64)
        .map(|n| n as f32)
        .unwrap_or(defaults.logit_scale_init_value);

    Ok(ClipConfig {
        text_config,
        vision_config,
        logit_scale_init_value,
        image_size: image_size_hint,
    })
}

/// Candle-based vision embedder for generating embeddings from images and videos.
#[cfg(feature = "vision-candle")]
pub struct CandleVisionEmbedder {
    config: VisionConfig,
    model: Arc<Mutex<ClipModel>>,
    device: Device,
    dimension: usize,
    #[cfg(feature = "video-processing")]
    video_processor: VideoProcessor,
}

#[cfg(feature = "vision-candle")]
impl std::fmt::Debug for CandleVisionEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleVisionEmbedder")
            .field("dimension", &self.dimension)
            .field("config", &self.config)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "vision-candle")]
impl CandleVisionEmbedder {
    /// Create a new Candle vision embedder with the given configuration.
    pub fn new(config: VisionConfig) -> Result<Self, DripstoneError> {
        let device = candle_device();

        let loader = HuggingFaceLoader::new(&config.model_repo, config.revision.as_deref())?;

        let config_path = loader.get_file("config.json")?;
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| DripstoneError::storage_with_source("failed to read model config", e))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| DripstoneError::storage_with_source("failed to parse model config", e))?;

        let clip_config = build_clip_config(&config_json, config.image_size)?;
        let dimension = clip_config.vision_config.projection_dim;

        let weights_path = loader.get_file("model.safetensors")?;
        let weights = candle_core::safetensors::load(&weights_path, &device)
            .map_err(|e| DripstoneError::storage_with_source("failed to load weights", e))?;
        let vb = VarBuilder::from_tensors(weights, candle_core::DType::F32, &device);

        let model = ClipModel::new(vb, &clip_config).map_err(|e| {
            DripstoneError::storage_with_source("failed to construct CLIP model", e)
        })?;

        Ok(Self {
            config,
            model: Arc::new(Mutex::new(model)),
            device,
            dimension,
            #[cfg(feature = "video-processing")]
            video_processor: VideoProcessor::new(),
        })
    }

    /// Create a new Candle vision embedder from a pre-loaded model.
    /// This allows the agent framework to manage model loading separately.
    pub fn from_model(
        model: Arc<Mutex<ClipModel>>,
        config: VisionConfig,
        device: Device,
        dimension: usize,
    ) -> Self {
        Self {
            config,
            model,
            device,
            dimension,
            #[cfg(feature = "video-processing")]
            video_processor: VideoProcessor::new(),
        }
    }

    fn normalize(tensor: &Tensor, should_normalize: bool) -> Result<Tensor, DripstoneError> {
        if !should_normalize {
            return Ok(tensor.clone());
        }

        // Compute L2 norm: sqrt(sum(x^2)) over the last dimension
        let squared = tensor
            .sqr()
            .map_err(|e| DripstoneError::storage_with_source("failed to square", e))?;
        let norm = squared
            .sum_keepdim(1)
            .map_err(|e| DripstoneError::storage_with_source("failed to sum", e))?
            .sqrt()
            .map_err(|e| DripstoneError::storage_with_source("failed to sqrt", e))?;
        // Normalize: x / norm
        tensor
            .broadcast_div(&norm)
            .map_err(|e| DripstoneError::storage_with_source("failed to normalize", e))
    }
}

#[cfg(feature = "vision-candle")]
#[async_trait::async_trait]
impl MediaGenerator for CandleVisionEmbedder {
    async fn generate_image(
        &self,
        images: &[ImageInput],
    ) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // Preprocess images first (async; uses tokio file IO under the hood).
        let mut processed_images = Vec::with_capacity(images.len());
        for image_input in images {
            let rgb_image = load_and_preprocess_image(
                image_input,
                self.config.image_size,
                self.config.preprocessor,
            )
            .await?;
            processed_images.push(rgb_image);
        }

        let config = self.config.clone();
        let model = self.model.clone();
        let device = self.device.clone();
        let normalize = config.normalize;

        tokio::task::spawn_blocking(move || -> Result<Vec<EmbeddingVector>, DripstoneError> {
            let mut batch_tensors = Vec::with_capacity(processed_images.len());
            for img in &processed_images {
                let (width, height) = img.dimensions();
                let pixels = img.as_raw();

                let tensor = match config.preprocessor {
                    ImagePreprocessor::Clip | ImagePreprocessor::SigLip => {
                        let (mean, std) = if matches!(config.preprocessor, ImagePreprocessor::Clip)
                        {
                            (&CLIP_MEAN, &CLIP_STD)
                        } else {
                            (&SIGLIP_MEAN, &SIGLIP_STD)
                        };
                        let normalized: Vec<f32> = pixels
                            .chunks_exact(3)
                            .flat_map(|rgb| {
                                [
                                    (f32::from(rgb[0]) / 255.0 - mean[0]) / std[0],
                                    (f32::from(rgb[1]) / 255.0 - mean[1]) / std[1],
                                    (f32::from(rgb[2]) / 255.0 - mean[2]) / std[2],
                                ]
                            })
                            .collect();
                        Tensor::from_vec(
                            normalized,
                            (1, 3, height as usize, width as usize),
                            &device,
                        )
                        .map_err(|e| {
                            DripstoneError::storage_with_source("failed to create image tensor", e)
                        })?
                    }
                    ImagePreprocessor::Simple => {
                        let normalized: Vec<f32> = pixels
                            .iter()
                            .map(|&p| (f32::from(p) / 255.0) - 0.5)
                            .collect();
                        Tensor::from_vec(
                            normalized,
                            (1, 3, height as usize, width as usize),
                            &device,
                        )
                        .map_err(|e| {
                            DripstoneError::storage_with_source("failed to create image tensor", e)
                        })?
                    }
                };
                batch_tensors.push(tensor);
            }

            let batch_tensor = Tensor::cat(&batch_tensors, 0).map_err(|e| {
                DripstoneError::storage_with_source("failed to concatenate batch", e)
            })?;

            let image_features = {
                let model = model.lock();
                model.get_image_features(&batch_tensor).map_err(|e| {
                    DripstoneError::storage_with_source("CLIP get_image_features failed", e)
                })?
            };

            let features = Self::normalize(&image_features, normalize)?;

            let rows = features.to_vec2::<f32>().map_err(|e| {
                DripstoneError::storage_with_source("failed to extract features", e)
            })?;
            Ok(rows.into_iter().map(EmbeddingVector::from).collect())
        })
        .await
        .map_err(|e| DripstoneError::storage_with_source("task join failed", e))?
    }

    #[cfg(feature = "video-processing")]
    async fn generate_video(
        &self,
        video: &VideoInput,
        opts: VideoOptions,
    ) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        match opts.strategy {
            VideoStrategy::FrameExtraction { aggregation } => {
                // Create temporary directory for frames
                let temp_dir = std::env::temp_dir().join(format!(
                    "dripstone_frames_{}",
                    SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_err(|e| DripstoneError::storage_with_source("time error", e))?
                        .as_secs()
                ));

                // Extract frames
                let frame_paths = self
                    .video_processor
                    .extract_frames(video, &temp_dir, opts.fps_sample, opts.max_frames)
                    .await?;

                // Generate embeddings for each frame
                let frame_inputs: Vec<ImageInput> = frame_paths
                    .iter()
                    .map(|p| ImageInput::from_path(p.clone()))
                    .collect();

                let frame_embeddings = self.generate_image(&frame_inputs).await?;

                // Clean up temporary frames
                let _ = std::fs::remove_dir_all(&temp_dir);

                // Aggregate frame embeddings
                self.video_processor
                    .aggregate_embeddings(&frame_embeddings, aggregation)
            }
            VideoStrategy::DedicatedModel => Err(DripstoneError::Misconfiguration(
                crate::multimodal::video::DEDICATED_VIDEO_UNSUPPORTED.to_string(),
            )),
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        "candle-vision"
    }
}

#[cfg(not(feature = "vision-candle"))]
pub struct CandleVisionEmbedder;

#[cfg(not(feature = "vision-candle"))]
impl CandleVisionEmbedder {
    pub fn new(_config: VisionConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "Candle vision embedder requires the 'vision-candle' feature".to_string(),
        ))
    }
}
