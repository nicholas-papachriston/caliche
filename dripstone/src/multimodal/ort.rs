//! ONNX Runtime-based vision embedding generation.

use crate::embedding::EmbeddingVector;
use crate::errors::DripstoneError;
use crate::loader::huggingface::{onnx_model_candidates, resolve_local_onnx_model_path};
use crate::loader::{HuggingFaceLoader, ModelSource, OrtSessionConfig, build_ort_session};
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
use std::sync::Arc;
#[cfg(feature = "video-processing")]
use std::time::SystemTime;

#[cfg(feature = "vision-ort")]
use ndarray;
#[cfg(feature = "vision-ort")]
use ort::session::Session;
#[cfg(feature = "vision-ort")]
use ort::value::TensorRef;

/// Best-effort inspection of an ORT session's declared outputs to extract
/// the trailing embedding dimension. Returns `None` for fully dynamic shapes
/// so callers can fall back to a sensible default rather than silently
/// assuming the wrong dimension.
#[cfg(feature = "vision-ort")]
fn infer_output_embedding_dim(session: &Session) -> Option<usize> {
    let outputs = session.outputs();
    let first = outputs.first()?;
    let shape = match first.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape,
        _ => return None,
    };
    // Trailing dim is the embedding dim for both (batch, dim) and
    // (batch, seq, dim) shapes. Some models return -1 for dynamic axes;
    // treat any non-positive value as unknown so we fall through to the
    // default instead of picking a bogus dimension.
    let last = shape.iter().last().copied()?;
    if last > 0 { Some(last as usize) } else { None }
}

/// ONNX Runtime-based vision embedder for generating embeddings from images and videos.
#[cfg(feature = "vision-ort")]
pub struct OrtVisionEmbedder {
    config: VisionConfig,
    session: Arc<parking_lot::Mutex<Session>>,
    dimension: usize,
    #[cfg(feature = "video-processing")]
    video_processor: VideoProcessor,
}

#[cfg(feature = "vision-ort")]
impl OrtVisionEmbedder {
    /// Create a new ORT vision embedder with the given configuration.
    pub fn new(config: VisionConfig) -> Result<Self, DripstoneError> {
        use std::path::PathBuf;

        // Use ModelSource to detect local vs HuggingFace
        let model_source = ModelSource::from_str(&config.model_repo);

        let model_path: PathBuf = match &model_source {
            ModelSource::Local { path } => {
                // Validate local path
                model_source.validate_local()?;
                let candidates = onnx_model_candidates(&[
                    "vision_model.onnx",
                    "onnx/vision_model.onnx",
                    "models/model.onnx",
                    "clip.onnx",
                    "encoder.onnx",
                ]);
                resolve_local_onnx_model_path(path, &candidates)?
            }
            ModelSource::HuggingFace { repo, revision } => {
                // Use shared HuggingFace loader with fallback filenames
                let loader = HuggingFaceLoader::new(repo, revision.as_deref())?;
                loader.get_onnx_model_file(&[
                    "vision_model.onnx",
                    "onnx/vision_model.onnx",
                    "models/model.onnx",
                    "clip.onnx",
                    "encoder.onnx",
                ])?
            }
        };

        // Build ORT session using shared utility
        let session_config = OrtSessionConfig::default();
        let session = build_ort_session(&model_path, &session_config)?;

        // Prefer inferring the output dimension from the session's declared
        // output tensor shape. Vision encoders usually expose a 2D output of
        // shape (batch, hidden) or a 3D output of shape (batch, seq, hidden);
        // the last dimension is the embedding size in both cases. Fall back to
        // 512 only when the shape has no static trailing dimension.
        let dimension = infer_output_embedding_dim(&session).unwrap_or(512);

        Ok(Self {
            config,
            session: Arc::new(parking_lot::Mutex::new(session)),
            dimension,
            #[cfg(feature = "video-processing")]
            video_processor: VideoProcessor::new(),
        })
    }

    /// Create a new ORT vision embedder from a pre-loaded session.
    /// This allows the agent framework to manage model loading separately.
    pub fn from_session(
        session: Arc<parking_lot::Mutex<Session>>,
        config: VisionConfig,
        dimension: usize,
    ) -> Self {
        Self {
            config,
            session,
            dimension,
            #[cfg(feature = "video-processing")]
            video_processor: VideoProcessor::new(),
        }
    }

    fn normalize(embeddings: &mut [f32], dimension: usize) {
        let num_embeddings = embeddings.len() / dimension;
        for i in 0..num_embeddings {
            let start = i * dimension;
            let end = start + dimension;
            let slice = &mut embeddings[start..end];

            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in slice.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }
}

#[cfg(feature = "vision-ort")]
fn pack_vision_pixels(
    images: &[image::RgbImage],
    preprocessor: ImagePreprocessor,
    image_size: usize,
) -> Vec<f32> {
    let batch_size = images.len();
    let mut batch_data = Vec::with_capacity(batch_size * 3 * image_size * image_size);

    for img in images {
        let pixels = img.as_raw();
        let (mean, std) = match preprocessor {
            ImagePreprocessor::Clip => (&CLIP_MEAN, &CLIP_STD),
            ImagePreprocessor::SigLip => (&SIGLIP_MEAN, &SIGLIP_STD),
            ImagePreprocessor::Simple => {
                for &pixel in pixels {
                    batch_data.push((pixel as f32 / 255.0) - 0.5);
                }
                continue;
            }
        };

        for rgb in pixels.chunks_exact(3) {
            batch_data.push((rgb[0] as f32 / 255.0 - mean[0]) / std[0]);
            batch_data.push((rgb[1] as f32 / 255.0 - mean[1]) / std[1]);
            batch_data.push((rgb[2] as f32 / 255.0 - mean[2]) / std[2]);
        }
    }

    batch_data
}

#[cfg(feature = "vision-ort")]
fn embeddings_from_pooled(
    mut embeddings_vec: Vec<f32>,
    batch_count: usize,
    embedding_dim: usize,
    normalize: bool,
) -> Vec<EmbeddingVector> {
    if normalize {
        OrtVisionEmbedder::normalize(&mut embeddings_vec, embedding_dim);
    }
    let mut result: Vec<EmbeddingVector> = Vec::with_capacity(batch_count);
    for i in 0..batch_count {
        let start = i * embedding_dim;
        let end = start + embedding_dim;
        result.push(EmbeddingVector::from(&embeddings_vec[start..end]));
    }
    result
}

#[cfg(feature = "vision-ort")]
fn embeddings_from_ort_output(
    output_shape: &[i64],
    output_data: &[f32],
    normalize: bool,
) -> Result<Vec<EmbeddingVector>, DripstoneError> {
    if output_shape.len() == 2 {
        let batch_size_actual = output_shape[0] as usize;
        let embedding_dim = output_shape[1] as usize;
        let embeddings_vec: Vec<f32> = output_data.iter().copied().collect();
        return Ok(embeddings_from_pooled(
            embeddings_vec,
            batch_size_actual,
            embedding_dim,
            normalize,
        ));
    }
    if output_shape.len() == 3 {
        let seq_len = output_shape[1] as usize;
        let hidden_size = output_shape[2] as usize;
        let batch_count = output_shape[0] as usize;
        let mut pooled = vec![0.0f32; batch_count * hidden_size];

        for b in 0..batch_count {
            for s in 0..seq_len {
                for h in 0..hidden_size {
                    let idx = (b * seq_len + s) * hidden_size + h;
                    pooled[b * hidden_size + h] += output_data[idx] / seq_len as f32;
                }
            }
        }

        let pooled_array = ndarray::Array::from_shape_vec((batch_count, hidden_size), pooled)
            .map_err(|e| DripstoneError::storage_with_source("failed to create pooled array", e))?;

        let pooled_view = pooled_array.view();
        let embeddings_vec: Vec<f32> = pooled_view.iter().copied().collect();
        return Ok(embeddings_from_pooled(
            embeddings_vec,
            batch_count,
            hidden_size,
            normalize,
        ));
    }
    Err(DripstoneError::storage(
        "unexpected output shape from ONNX model",
    ))
}

#[cfg(feature = "vision-ort")]
fn run_vision_inference(
    processed_images: &[image::RgbImage],
    config: &VisionConfig,
    session: &parking_lot::Mutex<Session>,
) -> Result<Vec<EmbeddingVector>, DripstoneError> {
    let batch_size = processed_images.len();
    let image_size = config.image_size;
    let batch_data = pack_vision_pixels(processed_images, config.preprocessor, image_size);

    let input_array =
        ndarray::Array::from_shape_vec((batch_size, 3, image_size, image_size), batch_data)
            .map_err(|e| DripstoneError::storage_with_source("failed to create input array", e))?;

    let input_value = TensorRef::<f32>::from_array_view(&input_array)
        .map_err(|e| DripstoneError::storage_with_source("failed to create input value", e))?;

    let mut session_guard = session.lock();
    let outputs = session_guard
        .run(ort::inputs![input_value])
        .map_err(|e| DripstoneError::storage_with_source("ONNX inference failed", e))?;

    let (output_shape, output_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| DripstoneError::storage_with_source("failed to extract output tensor", e))?;

    let output_shape: &[i64] = &*output_shape;
    embeddings_from_ort_output(output_shape, output_data, config.normalize)
}

#[cfg(feature = "vision-ort")]
#[async_trait::async_trait]
impl MediaGenerator for OrtVisionEmbedder {
    async fn generate_image(
        &self,
        images: &[ImageInput],
    ) -> Result<Vec<EmbeddingVector>, DripstoneError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // Preprocess images first (async)
        let mut processed_images = Vec::new();
        for image_input in images {
            let rgb_image = load_and_preprocess_image(
                image_input,
                self.config.image_size,
                self.config.preprocessor,
            )
            .await?;
            processed_images.push(rgb_image);
        }

        // Then process in blocking task
        let processed_images = processed_images;
        let config = self.config.clone();
        let session = self.session.clone();

        tokio::task::spawn_blocking(move || {
            run_vision_inference(&processed_images, &config, &session)
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
        "ort-vision"
    }
}

#[cfg(not(feature = "vision-ort"))]
pub struct OrtVisionEmbedder;

#[cfg(not(feature = "vision-ort"))]
impl OrtVisionEmbedder {
    pub fn new(_config: VisionConfig) -> Result<Self, DripstoneError> {
        Err(DripstoneError::Misconfiguration(
            "ORT vision embedder requires the 'vision-ort' feature".to_string(),
        ))
    }
}
