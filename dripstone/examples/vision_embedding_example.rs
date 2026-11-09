//! Example demonstrating image and video embedding generation with `Dripstone`.
//!
//! # Supported Models
//!
//! ## ONNX Runtime (ORT) - Requires repositories with ONNX model files
//!
//! The ORT backend requires models in ONNX format. Standard `HuggingFace` repositories
//! (like `openai/clip-vit-base-patch32`) contain `PyTorch` weights, NOT ONNX models.
//!
//! ### Working ONNX Model Repositories:
//!
//! | Repository | Description | Dimension | Notes |
//! |------------|-------------|-----------|-------|
//! | `Qdrant/clip-ViT-B-32-vision` | Qdrant's CLIP ViT-B/32 | 512 | Optimized for embeddings |
//! | `Qdrant/resnet50-onnx` | `ResNet50` vision encoder | 2048 | General purpose |
//! | `Qdrant/Unicom-ViT-B-32` | Unicom ViT-B/32 | 512 | Good for face/person |
//! | `Qdrant/Unicom-ViT-B-16` | Unicom ViT-B/16 | 768 | Higher resolution |
//! | `jinaai/jina-clip-v1` | Jina's multimodal CLIP | 768 | Text+Image |
//! | Local path (e.g., `./model.onnx`) | Any exported ONNX model | varies | Custom models |
//!
//! ### Converting Models to ONNX:
//! ```bash
//! # Using optimum-cli to export HuggingFace models to ONNX
//! pip install optimum[exporters]
//! optimum-cli export onnx --model openai/clip-vit-base-patch32 ./clip-onnx/
//! ```
//!
//! ## Candle Backend - Uses native safetensors from `HuggingFace`
//!
//! The Candle backend loads models directly from `HuggingFace` using safetensors format.
//!
//! ### Working Candle Model Repositories:
//!
//! | Repository | Description | Dimension | Notes |
//! |------------|-------------|-----------|-------|
//! | `openai/clip-vit-base-patch32` | `OpenAI` CLIP ViT-B/32 | 512 | Standard CLIP |
//! | `openai/clip-vit-base-patch16` | `OpenAI` CLIP ViT-B/16 | 512 | Higher resolution |
//! | `openai/clip-vit-large-patch14` | `OpenAI` CLIP ViT-L/14 | 768 | Larger model |
//! | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | LAION CLIP | 512 | Large dataset |
//! | `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` | LAION CLIP Large | 768 | Higher quality |
//!
//! Run with:
//! ```bash
//! # Candle backend (Cargo.toml required-features)
//! cargo run --example vision_embedding_example --features "vision-candle,video-processing"
//!
//! # ORT backend (superset: also satisfies required-features)
//! cargo run --example vision_embedding_example --features "vision-ort,vision-candle,video-processing,memory-backend"
//! ```

macro_rules! outputln {
    () => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

#[cfg(feature = "vision-ort")]
use dripstone::{Chunk, MemoryBackend, traits::Backend};
#[cfg(feature = "vision-ort")]
use std::sync::Arc;

#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
use dripstone::{ImageInput, ImagePreprocessor, MediaGenerator, VisionConfig};

#[cfg(all(feature = "vision-ort", feature = "video-processing"))]
use dripstone::multimodal::{TemporalAggregation, VideoInput, VideoOptions, VideoStrategy};

#[cfg(feature = "vision-ort")]
use dripstone::OrtVisionEmbedder;

#[cfg(feature = "vision-candle")]
use dripstone::CandleVisionEmbedder;

/// Available ONNX models for ORT backend
#[cfg(feature = "vision-ort")]
mod ort_models {
    /// Qdrant's CLIP ViT-B/32 - optimized for vision embeddings
    /// Dimension: 512
    pub const QDRANT_CLIP_VIT_B32: &str = "Qdrant/clip-ViT-B-32-vision";

    /// Qdrant's ResNet50 - general purpose vision encoder
    /// Dimension: 2048
    pub const QDRANT_RESNET50: &str = "Qdrant/resnet50-onnx";

    /// Qdrant's Unicom ViT-B/32 - good for face/person recognition
    /// Dimension: 512
    pub const QDRANT_UNICOM_VIT_B32: &str = "Qdrant/Unicom-ViT-B-32";

    /// Qdrant's Unicom ViT-B/16 - higher resolution variant
    /// Dimension: 768
    #[expect(dead_code, reason = "a model id listed for reference")]
    pub const QDRANT_UNICOM_VIT_B16: &str = "Qdrant/Unicom-ViT-B-16";

    /// Jina's multimodal CLIP - supports both text and image
    /// Dimension: 768
    /// Note: May require specific ONNX file path within repo
    #[expect(dead_code, reason = "a model id listed for reference")]
    pub const JINA_CLIP_V1: &str = "jinaai/jina-clip-v1";
}

/// Available models for Candle backend (native safetensors)
#[cfg(feature = "vision-candle")]
mod candle_models {
    /// OpenAI CLIP ViT-B/32 - standard CLIP model
    /// Dimension: 512, Image size: 224
    pub const OPENAI_CLIP_VIT_B32: &str = "openai/clip-vit-base-patch32";

    /// OpenAI CLIP ViT-B/16 - higher resolution patches
    /// Dimension: 512, Image size: 224
    pub const OPENAI_CLIP_VIT_B16: &str = "openai/clip-vit-base-patch16";

    /// OpenAI CLIP ViT-L/14 - larger model, better quality
    /// Dimension: 768, Image size: 224
    pub const OPENAI_CLIP_VIT_L14: &str = "openai/clip-vit-large-patch14";

    /// LAION CLIP ViT-B/32 - trained on 2B image-text pairs
    /// Dimension: 512, Image size: 224
    pub const LAION_CLIP_VIT_B32: &str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K";

    /// LAION CLIP ViT-L/14 - larger LAION model
    /// Dimension: 768, Image size: 224
    pub const LAION_CLIP_VIT_L14: &str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K";
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Image embedding with ORT (ONNX Runtime)
    #[cfg(feature = "vision-ort")]
    {
        outputln!("=== Image Embedding Example (ORT) ===");
        outputln!("Using Qdrant's CLIP ViT-B/32 ONNX model\n");

        // Use Qdrant's pre-exported ONNX model - this is the recommended approach
        // as it's optimized for embeddings and guaranteed to have ONNX files
        let config = VisionConfig::default()
            .with_model_repo(ort_models::QDRANT_CLIP_VIT_B32)
            .with_image_size(224)
            .with_preprocessor(ImagePreprocessor::Clip)
            .with_normalize(true);

        match OrtVisionEmbedder::new(config) {
            Ok(embedder) => {
                outputln!(
                    "Created ORT vision embedder with dimension: {}",
                    embedder.dimension()
                );

                // Create image inputs - replace with actual image paths
                let images = vec![ImageInput::from_path("examples/test_image.jpg")];

                // Generate embeddings
                match embedder.generate_image(&images).await {
                    Ok(embeddings) => {
                        outputln!("Generated {} embeddings", embeddings.len());

                        // Store embeddings in vector store
                        use dripstone::{
                            Chunk, MemoryBackend, Metric, SearchQuery, traits::Backend,
                        };
                        use std::sync::Arc;

                        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

                        let mut chunks = Vec::new();
                        for (i, embedding) in embeddings.iter().enumerate() {
                            let chunk = Chunk::new(
                                &format!("img:{}", i),
                                0,
                                "",
                                embedding.clone(),
                                0,
                                0,
                                1,
                                serde_json::json!({
                                    "media_type": "image",
                                    "model": ort_models::QDRANT_CLIP_VIT_B32,
                                    "source": format!("test_image_{}.jpg", i)
                                }),
                                None,
                            );
                            chunks.push(chunk);
                        }
                        backend.upsert(&chunks).await?;

                        outputln!("Stored {} image embeddings", embeddings.len());

                        // Search for similar images
                        if !embeddings.is_empty() {
                            let query = SearchQuery::new(embeddings[0].clone())
                                .metric(Metric::Cosine)
                                .limit(5);
                            let results = backend.search(&query).await?;

                            outputln!("Found {} similar images", results.len());
                            for (i, result) in results.iter().enumerate() {
                                outputln!(
                                    "  {}. Score: {:.4}, ID: {}",
                                    i + 1,
                                    result.score,
                                    result.chunk.id
                                );
                            }
                        }
                    }
                    Err(e) => {
                        outputln!(
                            "Note: Could not generate embeddings (image file may not exist): {}",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                outputln!("Could not create ORT embedder: {}", e);
                outputln!("\nAvailable ORT-compatible models:");
                outputln!("  - Qdrant/clip-ViT-B-32-vision (recommended)");
                outputln!("  - Qdrant/resnet50-onnx");
                outputln!("  - Qdrant/Unicom-ViT-B-32");
                outputln!("  - Or use a local ONNX file path");
            }
        }
    }

    // Example 2: Image embedding with Candle
    #[cfg(feature = "vision-candle")]
    {
        outputln!("\n=== Image Embedding Example (Candle) ===");
        outputln!("Using OpenAI CLIP ViT-B/32 model\n");

        // Candle uses native HuggingFace safetensors - standard repos work!
        let config = VisionConfig::default()
            .with_model_repo(candle_models::OPENAI_CLIP_VIT_B32)
            .with_image_size(224)
            .with_preprocessor(ImagePreprocessor::Clip);

        match CandleVisionEmbedder::new(config) {
            Ok(embedder) => {
                outputln!(
                    "Created Candle vision embedder with dimension: {}",
                    embedder.dimension()
                );

                let images = vec![ImageInput::from_path("examples/test_image.jpg")];

                match embedder.generate_image(&images).await {
                    Ok(embeddings) => {
                        outputln!("Generated {} embeddings", embeddings.len());
                    }
                    Err(e) => {
                        outputln!("Note: Could not generate embeddings: {}", e);
                    }
                }
            }
            Err(e) => {
                outputln!("Could not create Candle embedder: {}", e);
                outputln!("\nAvailable Candle-compatible models:");
                outputln!("  - openai/clip-vit-base-patch32");
                outputln!("  - openai/clip-vit-base-patch16");
                outputln!("  - openai/clip-vit-large-patch14");
                outputln!("  - laion/CLIP-ViT-B-32-laion2B-s34B-b79K");
            }
        }
    }

    // Example 3: Video embedding with frame extraction
    #[cfg(all(feature = "vision-ort", feature = "video-processing"))]
    {
        outputln!("\n=== Video Embedding Example ===");
        outputln!("Using Qdrant's CLIP model for video frames\n");

        let config = VisionConfig::default()
            .with_model_repo(ort_models::QDRANT_CLIP_VIT_B32)
            .with_image_size(224)
            .with_preprocessor(ImagePreprocessor::Clip);

        match OrtVisionEmbedder::new(config) {
            Ok(embedder) => {
                let video = VideoInput::from_path("examples/test_video.mp4");

                // Extract frames and generate embeddings
                let video_options = VideoOptions::new()
                    .with_strategy(VideoStrategy::FrameExtraction {
                        aggregation: TemporalAggregation::Mean,
                    })
                    .with_max_frames(32)
                    .with_fps_sample(Some(1.0)); // Sample 1 frame per second

                match embedder.generate_video(&video, video_options).await {
                    Ok(video_embeddings) => {
                        outputln!(
                            "Generated {} video embeddings (aggregated)",
                            video_embeddings.len()
                        );

                        // Store video embedding
                        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

                        if !video_embeddings.is_empty() {
                            let chunk = Chunk::new(
                                "video:1",
                                0,
                                "",
                                video_embeddings[0].clone(),
                                0,
                                0,
                                1,
                                serde_json::json!({
                                    "media_type": "video",
                                    "model": ort_models::QDRANT_CLIP_VIT_B32,
                                    "source": "test_video.mp4"
                                }),
                                None,
                            );

                            backend.upsert(&[chunk]).await?;
                            outputln!("Stored video embedding");
                        }
                    }
                    Err(e) => {
                        outputln!("Note: Could not generate video embeddings: {}", e);
                        outputln!("(Video file may not exist or ffmpeg not available)");
                    }
                }
            }
            Err(e) => {
                outputln!("Could not create embedder for video: {}", e);
            }
        }
    }

    // Example 4: Comparing different ORT models
    #[cfg(feature = "vision-ort")]
    {
        outputln!("\n=== ORT Model Comparison ===");
        outputln!("Available ONNX models for ORT backend:\n");

        let models = [
            (
                ort_models::QDRANT_CLIP_VIT_B32,
                "CLIP ViT-B/32",
                512,
                "Best for general image similarity",
            ),
            (
                ort_models::QDRANT_RESNET50,
                "ResNet50",
                2048,
                "High-dimensional features",
            ),
            (
                ort_models::QDRANT_UNICOM_VIT_B32,
                "Unicom ViT-B/32",
                512,
                "Good for faces/people",
            ),
        ];

        for (repo, name, dim, desc) in models {
            outputln!("  {} ({}):", name, repo);
            outputln!("    Dimension: {}, Use case: {}", dim, desc);
        }
    }

    // Example 5: Comparing different Candle models
    #[cfg(feature = "vision-candle")]
    {
        outputln!("\n=== Candle Model Comparison ===");
        outputln!("Available models for Candle backend:\n");

        let models = [
            (
                candle_models::OPENAI_CLIP_VIT_B32,
                "OpenAI CLIP ViT-B/32",
                512,
                "Standard, fast",
            ),
            (
                candle_models::OPENAI_CLIP_VIT_B16,
                "OpenAI CLIP ViT-B/16",
                512,
                "Higher resolution",
            ),
            (
                candle_models::OPENAI_CLIP_VIT_L14,
                "OpenAI CLIP ViT-L/14",
                768,
                "Best quality",
            ),
            (
                candle_models::LAION_CLIP_VIT_B32,
                "LAION CLIP ViT-B/32",
                512,
                "Large training set",
            ),
        ];

        for (repo, name, dim, desc) in models {
            outputln!("  {} ({}):", name, repo);
            outputln!("    Dimension: {}, Use case: {}", dim, desc);
        }
    }

    // Example 6: Using a local ONNX model file
    #[cfg(feature = "vision-ort")]
    {
        outputln!("\n=== Local ONNX Model Example ===");
        outputln!("You can also use local ONNX model files:\n");

        outputln!("  let config = VisionConfig::default()");
        outputln!("      .with_model_repo(\"./models/my_clip.onnx\")");
        outputln!("      .with_image_size(224);");
        outputln!();
        outputln!("  // Or absolute path:");
        outputln!("  let config = VisionConfig::default()");
        outputln!("      .with_model_repo(\"/path/to/model.onnx\");");
        outputln!();
        outputln!("To export a model to ONNX:");
        outputln!("  pip install optimum[exporters]");
        outputln!("  optimum-cli export onnx --model openai/clip-vit-base-patch32 ./clip-onnx/");
    }

    // Example 7: Video frame extraction with all frames
    #[cfg(all(feature = "vision-ort", feature = "video-processing"))]
    {
        outputln!("\n=== Video Frame Extraction (All Frames) ===");

        let config = VisionConfig::default()
            .with_model_repo(ort_models::QDRANT_CLIP_VIT_B32)
            .with_image_size(224);

        if let Ok(embedder) = OrtVisionEmbedder::new(config) {
            let video = VideoInput::from_path("examples/test_video.mp4");

            // Get embeddings for all frames separately
            let video_options = VideoOptions::new()
                .with_strategy(VideoStrategy::FrameExtraction {
                    aggregation: TemporalAggregation::All,
                })
                .with_max_frames(10)
                .with_fps_sample(Some(2.0)); // Sample 2 frames per second

            match embedder.generate_video(&video, video_options).await {
                Ok(frame_embeddings) => {
                    outputln!("Extracted {} frame embeddings", frame_embeddings.len());

                    // Store each frame as a separate record
                    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

                    let mut chunks = Vec::new();
                    for (i, embedding) in frame_embeddings.iter().enumerate() {
                        let chunk = Chunk::new(
                            &format!("video_frame:{}", i),
                            0,
                            "",
                            embedding.clone(),
                            0,
                            0,
                            1,
                            serde_json::json!({
                                "media_type": "video_frame",
                                "source": "test_video.mp4",
                                "frame_index": i
                            }),
                            None,
                        );
                        chunks.push(chunk);
                    }
                    backend.upsert(&chunks).await?;

                    outputln!("Stored {} frame embeddings", frame_embeddings.len());
                }
                Err(e) => {
                    outputln!("Note: Video processing skipped: {}", e);
                }
            }
        }
    }

    outputln!("\n=== Examples completed ===");
    Ok(())
}
