//! Image loading and preprocessing utilities.

#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
use crate::errors::DripstoneError;
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
use crate::multimodal::config::ImagePreprocessor;
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
use crate::multimodal::media::ImageInput;

#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
use image::{DynamicImage, RgbImage, imageops};

/// Load and preprocess an image for vision models.
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub async fn load_and_preprocess_image(
    input: &ImageInput,
    target_size: usize,
    preprocessor: ImagePreprocessor,
) -> Result<RgbImage, DripstoneError> {
    let image = load_image(input).await?;
    preprocess_image(&image, target_size, preprocessor)
}

/// Load an image from various input sources.
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub async fn load_image(input: &ImageInput) -> Result<DynamicImage, DripstoneError> {
    match input {
        ImageInput::Path(path) => {
            let path = path.clone();
            tokio::task::spawn_blocking(move || {
                image::open(&path).map_err(|e| {
                    DripstoneError::storage_with_source(
                        format!("failed to load image from path: {:?}", path),
                        e,
                    )
                })
            })
            .await
            .map_err(|e| DripstoneError::storage_with_source("image loading task failed", e))?
        }
        ImageInput::Bytes(bytes) => {
            let bytes = bytes.clone();
            tokio::task::spawn_blocking(move || {
                image::load_from_memory(&bytes).map_err(|e| {
                    DripstoneError::storage_with_source("failed to load image from bytes", e)
                })
            })
            .await
            .map_err(|e| DripstoneError::storage_with_source("image loading task failed", e))?
        }
        ImageInput::Url(url) => {
            #[cfg(not(feature = "reqwest"))]
            {
                let _ = url;
                return Err(DripstoneError::Misconfiguration(
                    "Image URL loading requires reqwest feature".to_string(),
                ));
            }
            #[cfg(feature = "reqwest")]
            {
                let url = url.clone();
                let response = reqwest::get(&url).await.map_err(|e| {
                    DripstoneError::storage_with_source("failed to fetch image URL", e)
                })?;

                let bytes = response.bytes().await.map_err(|e| {
                    DripstoneError::storage_with_source("failed to read image bytes", e)
                })?;

                tokio::task::spawn_blocking(move || {
                    image::load_from_memory(&bytes).map_err(|e| {
                        DripstoneError::storage_with_source(
                            "failed to load image from URL bytes",
                            e,
                        )
                    })
                })
                .await
                .map_err(|e| DripstoneError::storage_with_source("image loading task failed", e))?
            }
        }
        ImageInput::RgbImage(img) => {
            return Ok(DynamicImage::ImageRgb8(img.clone()));
        }
    }
}

/// Preprocess an image for vision models.
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub fn preprocess_image(
    image: &DynamicImage,
    target_size: usize,
    preprocessor: ImagePreprocessor,
) -> Result<RgbImage, DripstoneError> {
    match preprocessor {
        ImagePreprocessor::Clip | ImagePreprocessor::SigLip => {
            // Resize to square maintaining aspect ratio, then center crop
            let resized = if image.width() > image.height() {
                image.resize(
                    target_size as u32,
                    (image.height() * target_size as u32 / image.width()) as u32,
                    imageops::FilterType::Lanczos3,
                )
            } else {
                image.resize(
                    (image.width() * target_size as u32 / image.height()) as u32,
                    target_size as u32,
                    imageops::FilterType::Lanczos3,
                )
            };

            // Center crop to exact size
            let width = resized.width();
            let height = resized.height();
            let x = (width.saturating_sub(target_size as u32)) / 2;
            let y = (height.saturating_sub(target_size as u32)) / 2;
            let cropped = resized.crop_imm(x, y, target_size as u32, target_size as u32);

            Ok(cropped.to_rgb8())
        }
        ImagePreprocessor::Simple => {
            // Simple resize to target size
            let resized = image.resize_exact(
                target_size as u32,
                target_size as u32,
                imageops::FilterType::Lanczos3,
            );
            Ok(resized.to_rgb8())
        }
    }
}

/// Normalize pixel values for CLIP/SigLIP models.
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
#[expect(dead_code, reason = "nothing calls it yet")]
pub fn normalize_pixels_clip(pixels: &[u8], mean: &[f32], std: &[f32]) -> Vec<f32> {
    pixels
        .chunks_exact(3)
        .flat_map(|rgb| {
            [
                (rgb[0] as f32 / 255.0 - mean[0]) / std[0],
                (rgb[1] as f32 / 255.0 - mean[1]) / std[1],
                (rgb[2] as f32 / 255.0 - mean[2]) / std[2],
            ]
        })
        .collect()
}

/// CLIP normalization constants (ImageNet mean and std).
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub const CLIP_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub const CLIP_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SigLIP normalization constants.
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];
