//! Media input types for images and videos.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
use image::RgbImage;

/// Input source for an image.
#[derive(Debug, Clone)]
#[cfg_attr(
    not(any(feature = "vision-candle", feature = "vision-ort")),
    derive(Serialize, Deserialize)
)]
#[cfg_attr(
    not(any(feature = "vision-candle", feature = "vision-ort")),
    serde(tag = "type")
)]
pub enum ImageInput {
    /// Image file path
    Path(PathBuf),
    /// Image as raw bytes
    Bytes(Vec<u8>),
    /// Image URL (will be downloaded)
    Url(String),
    /// Pre-loaded RGB image (not serializable)
    #[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
    RgbImage(RgbImage),
}

impl ImageInput {
    /// Create an image input from a file path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    /// Create an image input from bytes.
    #[must_use]
    pub const fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }

    /// Create an image input from a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    #[cfg(any(feature = "vision-candle", feature = "vision-ort"))]
    /// Create an image input from a pre-loaded RGB image.
    pub fn from_rgb_image(image: RgbImage) -> Self {
        Self::RgbImage(image)
    }
}

/// Input source for a video.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VideoInput {
    /// Video file path
    Path(PathBuf),
    /// Video URL (will be downloaded)
    Url(String),
}

impl VideoInput {
    /// Create a video input from a file path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    /// Create a video input from a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }
}
