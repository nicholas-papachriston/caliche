//! Configuration for vision/multimodal embedding generation.

use serde::{Deserialize, Serialize};

/// Image preprocessing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ImagePreprocessor {
    /// CLIP-style preprocessing (resize to square, center crop, normalize)
    #[default]
    Clip,
    /// SigLIP-style preprocessing (resize to square, center crop, normalize)
    SigLip,
    /// Simple resize and normalize (for custom models)
    Simple,
}

/// Configuration for vision embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// `HuggingFace` model repository ID (e.g., "openai/clip-vit-base-patch32")
    pub model_repo: String,
    /// Optional revision/branch/tag (defaults to "main")
    pub revision: Option<String>,
    /// Model's expected input size (224, 256, etc.)
    pub image_size: usize,
    /// Whether to normalize embeddings (L2 normalization)
    pub normalize: bool,
    /// Image preprocessing strategy
    pub preprocessor: ImagePreprocessor,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            model_repo: "openai/clip-vit-base-patch32".to_string(),
            revision: None,
            image_size: 224,
            normalize: true,
            preprocessor: ImagePreprocessor::default(),
        }
    }
}

impl VisionConfig {
    /// Create a new vision configuration with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model repository ID.
    #[must_use]
    pub fn with_model_repo(mut self, repo: impl Into<String>) -> Self {
        self.model_repo = repo.into();
        self
    }

    /// Set the revision/branch/tag.
    #[must_use]
    pub fn with_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Set the image size.
    #[must_use]
    pub const fn with_image_size(mut self, size: usize) -> Self {
        self.image_size = size;
        self
    }

    /// Set whether to normalize embeddings.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the preprocessing strategy.
    #[must_use]
    pub const fn with_preprocessor(mut self, preprocessor: ImagePreprocessor) -> Self {
        self.preprocessor = preprocessor;
        self
    }
}
